import os
import httpx
import json
import logging
from typing import Optional

from fastapi import FastAPI, Request, Response, BackgroundTasks, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import AsyncOpenAI
import asyncpg

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="WhatsApp Product Assistant")

# Credentials
VERIFY_TOKEN = os.environ.get("WHATSAPP_VERIFY_TOKEN", "")
WHATSAPP_API_TOKEN = os.environ.get("WHATSAPP_API_TOKEN", "")
PHONE_NUMBER_ID = os.environ.get("WHATSAPP_PHONE_NUMBER_ID", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DATABASE_URL = os.environ.get("DATABASE_URL", "")
MERCHANT_ID = os.environ.get("MERCHANT_ID", "a6acb9a9-9551-4b7f-98eb-22702963ded7")

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)


def _clean_db_url(url: str) -> str:
    """asyncpg expects postgresql:// but the .env has postgresql+asyncpg://"""
    if url.startswith("postgresql+asyncpg://"):
        return url.replace("postgresql+asyncpg://", "postgresql://", 1)
    return url


@app.get("/webhook")
async def verify_webhook(request: Request):
    """
    Handles Meta's initial Webhook verification challenge.
    """
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    if mode and token:
        if mode == "subscribe" and token == VERIFY_TOKEN:
            logger.info("Webhook verified successfully.")
            return Response(content=challenge, media_type="text/plain")
        else:
            raise HTTPException(status_code=403, detail="Verification failed")
    
    raise HTTPException(status_code=400, detail="Missing parameters")


@app.post("/webhook")
async def receive_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Receives messages from WhatsApp users.
    Returns 200 OK immediately so Meta doesn't retry, and processes the message in the background.
    """
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return Response(status_code=400)

    # Check if it's a WhatsApp status update or a message
    if body.get("object") == "whatsapp_business_account":
        for entry in body.get("entry", []):
            for change in entry.get("changes", []):
                value = change.get("value", {})
                
                # We only care about messages, not message statuses (sent/delivered/read)
                if "messages" in value:
                    for msg in value["messages"]:
                        # Extract basic info
                        phone_number = value["contacts"][0]["wa_id"]
                        msg_id = msg.get("id")
                        
                        # Handle text only for this MVP
                        if msg.get("type") == "text":
                            text = msg["text"]["body"]
                            logger.info(f"Received message from {phone_number}: {text}")
                            
                            # Fire and forget
                            background_tasks.add_task(process_bot_message, phone_number, text, msg_id)

    # Always return 200 OK
    return Response(status_code=200, content="EVENT_RECEIVED", media_type="text/plain")


async def process_bot_message(phone_number: str, message_text: str, message_id: str):
    """
    Background worker that runs the RAG logic using pgvector and sends a reply back.
    """
    try:
        # 1. Embed user message
        embed_resp = await openai_client.embeddings.create(
            input=message_text,
            model="text-embedding-3-small"
        )
        query_vector = embed_resp.data[0].embedding

        # 2. Query Postgres pgvector directly
        db_url = _clean_db_url(DATABASE_URL)
        conn = await asyncpg.connect(db_url)
        
        # We search the core.product table for this specific merchant
        # using <=> (cosine distance) for the top 5 closest items.
        query = """
            SELECT p.name, p.description, pp.price, p.sku
            FROM core.product p
            LEFT JOIN core.product_price pp ON pp.product_id = p.id
            WHERE p.merchant_id = $1::uuid
              AND p.embedding IS NOT NULL
              AND p.is_active = true
            ORDER BY p.embedding <=> $2::vector
            LIMIT 5
        """
        # asyncpg requires the vector string representation
        vector_str = str(query_vector)
        rows = await conn.fetch(query, MERCHANT_ID, vector_str)
        await conn.close()
        
        # 3. Build knowledge base context
        context_parts = []
        for r in rows:
            desc = r['description'] if r['description'] else "Sin descripción"
            price = float(r['price']) if r['price'] else 0.0
            context_parts.append(f"- {r['name']} (SKU: {r['sku']}) | Precio: ${price} | Specs: {desc}")
            
        kb_text = "\n".join(context_parts) if context_parts else "No se encontraron productos relacionados."
        logger.info(f"Semantic Search Context:\n{kb_text}")

        system_prompt = f"""
        Eres el amigable asistente virtual de ventas de Tu Caserito. 
        Tu tarea es responder preguntas de clientes en WhatsApp sobre nuestros productos.
        
        Usa ESTRICTAMENTE la siguiente lista de productos de nuestro inventario para responder al cliente.
        Si la respuesta no está en la lista de productos, di amablemente que no tienes esa información o que 
        lo verificarás más tarde (nunca inventes precios ni productos).
        Intenta ser conciso, usa emojis de vez en cuando, y siempre fomenta la venta.
        
        INVENTARIO DISPONIBLE (contexto actual):
        {kb_text}
        """

        # 4. Fetch conversation history from Postgres
        history_query = """
            SELECT m.direction, m.body
            FROM comm.thread_message m
            JOIN comm.thread t ON m.thread_id = t.id
            WHERE t.external_address = $1
              AND t.merchant_id = $2::uuid
            ORDER BY m.created_at DESC
            LIMIT 10
        """
        conn = await asyncpg.connect(_clean_db_url(DATABASE_URL))
        history_rows = await conn.fetch(history_query, phone_number, MERCHANT_ID)
        await conn.close()

        # Build prompt messages
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # history_rows are DESC (newest first). We need to append them chronologically.
        for row in reversed(history_rows):
            role_map = {"in": "user", "out": "assistant"}
            role = role_map.get(row['direction'], "user")
            body = row['body'] or ""
            # Don't add completely empty messages
            if body.strip():
                messages.append({"role": role, "content": body})
                
        # Append the new current message
        messages.append({"role": "user", "content": message_text})

        # 5. Ask OpenAI to draft the reply as a salesperson
        chat_resp = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3
        )
        
        reply_text = chat_resp.choices[0].message.content

        # 5. Send reply back to WhatsApp
        await send_whatsapp_message(phone_number, reply_text)

    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)
        # Optional: Send a fallback error message to the user, but usually better to fail silently in async


async def send_whatsapp_message(to_number: str, message_body: str):
    """
    Sends a text message using Meta Graph API.
    """
    if not WHATSAPP_API_TOKEN or not PHONE_NUMBER_ID:
        logger.warning(f"Simulating sending message to {to_number}: {message_body}")
        return

    url = f"https://graph.facebook.com/v18.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_API_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to_number,
        "type": "text",
        "text": {
            "preview_url": False,
            "body": message_body
        }
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(url, headers=headers, json=payload, timeout=10.0)
        
        if resp.status_code not in (200, 201):
            logger.error(f"Failed to send WA message: {resp.text}")
        else:
            logger.info(f"Reply successfully sent to {to_number}")

