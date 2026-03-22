import os
import asyncio
import logging
import tempfile
import traceback
from datetime import datetime

import pandas as pd
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from google import genai
from google.genai import types
from huggingface_hub import hf_hub_download, HfApi

# ================================
# ENV + Logging
# ================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_TOKEN = os.getenv("GEMINI_API_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")
RENDER_URL = os.getenv("RENDER_EXTERNAL_URL")

# ================================
# Stock Analyzer
# ================================
class StockAnalyzer:
    def __init__(self):
        self.gemini_client = genai.Client(api_key=GEMINI_API_TOKEN) if GEMINI_API_TOKEN else None
        self.repo_id = "ahashanahmed/csv"
        self.hf_api = HfApi() if HF_TOKEN else None
        self.hf_token = HF_TOKEN
        self.min_request_interval = 4
        self.max_requests_per_minute = 15
        self.global_request_times = []

    async def get_stock_data(self, symbol: str, rows=400):
        try:
            files = self.hf_api.list_repo_files(repo_id=self.repo_id, repo_type="dataset", token=self.hf_token)
            csv_file = next((f for f in files if f.endswith(".csv")), None)
            if not csv_file:
                return None

            with tempfile.TemporaryDirectory() as temp:
                path = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=csv_file,
                    repo_type="dataset",
                    token=self.hf_token,
                    local_dir=temp
                )
                df = pd.read_csv(path)
                col = next((c for c in df.columns if c.lower() in ["symbol", "ticker"]), None)
                if not col:
                    return None
                df = df[df[col].astype(str).str.upper() == symbol.upper()]
                return df.tail(rows)
        except Exception:
            logger.error(traceback.format_exc())
            return None

    async def analyze(self, symbol: str, df: pd.DataFrame):
        if not self.gemini_client or df is None:
            return "⚠️ Data/API error"

        now = asyncio.get_event_loop().time()
        self.global_request_times = [t for t in self.global_request_times if now - t < 60]
        if len(self.global_request_times) >= self.max_requests_per_minute:
            return "⚠️ Rate limit exceeded. Try later."

        wait = self.min_request_interval - (now - self.global_request_times[-1]) if self.global_request_times else 0
        if wait > 0:
            await asyncio.sleep(wait)

        self.global_request_times.append(asyncio.get_event_loop().time())

        try:
            df_tail = df.tail(200)
            csv_text = df_tail.to_csv(index=False)
            prompt = f"📊 {symbol} স্টক অ্যানালাইসিস\nডাটা:\n{csv_text}\nসংক্ষিপ্ত কিন্তু বিশদ বিশ্লেষণ দিন।"

            res = self.gemini_client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=prompt,
                config=types.GenerateContentConfig(max_output_tokens=4500, temperature=0.7)
            )
            return res.text if res else "⚠️ No response"

        except Exception:
            logger.error(traceback.format_exc())
            return "⚠️ Server error"

# ================================
# Telegram Bot Setup
# ================================
analyzer = StockAnalyzer()
bot_application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
user_last = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 হ্যালো! বট চালু আছে।\nস্টক সিম্বল পাঠান যেমন `AAPL` বিশ্লেষণের জন্য।"
    )

async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip().upper()
    user_id = update.effective_user.id

    now = asyncio.get_event_loop().time()
    if user_id in user_last and now - user_last[user_id] < 5:
        await update.message.reply_text("⏳ একটু অপেক্ষা করুন...")
        return
    user_last[user_id] = now

    msg = await update.message.reply_text("⏳ প্রসেসিং হচ্ছে...")
    df = await analyzer.get_stock_data(text)
    if df is None or df.empty:
        await msg.edit_text("❌ ডাটা পাওয়া যায়নি")
        return

    try:
        result = await analyzer.analyze(text, df)
        await msg.edit_text(result[:4000])
    except Exception:
        await msg.edit_text("❌ ত্রুটি হয়েছে")

bot_application.add_handler(CommandHandler("start", start))
bot_application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle))

# ================================
# ASGI App (Hypercorn compatible)
# ================================
from hypercorn.config import Config
from hypercorn.asyncio import serve

async def app(scope, receive, send):
    """ASGI app that handles both webhook and health checks"""
    
    if scope["type"] == "http":
        path = scope["path"]
        method = scope["method"]
        
        # Health check endpoint
        if path == "/" and method == "GET":
            await send({
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json")],
            })
            body = f'{{"status": "ok", "time": "{datetime.now().isoformat()}"}}'.encode()
            await send({
                "type": "http.response.body",
                "body": body,
            })
            return
        
        # Webhook endpoint
        elif path == "/webhook" and method == "POST":
            # Read request body
            body = b""
            more_body = True
            while more_body:
                message = await receive()
                if message["type"] == "http.request":
                    body += message.get("body", b"")
                    more_body = message.get("more_body", False)
            
            try:
                import json
                data = json.loads(body)
                update = Update.de_json(data, bot_application.bot)
                await bot_application.update_queue.put(update)
                
                await send({
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"text/plain")],
                })
                await send({
                    "type": "http.response.body",
                    "body": b"ok",
                })
            except Exception as e:
                logger.error(f"Webhook error: {e}")
                await send({
                    "type": "http.response.start",
                    "status": 500,
                    "headers": [(b"content-type", b"text/plain")],
                })
                await send({
                    "type": "http.response.body",
                    "body": b"error",
                })
            return
        
        # 404
        else:
            await send({
                "type": "http.response.start",
                "status": 404,
                "headers": [(b"content-type", b"text/plain")],
            })
            await send({
                "type": "http.response.body",
                "body": b"Not found",
            })
            return

# ================================
# Main
# ================================
async def main():
    """Main function to start the server"""
    # Set webhook
    webhook_url = f"{RENDER_URL}/webhook"
    await bot_application.bot.set_webhook(webhook_url)
    print(f"✅ Webhook set to {webhook_url}")
    print(f"🚀 Server starting on port {os.environ.get('PORT', 10000)}")
    
    # Start the ASGI server
    config = Config()
    port = int(os.environ.get("PORT", 10000))
    config.bind = [f"0.0.0.0:{port}"]
    await serve(app, config)

if __name__ == "__main__":
    asyncio.run(main())