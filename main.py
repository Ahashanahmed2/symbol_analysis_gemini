import os
import logging
import asyncio
import tempfile
import traceback
from datetime import datetime

import pandas as pd
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
from google import genai
from google.genai import types
from huggingface_hub import hf_hub_download, HfApi

# ================================
# ENV + Logging
# ================================
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_TOKEN = os.getenv("GEMINI_API_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")
RENDER_URL = os.getenv("RENDER_EXTERNAL_URL")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================================
# Flask app (health check)
# ================================
app = Flask(__name__)

@app.route("/")
def health():
    return jsonify({"status": "ok", "time": datetime.now().isoformat()})

# ================================
# Stock Analyzer (async + rate-limit)
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
                col = next((c for c in df.columns if c.lower() in ["symbol","ticker"]), None)
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

        # Rate limit
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
# Telegram Bot (async webhook)
# ================================
analyzer = StockAnalyzer()
application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
user_last = {}  # flood control

async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip().upper()
    user_id = update.effective_user.id

    now = asyncio.get_event_loop().time()
    if user_id in user_last and now - user_last[user_id] < 5:
        await update.message.reply_text("⏳ অপেক্ষা করুন...")
        return
    user_last[user_id] = now

    msg = await update.message.reply_text("⏳ Processing...")
    df = await analyzer.get_stock_data(text)
    if df is None:
        await msg.edit_text("❌ Data পাওয়া যায়নি")
        return

    try:
        result = await analyzer.analyze(text, df)
        await msg.edit_text(result[:4000])
    except Exception:
        await msg.edit_text("❌ Error হয়েছে")

application.add_handler(MessageHandler(filters.TEXT, handle))

# ================================
# Webhook endpoint
# ================================
@app.route("/webhook", methods=["POST"])
async def webhook():
    data = request.get_json()
    update = Update.de_json(data, application.bot)
    await application.update_queue.put(update)
    return "ok"

# ================================
# Main async startup
# ================================
if __name__ == "__main__":
    import hypercorn.asyncio
    from hypercorn.config import Config

    async def main():
        # Set Telegram webhook
        await application.bot.set_webhook(f"{RENDER_URL}/webhook")
        print(f"✅ Webhook set to {RENDER_URL}/webhook")

        # Serve Flask async
        config = Config()
        config.bind = ["0.0.0.0:10000"]
        await hypercorn.asyncio.serve(app, config)

    asyncio.run(main())