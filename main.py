import os
import logging
import time
import asyncio
import threading
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import pandas as pd
from google import genai
from google.genai import types
from huggingface_hub import HfApi, hf_hub_download
import tempfile
import traceback

# ================================
# ENV + Logging
# ================================
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
RENDER_URL = os.getenv("RENDER_EXTERNAL_URL")
GEMINI_API_TOKEN = os.getenv("GEMINI_API_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO_ID = "ahashanahmed/csv"

# ================================
# Flask (Health & Webhook)
# ================================
app = Flask(__name__)

@app.route('/')
def health():
    return jsonify({"status": "ok", "time": datetime.now().isoformat()})

# Telegram webhook endpoint
@app.route('/webhook', methods=['POST'])
async def webhook():
    data = request.get_json()
    update = Update.de_json(data, bot)
    await application.update_queue.put(update)
    return "ok"

# ================================
# HF Fallback
# ================================
class HFHandler:
    def __init__(self, repo_id, token):
        self.repo_id = repo_id
        self.token = token
        self.api = HfApi()

    def get_symbol_latest_rows(self, symbol, rows=400):
        try:
            files = self.api.list_repo_files(repo_id=self.repo_id, repo_type="dataset", token=self.token)
            csv_file = next((f for f in files if f.endswith(".csv")), None)
            if not csv_file:
                return None

            with tempfile.TemporaryDirectory() as temp:
                path = hf_hub_download(repo_id=self.repo_id, filename=csv_file, repo_type="dataset",
                                       token=self.token, local_dir=temp)
                df = pd.read_csv(path)
                col = next((c for c in df.columns if c.lower() in ["symbol","ticker"]), None)
                if not col:
                    return None
                df = df[df[col].astype(str).str.upper() == symbol.upper()]
                return df.tail(rows)
        except:
            return None

# ================================
# Stock Analyzer
# ================================
class StockAnalyzer:
    def __init__(self):
        self.gemini_client = genai.Client(api_key=GEMINI_API_TOKEN) if GEMINI_API_TOKEN else None
        self.hf_handler = HFHandler(HF_REPO_ID, HF_TOKEN) if HF_TOKEN else None
        self.global_request_times = []
        self.last_request_time = 0
        self.min_request_interval = 4
        self.max_requests_per_minute = 15

    def _check_rate_limit(self):
        now = time.time()
        self.global_request_times = [t for t in self.global_request_times if now - t < 60]
        if len(self.global_request_times) >= self.max_requests_per_minute:
            return False, "⚠️ Rate limit exceeded. Try later."
        return True, None

    def _record_request(self):
        self.global_request_times.append(time.time())

    def get_stock_data(self, symbol):
        if not self.hf_handler:
            return None
        return self.hf_handler.get_symbol_latest_rows(symbol, 400)

    def analyze(self, symbol, df):
        if not self.gemini_client or df is None:
            return "⚠️ Data/API error"

        ok, msg = self._check_rate_limit()
        if not ok:
            return msg

        wait = self.min_request_interval - (time.time() - self.last_request_time)
        if wait > 0:
            time.sleep(wait)

        try:
            df_tail = df.tail(200)
            csv_text = df_tail.to_csv(index=False)

            prompt = f"""📊 **{symbol} স্টক অ্যানালাইসিস**

ডাটা: {len(df_tail)} রো CSV ফরম্যাটে

{csv_text}

... (FULL TECHNICAL ANALYSIS PROMPT) ...
"""
            self.last_request_time = time.time()
            self._record_request()

            res = self.gemini_client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=prompt,
                config=types.GenerateContentConfig(max_output_tokens=4500, temperature=0.7)
            )
            return res.text if res else "⚠️ No response"

        except:
            logger.error(traceback.format_exc())
            return "⚠️ Server error"

# ================================
# Telegram Bot
# ================================
bot = Bot(token=TELEGRAM_BOT_TOKEN)
application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
analyzer = StockAnalyzer()
user_last = {}

async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    user_id = update.effective_user.id
    now = time.time()
    if user_id in user_last and now - user_last[user_id] < 5:
        return await update.message.reply_text("⏳ অপেক্ষা করুন...")

    user_last[user_id] = now
    msg = await update.message.reply_text("⏳ Processing...")

    try:
        df = analyzer.get_stock_data(text.upper())
        if df is None:
            return await msg.edit_text("❌ Data পাওয়া যায়নি")

        result = await asyncio.get_event_loop().run_in_executor(None, analyzer.analyze, text.upper(), df)
        await msg.edit_text(result[:4000])
    except:
        await msg.edit_text("❌ Error হয়েছে")

application.add_handler(MessageHandler(filters.TEXT, handle))

# ================================
# Webhook setup
# ================================
async def set_telegram_webhook():
    url = f"{RENDER_URL}/webhook"
    success = await bot.set_webhook(url)
    if success:
        logger.info(f"✅ Webhook set to {url}")
    else:
        logger.error("❌ Failed to set webhook")

# ================================
# MAIN
# ================================
if __name__ == "__main__":
    # Flask thread
    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)), threaded=True), daemon=True).start()
    
    # Bot thread
    def run_bot():
        asyncio.run(set_telegram_webhook())
        asyncio.run(application.initialize())
        asyncio.run(application.start())
        print("✅ Telegram Bot running")
        asyncio.run(application.updater.start_polling())  # fallback if webhook fails
        asyncio.run(asyncio.Event().wait())  # keep alive

    threading.Thread(target=run_bot, daemon=True).start()

    print("✅ Flask + Bot running")
    while True:
        time.sleep(60)