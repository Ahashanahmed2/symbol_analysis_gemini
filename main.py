import os
import logging
import asyncio
import traceback
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from google.genai import types
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from flask import Flask, request, jsonify
import pandas as pd
import tempfile
from huggingface_hub import hf_hub_download, HfApi

# ================================
# ENV + Logging
# ================================
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_KEY = os.getenv("GEMINI_API_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("hf_token")
REPO_ID = "ahashanahmed/csv"

# ================================
# Flask App
# ================================
app = Flask(__name__)

@app.route('/')
def health():
    return jsonify({"status": "ok", "time": datetime.now().isoformat()})

# ================================
# Stock Analyzer
# ================================
class StockAnalyzer:

    def __init__(self):
        self.gemini_client = None
        if GEMINI_KEY:
            self.gemini_client = genai.Client(api_key=GEMINI_KEY)
        self.hf_handler = self._hf_fallback() if HF_TOKEN else None
        self.min_request_interval = 4
        self.max_requests_per_minute = 15
        self.global_request_times = []
        self.last_request_time = 0

    def _hf_fallback(self):
        class Fallback:
            def __init__(self, repo_id, token):
                self.repo_id = repo_id
                self.token = token
                self.api = HfApi()
            def get_symbol_latest_rows(self, symbol, rows=400):
                try:
                    files = self.api.list_repo_files(repo_id=self.repo_id, repo_type="dataset", token=self.token)
                    csv_file = next((f for f in files if f.endswith(".csv")), None)
                    if not csv_file: return None
                    with tempfile.TemporaryDirectory() as temp:
                        path = hf_hub_download(repo_id=self.repo_id, filename=csv_file,
                                               repo_type="dataset", token=self.token, local_dir=temp)
                        df = pd.read_csv(path)
                        col = next((c for c in df.columns if c.lower() in ["symbol","ticker"]), None)
                        if not col: return None
                        df = df[df[col].astype(str).str.upper() == symbol.upper()]
                        return df.tail(rows)
                except:
                    return None
        return Fallback(REPO_ID, HF_TOKEN)

    def _check_rate_limit(self):
        import time
        now = time.time()
        self.global_request_times = [t for t in self.global_request_times if now - t < 60]
        if len(self.global_request_times) >= self.max_requests_per_minute:
            return False, "⚠️ Rate limit exceeded. Try later."
        return True, None

    def _record_request(self):
        import time
        self.global_request_times.append(time.time())

    def get_stock_data(self, symbol):
        try:
            if not self.hf_handler:
                return None
            return self.hf_handler.get_symbol_latest_rows(symbol, 400)
        except:
            return None

    def analyze(self, symbol, df):
        import time
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
            prompt = f"📊 {symbol} স্টক অ্যানালাইসিস (সর্বশেষ {len(df_tail)} দিন):\n\n{csv_text}\n\nসংক্ষেপে বিশ্লেষণ দিন।"
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
# Telegram Bot (Webhook)
# ================================
analyzer = StockAnalyzer()
bot = Bot(token=TELEGRAM_TOKEN)
application = Application.builder().bot(bot).build()

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip().upper()
    msg = await update.message.reply_text("⏳ Processing...")
    try:
        df = analyzer.get_stock_data(text)
        if df is None:
            return await msg.edit_text("❌ Data পাওয়া যায়নি")
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, analyzer.analyze, text, df)
        await msg.edit_text(result[:4000])
    except Exception as e:
        logger.error(traceback.format_exc())
        await msg.edit_text("❌ Error হয়েছে")

application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

# ================================
# Flask Webhook Endpoint
# ================================
@app.route(f'/webhook/{TELEGRAM_TOKEN}', methods=['POST'])
def webhook():
    from telegram import Update
    import asyncio
    try:
        update = Update.de_json(request.get_json(force=True), bot)
        asyncio.run(application.update(update))
    except Exception:
        logger.error(traceback.format_exc())
    return "ok"

# ================================
# Set Webhook (once)
# ================================
def set_webhook():
    webhook_url = f"{os.getenv('RENDER_EXTERNAL_URL')}/webhook/{TELEGRAM_TOKEN}"
    res = bot.set_webhook(webhook_url)
    logger.info(f"Webhook set: {res}")

# ================================
# MAIN
# ================================
if __name__ == "__main__":
    set_webhook()
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, threaded=True)