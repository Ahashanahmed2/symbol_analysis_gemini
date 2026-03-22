import os
import logging
import time
import asyncio
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from google.genai import types
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram import Update
import pandas as pd
import tempfile
from huggingface_hub import hf_hub_download, HfApi
import requests
import io
import traceback
import threading
from flask import Flask, jsonify

# ================================
# Flask (Render Health Check)
# ================================
app = Flask(__name__)

@app.route('/')
def health():
    return jsonify({
        "status": "ok",
        "time": datetime.now().isoformat()
    })

def run_flask():
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, threaded=True)

# ================================
# ENV + Logging
# ================================
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================================
# HF Import
# ================================
try:
    from hf_uploader import get_symbol_latest_data, HFStreamingHandler
except:
    get_symbol_latest_data = None
    HFStreamingHandler = None

# ================================
# Stock Analyzer
# ================================
class StockAnalyzer:

    def __init__(self):
        self.gemini_key = os.getenv("GEMINI_API_TOKEN")
        self.hf_token = os.getenv("HF_TOKEN") or os.getenv("hf_token")
        self.repo_id = "ahashanahmed/csv"

        self.gemini_model = "gemini-2.0-flash-exp"

        self.min_request_interval = 4
        self.max_requests_per_minute = 15
        self.global_request_times = []
        self.last_request_time = 0

        self.last_symbol = {}
        self.last_data = {}

        self.gemini_client = None
        if self.gemini_key:
            self.gemini_client = genai.Client(api_key=self.gemini_key)

        if self.hf_token:
            if HFStreamingHandler:
                self.hf_handler = HFStreamingHandler(repo_id=self.repo_id, token=self.hf_token)
            else:
                self.hf_handler = self._fallback()
        else:
            self.hf_handler = None

    # -------------------------------
    def _fallback(self):
        class Fallback:
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
                        path = hf_hub_download(
                            repo_id=self.repo_id,
                            filename=csv_file,
                            repo_type="dataset",
                            token=self.token,
                            local_dir=temp
                        )

                        df = pd.read_csv(path)

                        col = next((c for c in df.columns if c.lower() in ["symbol","ticker"]), None)
                        if not col:
                            return None

                        df = df[df[col].astype(str).str.upper() == symbol.upper()]
                        return df.tail(rows)

                except:
                    return None

        return Fallback(self.repo_id, self.hf_token)

    # -------------------------------
    def _check_rate_limit(self):
        now = time.time()
        self.global_request_times = [t for t in self.global_request_times if now - t < 60]

        if len(self.global_request_times) >= self.max_requests_per_minute:
            return False, "⚠️ Rate limit exceeded. Try later."

        return True, None

    def _record_request(self):
        self.global_request_times.append(time.time())

    # -------------------------------
    def get_stock_data(self, symbol):
        try:
            if not self.hf_handler:
                return None

            if get_symbol_latest_data:
                return get_symbol_latest_data(symbol=symbol, rows=400, repo_id=self.repo_id, token=self.hf_token)

            return self.hf_handler.get_symbol_latest_rows(symbol, 400)

        except:
            return None

    # -------------------------------
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

            prompt = self._create_analysis_prompt(symbol, df_tail, csv_text)

            self.last_request_time = time.time()
            self._record_request()

            res = self.gemini_client.models.generate_content(
                model=self.gemini_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=4500,
                    temperature=0.7
                )
            )

            return res.text if res else "⚠️ No response"

        except:
            logger.error(traceback.format_exc())
            return "⚠️ Server error"

    # ❗ PROMPT UNCHANGED
    def _create_analysis_prompt(self, symbol, df, csv_text):
        return f"""📊 **{symbol} স্টক অ্যানালাইসিস - প্রফেশনাল টেকনিক্যাল অ্যানালাইসিস**

ডাটা: {len(df)} রো CSV ফরম্যাটে (সর্বশেষ ৪০০ দিনের ডাটা)

{csv_text}

... (YOUR ORIGINAL PROMPT EXACT SAME) ...
"""

    # -------------------------------
    def answer_question(self, symbol, df, question):
        try:
            df_tail = df.tail(100)
            csv_text = df_tail.to_csv(index=False)

            prompt = f"""📊 {symbol} স্টকের সর্বশেষ {len(df_tail)} টি ডাটা:

{csv_text}

ইউজারের প্রশ্ন: {question}

এই ডাটার ভিত্তিতে সংক্ষিপ্ত কিন্তু বিস্তারিত উত্তর দিন।"""

            res = self.gemini_client.models.generate_content(
                model=self.gemini_model,
                contents=prompt
            )

            return res.text if res else "⚠️ No response"

        except:
            return "⚠️ Error"


# ================================
# Telegram Bot
# ================================
class TelegramBot:

    def __init__(self):
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.analyzer = StockAnalyzer()
        self.app = None
        self.user_last = {}

    def setup(self):
        self.app = Application.builder().token(self.token).build()
        self.app.add_handler(MessageHandler(filters.TEXT, self.handle))

    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        text = update.message.text.strip()
        user_id = update.effective_user.id

        now = time.time()
        if user_id in self.user_last and now - self.user_last[user_id] < 5:
            return await update.message.reply_text("⏳ অপেক্ষা করুন...")

        self.user_last[user_id] = now

        msg = await update.message.reply_text("⏳ Processing...")

        try:
            df = self.analyzer.get_stock_data(text.upper())
            if df is None:
                return await msg.edit_text("❌ Data পাওয়া যায়নি")

            self.analyzer.last_data[user_id] = df.tail(200)

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self.analyzer.analyze, text.upper(), df)

            await msg.edit_text(result[:4000])

        except:
            await msg.edit_text("❌ Error হয়েছে")


# ================================
# BOT START (Render safe)
# ================================
async def start_bot():
    bot = TelegramBot()
    bot.setup()

    await bot.app.initialize()
    await bot.app.start()

    print("✅ Bot started")

    while True:
        await asyncio.sleep(60)


# ================================
# MAIN (Render)
# ================================
if __name__ == "__main__":

    print("🚀 Starting Render service...")

    # Flask thread
    threading.Thread(target=run_flask, daemon=True).start()

    # Bot thread
    def run_bot():
        asyncio.run(start_bot())

    threading.Thread(target=run_bot, daemon=True).start()

    print("✅ Flask + Bot running")

    # keep alive
    while True:
        time.sleep(60)