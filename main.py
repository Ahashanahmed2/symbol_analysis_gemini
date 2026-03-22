import os
import logging
import asyncio
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from google.genai import types
from telegram import Bot
from telegram.ext import Application, ContextTypes
from flask import Flask, request, jsonify
import pandas as pd
import tempfile
from huggingface_hub import hf_hub_download, HfApi
import traceback

# ================================
# ENV + Logging
# ================================
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_TOKEN = os.getenv("GEMINI_API_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")
RENDER_URL = os.getenv("RENDER_EXTERNAL_URL")

# ================================
# Flask App
# ================================
app = Flask(__name__)

@app.route("/")
def health():
    return jsonify({"status": "ok", "time": datetime.now().isoformat()})

# ================================
# Stock Analyzer
# ================================
class StockAnalyzer:
    def __init__(self):
        self.gemini_key = GEMINI_API_TOKEN
        self.hf_token = HF_TOKEN
        self.repo_id = "ahashanahmed/csv"
        self.gemini_model = "gemini-2.0-flash-exp"
        self.min_request_interval = 2
        self.global_request_times = []
        self.last_request_time = 0
        self.gemini_client = genai.Client(api_key=self.gemini_key) if self.gemini_key else None
        self.hf_handler = self._fallback() if self.hf_token else None

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

    async def analyze(self, symbol: str, df: pd.DataFrame):
        if not self.gemini_client or df is None:
            return "⚠️ Data/API error"
        # Rate limit
        now = asyncio.get_event_loop().time()
        wait = max(0, self.min_request_interval - (now - self.last_request_time))
        if wait > 0:
            await asyncio.sleep(wait)
        try:
            df_tail = df.tail(200)
            csv_text = df_tail.to_csv(index=False)
            prompt = f"""📊 {symbol} স্টক অ্যানালাইসিস

ডাটা: {len(df_tail)} রো CSV ফরম্যাটে

{csv_text}

সংক্ষিপ্ত প্রফেশনাল টেকনিক্যাল অ্যানালাইসিস:"""
            self.last_request_time = asyncio.get_event_loop().time()
            res = self.gemini_client.models.generate_content(
                model=self.gemini_model,
                contents=prompt,
                config=types.GenerateContentConfig(max_output_tokens=4500, temperature=0.7)
            )
            return res.text if res else "⚠️ No response"
        except:
            logger.error(traceback.format_exc())
            return "⚠️ Server error"

    def get_stock_data(self, symbol: str):
        if not self.hf_handler:
            return None
        return self.hf_handler.get_symbol_latest_rows(symbol, 400)

# ================================
# Telegram Webhook
# ================================
analyzer = StockAnalyzer()
bot = Bot(token=TELEGRAM_BOT_TOKEN)

@app.route("/webhook", methods=["POST"])
async def webhook():
    try:
        data = request.get_json()
        if "message" not in data:
            return jsonify({"ok": True})

        chat_id = data["message"]["chat"]["id"]
        text = data["message"].get("text")
        if not text:
            await bot.send_message(chat_id, "❌ শুধু text message পাঠান।")
            return jsonify({"ok": True})

        df = analyzer.get_stock_data(text.upper())
        if df is None:
            await bot.send_message(chat_id, "❌ Data পাওয়া যায়নি")
            return jsonify({"ok": True})

        result = await analyzer.analyze(text.upper(), df)
        await bot.send_message(chat_id, result[:4000])
        return jsonify({"ok": True})
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"ok": False, "error": str(e)})

# ================================
# Set webhook on startup
# ================================
def set_telegram_webhook():
    url = f"{RENDER_URL}/webhook"
    success = bot.set_webhook(url)
    if success:
        logger.info(f"✅ Webhook set to {url}")
    else:
        logger.error("❌ Failed to set webhook")

# ================================
# MAIN
# ================================
if __name__ == "__main__":
    set_telegram_webhook()
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, threaded=True)