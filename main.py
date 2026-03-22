import os
import asyncio
import logging
import tempfile
import traceback
from datetime import datetime
import threading
import json

import pandas as pd
from flask import Flask, request, jsonify
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
# Flask app for UptimeRobot
# ================================
flask_app = Flask(__name__)

@flask_app.route('/')
def health():
    """Health check endpoint for UptimeRobot"""
    return jsonify({
        'status': 'active',
        'message': 'Stock Analysis Bot is running!',
        'timestamp': datetime.now().isoformat()
    })

@flask_app.route('/health')
def health_check():
    """Detailed health check"""
    return jsonify({
        'status': 'healthy',
        'bot_status': 'active',
        'timestamp': datetime.now().isoformat()
    }), 200

@flask_app.route('/ping')
def ping():
    """Ping endpoint for UptimeRobot"""
    return jsonify({'status': 'pong'}), 200

@flask_app.route('/webhook', methods=['POST'])
def webhook():
    """Telegram webhook endpoint"""
    try:
        data = request.get_json()
        if not data:
            return 'No data', 400
        
        # Process update in background to avoid blocking
        def process_update():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                update = Update.de_json(data, bot_application.bot)
                loop.run_until_complete(bot_application.update_queue.put(update))
            except Exception as e:
                logger.error(f"Error processing update: {e}")
            finally:
                loop.close()
        
        thread = threading.Thread(target=process_update)
        thread.start()
        
        return 'ok', 200
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return 'error', 500

def run_flask():
    """Run Flask server in a separate thread"""
    port = int(os.environ.get('PORT', 10000))
    flask_app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

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
    """Start command handler"""
    await update.message.reply_text(
        "👋 হ্যালো! বট চালু আছে।\n"
        "স্টক সিম্বল পাঠান যেমন `AAPL` বিশ্লেষণের জন্য।\n\n"
        "উদাহরণ: `AAPL`, `GOOGL`, `TSLA`"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Help command handler"""
    await update.message.reply_text(
        "📚 **সাহায্য গাইড**\n\n"
        "**কিভাবে ব্যবহার করবেন:**\n"
        "1. স্টক সিম্বল পাঠান (যেমন: AAPL)\n"
        "2. বট ডাটা বিশ্লেষণ করবে\n"
        "3. বিস্তারিত বিশ্লেষণ দেখাবে\n\n"
        "**উপলব্ধ কমান্ড:**\n"
        "/start - বট চালু করুন\n"
        "/help - সাহায্য দেখুন\n"
        "/about - বট সম্পর্কে জানুন\n\n"
        "**সাপোর্টেড স্টক:**\n"
        "যেকোনো স্টক সিম্বল সাপোর্ট করে যা ডাটাসেটে আছে।",
        parse_mode='Markdown'
    )

async def about_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """About command handler"""
    await update.message.reply_text(
        "🤖 **স্টক অ্যানালাইসিস বট**\n\n"
        "এই বটটি AI ব্যবহার করে স্টক মার্কেট বিশ্লেষণ করে।\n\n"
        "**ফিচারসমূহ:**\n"
        "• জেমিনি AI দ্বারা বিশ্লেষণ\n"
        "• রিয়েল-টাইম ডাটা প্রসেসিং\n"
        "• বিস্তারিত স্টক অ্যানালাইসিস\n"
        "• বাংলা ভাষায় রিপোর্ট\n\n"
        "**ক্রেডিট:**\n"
        "পাওয়ার্ড বাই: জেমিনি 2.0 ফ্ল্যাশ\n"
        "ডাটা সোর্স: Hugging Face Datasets",
        parse_mode='Markdown'
    )

async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text messages"""
    text = update.message.text.strip().upper()
    user_id = update.effective_user.id

    # Flood control
    now = asyncio.get_event_loop().time()
    if user_id in user_last and now - user_last[user_id] < 5:
        await update.message.reply_text("⏳ একটু অপেক্ষা করুন...")
        return
    user_last[user_id] = now

    # Send processing message
    msg = await update.message.reply_text("⏳ ডাটা সংগ্রহ করা হচ্ছে...")
    
    # Get stock data
    df = await analyzer.get_stock_data(text)
    if df is None or df.empty:
        await msg.edit_text(
            f"❌ `{text}` সিম্বলের জন্য ডাটা পাওয়া যায়নি।\n\n"
            "দয়া করে সঠিক স্টক সিম্বল ব্যবহার করুন।",
            parse_mode='Markdown'
        )
        return

    # Update message
    await msg.edit_text("🤖 AI বিশ্লেষণ করা হচ্ছে...")
    
    try:
        result = await analyzer.analyze(text, df)
        # Split long messages if needed
        if len(result) > 4096:
            for i in range(0, len(result), 4096):
                await msg.edit_text(result[i:i+4096])
                if i == 0:
                    msg = await update.message.reply_text("📊 **বাকি বিশ্লেষণ:**")
        else:
            await msg.edit_text(result[:4000])
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        await msg.edit_text("❌ বিশ্লেষণ করতে সমস্যা হয়েছে। পরে আবার চেষ্টা করুন।")

# Add handlers
bot_application.add_handler(CommandHandler("start", start))
bot_application.add_handler(CommandHandler("help", help_command))
bot_application.add_handler(CommandHandler("about", about_command))
bot_application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle))

# ================================
# Main function with webhook setup
# ================================
async def setup_webhook():
    """Set up the webhook for Telegram"""
    if RENDER_URL:
        webhook_url = f"{RENDER_URL}/webhook"
        try:
            await bot_application.bot.set_webhook(webhook_url)
            logger.info(f"✅ Webhook set to {webhook_url}")
            
            # Verify webhook
            webhook_info = await bot_application.bot.get_webhook_info()
            logger.info(f"Webhook info: {webhook_info.url}")
        except Exception as e:
            logger.error(f"Failed to set webhook: {e}")
    else:
        logger.warning("RENDER_EXTERNAL_URL not set, webhook not configured")

async def run_bot():
    """Run the bot with polling (for local development)"""
    logger.info("🤖 Starting bot with polling...")
    await bot_application.initialize()
    await bot_application.start()
    await bot_application.updater.start_polling()
    
    # Keep running
    while True:
        await asyncio.sleep(1)

async def main():
    """Main function to run both Flask and bot"""
    logger.info("🚀 Starting Stock Analysis Bot...")
    
    # Start Flask server in separate thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    logger.info(f"🌐 Flask server started on port {os.environ.get('PORT', 10000)}")
    
    # Setup webhook if running on Render
    if RENDER_URL:
        logger.info("📡 Running on Render, setting up webhook...")
        await setup_webhook()
        
        # Start the bot application with webhook
        await bot_application.initialize()
        await bot_application.start()
        
        # Keep the bot running
        logger.info("✅ Bot is running with webhook mode")
        while True:
            await asyncio.sleep(1)
    else:
        # Local development - use polling
        logger.info("💻 Local development mode, using polling...")
        await run_bot()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")