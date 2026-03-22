import os
import asyncio
import logging
import traceback
from datetime import datetime
import threading
import pandas as pd
from flask import Flask, request, jsonify
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from google import genai
from google.genai import types

# Import from your hf_uploader
from hf_uploader import HFStreamingHandler, get_symbol_latest_data

# ================================
# ENV + Logging
# ================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_TOKEN = os.getenv("GEMINI_API_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")
RENDER_URL = os.getenv("RENDER_EXTERNAL_URL")
REPO_ID = "ahashanahmed/csv"

# ================================
# Flask app for UptimeRobot
# ================================
flask_app = Flask(__name__)

@flask_app.route('/')
def health():
    return jsonify({
        'status': 'active',
        'message': 'Stock Analysis Bot is running!',
        'timestamp': datetime.now().isoformat()
    })

@flask_app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'bot_status': 'active',
        'timestamp': datetime.now().isoformat()
    }), 200

@flask_app.route('/ping')
def ping():
    return jsonify({'status': 'pong'}), 200

@flask_app.route('/debug')
def debug():
    """Debug endpoint to check symbols"""
    try:
        handler = HFStreamingHandler(repo_id=REPO_ID, token=HF_TOKEN)
        symbols = handler.get_symbol_list(max_symbols=50)
        return jsonify({
            'total_symbols': len(symbols),
            'sample_symbols': symbols[:20]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@flask_app.route('/webhook', methods=['POST'])
def webhook():
    try:
        data = request.get_json()
        if not data:
            return 'No data', 400

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
    port = int(os.environ.get('PORT', 10000))
    flask_app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

# ================================
# Stock Analyzer - Using HFStreamingHandler
# ================================
class StockAnalyzer:
    def __init__(self):
        self.gemini_client = genai.Client(api_key=GEMINI_API_TOKEN) if GEMINI_API_TOKEN else None
        self.hf_handler = HFStreamingHandler(repo_id=REPO_ID, token=HF_TOKEN)
        self.min_request_interval = 2
        self.max_requests_per_minute = 20
        self.global_request_times = []
        self.symbols_cache = None

    async def get_available_symbols(self):
        """Get list of available symbols"""
        if self.symbols_cache is None:
            try:
                # Run in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                self.symbols_cache = await loop.run_in_executor(
                    None, 
                    self.hf_handler.get_symbol_list, 
                    200  # max symbols
                )
                logger.info(f"Loaded {len(self.symbols_cache)} symbols")
            except Exception as e:
                logger.error(f"Error loading symbols: {e}")
                self.symbols_cache = []
        return self.symbols_cache

    async def get_stock_data(self, symbol: str, rows=400):
        """Get stock data using HFStreamingHandler"""
        try:
            logger.info(f"Fetching data for {symbol}...")
            
            # Run synchronous method in thread pool
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(
                None,
                self.hf_handler.get_symbol_latest_rows,
                symbol,
                rows
            )
            
            if df is not None and len(df) > 0:
                logger.info(f"Found {len(df)} rows for {symbol}")
                return df
            else:
                logger.warning(f"No data found for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching data: {traceback.format_exc()}")
            return None

    async def analyze(self, symbol: str, df: pd.DataFrame):
        """Analyze stock data using Gemini AI"""
        if not self.gemini_client or df is None:
            return "⚠️ Data/API error"

        # Rate limiting
        now = asyncio.get_event_loop().time()
        self.global_request_times = [t for t in self.global_request_times if now - t < 60]
        
        if len(self.global_request_times) >= self.max_requests_per_minute:
            return "⚠️ Rate limit exceeded. Please try again in a minute."
        
        if self.global_request_times:
            wait = self.min_request_interval - (now - self.global_request_times[-1])
            if wait > 0:
                await asyncio.sleep(wait)
        
        self.global_request_times.append(asyncio.get_event_loop().time())

        try:
            # Prepare data for analysis
            df_tail = df.tail(200)
            
            # Get column info
            columns = list(df_tail.columns)
            date_col = None
            for col in columns:
                if 'date' in col.lower() or 'time' in col.lower():
                    date_col = col
                    break
            
            # Create summary
            summary = f"""
📊 **Stock: {symbol}**
📅 Total Records: {len(df_tail)}
📋 Columns: {', '.join(columns[:8])}

"""
            
            if date_col:
                summary += f"📅 Date Range: {df_tail[date_col].min()} to {df_tail[date_col].max()}\n\n"
            
            # Add sample data
            csv_text = df_tail.to_csv(index=False)
            
            prompt = f"""{summary}

**Sample Data (last {len(df_tail)} records):**
{csv_text[:3500]}

Please provide a comprehensive analysis in Bengali covering:
1. Price trends and patterns
2. Support and resistance levels
3. Volume analysis (if available)
4. Technical indicators summary
5. Trading recommendation (Buy/Sell/Hold)
6. Risk factors to consider

Keep the analysis concise but informative. Use emojis and bullet points for better readability."""

            res = self.gemini_client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=4000,
                    temperature=0.7
                )
            )
            return res.text if res else "⚠️ No response from AI"

        except Exception as e:
            logger.error(f"Analysis error: {traceback.format_exc()}")
            return f"⚠️ Error during analysis: {str(e)}"

# ================================
# Telegram Bot Setup
# ================================
analyzer = StockAnalyzer()
bot_application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
user_last = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start command handler"""
    user = update.effective_user
    
    # Get available symbols
    symbols = await analyzer.get_available_symbols()
    sample_symbols = symbols[:10] if symbols else ["AAPL", "GOOGL", "TSLA"]
    
    text = f"""
👋 **স্বাগতম {user.first_name}!** 

🤖 আমি একটি **স্টক অ্যানালাইসিস বট**।

**কিভাবে ব্যবহার করবেন:**
1. 📈 যেকোনো স্টক সিম্বল পাঠান
2. 📊 আমি Hugging Face থেকে ডাটা আনব
3. 🤖 Gemini AI দিয়ে বিশ্লেষণ করব
4. 📝 বিস্তারিত রিপোর্ট দেব

**উপলব্ধ সিম্বল (স্যাম্পল):**
`{'`, `'.join(sample_symbols)}`

**কমান্ডসমূহ:**
/start - বট চালু করুন
/help - সাহায্য দেখুন
/symbols - সব সিম্বল দেখুন
/about - বট সম্পর্কে জানুন

এখন আপনার পছন্দের স্টক সিম্বল পাঠান!
"""
    await update.message.reply_text(text, parse_mode='Markdown')

async def symbols_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show available symbols"""
    await update.message.reply_text("🔄 সিম্বল লিস্ট লোড হচ্ছে...")
    
    symbols = await analyzer.get_available_symbols()
    
    if symbols:
        # Group symbols in chunks
        chunks = [symbols[i:i+20] for i in range(0, len(symbols), 20)]
        
        text = f"📊 **মোট {len(symbols)} টি সিম্বল উপলব্ধ:**\n\n"
        
        for i, chunk in enumerate(chunks[:5]):  # Show first 100 symbols
            text += f"`{'`, `'.join(chunk)}`\n\n"
        
        if len(chunks) > 5:
            text += f"\n... এবং আরও {len(symbols) - 100} টি সিম্বল"
        
        await update.message.reply_text(text, parse_mode='Markdown')
    else:
        await update.message.reply_text("❌ সিম্বল লিস্ট লোড করতে পারেনি")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Help command handler"""
    text = """
📚 **সাহায্য গাইড**

**কিভাবে ব্যবহার করবেন:**
1. স্টক সিম্বল পাঠান (যেমন: AAPL, GOOGL)
2. বট ডাটা সংগ্রহ করবে
3. AI বিশ্লেষণ করবে
4. বিস্তারিত রিপোর্ট দেবে

**উপলব্ধ কমান্ড:**
/start - বট চালু করুন
/help - এই সাহায্য দেখুন
/symbols - সব সিম্বল দেখুন
/about - বট সম্পর্কে জানুন

**টিপস:**
• সিম্বল বড় হাতের অক্ষরে লিখুন
• সঠিক সিম্বল ব্যবহার করুন
• প্রতিটি রিকোয়েস্টের মধ্যে ৫ সেকেন্ড ব্যবধান রাখুন
"""
    await update.message.reply_text(text, parse_mode='Markdown')

async def about_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """About command handler"""
    text = """
🤖 **স্টক অ্যানালাইসিস বট v2.0**

**টেকনোলজি স্ট্যাক:**
• 🐍 Python 3.11
• 🤖 Telegram Bot API
• 📊 Hugging Face Datasets (Streaming)
• 🧠 Google Gemini AI
• 🌐 Flask (Health Check)

**ফিচারসমূহ:**
✅ স্মার্ট ডাটা স্ট্রিমিং
✅ AI-পাওয়ার্ড অ্যানালাইসিস
✅ বাংলা ভাষায় রিপোর্ট
✅ UptimeRobot মনিটরিং
✅ ফ্লাড কন্ট্রোল
✅ 400+ সিম্বল সাপোর্ট

**ক্রেডিট:**
• Data Source: Hugging Face (ahashanahmed/csv)
• AI Model: Gemini 2.0 Flash
• Hosting: Render.com

**লিমিটেশন:**
• প্রতি মিনিটে ২০টি রিকোয়েস্ট
• বিশ্লেষণে ৫-১০ সেকেন্ড সময় লাগে
"""
    await update.message.reply_text(text, parse_mode='Markdown')

async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle stock symbol messages"""
    symbol = update.message.text.strip().upper()
    user_id = update.effective_user.id

    # Flood control
    now = asyncio.get_event_loop().time()
    if user_id in user_last and now - user_last[user_id] < 5:
        await update.message.reply_text("⏳ **একটু অপেক্ষা করুন!**\nআগের রিকোয়েস্ট প্রসেসিং হচ্ছে...", parse_mode='Markdown')
        return
    user_last[user_id] = now

    # Send initial message
    msg = await update.message.reply_text(
        f"🔍 **{symbol}** বিশ্লেষণ করা হচ্ছে...\n\n"
        "📡 **ধাপ 1/3:** Hugging Face থেকে ডাটা সংগ্রহ করা হচ্ছে...",
        parse_mode='Markdown'
    )
    
    try:
        # Get stock data using HFStreamingHandler
        await msg.edit_text(
            f"📊 **{symbol}**\n\n"
            "✅ **ধাপ 1/3:** ডাটা সংগ্রহ সম্পূর্ণ!\n"
            "🤖 **ধাপ 2/3:** AI বিশ্লেষণ করা হচ্ছে...\n"
            "⏳ ৫-১০ সেকেন্ড সময় লাগতে পারে..."
        )
        
        df = await analyzer.get_stock_data(symbol, rows=400)
        
        if df is None or df.empty:
            # Show available symbols for suggestion
            symbols = await analyzer.get_available_symbols()
            similar = [s for s in symbols if symbol in s][:5]
            
            suggestion = ""
            if similar:
                suggestion = f"\n\n💡 **আপনি কি বোঝাতে চেয়েছেন?**\n`{'`, `'.join(similar)}`"
            
            await msg.edit_text(
                f"❌ **{symbol}** এর জন্য কোনো ডাটা পাওয়া যায়নি!\n\n"
                f"🔍 **কারণ:**\n"
                f"• সিম্বলটি সঠিক নাও হতে পারে\n"
                f"• ডাটাসেটে এই সিম্বল নেই{suggestion}\n\n"
                f"💡 **সঠিক সিম্বল জানতে:** `/symbols` কমান্ড ব্যবহার করুন",
                parse_mode='Markdown'
            )
            return
        
        # Analyze with Gemini
        await msg.edit_text(
            f"🧠 **{symbol}**\n\n"
            "✅ **ধাপ 2/3:** ডাটা সংগ্রহ সম্পূর্ণ!\n"
            "🤖 **ধাপ 3/3:** AI বিশ্লেষণ করা হচ্ছে...\n"
            "⏳ শেষ ধাপ, আরও কয়েক সেকেন্ড..."
        )
        
        analysis = await analyzer.analyze(symbol, df)
        
        # Send final result
        if analysis.startswith("⚠️"):
            await msg.edit_text(f"⚠️ **{symbol}**\n\n{analysis}")
        else:
            final_message = f"📈 **{symbol} বিশ্লেষণ রিপোর্ট**\n\n{analysis}"
            
            if len(final_message) > 4096:
                await msg.delete()
                for i in range(0, len(final_message), 4096):
                    await update.message.reply_text(
                        final_message[i:i+4096],
                        parse_mode='Markdown'
                    )
            else:
                await msg.edit_text(final_message, parse_mode='Markdown')
                
    except Exception as e:
        logger.error(f"Error processing {symbol}: {traceback.format_exc()}")
        await msg.edit_text(
            f"❌ **{symbol}** বিশ্লেষণ করতে সমস্যা হয়েছে!\n\n"
            f"🔧 **ত্রুটি:** {str(e)[:200]}\n\n"
            "পরে আবার চেষ্টা করুন।"
        )

# ================================
# Main Setup
# ================================
def setup_bot():
    bot_application.add_handler(CommandHandler("start", start))
    bot_application.add_handler(CommandHandler("help", help_command))
    bot_application.add_handler(CommandHandler("symbols", symbols_command))
    bot_application.add_handler(CommandHandler("about", about_command))
    bot_application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle))

async def setup_webhook():
    if RENDER_URL:
        webhook_url = f"{RENDER_URL}/webhook"
        try:
            await bot_application.bot.set_webhook(webhook_url)
            logger.info(f"✅ Webhook set to {webhook_url}")
            webhook_info = await bot_application.bot.get_webhook_info()
            logger.info(f"Webhook info: {webhook_info.url}")
        except Exception as e:
            logger.error(f"Failed to set webhook: {e}")

async def main():
    logger.info("🚀 Starting Stock Analysis Bot...")
    
    # Start Flask server
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    logger.info(f"🌐 Flask server started")
    
    # Setup bot
    setup_bot()
    
    if RENDER_URL:
        logger.info("📡 Running on Render, setting up webhook...")
        await setup_webhook()
        await bot_application.initialize()
        await bot_application.start()
        logger.info("✅ Bot is running with webhook mode")
        while True:
            await asyncio.sleep(1)
    else:
        logger.info("💻 Local mode, using polling...")
        await bot_application.initialize()
        await bot_application.start()
        await bot_application.updater.start_polling()
        while True:
            await asyncio.sleep(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped")
    except Exception as e:
        logger.error(f"Fatal error: {e}")