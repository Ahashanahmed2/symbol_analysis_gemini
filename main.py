import os
import asyncio
import logging
import tempfile
import traceback
from datetime import datetime
import threading
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
    return jsonify({
        'status': 'active',
        'message': 'Stock Analysis Bot is running!',
        'timestamp': datetime.now().isoformat()
    })

@flask_app.route('/health')
def health_check():
    return jsonify({'status': 'healthy'}), 200

@flask_app.route('/ping')
def ping():
    return jsonify({'status': 'pong'}), 200

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
# Stock Analyzer with Better Error Handling
# ================================
class StockAnalyzer:
    def __init__(self):
        # Initialize Gemini with proper error handling
        self.gemini_client = None
        if GEMINI_API_TOKEN:
            try:
                self.gemini_client = genai.Client(api_key=GEMINI_API_TOKEN)
                logger.info("✅ Gemini client initialized successfully")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Gemini: {e}")
        
        self.repo_id = "ahashanahmed/csv"
        self.hf_api = HfApi() if HF_TOKEN else None
        self.hf_token = HF_TOKEN
        self.min_request_interval = 2
        self.max_requests_per_minute = 15
        self.global_request_times = []

    async def get_stock_data(self, symbol: str, rows=400):
        """Get stock data from Hugging Face"""
        try:
            logger.info(f"Fetching data for {symbol}...")
            
            # List files in the dataset
            files = self.hf_api.list_repo_files(
                repo_id=self.repo_id, 
                repo_type="dataset", 
                token=self.hf_token
            )
            
            # Find CSV file
            csv_file = next((f for f in files if f.endswith(".csv")), None)
            if not csv_file:
                logger.error("No CSV file found")
                return None

            # Download and read CSV
            with tempfile.TemporaryDirectory() as temp:
                path = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=csv_file,
                    repo_type="dataset",
                    token=self.hf_token,
                    local_dir=temp
                )
                
                # Try different encodings
                try:
                    df = pd.read_csv(path, encoding='utf-8')
                except:
                    try:
                        df = pd.read_csv(path, encoding='latin-1')
                    except:
                        df = pd.read_csv(path, encoding='utf-8-sig')
                
                # Find symbol column
                symbol_col = None
                possible_cols = ['symbol', 'Symbol', 'SYMBOL', 'ticker', 'Ticker', 
                                'stock', 'Stock', 'name', 'Name', 'code', 'Code']
                
                for col in possible_cols:
                    if col in df.columns:
                        symbol_col = col
                        break
                
                if not symbol_col:
                    # Use first column as symbol column
                    symbol_col = df.columns[0]
                    logger.warning(f"No symbol column found, using: {symbol_col}")
                
                # Convert to string and filter
                df[symbol_col] = df[symbol_col].astype(str).str.strip().str.upper()
                filtered_df = df[df[symbol_col] == symbol.upper()]
                
                if filtered_df.empty:
                    # Try partial match
                    filtered_df = df[df[symbol_col].str.contains(symbol.upper(), na=False)]
                    if filtered_df.empty:
                        logger.warning(f"No data found for {symbol}")
                        return None
                
                logger.info(f"Found {len(filtered_df)} rows for {symbol}")
                return filtered_df.tail(rows)
                
        except Exception as e:
            logger.error(f"Error fetching data: {traceback.format_exc()}")
            return None

    async def analyze(self, symbol: str, df: pd.DataFrame):
        """Analyze stock data using Gemini AI"""
        
        # Check if Gemini client is available
        if not self.gemini_client:
            error_msg = "⚠️ Gemini API not configured. Please set GEMINI_API_TOKEN environment variable."
            logger.error(error_msg)
            return error_msg
        
        if df is None or df.empty:
            return "⚠️ No data available for analysis"

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
            
            # Get data summary
            columns = list(df_tail.columns)
            numeric_cols = df_tail.select_dtypes(include=['number']).columns.tolist()
            
            # Create a summary instead of sending all data
            summary = f"""
📊 **Stock: {symbol}**
📅 Total Records: {len(df_tail)}
📋 Columns: {', '.join(columns[:10])}

"""
            
            # Add numeric summary if available
            if numeric_cols:
                summary += f"📈 **Numeric Columns Summary:**\n"
                for col in numeric_cols[:5]:  # First 5 numeric columns
                    try:
                        summary += f"• {col}: Min={df_tail[col].min():.2f}, Max={df_tail[col].max():.2f}, Mean={df_tail[col].mean():.2f}\n"
                    except:
                        pass
                summary += "\n"
            
            # Add sample data (last 20 rows only)
            sample_data = df_tail.tail(20).to_string()
            
            prompt = f"""{summary}

**Last 20 Records:**
{sample_data}

Please provide a comprehensive analysis in Bengali covering:
1. 📈 Price trends and patterns
2. 🎯 Support and resistance levels
3. ⚡ Volatility assessment
4. 💡 Trading recommendation (Buy/Sell/Hold)
5. ⚠️ Risk factors to consider

Keep the analysis concise but informative. Use emojis and bullet points.
"""

            # Call Gemini API with timeout
            logger.info(f"Sending request to Gemini for {symbol}...")
            
            try:
                response = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.gemini_client.models.generate_content(
                            model="gemini-2.0-flash-lite",
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                max_output_tokens=3000,
                                temperature=0.7
                            )
                        )
                    ),
                    timeout=30.0  # 30 second timeout
                )
                
                if response and response.text:
                    logger.info(f"Gemini analysis successful for {symbol}")
                    return response.text
                else:
                    return "⚠️ No response from AI model"
                    
            except asyncio.TimeoutError:
                logger.error(f"Gemini API timeout for {symbol}")
                return "⚠️ Gemini API timeout. Please try again."
                
        except Exception as e:
            logger.error(f"Analysis error: {traceback.format_exc()}")
            return f"⚠️ Error: {str(e)}"

# ================================
# Telegram Bot Setup
# ================================
analyzer = StockAnalyzer()
bot_application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
user_last = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    text = f"""
👋 **স্বাগতম {user.first_name}!**

🤖 আমি একটি **স্টক অ্যানালাইসিস বট**।

**কিভাবে ব্যবহার করবেন:**
1. 📈 যেকোনো স্টক সিম্বল পাঠান (যেমন: GP, AAPL)
2. 📊 আমি Hugging Face থেকে ডাটা আনব
3. 🤖 Gemini AI দিয়ে বিশ্লেষণ করব
4. 📝 বিস্তারিত রিপোর্ট দেব

**কমান্ডসমূহ:**
/start - বট চালু করুন
/help - সাহায্য দেখুন
/about - বট সম্পর্কে জানুন
/test - API চেক করুন

এখন আপনার পছন্দের স্টক সিম্বল পাঠান!
"""
    await update.message.reply_text(text, parse_mode='Markdown')

async def test_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Test API connectivity"""
    await update.message.reply_text("🔍 API চেক করা হচ্ছে...")
    
    # Check Gemini API
    if analyzer.gemini_client:
        try:
            test_response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: analyzer.gemini_client.models.generate_content(
                    model="gemini-2.0-flash-lite",
                    contents="Say 'API working'",
                    config=types.GenerateContentConfig(max_output_tokens=10)
                )
            )
            if test_response and test_response.text:
                await update.message.reply_text("✅ Gemini API is working!")
            else:
                await update.message.reply_text("❌ Gemini API returned empty response")
        except Exception as e:
            await update.message.reply_text(f"❌ Gemini API error: {str(e)}")
    else:
        await update.message.reply_text("❌ Gemini API not configured")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = """
📚 **সাহায্য গাইড**

**কিভাবে ব্যবহার করবেন:**
1. স্টক সিম্বল পাঠান (যেমন: GP, AAPL)
2. বট ডাটা সংগ্রহ করবে
3. AI বিশ্লেষণ করবে
4. বিস্তারিত রিপোর্ট দেবে

**উপলব্ধ কমান্ড:**
/start - বট চালু করুন
/help - এই সাহায্য দেখুন
/about - বট সম্পর্কে জানুন
/test - API চেক করুন

**সাপোর্টেড স্টক:**
GP, ACI, BEXIMCO, SQUARE, BRACBANK, CITYBANK সহ আরও অনেক
"""
    await update.message.reply_text(text, parse_mode='Markdown')

async def about_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = """
🤖 **স্টক অ্যানালাইসিস বট v1.0**

**টেকনোলজি স্ট্যাক:**
• 🐍 Python 3.11
• 🤖 Telegram Bot API
• 📊 Hugging Face Datasets
• 🧠 Google Gemini AI
• 🌐 Flask (Health Check)

**ফিচারসমূহ:**
✅ স্টক ডাটা সংগ্রহ
✅ AI-পাওয়ার্ড অ্যানালাইসিস
✅ বাংলা ভাষায় রিপোর্ট
✅ UptimeRobot মনিটরিং
✅ ফ্লাড কন্ট্রোল

**ক্রেডিট:**
• Data Source: Hugging Face
• AI Model: gemini-2.0-flash-lite
• Hosting: Render.com
"""
    await update.message.reply_text(text, parse_mode='Markdown')

async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle stock symbol messages"""
    symbol = update.message.text.strip().upper()
    user_id = update.effective_user.id

    # Flood control
    now = asyncio.get_event_loop().time()
    if user_id in user_last and now - user_last[user_id] < 5:
        await update.message.reply_text("⏳ একটু অপেক্ষা করুন...")
        return
    user_last[user_id] = now

    # Send initial message
    msg = await update.message.reply_text(
        f"🔍 **{symbol}** বিশ্লেষণ করা হচ্ছে...\n\n"
        "📡 **ধাপ 1/3:** Hugging Face থেকে ডাটা সংগ্রহ করা হচ্ছে...",
        parse_mode='Markdown'
    )
    
    try:
        # Get stock data
        await msg.edit_text(
            f"📊 **{symbol}**\n\n"
            "✅ **ধাপ 1/3:** ডাটা সংগ্রহ সম্পূর্ণ!\n"
            "🤖 **ধাপ 2/3:** AI বিশ্লেষণ করা হচ্ছে...\n"
            "⏳ ৫-১০ সেকেন্ড সময় লাগতে পারে..."
        )
        
        df = await analyzer.get_stock_data(symbol, rows=300)
        
        if df is None or df.empty:
            await msg.edit_text(
                f"❌ **{symbol}** সিম্বলের জন্য ডাটা পাওয়া যায়নি।\n\n"
                "💡 সঠিক স্টক সিম্বল ব্যবহার করুন যেমন: GP, ACI, BEXIMCO",
                parse_mode='Markdown'
            )
            return
        
        # Analyze with Gemini
        analysis = await analyzer.analyze(symbol, df)
        
        # Send result
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
        logger.error(f"Error: {traceback.format_exc()}")
        await msg.edit_text(
            f"❌ **{symbol}** বিশ্লেষণ করতে সমস্যা হয়েছে!\n\n"
            f"ত্রুটি: {str(e)[:200]}\n\n"
            "পরে আবার চেষ্টা করুন।"
        )

# Add handlers
bot_application.add_handler(CommandHandler("start", start))
bot_application.add_handler(CommandHandler("help", help_command))
bot_application.add_handler(CommandHandler("about", about_command))
bot_application.add_handler(CommandHandler("test", test_command))
bot_application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle))

# ================================
# Main
# ================================
async def setup_webhook():
    if RENDER_URL:
        webhook_url = f"{RENDER_URL}/webhook"
        try:
            await bot_application.bot.set_webhook(webhook_url)
            logger.info(f"✅ Webhook set to {webhook_url}")
        except Exception as e:
            logger.error(f"Failed to set webhook: {e}")

async def main():
    logger.info("🚀 Starting Stock Analysis Bot...")
    
    # Start Flask
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    
    # Check API keys
    if not GEMINI_API_TOKEN:
        logger.error("❌ GEMINI_API_TOKEN not set!")
    if not TELEGRAM_BOT_TOKEN:
        logger.error("❌ TELEGRAM_BOT_TOKEN not set!")
    
    if RENDER_URL:
        await setup_webhook()
        await bot_application.initialize()
        await bot_application.start()
        logger.info("✅ Bot running with webhook")
        while True:
            await asyncio.sleep(1)
    else:
        await bot_application.initialize()
        await bot_application.start()
        await bot_application.updater.start_polling()
        logger.info("✅ Bot running with polling")
        while True:
            await asyncio.sleep(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped")