import os
import asyncio
import logging
import tempfile
import traceback
from datetime import datetime
import threading
import pandas as pd
from flask import Flask, request, jsonify, send_file
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
from huggingface_hub import hf_hub_download, HfApi
import io
import urllib.parse
import difflib

# ================================
# এনভায়রনমেন্ট + লগিং
# ================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")
RENDER_URL = os.getenv("RENDER_EXTERNAL_URL")

# ================================
# ফ্লাস্ক অ্যাপ
# ================================
flask_app = Flask(__name__)

@flask_app.route('/')
def health():
    return jsonify({
        'status': 'active',
        'message': 'স্টক অ্যানালাইসিস বট সচল আছে!',
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
            return 'কোন ডাটা নেই', 400

        def process_update():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                update = Update.de_json(data, bot_application.bot)
                loop.run_until_complete(bot_application.update_queue.put(update))
            except Exception as e:
                logger.error(f"আপডেট প্রসেসিং এ সমস্যা: {e}")
            finally:
                loop.close()

        thread = threading.Thread(target=process_update)
        thread.start()
        return 'ok', 200
    except Exception as e:
        logger.error(f"ওয়েবহুক ত্রুটি: {e}")
        return 'error', 500

def run_flask():
    port = int(os.environ.get('PORT', 10000))
    flask_app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

# ================================
# স্টক ডাটা ফেচার
# ================================
class StockDataFetcher:
    def __init__(self):
        self.repo_id = "ahashanahmed/csv"
        self.filename = "mongodb.csv"
        self.hf_api = HfApi() if HF_TOKEN else None
        self.hf_token = HF_TOKEN
        self.all_symbols = None

    def _find_symbol_column(self, df):
        possible_columns = ['symbol', 'Symbol', 'SYMBOL', 'ticker', 'Ticker', 
                           'stock', 'Stock', 'name', 'Name', 'code', 'Code']

        for col in possible_columns:
            if col in df.columns:
                return col

        for col in df.columns:
            if df[col].dtype == 'object':
                sample = df[col].dropna().astype(str).head(10)
                if all(2 <= len(str(x)) <= 15 for x in sample):
                    return col

        return df.columns[0] if len(df.columns) > 0 else None

    def _find_date_column(self, df):
        possible_columns = ['date', 'Date', 'DATE', 'datetime', 'DateTime', 
                           'timestamp', 'Timestamp', 'time', 'Time']

        for col in possible_columns:
            if col in df.columns:
                return col

        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                return col

        return None

    async def get_all_symbols(self):
        if self.all_symbols is not None:
            return self.all_symbols

        try:
            if not self.hf_token:
                return []

            with tempfile.TemporaryDirectory() as temp:
                path = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=self.filename,
                    repo_type="dataset",
                    token=self.hf_token,
                    local_dir=temp,
                    local_dir_use_symlinks=False
                )

                df = pd.read_csv(path, nrows=50000, encoding='utf-8-sig')

                symbol_col = self._find_symbol_column(df)
                if symbol_col:
                    symbols = df[symbol_col].astype(str).str.strip().str.upper().unique()
                    symbols = [s for s in symbols if s and s != 'nan' and len(s) > 0]
                    self.all_symbols = sorted(symbols)
                    logger.info(f"মোট {len(self.all_symbols)}টি সিম্বল পাওয়া গেছে")

                return self.all_symbols

        except Exception as e:
            logger.error(f"সিম্বল লিস্ট পেতে সমস্যা: {e}")
            return []

    def find_similar_symbols(self, query: str, limit=10):
        """query এর সাথে মিলে এমন সিম্বল খুঁজে বের করে"""
        if not self.all_symbols:
            return []

        query_upper = query.upper()

        exact_matches = [s for s in self.all_symbols if s == query_upper]
        if exact_matches:
            return exact_matches[:limit]

        partial_matches = [s for s in self.all_symbols if query_upper in s]

        fuzzy_matches = difflib.get_close_matches(query_upper, self.all_symbols, n=limit, cutoff=0.6)

        all_matches = list(dict.fromkeys(partial_matches + fuzzy_matches))

        return all_matches[:limit]

    async def get_stock_data(self, symbol: str, rows=400):
        try:
            logger.info(f"{symbol} এর জন্য ডাটা সংগ্রহ করা হচ্ছে...")

            if not self.hf_token:
                return None, 0

            with tempfile.TemporaryDirectory() as temp:
                path = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=self.filename,
                    repo_type="dataset",
                    token=self.hf_token,
                    local_dir=temp,
                    local_dir_use_symlinks=False
                )

                try:
                    df = pd.read_csv(path, encoding='utf-8-sig')
                except:
                    try:
                        df = pd.read_csv(path, encoding='utf-8')
                    except:
                        df = pd.read_csv(path, encoding='latin-1')

                symbol_col = self._find_symbol_column(df)
                if not symbol_col:
                    logger.error("কোন সিম্বল কলাম পাওয়া যায়নি!")
                    return None, 0

                df[symbol_col] = df[symbol_col].astype(str).str.strip().str.upper()

                filtered_df = df[df[symbol_col] == symbol.upper()]

                if filtered_df.empty:
                    logger.info(f"{symbol} এর জন্য কোন ডাটা পাওয়া যায়নি")
                    return None, 0

                total_rows = len(filtered_df)
                logger.info(f"{symbol} এর জন্য মোট {total_rows}টি রো পাওয়া গেছে")

                date_col = self._find_date_column(filtered_df)
                if date_col:
                    filtered_df[date_col] = pd.to_datetime(filtered_df[date_col], errors='coerce')
                    filtered_df = filtered_df.sort_values(date_col, ascending=False)
                    latest_df = filtered_df.head(rows).copy()
                    latest_df = latest_df.sort_values(date_col, ascending=True)
                else:
                    latest_df = filtered_df.tail(rows).copy()

                logger.info(f"{symbol} এর জন্য {len(latest_df)}টি সর্বশেষ রো নেওয়া হয়েছে")
                return latest_df, total_rows

        except Exception as e:
            logger.error(f"ডাটা ফেচিং এ সমস্যা: {traceback.format_exc()}")
            return None, 0

    def create_full_prompt(self, symbol: str, df: pd.DataFrame, total_rows: int):
        """সম্পূর্ণ প্রম্পট তৈরি করুন - সম্পূর্ণ ডাটা সহ"""
        if df is None or df.empty:
            return None

        timestamp = datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")

        # সম্পূর্ণ ডাটা টেক্সট ফরম্যাটে কনভার্ট করুন
        df_clean = df.copy()
        df_clean = df_clean.fillna('')
        
        # সম্পূর্ণ ডাটা স্ট্রিং - এটাই আগে ছিল না
        data_string = df_clean.to_string()
        
        columns_list = list(df.columns)

        price_col = None
        for col in ['close', 'Close', 'price', 'Price', 'last', 'Last']:
            if col in df.columns:
                price_col = col
                break

        current_price = df[price_col].iloc[-1] if price_col and len(df) > 0 else 'N/A'
        date_col = self._find_date_column(df)

        time_period = ""
        if date_col and date_col in df.columns:
            start_date = df[date_col].min()
            end_date = df[date_col].max()
            time_period = f"📅 সময়কাল: {start_date} থেকে {end_date}"

        # সম্পূর্ণ প্রম্পট - সম্পূর্ণ ডাটা সহ
        prompt = f"""🤖 **ভূমিকা:** আপনি একজন বিশ্বসেরা প্রফেশনাল টেকনিক্যাল অ্যানালিস্ট, চার্ট রিডার এবং ট্রেডার। আপনার কাজ হলো প্রদত্ত OHLCV ডাটা, প্রাইস মুভমেন্ট এবং মার্কেট স্ট্রাকচার বিশ্লেষণ করে একটি পূর্ণাঙ্গ, প্রমাণভিত্তিক এবং অ্যাকশনেবল টেকনিক্যাল রিপোর্ট তৈরি করা। আপনার প্রতিটি মন্তব্য যুক্তিসঙ্গত, ডাটা-ড্রিভেন এবং প্যাটার্ন-ভিত্তিক হতে হবে।

⚠️ **গুরুত্বপূর্ণ নির্দেশনা:** আপনার সম্পূর্ণ উত্তর **বাংলা ভাষায়** দিন। ইমোজি ব্যবহার করুন। মার্কডাউন ফরম্যাটে উত্তর দিন।

📊 **ইনপুট ডাটা:**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• **সিম্বল:** {symbol}
• **মোট উপলব্ধ রেকর্ড:** {total_rows}টি
• **বিশ্লেষণে ব্যবহৃত:** {len(df)}টি সর্বশেষ রেকর্ড
• {time_period}
• **বর্তমান মূল্য:** {current_price}
• **কলাম সংখ্যা:** {len(df.columns)}টি
• **কলাম সমূহ:** {', '.join(columns_list[:15])}{'...' if len(columns_list) > 15 else ''}

📋 **সম্পূর্ণ ডাটা (সব কলাম সহ):**