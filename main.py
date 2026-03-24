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
        self.hf_token = HF_TOKEN
        self.all_symbols = []
        self.df_cache = None

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

    async def load_all_symbols(self):
        """সব সিম্বল লোড করুন"""
        try:
            if not self.hf_token:
                logger.error("HF_TOKEN সেট করা নেই!")
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
                
                self.df_cache = pd.read_csv(path, encoding='utf-8-sig')
                logger.info(f"মোট {len(self.df_cache)}টি রো লোড করা হয়েছে")
                
                symbol_col = self._find_symbol_column(self.df_cache)
                if symbol_col:
                    symbols = self.df_cache[symbol_col].astype(str).str.strip().str.upper().unique()
                    symbols = [s for s in symbols if s and s != 'nan' and len(s) > 0]
                    self.all_symbols = sorted(symbols)
                    logger.info(f"মোট {len(self.all_symbols)}টি সিম্বল পাওয়া গেছে")
                
                return self.all_symbols
                
        except Exception as e:
            logger.error(f"সিম্বল লোড করতে সমস্যা: {traceback.format_exc()}")
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
        """সিম্বল অনুযায়ী ডাটা ফিল্টার করুন"""
        try:
            logger.info(f"{symbol} এর জন্য ডাটা সংগ্রহ করা হচ্ছে...")
            
            if self.df_cache is None:
                with tempfile.TemporaryDirectory() as temp:
                    path = hf_hub_download(
                        repo_id=self.repo_id,
                        filename=self.filename,
                        repo_type="dataset",
                        token=self.hf_token,
                        local_dir=temp,
                        local_dir_use_symlinks=False
                    )
                    self.df_cache = pd.read_csv(path, encoding='utf-8-sig')
            
            symbol_col = self._find_symbol_column(self.df_cache)
            if not symbol_col:
                logger.error("কোন সিম্বল কলাম পাওয়া যায়নি!")
                return None, 0
            
            # সিম্বল কলাম আপারকেসে কনভার্ট করুন
            self.df_cache[symbol_col] = self.df_cache[symbol_col].astype(str).str.strip().str.upper()
            
            # ফিল্টার করুন
            filtered_df = self.df_cache[self.df_cache[symbol_col] == symbol.upper()]
            
            if filtered_df.empty:
                logger.info(f"{symbol} এর জন্য কোন ডাটা পাওয়া যায়নি")
                return None, 0
            
            total_rows = len(filtered_df)
            logger.info(f"{symbol} এর জন্য মোট {total_rows}টি রো পাওয়া গেছে")
            
            # ডেট কলাম অনুযায়ী সাজান
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

    def create_full_file(self, symbol: str, df: pd.DataFrame, total_rows: int):
        """সম্পূর্ণ ডাটা + প্রম্পট সহ ফাইল তৈরি করুন"""
        if df is None or df.empty:
            return None
        
        timestamp = datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")
        
        # সম্পূর্ণ ডাটা টেক্সট ফরম্যাটে কনভার্ট করুন
        df_clean = df.copy()
        df_clean = df_clean.fillna('')
        
        # ডাটা স্ট্রিং তৈরি করুন
        data_string = df_clean.to_string()
        logger.info(f"ডাটা স্ট্রিং তৈরি করা হয়েছে: {len(data_string)} অক্ষর")
        
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
        
        # সম্পূর্ণ ফাইল কন্টেন্ট তৈরি করুন
        file_content = f"""🤖 **ভূমিকা:** আপনি একজন বিশ্বসেরা প্রফেশনাল টেকনিক্যাল অ্যানালিস্ট, চার্ট রিডার এবং ট্রেডার। আপনার কাজ হলো প্রদত্ত OHLCV ডাটা, প্রাইস মুভমেন্ট এবং মার্কেট স্ট্রাকচার বিশ্লেষণ করে একটি পূর্ণাঙ্গ, প্রমাণভিত্তিক এবং অ্যাকশনেবল টেকনিক্যাল রিপোর্ট তৈরি করা। আপনার প্রতিটি মন্তব্য যুক্তিসঙ্গত, ডাটা-ড্রিভেন এবং প্যাটার্ন-ভিত্তিক হতে হবে।

⚠️ **গুরুত্বপূর্ণ নির্দেশনা:** আপনার সম্পূর্ণ উত্তর **বাংলা ভাষায়** দিন। ইমোজি ব্যবহার করুন। মার্কডাউন ফরম্যাটে উত্তর দিন। dsex স্টক মার্কেট,সর্ট সেল নায়।

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

{data_string}


═══════════════════════════════════════════════════════════
🌊 **সেকশন ১: এলিয়ট ওয়েভ অ্যানালাইসিস (সম্পূর্ণ প্যাটার্ন লাইব্রেরি)**
═══════════════════════════════════════════════════════════

【মোটিভ ওয়েভ প্যাটার্নস - ট্রেন্ড ডিরেকশন】
□ ইম্পালস ওয়েভ (Impulse Wave) - ৫টি সাব-ওয়েভ (১-২-৩-৪-৫)
  → Wave 1: ইনিশিয়াল মুভমেন্ট, ভলিউম কনফার্মেশন চেক
  → Wave 2: শ্যালো/ডিপ রিট্রেসমেন্ট, ইনভ্যালিডেশন লেভেল
  → Wave 3: এক্সটেনশন পটেনশিয়াল, স্ট্রং মোমেন্টাম
  → Wave 4: কমপ্লেক্স/সিম্পল কারেকশন, ট্রায়াঙ্গেল পটেনশিয়াল
  → Wave 5: ডাইভারজেন্স চেক, ট্রাঙ্কেশন পটেনশিয়াল

□ ডায়াগোনাল প্যাটার্নস:
  → লিডিং ডায়াগোনাল (Leading Diagonal): Wave 1 বা A তে, ৫-৩-৫-৩-৫ স্ট্রাকচার
  → এন্ডিং ডায়াগোনাল (Ending Diagonal): Wave 5 বা C তে, টার্মিনাল প্যাটার্ন

□ ওয়েভ এক্সটেনশন ভেরিয়েশনস:
  → 3rd Wave Extension: মোস্ট কমন, ১.৬১৮-২.৬১৮ ফিবো টার্গেট
  → 5th Wave Extension: টার্মিনাল মুভমেন্ট

【কারেক্টিভ ওয়েভ প্যাটার্নস - কাউন্টার-ট্রেন্ড】
□ জিগজ্যাগ ফ্যামিলি (৫-৩-৫):
  → সিঙ্গেল জিগজ্যাগ: শার্প কারেকশন
  → ডাবল জিগজ্যাগ (W-X-Y): দুটি জিগজ্যাগ + এক্স ওয়েভ

□ ফ্ল্যাট ফ্যামিলি (৩-৩-৫):
  → রেগুলার ফ্ল্যাট: B=A, C=A
  → এক্সপ্যান্ডেড ফ্ল্যাট: B>A, C>B
  → রানিং ফ্ল্যাট: B>A, C<A

□ ট্রায়াঙ্গেল ফ্যামিলি (৩-৩-৩-৩-৩):
  → কন্ট্রাক্টিং ট্রায়াঙ্গেল: ক্রমশ সংকুচিত
  → এক্সপ্যান্ডিং ট্রায়াঙ্গেল: ক্রমশ প্রসারিত

【এলিয়ট ওয়েভ ভ্যালিডেশন চেকলিস্ট】
✓ ওয়েভ ২ কখনো ওয়েভ ১ এর ১০০% রিট্রেস করতে পারবে না
✓ ওয়েভ ৩ কখনো শর্টেস্ট হতে পারবে না
✓ ওয়েভ ৪ কখনো ওয়েভ ১ এর প্রাইস জোনে ঢুকবে না

বিশেষ ভাবে দিতে হবে নিচের গুলো
১.কোন ওয়েবের কোন সাব-ওয়েবে আছে। 
২. সাব-ওয়েবের পর কোন সাব-ওয়েব আসতে পারে। 
৩. ওয়েবের পর কোন ওয়েব আসবে।
৪. এন্টি-নেওয়ার সময় ও কোথায় নিতে হবে।
৫. চার্ট,টেবিলে ওয়েব,সাব-ওয়েব ও ওয়েবের সময়
    *এলিয়ট ওয়েব বিশ্লেষণ ২ ভাবে হবে*
৬. লার্জার স্ট্রাকচার (Higher Timeframe)】
৭.【বর্তমান স্ট্রাকচার (Current Position)】
═══════════════════════════════════════════════════════════
💰 **সেকশন ২: স্মার্ট মানি কনসেপ্টস (SMC)**
═══════════════════════════════════════════════════════════

【অর্ডার ব্লক (Order Block)】
  → বুলিশ OB: লাস্ট ডাউন ক্যান্ডেল আপট্রেন্ডে
  → বেয়ারিশ OB: লাস্ট আপ ক্যান্ডেল ডাউনট্রেন্ডে
  → ব্রেকার ব্লক: ফেইল্ড OB যেটা ব্রেক হয়ে রিটেস্ট হয়েছে

【ফেয়ার ভ্যালু গ্যাপ (FVG)】
  → বুলিশ FVG: ৩-ক্যান্ডেল প্যাটার্ন, লো-টু-হাই গ্যাপ
  → বেয়ারিশ FVG: ৩-ক্যান্ডেল প্যাটার্ন, হাই-টু-লো গ্যাপ

【লিকুইডিটি পুলস】
  → BSL (Buy Side Liquidity): ইকুয়াল হাই
  → SSL (Sell Side Liquidity): ইকুয়াল লো

【মার্কেট স্ট্রাকচার】
  → BOS (Break of Structure): ট্রেন্ড কন্টিনিউয়েশন
  → CHOCH (Change of Character): পটেনশিয়াল রিভার্সাল
  → MSS (Market Structure Shift): ট্রেন্ড চেঞ্জ

【এন্ট্রি মডেল】
  → OTE (Optimal Trade Entry): ০.৬১৮-০.৭৮৬ ফিবো
  → স্নাইপার এন্ট্রি: কনফ্লুয়েন্স জোন + অর্ডার ব্লক + FVG

═══════════════════════════════════════════════════════════
🕯️ **সেকশন ৩: প্রাইস অ্যাকশন ক্যান্ডেলস্টিক প্যাটার্নস**
═══════════════════════════════════════════════════════════

【সিঙ্গেল ক্যান্ডেল】
  → হ্যামার (Hammer): ডাউনট্রেন্ডে, লং লোয়ার শ্যাডো, বুলিশ
  → শুটিং স্টার (Shooting Star): আপট্রেন্ডে, লং আপার শ্যাডো, বেয়ারিশ
  → ডজি (Doji): ওপেন=ক্লোজ, ইনডিসিশন
  → পিন বার (Pin Bar): রিজেকশন ক্যান্ডেল

【ডাবল ক্যান্ডেল】
  → বুলিশ এনগালফিং: ডাউনট্রেন্ডে, ২য় ক্যান্ডেল ১ম কে পূর্ণ কভার
  → বেয়ারিশ এনগালফিং: আপট্রেন্ডে, ২য় ক্যান্ডেল ১ম কে পূর্ণ কভার
  → টুইজার টপ/বটম: সমান হাই/লো

【ট্রিপল ক্যান্ডেল】
  → মর্নিং স্টার: ডাউনট্রেন্ডে, বড় ডাউন + ছোট + বড় আপ
  → ইভনিং স্টার: আপট্রেন্ডে, বড় আপ + ছোট + বড় ডাউন
  → থ্রি হোয়াইট সোলজার্স: তিনটি বুলিশ ক্যান্ডেল

═══════════════════════════════════════════════════════════
📐 **সেকশন ৪: চার্ট প্যাটার্নস**
═══════════════════════════════════════════════════════════

【রিভার্সাল প্যাটার্ন】
  → হেড অ্যান্ড শোল্ডারস (Bearish)
  → ইনভার্স হেড অ্যান্ড শোল্ডারস (Bullish)
  → ডাবল টপ/বটম
  → ট্রিপল টপ/বটম
  → রাউন্ডিং টপ/বটম
  → আইল্যান্ড রিভার্সাল

【কন্টিনিউয়েশন প্যাটার্ন】
  → ফ্ল্যাগ / পেন্যান্ট
  → কাপ অ্যান্ড হ্যান্ডেল
  → আসেন্ডিং/ডিসেন্ডিং ট্রায়াঙ্গেল
  → রেক্ট্যাঙ্গেল
  → প্রাইস চ্যানেল

【বাইল্যাটারাল প্যাটার্ন】
  → সিমেট্রিক্যাল ট্রায়াঙ্গেল
  → ব্রডেনিং ওয়েজ

【ওয়েজ প্যাটার্ন】
  → রাইজিং ওয়েজ (Bearish)
  → ফলিং ওয়েজ (Bullish)

═══════════════════════════════════════════════════════════
📏 **সেকশন ৫: ফিবোনাচ্চি অ্যানালাইসিস**
═══════════════════════════════════════════════════════════

【ফিবোনাচ্চি রিট্রেসমেন্ট】
  → 0.236: শ্যালো রিট্রেসমেন্ট
  → 0.382: মডারেট রিট্রেসমেন্ট
  → 0.500: ইকুইলিব্রিয়াম
  → 0.618: গোল্ডেন রেশিও (প্রিমিয়াম এন্ট্রি)
  → 0.786: ডিপ রিট্রেসমেন্ট

【ফিবো এক্সটেনশন】
  → 1.272: ফার্স্ট টার্গেট
  → 1.618: মেজর টার্গেট
  → 2.618: এক্সটেন্ডেড টার্গেট

═══════════════════════════════════════════════════════════
🎯 **সেকশন ৬: সাপোর্ট-রেজিস্ট্যান্স**
═══════════════════════════════════════════════════════════

【মেজর লেভেলস】
  → R3: স্ট্রং সেলিং জোন
  → R2: মিডিয়াম সেলিং জোন
  → R1: নিকটবর্তী সেলিং জোন
  → P: বর্তমান প্রাইস
  → S1: নিকটবর্তী বায়িং জোন
  → S2: মিডিয়াম বায়িং জোন
  → S3: স্ট্রং বায়িং জোন

【ডাইনামিক লেভেলস】
  → 20/50/100/200 EMA
  → ট্রেন্ডলাইন

═══════════════════════════════════════════════════════════
📊 **সেকশন ৭: ভলিউম, মোমেন্টাম এবং ডাইভারজেন্স অ্যানালাইসিস**
═══════════════════════════════════════════════════════════

【ভলিউম প্যাটার্নস】
  → ভলিউম স্পাইক: অস্বাভাবিক হাই ভলিউম
  → ভলিউম ডাইভারজেন্স: প্রাইস হাইয়ার হাই কিন্তু ভলিউম লোয়ার
  → ভলিউম কনফার্মেশন: ব্রেকআউটের সাথে হাই ভলিউম

【RSI ডাইভারজেন্স】
  → ক্লাসিক বুলিশ: প্রাইস লোয়ার লো, RSI হায়ার লো
  → হিডেন বুলিশ: প্রাইস হায়ার লো, RSI লোয়ার লো (ট্রেন্ড কন্টিনিউ)
  → ক্লাসিক বেয়ারিশ: প্রাইস হায়ার হাই, RSI লোয়ার হাই
  → হিডেন বেয়ারিশ: প্রাইস লোয়ার হাই, RSI হায়ার হাই (ট্রেন্ড কন্টিনিউ)

【MACD ডাইভারজেন্স】
  → বুলিশ: প্রাইস লোয়ার লো, MACD হিস্টোগ্রাম হায়ার লো
  → বেয়ারিশ: প্রাইস হায়ার হাই, MACD হিস্টোগ্রাম লোয়ার হাই

【OBV ডাইভারজেন্স】
  → বুলিশ: প্রাইস ডাউন, OBV আপ = একুমুলেশন
  → বেয়ারিশ: প্রাইস আপ, OBV ডাউন = ডিস্ট্রিবিউশন

═══════════════════════════════════════════════════════════
💼 **সেকশন ৮: ট্রেডিং প্ল্যান টেমপ্লেট**
═══════════════════════════════════════════════════════════

【এন্ট্রি কনফার্মেশন (৩/৫ চেক)】
  ✓ প্যাটার্ন ক্লোজ কনফার্মেশন
  ✓ ভলিউম কনফার্মেশন
  ✓ RSI/MACD ডাইভারজেন্স
  ✓ কনফ্লুয়েন্স জোন
  ✓ মোমেন্টাম কনফার্মেশন

【রিস্ক ম্যানেজমেন্ট】
  → স্টপ লস: স্ট্রাকচার-বেসড / ATR-বেসড
  → টেক প্রফিট: TP1 (1:1.5), TP2 (1:2.5), TP3 (1:4+)
  → পজিশন সাইজ: (অ্যাকাউন্ট রিস্ক %) / (এন্ট্রি-স্টপ ডিসটেন্স %)

═══════════════════════════════════════════════════════════
🎯 **সেকশন ৯: ফাইনাল ডিসিশন**
═══════════════════════════════════════════════════════════

【অ্যাকশন রিকমেন্ডেশন】
  → 🟢 STRONG BUY: ৮০%+ বুলিশ কনফ্লুয়েন্স + ডাইভারজেন্স
  → 🟡 BUY: ৬০-৭৯% বুলিশ কনফ্লুয়েন্স
  → ⚪ HOLD: ৪০-৫৯% ব্যালেন্স
  → 🟠 SELL: ৬০-৭৯% বেয়ারিশ কনফ্লুয়েন্স
  → 🔴 STRONG SELL: ৮০%+ বেয়ারিশ কনফ্লুয়েন্স + ডাইভারজেন্স

【আউটপুট ফরম্যাট - বাংলায় উত্তর দিন】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔹 **সিম্বল:** {symbol} | **টাইমফ্রেম:** ডেইলি | **বিশ্লেষণের সময়:** {timestamp}

🌊 **এলিয়ট ওয়েভ কাউন্ট:** [ডাটা অনুযায়ী নির্ধারণ করুন]
💰 **SMC জোন:** [অর্ডার ব্লক + FVG + লিকুইডিটি জোন]
🕯️ **প্রাইস অ্যাকশন:** [শনাক্তকৃত ক্যান্ডেল প্যাটার্ন]
📐 **চার্ট প্যাটার্ন:** [শনাক্তকৃত প্যাটার্ন + সিগন্যাল]
📐 **কি লেভেলস:** সাপোর্ট: S1/S2/S3 | রেজিস্ট্যান্স: R1/R2/R3

📊 **ডাইভারজেন্স অ্যানালাইসিস:**
• RSI ডাইভারজেন্স: [ক্লাসিক/হিডেন/কোনটি নেই] - [বুলিশ/বেয়ারিশ]
• MACD ডাইভারজেন্স: [ক্লাসিক/হিডেন/কোনটি নেই] - [বুলিশ/বেয়ারিশ]
• ভলিউম ডাইভারজেন্স: [হ্যাঁ/না]

🎯 **ট্রেডিং সেটআপ:**
• এন্ট্রি জোন: [প্রাইস রেঞ্জ]
• স্টপ লস: [প্রাইস] (রিস্ক: [X]%)
• টেক প্রফিট: TP1: [P] (R:R 1:1.5) | TP2: [P] (R:R 1:2.5) | TP3: [P] (R:R 1:4+)
• রিস্ক-রিওয়ার্ড: 1:[X]

📊 **কনফিডেন্স স্কোর:** [X]%

⚡ **কুইক ইনসাইট:**
• [পয়েন্ট ১ - মূল টেকনিক্যাল ফ্যাক্টর]
• [পয়েন্ট ২ - শনাক্তকৃত প্যাটার্ন]
• [পয়েন্ট ৩ - ট্রেডিং সেটআপ]

⚠️ **রিস্ক নোট:**
• [সতর্কতা ১]
• [সতর্কতা ২]

💡 **নোট:** এই বিশ্লেষণ সম্পূর্ণ টেকনিক্যাল ডাটার উপর ভিত্তি করে। সবসময় নিজস্ব রিসার্চ এবং রিস্ক ম্যানেজমেন্ট প্রয়োগ করুন。

═══════════════════════════════════════════════════════════

⚠️ **আবারও মনে রাখবেন: আপনার সম্পূর্ণ উত্তর বাংলা ভাষায় দিন! ইমোজি ব্যবহার করুন!**
"""
        
        logger.info(f"ফাইল তৈরি করা হয়েছে: {len(data_string)} অক্ষরের ডাটা সহ")
        
        return file_content

# ================================
# টেলিগ্রাম বট সেটআপ
# ================================
fetcher = StockDataFetcher()
bot_application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
user_last = {}
generated_files = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    text = f"""
👋 **স্বাগতম {user.first_name}!**

🤖 আমি একটি **প্রফেশনাল স্টক টেকনিক্যাল অ্যানালাইসিস বট**।

**কিভাবে ব্যবহার করবেন:**
1. 📈 যেকোনো স্টক সিম্বল পাঠান (যেমন: GP, ACI, SQUARE, RUPALILIFE)
2. 📊 আমি হাগিং ফেস থেকে সম্পূর্ণ ডাটা আনব (সব কলাম সহ)
3. 📝 সম্পূর্ণ প্রফেশনাল অ্যানালাইসিস প্রম্পট তৈরি করব
4. 📥 ফাইল ডাউনলোড লিংক পাবেন
5. 🤖 AI টুলে ফাইল আপলোড করে বিশ্লেষণ করান

**AI টুলে ব্যবহারের সময় বলুন:**
"এই ডাটা এবং প্রম্পট অনুযায়ী সম্পূর্ণ টেকনিক্যাল অ্যানালাইসিস করুন। উত্তর বাংলায় দিন。"

**কমান্ডসমূহ:**
/start - বট চালু করুন
/help - সাহায্য দেখুন
/about - বট সম্পর্কে জানুন
/symbols - উপলব্ধ সিম্বল দেখুন

এখন আপনার পছন্দের স্টক সিম্বল পাঠান!
"""
    await update.message.reply_text(text, parse_mode='Markdown')

async def symbols_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("🔍 সিম্বল লিস্ট সংগ্রহ করা হচ্ছে...\n\n⏳ প্রথমবার হতে ৩০-৪০ সেকেন্ড সময় লাগতে পারে...")

    symbols = await fetcher.load_all_symbols()

    if symbols and len(symbols) > 0:
        symbol_text = "📊 **উপলব্ধ স্টক সিম্বল:**\n\n"

        for i in range(0, min(len(symbols), 100), 20):
            group = symbols[i:i+20]
            symbol_text += "• " + " • ".join(group) + "\n"

        if len(symbols) > 100:
            symbol_text += f"\n... এবং আরও {len(symbols) - 100}টি সিম্বল"

        symbol_text += "\n\n💡 উপরের যেকোনো সিম্বল ব্যবহার করতে পারেন।"

        await msg.edit_text(symbol_text, parse_mode='Markdown')
    else:
        await msg.edit_text("❌ সিম্বল লিস্ট পাওয়া যায়নি।")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = """
📚 **সাহায্য গাইড**

**কিভাবে ব্যবহার করবেন:**

১️⃣ **স্টক সিম্বল পাঠান**
যেমন: GP, ACI, SQUARE, BRACBANK, RUPALILIFE
(সম্পূর্ণ লিস্ট দেখতে /symbols)

২️⃣ **ফাইল ডাউনলোড করুন**
বট ডাটা + সম্পূর্ণ প্রম্পট একসাথে ফাইল তৈরি করবে

৩️⃣ **AI টুলে আপলোড করুন**
• Gemini AI (gemini.google.com)
• Groq (groq.com)
• ChatGPT (chat.openai.com)
• Claude (claude.ai)

৪️⃣ **বাংলায় বিশ্লেষণ করতে বলুন**
AI কে বলুন: "এই ডাটা এবং প্রম্পট অনুযায়ী সম্পূর্ণ টেকনিক্যাল অ্যানালাইসিস করুন। উত্তর **বাংলা ভাষায়** দিন。"

**কমান্ড:**
/start - বট চালু
/help - সাহায্য
/about - তথ্য
/symbols - সিম্বল লিস্ট
"""
    await update.message.reply_text(text, parse_mode='Markdown')

async def about_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = """
🤖 **স্টক টেকনিক্যাল অ্যানালাইসিস বট v3.0**

**টেকনোলজি:**
• Python 3.11
• Telegram Bot API
• Hugging Face Datasets
• Flask

**ফিচার:**
✅ সম্পূর্ণ ডাটা (সব কলাম)
✅ ৪০০টি লেটেস্ট রো
✅ সম্পূর্ণ প্রম্পট (ইমোজি সহ সাজানো)
✅ বাংলায় উত্তর দেওয়ার নির্দেশনা
✅ ১ ক্লিকে ডাউনলোড
✅ সিমিলার সিম্বল সাজেশন

**ডাইভারজেন্স লাইব্রেরি:**
• RSI ক্লাসিক/হিডেন ডাইভারজেন্স
• MACD ক্লাসিক/হিডেন ডাইভারজেন্স
• OBV ডাইভারজেন্স
• ভলিউম ডাইভারজেন্স

**ডাটা সোর্স:**
huggingface.co/datasets/ahashanahmed/csv

**হোস্টিং:**
Render.com
"""
    await update.message.reply_text(text, parse_mode='Markdown')

async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbol = update.message.text.strip().upper()
    user_id = update.effective_user.id

    now = asyncio.get_event_loop().time()
    if user_id in user_last and now - user_last[user_id] < 5:
        await update.message.reply_text("⏳ একটু অপেক্ষা করুন...")
        return
    user_last[user_id] = now

    # প্রথমে সিম্বল লিস্ট লোড করুন
    if not fetcher.all_symbols:
        await fetcher.load_all_symbols()

    similar = fetcher.find_similar_symbols(symbol)

    if similar and similar[0] != symbol:
        keyboard = []
        for s in similar[:10]:
            keyboard.append([InlineKeyboardButton(s, callback_data=f"select_{s}")])
        keyboard.append([InlineKeyboardButton("❌ বাতিল", callback_data="cancel")])

        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(
            f"🔍 **'{symbol}'** সিম্বলটি পাওয়া যায়নি।\n\n"
            f"আপনি কি এইগুলোর মধ্যে একটি বোঝাতে চেয়েছেন?\n\n"
            f"💡 সম্পূর্ণ লিস্ট দেখতে /symbols ব্যবহার করুন।",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
        return

    await process_symbol(update, symbol)

async def process_symbol(update: Update, symbol: str, is_callback=False):
    if is_callback:
        query = update.callback_query
        await query.answer()
        msg = await query.edit_message_text(
            f"🔍 **{symbol}** ডাটা সংগ্রহ করা হচ্ছে...\n\n⏳ ১০-২০ সেকেন্ড সময় লাগতে পারে...",
            parse_mode='Markdown'
        )
        user_id = query.from_user.id
    else:
        msg = await update.message.reply_text(
            f"🔍 **{symbol}** ডাটা সংগ্রহ করা হচ্ছে...\n\n⏳ ১০-২০ সেকেন্ড সময় লাগতে পারে...",
            parse_mode='Markdown'
        )
        user_id = update.effective_user.id

    try:
        df, total_rows = await fetcher.get_stock_data(symbol, rows=400)

        if df is None or df.empty:
            await msg.edit_text(
                f"❌ **{symbol}** সিম্বলের জন্য কোন ডাটা পাওয়া যায়নি!\n\n"
                "📋 সম্পূর্ণ সিম্বল লিস্ট দেখতে /symbols কমান্ড ব্যবহার করুন।",
                parse_mode='Markdown'
            )
            return

        await msg.edit_text(
            f"📊 **{symbol}**\n\n"
            f"✅ ডাটা সংগ্রহ সম্পূর্ণ!\n"
            f"📈 মোট রো: {total_rows}টি\n"
            f"📋 ব্যবহৃত: {len(df)}টি সর্বশেষ রো\n"
            f"📝 ফাইল তৈরি হচ্ছে...",
            parse_mode='Markdown'
        )

        file_content = fetcher.create_full_file(symbol, df, total_rows)

        if not file_content:
            await msg.edit_text(f"❌ **{symbol}** ফাইল তৈরি করতে সমস্যা!", parse_mode='Markdown')
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{symbol}_analysis_{timestamp}.txt"

        file_id = f"{user_id}_{symbol}_{timestamp}"
        generated_files[file_id] = {
            'content': file_content,
            'filename': filename,
            'symbol': symbol
        }

        if RENDER_URL:
            download_url = f"{RENDER_URL}/download/{file_id}"
        else:
            download_url = f"http://localhost:10000/download/{file_id}"

        price_col = None
        for col in ['close', 'Close', 'price', 'Price', 'last', 'Last']:
            if col in df.columns:
                price_col = col
                break

        current_price = df[price_col].iloc[-1] if price_col and len(df) > 0 else 'N/A'

        response = f"""✅ **{symbol}** ফাইল তৈরি সম্পূর্ণ!

📊 **ডাটা সারাংশ:**
• সিম্বল: {symbol}
• বর্তমান মূল্য: {current_price}
• মোট রেকর্ড: {len(df)}টি সর্বশেষ ডাটা
• কলাম সংখ্যা: {len(df.columns)}টি
• তৈরি করা হয়েছে: {datetime.now().strftime('%d %B, %Y - %I:%M %p')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📥 **ডাউনলোড লিংক:**
[📄 {symbol}_analysis.txt]({download_url})

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🤖 **AI টুলে ব্যবহার করুন:**

ফাইলটি আপনার পছন্দের AI টুলে আপলোড করে বলুন:

**"এই ডাটা এবং প্রম্পট অনুযায়ী সম্পূর্ণ টেকনিক্যাল অ্যানালাইসিস করুন। উত্তর বাংলা ভাষায় দিন।"**

• [✨ Gemini AI](https://gemini.google.com)
• [⚡ Groq](https://groq.com)
• [💬 ChatGPT](https://chat.openai.com)
• [🧠 Claude](https://claude.ai)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📄 **ফাইলে যা থাকছে (সম্পূর্ণ):**
✅ সম্পূর্ণ ডাটা ({len(df)}টি রো, {len(df.columns)}টি কলাম)
✅ এলিয়ট ওয়েভ সম্পূর্ণ লাইব্রেরি
✅ SMC সম্পূর্ণ লাইব্রেরি
✅ প্রাইস অ্যাকশন সম্পূর্ণ লাইব্রেরি
✅ চার্ট প্যাটার্ন সম্পূর্ণ লাইব্রেরি
✅ ফিবোনাচ্চি অ্যানালাইসিস
✅ RSI/MACD/OBV ডাইভারজেন্স সম্পূর্ণ লাইব্রেরি
✅ ট্রেডিং প্ল্যান টেমপ্লেট
✅ রিস্ক ম্যানেজমেন্ট ফ্রেমওয়ার্ক
✅ **বাংলায় উত্তর দেওয়ার নির্দেশনা**

⏰ লিংক ৩০ মিনিটের জন্য সক্রিয় থাকবে।
"""

        await msg.edit_text(response, parse_mode='Markdown', disable_web_page_preview=False)

        async def delete_file():
            await asyncio.sleep(1800)
            if file_id in generated_files:
                del generated_files[file_id]

        asyncio.create_task(delete_file())

    except Exception as e:
        logger.error(f"ত্রুটি: {traceback.format_exc()}")
        await msg.edit_text(
            f"❌ **{symbol}** প্রক্রিয়াকরণে সমস্যা!\n\nত্রুটি: {str(e)[:200]}",
            parse_mode='Markdown'
        )

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    data = query.data

    if data == "cancel":
        await query.answer()
        await query.edit_message_text("❌ বাতিল করা হয়েছে।")
        return

    if data.startswith("select_"):
        symbol = data.replace("select_", "")
        await process_symbol(update, symbol, is_callback=True)

@flask_app.route('/download/<file_id>')
def download_file(file_id):
    if file_id not in generated_files:
        return jsonify({'error': 'ফাইল পাওয়া যায়নি বা মেয়াদ উত্তীর্ণ হয়েছে'}), 404

    file_data = generated_files[file_id]
    content = file_data['content']
    filename = file_data['filename']

    file_stream = io.BytesIO(content.encode('utf-8-sig'))
    file_stream.seek(0)

    return send_file(
        file_stream,
        as_attachment=True,
        download_name=filename,
        mimetype='text/plain; charset=utf-8'
    )

bot_application.add_handler(CommandHandler("start", start))
bot_application.add_handler(CommandHandler("help", help_command))
bot_application.add_handler(CommandHandler("about", about_command))
bot_application.add_handler(CommandHandler("symbols", symbols_command))
bot_application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle))
bot_application.add_handler(CallbackQueryHandler(handle_callback))

# ================================
# মেইন
# ================================
async def setup_webhook():
    if RENDER_URL:
        webhook_url = f"{RENDER_URL}/webhook"
        try:
            await bot_application.bot.set_webhook(webhook_url)
            logger.info(f"✅ ওয়েবহুক সেট করা হয়েছে: {webhook_url}")
        except Exception as e:
            logger.error(f"ওয়েবহুক সেট করতে ব্যর্থ: {e}")

async def main():
    logger.info("🚀 স্টক টেকনিক্যাল অ্যানালাইসিস বট চালু হচ্ছে...")

    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    if not TELEGRAM_BOT_TOKEN:
        logger.error("❌ TELEGRAM_BOT_TOKEN সেট করা নেই!")
    if not HF_TOKEN:
        logger.warning("⚠️ HF_TOKEN সেট করা নেই!")

    if RENDER_URL:
        await setup_webhook()
        await bot_application.initialize()
        await bot_application.start()
        logger.info("✅ বট ওয়েবহুক মোডে চালু আছে")
        while True:
            await asyncio.sleep(1)
    else:
        await bot_application.initialize()
        await bot_application.start()
        await bot_application.updater.start_polling()
        logger.info("✅ বট পোলিং মোডে চালু আছে")
        while True:
            await asyncio.sleep(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 বট বন্ধ করা হয়েছে")