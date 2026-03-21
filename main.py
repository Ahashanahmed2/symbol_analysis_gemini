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
import numpy as np
import tempfile
from huggingface_hub import hf_hub_download, HfApi
import requests
import io
import traceback
import threading
from flask import Flask, jsonify
import json

# Flask app
app = Flask(__name__)

@app.route('/')
def health():
    return jsonify({
        "status": "ok",
        "message": "Stock Analysis Bot is running",
        "time": datetime.now().isoformat()
    })

@app.route('/debug')
def debug():
    return jsonify({
        "gemini_key_set": bool(os.getenv("GEMINI_API_TOKEN")),
        "hf_token_set": bool(os.getenv("HF_TOKEN") or os.getenv("hf_token")),
        "telegram_token_set": bool(os.getenv("TELEGRAM_BOT_TOKEN"))
    })

def run_flask():
    """Flask সার্ভার চালানোর ফাংশন"""
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

# hf_uploader থেকে ইম্পোর্ট করুন
try:
    from hf_uploader import get_symbol_latest_data, HFStreamingHandler
    print("✅ hf_uploader imported successfully")
except Exception as e:
    print(f"⚠️ hf_uploader import error: {e}")
    get_symbol_latest_data = None
    HFStreamingHandler = None

# লোড এনভায়রনমেন্ট
load_dotenv()

# লগিং
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TechnicalAnalyzer:
    """টেকনিক্যাল অ্যানালাইসিস ক্যালকুলেশন ক্লাস"""
    
    @staticmethod
    def find_swing_points(df, column='close', order=5):
        """সুইং হাই এবং সুইং লো ফাইন্ড করুন"""
        highs = []
        lows = []
        
        for i in range(order, len(df) - order):
            if df[column].iloc[i] == max(df[column].iloc[i-order:i+order+1]):
                highs.append((df.index[i], df[column].iloc[i]))
            if df[column].iloc[i] == min(df[column].iloc[i-order:i+order+1]):
                lows.append((df.index[i], df[column].iloc[i]))
        
        return highs, lows
    
    @staticmethod
    def count_hh_hl_lh_ll(df, column='close', order=5):
        """HH, HL, LH, LL কাউন্ট করুন"""
        highs, lows = TechnicalAnalyzer.find_swing_points(df, column, order)
        
        hh_count = 0
        hl_count = 0
        lh_count = 0
        ll_count = 0
        
        last_high = None
        last_low = None
        
        for i, (idx, price) in enumerate(highs):
            if last_high is not None:
                if price > last_high:
                    hh_count += 1
            last_high = price
        
        for i, (idx, price) in enumerate(lows):
            if last_low is not None:
                if price > last_low:
                    hl_count += 1
            last_low = price
        
        for i, (idx, price) in enumerate(highs):
            if last_low is not None:
                if price < last_low:
                    lh_count += 1
        
        for i, (idx, price) in enumerate(lows):
            if last_high is not None:
                if price < last_high:
                    ll_count += 1
        
        return {
            'hh_count': hh_count,
            'hl_count': hl_count,
            'lh_count': lh_count,
            'll_count': ll_count,
            'last_hh': highs[-1][1] if highs else None,
            'last_hl': lows[-1][1] if lows else None,
            'last_lh': highs[-1][1] if highs else None,
            'last_ll': lows[-1][1] if lows else None
        }
    
    @staticmethod
    def detect_bos_choch(df, column='close', order=5):
        """BOS (Break of Structure) এবং CHOCH (Change of Character) ডিটেক্ট করুন"""
        highs, lows = TechnicalAnalyzer.find_swing_points(df, column, order)
        
        bos = False
        choch = False
        bos_level = None
        choch_level = None
        
        if len(highs) >= 2 and len(lows) >= 2:
            if highs[-1][1] > highs[-2][1]:
                bos = True
                bos_level = highs[-1][1]
            
            if lows[-1][1] > lows[-2][1] and highs[-1][1] < highs[-2][1]:
                choch = True
                choch_level = lows[-1][1]
        
        return {'bos': bos, 'choch': choch, 'bos_level': bos_level, 'choch_level': choch_level}
    
    @staticmethod
    def calculate_fibonacci_levels(df, column='close'):
        """ফিবোনাচ্চি লেভেল ক্যালকুলেট করুন"""
        high = df[column].max()
        low = df[column].min()
        diff = high - low
        
        return {
            'fib_236': low + diff * 0.236,
            'fib_382': low + diff * 0.382,
            'fib_500': low + diff * 0.5,
            'fib_618': low + diff * 0.618,
            'fib_786': low + diff * 0.786,
            'fib_1272': high + diff * 0.272,
            'fib_1414': high + diff * 0.414,
            'fib_1618': high + diff * 0.618,
            'fib_2000': high + diff * 1.0,
            'fib_2618': high + diff * 1.618
        }
    
    @staticmethod
    def find_support_resistance(df, column='close', n_levels=3):
        """সাপোর্ট এবং রেজিস্ট্যান্স লেভেল ফাইন্ড করুন"""
        try:
            from scipy.signal import argrelextrema
            import numpy as np
            
            prices = df[column].values
            
            # লোকাল মিনিমা (সাপোর্ট)
            local_min = argrelextrema(prices, np.less, order=10)[0]
            supports = sorted(set(prices[local_min]))[:n_levels]
            
            # লোকাল ম্যাক্সিমা (রেজিস্ট্যান্স)
            local_max = argrelextrema(prices, np.greater, order=10)[0]
            resistances = sorted(set(prices[local_max]), reverse=True)[:n_levels]
            
            while len(supports) < n_levels:
                supports.append(prices.min() - (prices.max() - prices.min()) * 0.1)
            while len(resistances) < n_levels:
                resistances.append(prices.max() + (prices.max() - prices.min()) * 0.1)
            
            return supports[:n_levels], resistances[:n_levels]
        except:
            # Fallback if scipy not available
            prices = df[column].values
            supports = [prices.min(), prices.min() * 0.98, prices.min() * 0.96]
            resistances = [prices.max(), prices.max() * 1.02, prices.max() * 1.04]
            return supports[:n_levels], resistances[:n_levels]
    
    @staticmethod
    def calculate_volume_analysis(df):
        """ভলিউম অ্যানালাইসিস"""
        avg_volume = df['volume'].mean()
        current_volume = df['volume'].iloc[-1]
        
        volume_bullish = current_volume > avg_volume and df['close'].iloc[-1] > df['close'].iloc[-2]
        volume_bearish = current_volume > avg_volume and df['close'].iloc[-1] < df['close'].iloc[-2]
        
        return {
            'current_volume': current_volume,
            'avg_volume': avg_volume,
            'volume_bullish': volume_bullish,
            'volume_bearish': volume_bearish
        }
    
    @staticmethod
    def detect_patterns(df):
        """বেসিক প্যাটার্ন ডিটেক্ট করুন"""
        patterns = {
            'cup_handle_forming': False,
            'cup_handle_complete': False,
            'cup_handle_breakout': None,
            'double_top': False,
            'double_bottom': False,
            'neckline': None,
            'bull_flag': False,
            'bear_flag': False,
            'rising_wedge': False,
            'falling_wedge': False
        }
        
        # সিম্পল কাপ অ্যান্ড হ্যান্ডেল ডিটেকশন
        last_20 = df['close'].tail(20)
        if len(last_20) == 20:
            if last_20.max() - last_20.min() > last_20.mean() * 0.1:
                patterns['cup_handle_forming'] = True
                if last_20.iloc[-1] > last_20.iloc[-5]:
                    patterns['cup_handle_complete'] = True
                    patterns['cup_handle_breakout'] = last_20.max()
        
        # ডাবল টপ/বটম ডিটেকশন
        last_50 = df['close'].tail(50)
        if len(last_50) >= 50:
            peaks = last_50[last_50 == last_50.rolling(10, center=True).max()].dropna()
            if len(peaks) >= 2:
                if abs(peaks.iloc[-1] - peaks.iloc[-2]) / peaks.iloc[-2] < 0.03:
                    patterns['double_top'] = True
                    patterns['neckline'] = last_50.min()
            
            troughs = last_50[last_50 == last_50.rolling(10, center=True).min()].dropna()
            if len(troughs) >= 2:
                if abs(troughs.iloc[-1] - troughs.iloc[-2]) / troughs.iloc[-2] < 0.03:
                    patterns['double_bottom'] = True
                    patterns['neckline'] = last_50.max()
        
        return patterns
    
    @staticmethod
    def calculate_risk_reward(current_price, entry_zone_low, entry_zone_high, target_1, stop_loss):
        """Risk/Reward Ratio ক্যালকুলেট করুন"""
        risk = abs(entry_zone_low - stop_loss)
        reward_1 = abs(target_1 - entry_zone_high)
        
        if risk > 0:
            rr_ratio = reward_1 / risk
        else:
            rr_ratio = 0
        
        return rr_ratio


class StockAnalyzer:
    def __init__(self):
        self.gemini_key = os.getenv("GEMINI_API_TOKEN")
        self.hf_token = os.getenv("HF_TOKEN") or os.getenv("hf_token")
        self.repo_id = "ahashanahmed/csv"
        self.technical = TechnicalAnalyzer()

        # Gemini মডেল - নতুন SDK ব্যবহার
        self.gemini_model = "gemini-2.0-flash-exp"
        self.last_request_time = 0
        self.min_request_interval = 20

        # গ্লোবাল রেট লিমিট ট্র্যাকিং
        self.global_request_times = []
        self.max_requests_per_minute = 15

        # ফলোআপ প্রশ্নের জন্য স্টোরেজ
        self.last_symbol = {}
        self.last_data = {}

        print(f"🔧 Initializing StockAnalyzer...")
        print(f"   GEMINI_API_TOKEN: {'✅ Found' if self.gemini_key else '❌ Not found'}")
        print(f"   Gemini Model: {self.gemini_model}")
        print(f"   HF Token: {'✅ Found' if self.hf_token else '❌ Not found'}")

        # NEW: Google GenAI ক্লায়েন্ট (google-genai SDK)
        self.gemini_client = None
        if self.gemini_key:
            try:
                # নতুন SDK ব্যবহার করে ক্লায়েন্ট ইনিশিয়ালাইজ
                self.gemini_client = genai.Client(api_key=self.gemini_key)
                print("✅ Google GenAI client initialized (google-genai SDK)")
            except Exception as e:
                print(f"❌ Gemini client error: {e}")

        # HF হ্যান্ডলার
        self.hf_handler = None
        if self.hf_token:
            try:
                if HFStreamingHandler:
                    self.hf_handler = HFStreamingHandler(repo_id=self.repo_id, token=self.hf_token)
                    print("✅ HF Streaming Handler initialized")
                else:
                    self.hf_handler = self._create_fallback_handler()
                    print("⚠️ Using fallback handler")
            except Exception as e:
                print(f"❌ HF handler error: {e}")
                self.hf_handler = self._create_fallback_handler()
        else:
            print("❌ HF Token not found!")

    def _check_rate_limit(self):
        """গ্লোবাল রেট লিমিট চেক"""
        current_time = time.time()
        self.global_request_times = [t for t in self.global_request_times if current_time - t < 60]
        
        if len(self.global_request_times) >= self.max_requests_per_minute:
            oldest_request = min(self.global_request_times)
            wait_time = 60 - (current_time - oldest_request)
            return False, f"⚠️ অনেক রিকোয়েস্ট হয়েছে। {wait_time:.0f} সেকেন্ড অপেক্ষা করে আবার চেষ্টা করুন।"
        
        return True, None

    def _record_request(self):
        """রিকোয়েস্ট রেকর্ড করুন"""
        self.global_request_times.append(time.time())

    def _create_fallback_handler(self):
        """ফলব্যাক হ্যান্ডলার"""
        class FallbackHandler:
            def __init__(self, repo_id, token):
                self.repo_id = repo_id
                self.token = token
                self.api = HfApi()

            def get_symbol_latest_rows(self, symbol, rows=400):
                return self._get_data_direct(symbol, rows)

            def _get_data_direct(self, symbol, rows=400):
                try:
                    print(f"🔍 Loading {symbol}...")
                    files = self.api.list_repo_files(
                        repo_id=self.repo_id,
                        repo_type="dataset",
                        token=self.token
                    )
                    
                    csv_files = [f for f in files if f.endswith('.csv')]
                    if not csv_files:
                        print("❌ No CSV files found")
                        return None
                    
                    target_file = "mongodb.csv" if "mongodb.csv" in csv_files else csv_files[0]
                    
                    with tempfile.TemporaryDirectory() as temp_dir:
                        downloaded_file = hf_hub_download(
                            repo_id=self.repo_id,
                            filename=target_file,
                            repo_type="dataset",
                            token=self.token,
                            local_dir=temp_dir,
                            local_dir_use_symlinks=False
                        )
                        
                        sample_df = pd.read_csv(downloaded_file, nrows=100, encoding='utf-8-sig')
                        symbol_col = None
                        for col in ['symbol', 'Symbol', 'SYMBOL', 'ticker', 'Ticker']:
                            if col in sample_df.columns:
                                symbol_col = col
                                break
                        
                        if not symbol_col:
                            print("❌ Symbol column not found")
                            return None
                        
                        chunk_size = 50000
                        symbol_data = []
                        
                        for chunk in pd.read_csv(downloaded_file, chunksize=chunk_size, encoding='utf-8-sig'):
                            mask = chunk[symbol_col].astype(str).str.upper() == symbol.upper()
                            filtered = chunk[mask]
                            if len(filtered) > 0:
                                symbol_data.append(filtered)
                        
                        if not symbol_data:
                            print(f"❌ No data for {symbol}")
                            return None
                        
                        df = pd.concat(symbol_data, ignore_index=True)
                        print(f"✅ Found {len(df)} rows for {symbol}")
                        
                        date_col = None
                        for col in ['date', 'Date', 'DATE', 'timestamp']:
                            if col in df.columns:
                                date_col = col
                                break
                        
                        if date_col:
                            df[date_col] = pd.to_datetime(df[date_col])
                            df = df.sort_values(date_col, ascending=False)
                            df = df.head(rows)
                            df = df.sort_values(date_col)
                        else:
                            df = df.tail(rows)
                        
                        return df
                        
                except Exception as e:
                    print(f"❌ Error: {e}")
                    return None
            
            def get_symbol_list(self, max_symbols=400):
                try:
                    files = self.api.list_repo_files(
                        repo_id=self.repo_id,
                        repo_type="dataset",
                        token=self.token
                    )
                    
                    csv_files = [f for f in files if f.endswith('.csv')]
                    if not csv_files:
                        return []
                    
                    target_file = "mongodb.csv" if "mongodb.csv" in csv_files else csv_files[0]
                    
                    with tempfile.TemporaryDirectory() as temp_dir:
                        downloaded_file = hf_hub_download(
                            repo_id=self.repo_id,
                            filename=target_file,
                            repo_type="dataset",
                            token=self.token,
                            local_dir=temp_dir,
                            local_dir_use_symlinks=False
                        )
                        
                        df = pd.read_csv(downloaded_file, encoding='utf-8-sig')
                        
                        symbol_col = None
                        for col in ['symbol', 'Symbol', 'SYMBOL', 'ticker', 'Ticker']:
                            if col in df.columns:
                                symbol_col = col
                                break
                        
                        if symbol_col:
                            symbols = df[symbol_col].dropna().unique().tolist()
                            symbols = [str(s).strip().upper() for s in symbols if str(s).strip()]
                            return sorted(list(set(symbols)))[:max_symbols]
                        
                        return []
                        
                except Exception as e:
                    print(f"❌ Error: {e}")
                    return []
        
        return FallbackHandler(self.repo_id, self.hf_token)

    def get_stock_data(self, symbol, rows=400):
        """স্টক ডাটা লোড"""
        try:
            if not self.hf_handler:
                print("❌ HF handler not initialized")
                return None
            
            print(f"🔍 Loading {symbol} (last {rows} rows)...")
            
            if get_symbol_latest_data:
                df = get_symbol_latest_data(
                    symbol=symbol,
                    rows=rows,
                    repo_id=self.repo_id,
                    token=self.hf_token
                )
            else:
                df = self.hf_handler.get_symbol_latest_rows(symbol, rows)
            
            if df is None or df.empty:
                print(f"❌ No data found for {symbol}")
                return None
            
            print(f"✅ Found {len(df)} rows for {symbol}")
            return df
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def prepare_technical_data(self, df):
        """টেকনিক্যাল ডাটা প্রিপ্রসেস করুন প্রম্পটের জন্য"""
        if df is None or df.empty:
            return {}
        
        # প্রাইস কলাম ডিটেক্ট করুন
        price_col = 'close'
        for col in ['close', 'Close', 'CLOSE', 'price', 'Price']:
            if col in df.columns:
                price_col = col
                break
        
        # HH/HL/LH/LL ক্যালকুলেশন
        swing_data = self.technical.count_hh_hl_lh_ll(df, price_col)
        
        # BOS/CHOCH ডিটেকশন
        structure_data = self.technical.detect_bos_choch(df, price_col)
        
        # ফিবোনাচ্চি লেভেল
        fib_data = self.technical.calculate_fibonacci_levels(df, price_col)
        
        # সাপোর্ট/রেজিস্ট্যান্স
        supports, resistances = self.technical.find_support_resistance(df, price_col)
        
        # ভলিউম অ্যানালাইসিস
        volume_data = self.technical.calculate_volume_analysis(df)
        
        # প্যাটার্ন ডিটেকশন
        patterns = self.technical.detect_patterns(df)
        
        # বর্তমান প্রাইস
        current_price = df[price_col].iloc[-1]
        
        # ট্রেন্ড ডিটেকশন
        sma_20 = df[price_col].rolling(20).mean().iloc[-1]
        sma_50 = df[price_col].rolling(50).mean().iloc[-1] if len(df) > 50 else current_price
        
        uptrend = current_price > sma_20 and sma_20 > sma_50
        downtrend = current_price < sma_20 and sma_20 < sma_50
        
        # Risk/Reward ক্যালকুলেশন
        entry_zone_low = supports[0] if supports else current_price * 0.98
        entry_zone_high = supports[0] * 1.02 if supports else current_price
        target_1 = resistances[0] if resistances else current_price * 1.05
        stop_loss = supports[0] * 0.98 if supports else current_price * 0.97
        
        rr_ratio = self.technical.calculate_risk_reward(
            current_price, entry_zone_low, entry_zone_high, target_1, stop_loss
        )
        
        return {
            'current_price': current_price,
            'swing_data': swing_data,
            'structure_data': structure_data,
            'fib_data': fib_data,
            'supports': supports,
            'resistances': resistances,
            'volume_data': volume_data,
            'patterns': patterns,
            'uptrend': uptrend,
            'downtrend': downtrend,
            'sma_20': sma_20,
            'sma_50': sma_50,
            'entry_zone_low': entry_zone_low,
            'entry_zone_high': entry_zone_high,
            'target_1': target_1,
            'stop_loss': stop_loss,
            'rr_ratio': rr_ratio
        }
    
    def analyze(self, symbol, df, progress_callback=None):
        """Gemini দিয়ে বিশ্লেষণ - google-genai SDK ব্যবহার করে"""
        if not self.gemini_client:
            return "⚠️ Gemini API কনফিগার করা নেই।"
        
        if df is None or df.empty:
            return "⚠️ বিশ্লেষণের জন্য ডাটা নেই।"
        
        # গ্লোবাল রেট লিমিট চেক
        ok, msg = self._check_rate_limit()
        if not ok:
            return msg
        
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            wait_time = self.min_request_interval - time_since_last
            if progress_callback:
                progress_callback(f"⏳ রেট লিমিট, {wait_time:.0f} সেকেন্ড অপেক্ষা...")
            time.sleep(wait_time)
        
        try:
            if progress_callback:
                progress_callback("📊 CSV ডাটা প্রস্তুত করা হচ্ছে...")
            
            # টেকনিক্যাল ডাটা প্রিপ্রসেস
            tech_data = self.prepare_technical_data(df)
            
            csv_buffer = io.StringIO()
            df_tail = df.tail(400)
            df_tail.to_csv(csv_buffer, index=False)
            csv_text = csv_buffer.getvalue()
            
            if len(csv_text) > 30000:
                csv_text = csv_text[-30000:]
            
            print(f"📊 Analysis CSV: {len(csv_text)} chars, {len(df_tail)} rows")
            
            if progress_callback:
                progress_callback(f"🤖 Gemini AI বিশ্লেষণ করছে...\n⏳ এটি ২০-৩০ সেকেন্ড সময় নিতে পারে...")
            
            # প্রম্পট তৈরি
            prompt = self._create_analysis_prompt(symbol, df_tail, csv_text, tech_data)
            
            self.last_request_time = time.time()
            self._record_request()
            
            # NEW: google-genai SDK ব্যবহার করে কন্টেন্ট জেনারেশন
            response = self.gemini_client.models.generate_content(
                model=self.gemini_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=4500,
                    temperature=0.7
                )
            )
            
            if response and response.text:
                analysis = response.text
                print(f"✅ Analysis received: {len(analysis)} chars")
                return analysis
            else:
                return "⚠️ বিশ্লেষণ তৈরি করা যায়নি।"
                
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Gemini error: {error_msg}")
            if "rate_limit" in error_msg.lower():
                return "⚠️ অনেক রিকোয়েস্ট হয়েছে। ১ মিনিট অপেক্ষা করে আবার চেষ্টা করুন।"
            else:
                return f"⚠️ বিশ্লেষণ করতে সমস্যা: {error_msg[:100]}"
    
    def _create_analysis_prompt(self, symbol, df, csv_text, tech_data):
        """প্রফেশনাল অ্যানালাইসিস প্রম্পট তৈরি করুন"""
        
        prompt = f"""📊 **{symbol} স্টক অ্যানালাইসিস - প্রফেশনাল টেকনিক্যাল অ্যানালাইসিস**

ডাটা: {len(df)} রো CSV ফরম্যাটে (সর্বশেষ ৪০০ দিনের ডাটা)

{csv_text}

🔍 **টেকনিক্যাল ডাটা সামারি:**
• বর্তমান মূল্য: {tech_data.get('current_price', 0):.2f}
• ২০ দিনের SMA: {tech_data.get('sma_20', 0):.2f}
• ৫০ দিনের SMA: {tech_data.get('sma_50', 0):.2f}
• ট্রেন্ড: {'📈 আপট্রেন্ড' if tech_data.get('uptrend') else '📉 ডাউনট্রেন্ড' if tech_data.get('downtrend') else '🔄 সাইডওয়েজ'}

═══════════════════════════════════════════════════════════
1️⃣ **ELLIOT WAVE অ্যানালাইসিস (সবচেয়ে গুরুত্বপূর্ণ)**
═══════════════════════════════════════════════════════════

উপরের CSV ডাটা এবং প্রাইস অ্যাকশন বিশ্লেষণ করে নিচের বিষয়গুলো নির্ধারণ করুন:

• **বর্তমান Wave অবস্থান:** (Wave 1, 2, 3, 4, 5 / A, B, C)
• **Impulse Wave:** Wave 1, 2, 3, 4, 5 এর মধ্যে কোনটি চলছে?
• **Corrective Wave:** A, B, C এর মধ্যে কোনটি চলছে?
• **Leading Diagonal:** শুরুতে Diagonal Pattern আছে কিনা?
• **Ending Diagonal:** শেষের দিকে Diagonal Pattern আছে কিনা?
• **Extension:** কোন Wave এ Extension হয়েছে? (1, 3, বা 5)
• **Truncation:** ৫ম Wave ট্রাঙ্কেট হয়েছে কিনা?

**কোরেক্টিভ প্যাটার্ন ডিটেকশন:**
• **Zigzag (5-3-5):** দেখা যাচ্ছে কিনা?
• **Flat (3-3-5):** 
  - Regular Flat: আছে কিনা?
  - Expanded Flat: আছে কিনা?
  - Running Flat: আছে কিনা?
• **Triangle:**
  - Ascending Triangle: ফর্মিং হচ্ছে কিনা?
  - Descending Triangle: ফর্মিং হচ্ছে কিনা?
  - Symmetrical Triangle: ফর্মিং হচ্ছে কিনা?
  - Expanding Triangle: ফর্মিং হচ্ছে কিনা?
• **Double Three (WXY):** কমপ্লেক্স কোরেকশন আছে কিনা?
• **Triple Three (WXYZ):** আছে কিনা?

**পরবর্তী Wave পূর্বাভাস:**
• পরবর্তী Wave কোনটি আসবে? (Wave 3 / C / 5)
• সম্ভাব্য Target প্রাইস কত?
• Invalidation Level কোথায়?

═══════════════════════════════════════════════════════════
2️⃣ **চার্ট প্যাটার্ন অ্যানালাইসিস**
═══════════════════════════════════════════════════════════

**ক্লাসিক্যাল প্যাটার্ন:**
• **কাপ অ্যান্ড হ্যান্ডেল:** {'ফর্মিং' if tech_data.get('patterns', {}).get('cup_handle_forming') else 'কমপ্লিট' if tech_data.get('patterns', {}).get('cup_handle_complete') else 'নেই'} - ব্রেকআউট লেভেল: {tech_data.get('patterns', {}).get('cup_handle_breakout', 'N/A')}
• **ডাবল টপ / ডাবল বটম:** {'ডাবল টপ' if tech_data.get('patterns', {}).get('double_top') else 'ডাবল বটম' if tech_data.get('patterns', {}).get('double_bottom') else 'নেই'} - নেকলাইন: {tech_data.get('patterns', {}).get('neckline', 'N/A')}
• **হেড অ্যান্ড শোল্ডারস:** আছে কিনা? (Regular/Inverse)
• **বুলিশ ফ্ল্যাগ / বিয়ারিশ ফ্ল্যাগ:** {'বুলিশ' if tech_data.get('patterns', {}).get('bull_flag') else 'বিয়ারিশ' if tech_data.get('patterns', {}).get('bear_flag') else 'নেই'}
• **ওয়েজ প্যাটার্ন:** {'রাইজিং ওয়েজ' if tech_data.get('patterns', {}).get('rising_wedge') else 'ফলিং ওয়েজ' if tech_data.get('patterns', {}).get('falling_wedge') else 'নেই'}

═══════════════════════════════════════════════════════════
3️⃣ **প্রাইস অ্যাকশন স্ট্রাকচার (HH/HL/LH/LL/BOS/CHOCH)**
═══════════════════════════════════════════════════════════

**ট্রেন্ড স্ট্রাকচার:**
• Higher High (HH): {tech_data.get('swing_data', {}).get('hh_count', 0)} টি - সর্বশেষ HH: {tech_data.get('swing_data', {}).get('last_hh', 'N/A')}
• Higher Low (HL): {tech_data.get('swing_data', {}).get('hl_count', 0)} টি - সর্বশেষ HL: {tech_data.get('swing_data', {}).get('last_hl', 'N/A')}
• Lower High (LH): {tech_data.get('swing_data', {}).get('lh_count', 0)} টি - সর্বশেষ LH: {tech_data.get('swing_data', {}).get('last_lh', 'N/A')}
• Lower Low (LL): {tech_data.get('swing_data', {}).get('ll_count', 0)} টি - সর্বশেষ LL: {tech_data.get('swing_data', {}).get('last_ll', 'N/A')}

**স্ট্রাকচার ব্রেক:**
• **BOS (Break of Structure):** {'হ্যাঁ' if tech_data.get('structure_data', {}).get('bos') else 'না'} - লেভেল: {tech_data.get('structure_data', {}).get('bos_level', 'N/A')}
• **CHOCH (Change of Character):** {'হ্যাঁ' if tech_data.get('structure_data', {}).get('choch') else 'না'} - লেভেল: {tech_data.get('structure_data', {}).get('choch_level', 'N/A')}

═══════════════════════════════════════════════════════════
4️⃣ **ফিবোনাচ্চি অ্যানালাইসিস**
═══════════════════════════════════════════════════════════

**রিট্রেসমেন্ট লেভেল:**
• 0.236: {tech_data.get('fib_data', {}).get('fib_236', 0):.2f}
• 0.382: {tech_data.get('fib_data', {}).get('fib_382', 0):.2f}
• 0.500: {tech_data.get('fib_data', {}).get('fib_500', 0):.2f}
• 0.618: {tech_data.get('fib_data', {}).get('fib_618', 0):.2f} (গোল্ডেন রেশিও)
• 0.786: {tech_data.get('fib_data', {}).get('fib_786', 0):.2f}

**এক্সটেনশন লেভেল:**
• 1.272: {tech_data.get('fib_data', {}).get('fib_1272', 0):.2f}
• 1.414: {tech_data.get('fib_data', {}).get('fib_1414', 0):.2f}
• 1.618: {tech_data.get('fib_data', {}).get('fib_1618', 0):.2f} (প্রাইম টার্গেট)
• 2.000: {tech_data.get('fib_data', {}).get('fib_2000', 0):.2f}
• 2.618: {tech_data.get('fib_data', {}).get('fib_2618', 0):.2f}

═══════════════════════════════════════════════════════════
5️⃣ **সাপোর্ট-রেজিস্ট্যান্স ও লিকুইডিটি অ্যানালাইসিস**
═══════════════════════════════════════════════════════════

**মেজর সাপোর্ট লেভেল:**
1. S1: {tech_data.get('supports', [0,0,0])[0]:.2f}
2. S2: {tech_data.get('supports', [0,0,0])[1]:.2f}
3. S3: {tech_data.get('supports', [0,0,0])[2]:.2f}

**মেজর রেজিস্ট্যান্স লেভেল:**
1. R1: {tech_data.get('resistances', [0,0,0])[0]:.2f}
2. R2: {tech_data.get('resistances', [0,0,0])[1]:.2f}
3. R3: {tech_data.get('resistances', [0,0,0])[2]:.2f}

**লিকুইডিটি অ্যানালাইসিস:**
• প্রাইস অ্যাকশন দেখে লিকুইডিটি Grab/Sweep হয়েছে কিনা বিশ্লেষণ করুন
• স্টপ হান্টিং এর迹象 আছে কিনা দেখুন
• অর্ডার ব্লক (Order Block) কোথায় হতে পারে?

═══════════════════════════════════════════════════════════
6️⃣ **ভলিউম অ্যানালাইসিস**
═══════════════════════════════════════════════════════════

• বর্তমান ভলিউম: {tech_data.get('volume_data', {}).get('current_volume', 0):,.0f}
• গড় ভলিউম: {tech_data.get('volume_data', {}).get('avg_volume', 0):,.0f}
• ভলিউম ট্রেন্ড: {'🟢 বুলিশ' if tech_data.get('volume_data', {}).get('volume_bullish') else '🔴 বিয়ারিশ' if tech_data.get('volume_data', {}).get('volume_bearish') else '⚪ নিউট্রাল'}

═══════════════════════════════════════════════════════════
7️⃣ **ট্রেডিং প্ল্যান (Risk/Reward সহ)**
═══════════════════════════════════════════════════════════

**এন্ট্রি প্ল্যান:**
• এন্ট্রি জোন: {tech_data.get('entry_zone_low', 0):.2f} - {tech_data.get('entry_zone_high', 0):.2f}
• স্টপ লস (SL): {tech_data.get('stop_loss', 0):.2f}

**টার্গেট লেভেল:**
• টার্গেট ১ (TP1): {tech_data.get('target_1', 0):.2f}
• টার্গেট ২ (TP2): {tech_data.get('resistances', [0,0,0])[1] if len(tech_data.get('resistances', [])) > 1 else tech_data.get('target_1', 0) * 1.05:.2f}
• টার্গেট ৩ (TP3): {tech_data.get('resistances', [0,0,0])[2] if len(tech_data.get('resistances', [])) > 2 else tech_data.get('target_1', 0) * 1.10:.2f}

**Risk/Reward Ratio:** 1:{tech_data.get('rr_ratio', 0):.1f}

═══════════════════════════════════════════════════════════
8️⃣ **সামগ্রিক মতামত ও পরামর্শ**
═══════════════════════════════════════════════════════════

উপরের সব বিশ্লেষণের ভিত্তিতে দিন:

• **বর্তমান Action:** BUY / SELL / HOLD / WAIT
• **কনফিডেন্স লেভেল:** (%)
• **টাইমফ্রেম:** শর্ট টার্ম / মিড টার্ম / লং টার্ম

**মূল পয়েন্টসমূহ:**
(৩টি মূল পয়েন্ট লিখুন)

**সতর্কতা:**
(২টি সতর্কতা লিখুন)

**পরামর্শ:**
(ট্রেডারদের জন্য বিস্তারিত পরামর্শ দিন)

═══════════════════════════════════════════════════════════
💡 **নোট:** এই বিশ্লেষণ শুধুমাত্র টেকনিক্যাল ডাটার উপর ভিত্তি করে। বাস্তব ট্রেড করার আগে ফান্ডামেন্টাল অ্যানালাইসিস এবং নিজস্ব রিসার্চ করুন।
"""
        
        return prompt
    
    def answer_question(self, symbol, df, question):
        """ফলোআপ প্রশ্নের উত্তর - google-genai SDK ব্যবহার করে"""
        if not self.gemini_client:
            return "⚠️ Gemini API কনফিগার করা নেই।"
        
        ok, msg = self._check_rate_limit()
        if not ok:
            return msg
        
        try:
            tech_data = self.prepare_technical_data(df)
            
            csv_buffer = io.StringIO()
            df_tail = df.tail(100)
            df_tail.to_csv(csv_buffer, index=False)
            csv_text = csv_buffer.getvalue()
            
            if len(csv_text) > 15000:
                csv_text = csv_text[-15000:]
            
            prompt = f"""📊 {symbol} স্টকের সর্বশেষ {len(df_tail)} টি ডাটা:

{csv_text}

**টেকনিক্যাল ডাটা:**
• বর্তমান মূল্য: {tech_data.get('current_price', 0):.2f}
• সাপোর্ট: {tech_data.get('supports', [0])[0]:.2f}
• রেজিস্ট্যান্স: {tech_data.get('resistances', [0])[0]:.2f}

ইউজারের প্রশ্ন: {question}

এই ডাটার ভিত্তিতে সংক্ষিপ্ত কিন্তু বিস্তারিত উত্তর দিন। উত্তর বাংলায়, সরাসরি ও ব্যবহারিক হোক।"""
            
            self._record_request()
            
            # NEW: google-genai SDK ব্যবহার করে উত্তর জেনারেশন
            response = self.gemini_client.models.generate_content(
                model=self.gemini_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=1500,
                    temperature=0.7
                )
            )
            
            if response and response.text:
                return response.text
            else:
                return "⚠️ প্রশ্নের উত্তর দিতে পারিনি।"
                
        except Exception as e:
            print(f"❌ Question error: {e}")
            return f"⚠️ সমস্যা: {str(e)[:100]}"


class TelegramBot:
    def __init__(self):
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.analyzer = StockAnalyzer()
        self.app = None
        self.user_last_request = {}
        
        if not self.token:
            print("❌ TELEGRAM_BOT_TOKEN not found!")
    
    def _is_followup_question(self, text, user_id):
        """চেক করে এটি প্রশ্ন কিনা"""
        text_lower = text.lower()
        if '?' in text:
            return True
        question_words = ['কি', 'কী', 'কেমন', 'কবে', 'কোথায়', 'কেন', 'কখন',
            'আছে', 'নাই', 'হবে', 'যাবে', 'কেনা', 'বেচা', 'কেন', 'বেচ', 'হোল্ড', 
            'এন্ট্রি', 'টার্গেট', 'স্টপ', 'সাপোর্ট', 'রেজিস্ট্যান্স', 'ট্রেন্ড', 
            'ভলিউম', 'কাপ', 'হ্যান্ডেল', 'প্যাটার্ন', 'এলিয়ট', 'রিস্ক', 'রিওয়ার্ড',
            'ওয়েভ', 'ফিবো', 'করে', 'করে?', 'কিভাবে', 'কখন']
        for word in question_words:
            if word in text_lower:
                return True
        if len(text.split()) >= 2 and len(text) > 5:
            return True
        return False
    
    def setup(self):
        """বট সেটআপ"""
        self.app = Application.builder().token(self.token).build()
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(CommandHandler("help", self.help))
        self.app.add_handler(CommandHandler("test", self.test))
        self.app.add_handler(CommandHandler("symbols", self.show_symbols))
        self.app.add_handler(CommandHandler("clear", self.clear_session))
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle))
        self.app.add_error_handler(self.error)
        print("✅ Bot handlers registered")
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        msg = """🤖 *স্টক অ্যানালাইসিস বট - প্রফেশনাল ভার্সন (Gemini AI)*

আমি স্টক মার্কেটের প্রফেশনাল টেকনিক্যাল অ্যানালাইসিস করি।

*ফিচারসমূহ:*
📊 Elliot Wave Analysis (Impulse, Corrective, Diagonals)
📈 চার্ট প্যাটার্ন (Cup & Handle, Triangles, Flags)
🎯 HH/HL/LH/LL, BOS, CHOCH Analysis
📐 Fibonacci Retracement & Extension
💰 Risk/Reward Ratio সহ ট্রেডিং প্ল্যান
💡 ইন্টারেক্টিভ প্রশ্নোত্তর

*কমান্ডসমূহ:*
/help - সাহায্য
/test - সিস্টেম চেক
/symbols - উপলব্ধ সিম্বল দেখান
/clear - সেশনের ডাটা ক্লিয়ার করুন

*কিভাবে ব্যবহার করবেন:*
1. প্রথমে স্টক সিম্বল লিখুন: `GP`, `BRACBANK`, `GPH`
2. বিস্তারিত বিশ্লেষণ পাবেন (Elliot Wave, Patterns, Risk/Reward সহ)
3. বিশ্লেষণ পাওয়ার পর ঐ স্টক নিয়ে প্রশ্ন করতে পারেন

*নোট:* API রেট লিমিটের জন্য ১ মিনিটে ১৫টি রিকোয়েস্ট করতে পারবেন।"""
        await update.message.reply_text(msg, parse_mode='Markdown')
    
    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        msg = """📚 *সাহায্য*

*স্টক সিম্বল লিখুন:*
- GP (গ্রামীণফোন)
- BRACBANK (ব্র্যাক ব্যাংক)
- GPH (জিপি হাউজিং)

*বিশ্লেষণে যা পাবেন:*
1️⃣ Elliot Wave (1,2,3,4,5,A,B,C)
2️⃣ চার্ট প্যাটার্ন (Cup & Handle, Triangles)
3️⃣ HH/HL/LH/LL Structure
4️⃣ BOS & CHOCH Analysis
5️⃣ Fibonacci Levels
6️⃣ Support/Resistance
7️⃣ Risk/Reward Ratio
8️⃣ ট্রেডিং প্ল্যান

*কমান্ড:*
/test - API সংযোগ চেক করুন
/symbols - উপলব্ধ সিম্বল দেখান
/clear - সেশনের ডাটা ক্লিয়ার করুন

*নোট:* ১ মিনিটে ১৫টি রিকোয়েস্ট করতে পারবেন।"""
        await update.message.reply_text(msg, parse_mode='Markdown')
    
    async def clear_session(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        if user_id in self.analyzer.last_symbol:
            del self.analyzer.last_symbol[user_id]
        if user_id in self.analyzer.last_data:
            del self.analyzer.last_data[user_id]
        await update.message.reply_text("✅ আপনার সেশন ডাটা ক্লিয়ার করা হয়েছে।")
    
    async def show_symbols(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        wait = await update.message.reply_text("⏳ সিম্বল লোড করা হচ্ছে...")
        try:
            if self.analyzer.hf_handler:
                symbols = self.analyzer.hf_handler.get_symbol_list(max_symbols=200)
                if symbols and len(symbols) > 0:
                    chunks = [symbols[i:i+15] for i in range(0, len(symbols), 15)]
                    msg = f"📊 *উপলব্ধ সিম্বল (মোট {len(symbols)} টি):*\n\n"
                    for chunk in chunks[:5]:
                        msg += " | ".join(chunk) + "\n\n"
                    await wait.delete()
                    await update.message.reply_text(msg, parse_mode='Markdown')
                else:
                    await wait.edit_text("❌ কোন সিম্বল পাওয়া যায়নি")
            else:
                await wait.edit_text("❌ HF হ্যান্ডলার কাজ করছে না")
        except Exception as e:
            await wait.edit_text(f"❌ সিম্বল লোড করতে সমস্যা হয়েছে")
    
    async def test(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        wait = await update.message.reply_text("⏳ টেস্ট চলছে...")
        try:
            gemini_ok = "❌"
            if self.analyzer.gemini_client:
                try:
                    # NEW: google-genai SDK ব্যবহার করে টেস্ট
                    test_response = self.analyzer.gemini_client.models.generate_content(
                        model=self.analyzer.gemini_model,
                        contents="ok",
                        config=types.GenerateContentConfig(max_output_tokens=5)
                    )
                    if test_response and test_response.text:
                        gemini_ok = "✅"
                    else:
                        gemini_ok = "⚠️"
                except Exception as e:
                    gemini_ok = f"❌ ({str(e)[:30]})"
            
            hf_ok = "❌"
            if self.analyzer.hf_handler:
                try:
                    test_df = self.analyzer.get_stock_data("GP", rows=5)
                    if test_df is not None and len(test_df) > 0:
                        hf_ok = f"✅ ({len(test_df)} rows)"
                    else:
                        hf_ok = "⚠️"
                except Exception as e:
                    hf_ok = "❌"
            
            result = f"""📊 *টেস্ট রেজাল্ট*

🤖 Gemini API: {gemini_ok}
📊 Gemini Model: `{self.analyzer.gemini_model}`
💾 Hugging Face: {hf_ok}

*ফিচার সমূহ:*
✅ Elliot Wave (1,2,3,4,5,ABC)
✅ Chart Patterns (Cup & Handle, Triangles)
✅ HH/HL/LH/LL Structure
✅ BOS & CHOCH Analysis
✅ Fibonacci Analysis
✅ Risk/Reward Ratio

*রেট লিমিট:* 1 মিনিটে 15টি রিকোয়েস্ট

*স্ট্যাটাস:* {'✅ সিস্টেম ওকে' if '✅' in gemini_ok and '✅' in hf_ok else '⚠️ কিছু সমস্যা আছে'}"""
            await wait.delete()
            await update.message.reply_text(result, parse_mode='Markdown')
        except Exception as e:
            await wait.delete()
            await update.message.reply_text(f"❌ টেস্ট চলাকালীন সমস্যা হয়েছে: {str(e)[:100]}")
    
    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        text = update.message.text.strip()
        user = update.effective_user.username or update.effective_user.first_name
        user_id = update.effective_user.id
        print(f"📩 Received: '{text}' from {user}")
        
        if text.startswith('/'):
            return
        
        current_time = time.time()
        last_request = self.user_last_request.get(user_id, 0)
        if current_time - last_request < 20:
            wait_time = 20 - (current_time - last_request)
            await update.message.reply_text(f"⏳ দয়া করে {wait_time:.0f} সেকেন্ড অপেক্ষা করুন।")
            return
        self.user_last_request[user_id] = current_time
        
        is_question = self._is_followup_question(text, user_id)
        if is_question and user_id in self.analyzer.last_symbol:
            symbol = self.analyzer.last_symbol[user_id]
            df = self.analyzer.last_data.get(user_id)
            if df is not None and not df.empty:
                await self._handle_question(update, symbol, df, text)
                return
        
        symbol = text.upper()
        await self._handle_symbol(update, symbol, user_id)
    
    async def _handle_symbol(self, update: Update, symbol: str, user_id: int):
        msg = await update.message.reply_text(
            f"📊 *{symbol}* এর ডাটা লোড করা হচ্ছে...\n⏳ একটু অপেক্ষা করুন...", 
            parse_mode='Markdown'
        )
        try:
            df = self.analyzer.get_stock_data(symbol, rows=400)
            if df is None or df.empty:
                await msg.edit_text(f"❌ *{symbol}* এর তথ্য পাওয়া যায়নি", parse_mode='Markdown')
                return
            
            self.analyzer.last_symbol[user_id] = symbol
            self.analyzer.last_data[user_id] = df
            
            await msg.edit_text(
                f"✅ *{symbol}* এর {len(df)} টি ডাটা পাওয়া গেছে\n\n🤖 Gemini AI প্রফেশনাল অ্যানালাইসিস করছে...\n⏳ এটি ৩০-৪০ সেকেন্ড সময় নিতে পারে...\n\n📊 *অ্যানালাইসিসে যা পাবেন:*\n• Elliot Wave (1,2,3,4,5,ABC)\n• চার্ট প্যাটার্ন\n• HH/HL/LH/LL Structure\n• BOS & CHOCH\n• Fibonacci Levels\n• Risk/Reward Ratio\n• ট্রেডিং প্ল্যান", 
                parse_mode='Markdown'
            )
            
            def sync_analyze():
                return self.analyzer.analyze(symbol, df, None)
            
            analysis = await asyncio.get_event_loop().run_in_executor(None, sync_analyze)
            await msg.delete()
            
            if analysis and not analysis.startswith("⚠️"):
                # Split long message if needed (Telegram has 4096 char limit)
                if len(analysis) > 4000:
                    parts = [analysis[i:i+4000] for i in range(0, len(analysis), 4000)]
                    for i, part in enumerate(parts):
                        if i == 0:
                            await update.message.reply_text(
                                f"📈 *{symbol}* প্রফেশনাল অ্যানালাইসিস (পার্ট {i+1}/{len(parts)}):\n\n{part}\n\n💡 আরও জানতে চাইলে প্রশ্ন করতে পারেন।", 
                                parse_mode='Markdown'
                            )
                        else:
                            await update.message.reply_text(
                                f"📈 *{symbol}* প্রফেশনাল অ্যানালাইসিস (পার্ট {i+1}/{len(parts)}):\n\n{part}", 
                                parse_mode='Markdown'
                            )
                else:
                    await update.message.reply_text(
                        f"📈 *{symbol}* প্রফেশনাল অ্যানালাইসিস:\n\n{analysis}\n\n💡 আরও জানতে চাইলে প্রশ্ন করতে পারেন।", 
                        parse_mode='Markdown'
                    )
            else:
                await update.message.reply_text(f"⚠️ *{symbol}* বিশ্লেষণ করা যায়নি\n\n{analysis}")
                
        except Exception as e:
            await msg.edit_text(f"❌ টেকনিক্যাল সমস্যা হয়েছে\n\nError: {str(e)[:100]}")
    
    async def _handle_question(self, update: Update, symbol: str, df, question: str):
        msg = await update.message.reply_text(
            f"🤔 *{symbol}* নিয়ে আপনার প্রশ্ন: {question}\n\n⏳ উত্তর তৈরি করা হচ্ছে...", 
            parse_mode='Markdown'
        )
        try:
            answer = self.analyzer.answer_question(symbol, df, question)
            await msg.delete()
            
            if answer and not answer.startswith("⚠️"):
                await update.message.reply_text(
                    f"💬 *{symbol}* নিয়ে আপনার প্রশ্নের উত্তর:*\n\n{answer}\n\n❓ আরও জানতে চাইলে প্রশ্ন করতে পারেন।", 
                    parse_mode='Markdown'
                )
            else:
                await update.message.reply_text(f"⚠️ প্রশ্নের উত্তর দিতে পারিনি।\n\n{answer}")
                
        except Exception as e:
            await msg.edit_text(f"❌ প্রশ্নের উত্তর দিতে সমস্যা হয়েছে\n\nError: {str(e)[:100]}")
    
    async def error(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        print(f"Error: {context.error}")

    def run_polling(self):
        """পোলিং মোডে বট চালান"""
        if self.app:
            print("🚀 Starting bot in polling mode...")
            try:
                self.app.stop_running()
            except:
                pass
            time.sleep(1)
            self.app.run_polling(
                allowed_updates=Update.ALL_TYPES,
                drop_pending_updates=True,
                poll_interval=1.0,
                timeout=60
            )


def main():
    """মেইন ফাংশন"""
    print("="*60)
    print("📊 স্টক অ্যানালাইসিস বট - প্রফেশনাল ভার্সন (Google GenAI SDK)")
    print("🤖 Elliot Wave + Chart Patterns + Price Action + Risk/Reward")
    print("="*60)
    
    token = os.getenv('TELEGRAM_BOT_TOKEN', '')
    if not token:
        print("❌ TELEGRAM_BOT_TOKEN not found!")
        return
    
    # ফোর্স ওয়েবহুক ডিলিট
    print("🔄 Force deleting webhook...")
    for attempt in range(5):
        try:
            url = f"https://api.telegram.org/bot{token}/deleteWebhook?drop_pending_updates=true"
            response = requests.get(url, timeout=10)
            print(f"✅ Attempt {attempt + 1}: {response.json()}")
        except Exception as e:
            print(f"⚠️ Attempt {attempt + 1} failed: {e}")
        time.sleep(2)
    
    print("⏳ Waiting 30 seconds for Telegram API to fully reset...")
    time.sleep(30)
    
    # ওয়েবহুক স্ট্যাটাস চেক
    try:
        url = f"https://api.telegram.org/bot{token}/getWebhookInfo"
        response = requests.get(url, timeout=10)
        info = response.json()
        print(f"📡 Webhook info: {info}")
        if info.get('result', {}).get('url'):
            print("⚠️ Webhook still exists! Force deleting again...")
            requests.get(f"https://api.telegram.org/bot{token}/deleteWebhook?drop_pending_updates=true")
            time.sleep(10)
    except Exception as e:
        print(f"⚠️ Could not get webhook info: {e}")
    
    # বট ইনিশিয়ালাইজ
    print("🔧 Initializing TelegramBot...")
    global telegram_bot
    telegram_bot = TelegramBot()
    telegram_bot.setup()
    
    print("🚀 Starting bot in polling mode...")
    print("📊 Features:")
    print("   ✅ Elliot Wave (1,2,3,4,5,ABC)")
    print("   ✅ Chart Patterns (Cup & Handle, Triangles, Flags)")
    print("   ✅ Price Action (HH/HL/LH/LL, BOS, CHOCH)")
    print("   ✅ Fibonacci Analysis")
    print("   ✅ Risk/Reward Ratio")
    print("   ✅ Interactive Q&A")
    print("   ✅ Google GenAI SDK (google-genai)")
    print("⏳ Rate Limit: 15 requests per minute")
    print("✅ Bot started. Waiting for messages...")
    
    time.sleep(5)
    
    # পোলিং শুরু
    print("🔄 Starting polling...")
    try:
        telegram_bot.run_polling()
        print("✅ Polling started successfully")
    except Exception as e:
        print(f"❌ Polling error: {e}")
        traceback.print_exc()
        while True:
            print("🔄 Bot is running but polling failed...")
            time.sleep(60)


if __name__ == "__main__":
    # Flask সার্ভার আলাদা থ্রেডে চালান
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    print(f"✅ Flask server starting on port {os.environ.get('PORT', 10000)}")
    
    time.sleep(2)
    
    # বট চালান
    main()
