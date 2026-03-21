from huggingface_hub import login, upload_folder, snapshot_download, HfApi, hf_hub_download
import os
import shutil
import time
import pandas as pd
import hashlib
import json
from datetime import datetime
from dotenv import load_dotenv
import tempfile

load_dotenv()

HF_TOKEN = os.getenv("hf_token")
USERNAME = "ahashanahmed"
REPO_NAME = "csv"
REPO_ID = f"{USERNAME}/{REPO_NAME}"

# ==================== বেসিক ফাংশন ====================

def hf_login(token=None):
    """Hugging Face লগইন"""
    if token:
        try:
            login(token=token)
            print("🔐 HF login সফল হয়েছে।")
            return True
        except Exception as e:
            print(f"❌ HF login ব্যর্থ: {e}")
            return False
    return False

def is_valid_directory(local_dir: str) -> bool:
    """ডিরেক্টরি ভ্যালিড কিনা চেক"""
    return os.path.isdir(local_dir) and len(os.listdir(local_dir)) > 0

def create_repo_if_not_exists(repo_id: str = REPO_ID, token: str = HF_TOKEN):
    """রিপোজিটরি তৈরি (যদি না থাকে)"""
    api = HfApi()
    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset", token=token)
        print(f"ℹ️ Repo '{repo_id}' আগে থেকেই আছে।")
        return True
    except Exception:
        try:
            api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, token=token)
            print(f"✅ নতুন Repo তৈরি হয়েছে: {repo_id}")
            return True
        except Exception as e:
            print(f"❌ Repo তৈরি ব্যর্থ: {e}")
            return False

# ==================== স্ট্রিমিং হ্যান্ডলার ক্লাস ====================

class HFStreamingHandler:
    """
    Hugging Face থেকে স্ট্রিমিং পদ্ধতিতে ডাটা আনে
    লোকালে কিছু সেভ না করে সরাসরি মেমোরিতে
    """

    def __init__(self, repo_id=REPO_ID, token=HF_TOKEN):
        self.repo_id = repo_id
        self.token = token
        self.api = HfApi()

    def get_symbol_latest_rows(self, symbol, rows=400):
        """
        নির্দিষ্ট সিম্বলের সর্বশেষ (লেটেস্ট) 'rows' সংখ্যক রো রিটার্ন করে
        এটি শুধুমাত্র সবচেয়ে নতুন ডাটা রিটার্ন করবে
        
        Args:
            symbol (str): সিম্বল নাম (যেমন: 'AAPL', 'BTC-USD')
            rows (int): কতটি সর্বশেষ রো চাই (ডিফল্ট: 400)
        
        Returns:
            pandas.DataFrame: সিম্বলের সর্বশেষ ডাটা (নতুন থেকে পুরনো ক্রমে)
        """
        print(f"🚀 {symbol} এর সর্বশেষ {rows} টি রো আনতে হচ্ছে...")

        try:
            # 1. রিপোজিটরির ফাইল লিস্ট
            files = self.api.list_repo_files(
                repo_id=self.repo_id,
                repo_type="dataset",
                token=self.token
            )

            csv_files = [f for f in files if f.endswith('.csv')]
            if not csv_files:
                print("❌ কোন CSV ফাইল নেই")
                return None

            # 2. mongodb.csv টার্গেট
            target_file = "mongodb.csv" if "mongodb.csv" in csv_files else csv_files[0]
            print(f"📁 টার্গেট ফাইল: {target_file}")

            # 3. টেম্প ডিরেক্টরিতে ডাউনলোড
            with tempfile.TemporaryDirectory() as temp_dir:
                downloaded_file = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=target_file,
                    repo_type="dataset",
                    token=self.token,
                    local_dir=temp_dir,
                    local_dir_use_symlinks=False
                )

                print(f"📥 ডাউনলোড সম্পন্ন, প্রসেসিং শুরু...")

                # 4. সিম্বল কলাম ডিটেক্ট
                sample_df = pd.read_csv(downloaded_file, nrows=100, encoding='utf-8-sig')
                symbol_col = self._find_symbol_column(sample_df)

                if not symbol_col:
                    print("❌ সিম্বল কলাম খুঁজে পাওয়া যায়নি")
                    return None

                print(f"🔖 সিম্বল কলাম: {symbol_col}")

                # 5. শুধু নির্দিষ্ট সিম্বলের ডাটা পড়ি
                chunk_size = 50000
                symbol_rows = []

                for chunk in pd.read_csv(downloaded_file, chunksize=chunk_size, encoding='utf-8-sig'):
                    # সিম্বল ফিল্টার
                    mask = chunk[symbol_col].astype(str).str.upper() == symbol.upper()
                    filtered = chunk[mask]

                    if len(filtered) > 0:
                        symbol_rows.append(filtered)

                if not symbol_rows:
                    print(f"❌ {symbol} এর কোন ডাটা পাওয়া যায়নি")
                    return None

                # 6. সব ডাটা কনক্যাট
                df = pd.concat(symbol_rows, ignore_index=True)
                print(f"✅ মোট {len(df)} টি রো পাওয়া গেছে")

                # 7. 'date' কলাম চেক
                date_col = self._find_date_column(df)

                if date_col:
                    # ডেট টাইমে কনভার্ট
                    df[date_col] = pd.to_datetime(df[date_col])

                    # সর্বশেষ (লেটেস্ট) rows নেওয়া
                    # নতুন থেকে পুরনো সাজিয়ে সবচেয়ে নতুন rows নিই
                    df_sorted_desc = df.sort_values(date_col, ascending=False)
                    latest_rows = df_sorted_desc.head(rows)

                    # ফলাফল নতুন থেকে পুরনো ক্রমে থাকবে
                    print(f"🎯 {symbol} এর সর্বশেষ {len(latest_rows)} টি রো প্রস্তুত")
                    if len(latest_rows) > 0:
                        print(f"   📅 সর্বশেষ তারিখ: {latest_rows[date_col].max()}")
                        print(f"   📅 পুরনোতম তারিখ: {latest_rows[date_col].min()}")

                    return latest_rows
                else:
                    # date কলাম না থাকলে, শেষের rows নিই
                    print("⚠️ 'date' কলাম নেই, শেষের rows নেওয়া হচ্ছে")
                    return df.tail(rows)

        except Exception as e:
            print(f"❌ এরর: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_symbol_latest_by_date(self, symbol, rows=400):
        """
        নির্দিষ্ট সিম্বলের সর্বশেষ (লেটেস্ট) ডাটা রিটার্ন করে
        date কলাম অনুযায়ী সাজানো
        
        Args:
            symbol (str): সিম্বল নাম
            rows (int): কতটি সর্বশেষ রো চাই
        
        Returns:
            pandas.DataFrame: নতুন থেকে পুরনো ক্রমে সাজানো ডাটা
        """
        df = self.get_symbol_latest_rows(symbol, rows)

        if df is not None and len(df) > 0:
            # নিশ্চিত করা যে ডাটা নতুন থেকে পুরনো ক্রমে আছে
            date_col = self._find_date_column(df)
            if date_col:
                df = df.sort_values(date_col, ascending=False)
                df = df.reset_index(drop=True)
                print(f"📊 ডাটা সাজানো হয়েছে: নতুন → পুরনো")

        return df

    def _find_symbol_column(self, df):
        """ডাটাফ্রেমে সিম্বল কলাম খুঁজে বের করে"""
        possible_columns = ['symbol', 'Symbol', 'SYMBOL', 'ticker', 'Ticker', 
                           'inv_symbol', 'Inv_Symbol', 'code', 'Code', 'stock', 'Stock']

        for col in possible_columns:
            if col in df.columns:
                return col

        # না পেলে, যেকোনো কলাম যার নামে 'symbol' আছে
        for col in df.columns:
            if 'symbol' in col.lower():
                return col

        return None

    def _find_date_column(self, df):
        """ডাটাফ্রেমে ডেট কলাম খুঁজে বের করে"""
        possible_columns = ['date', 'Date', 'DATE', 'datetime', 'DateTime', 
                           'timestamp', 'Timestamp', 'time', 'Time']

        for col in possible_columns:
            if col in df.columns:
                return col

        # না পেলে, যেকোনো কলাম যার নামে 'date' বা 'time' আছে
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                return col

        return None

    def get_symbol_list(self, max_symbols=100):
        """
        রিপোজিটরিতে উপলব্ধ সব সিম্বলের লিস্ট রিটার্ন করে
        """
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

                df = pd.read_csv(downloaded_file, nrows=10000, encoding='utf-8-sig')
                symbol_col = self._find_symbol_column(df)

                if symbol_col:
                    symbols = df[symbol_col].dropna().unique().tolist()
                    symbols = [str(s).strip().upper() for s in symbols if str(s).strip()]
                    symbols = sorted(list(set(symbols)))

                    return symbols[:max_symbols]
                else:
                    return []

        except Exception as e:
            print(f"সিম্বল লিস্ট পেতে সমস্যা: {e}")
            return []

    def search_symbols(self, query, limit=10):
        """
        সিম্বল সার্চ করে (query দিয়ে)
        """
        try:
            all_symbols = self.get_symbol_list(limit=100)
            query_upper = query.upper()

            matches = [s for s in all_symbols if query_upper in s]
            return matches[:limit]

        except Exception as e:
            print(f"সার্চ সমস্যা: {e}")
            return []

# ==================== HF আপলোডার ফাংশন ====================

def get_symbol_latest_data(symbol, rows=400, repo_id=REPO_ID, token=HF_TOKEN):
    """
    নির্দিষ্ট সিম্বলের সর্বশেষ (লেটেস্ট) rows সংখ্যক ডাটা রিটার্ন করে
    এটি hf_uploader এ কল করার জন্য প্রধান ফাংশন
    
    Args:
        symbol (str): সিম্বল নাম (যেমন: 'AAPL', 'BTC-USD')
        rows (int): কতটি সর্বশেষ রো চাই (ডিফল্ট: 400)
        repo_id (str): Hugging Face রিপোজিটরি আইডি
        token (str): HF টোকেন
    
    Returns:
        pandas.DataFrame: সিম্বলের সর্বশেষ ডাটা (নতুন থেকে পুরনো ক্রমে)
        None: যদি কোনো ডাটা না পাওয়া যায়
    
    Example:
        >>> df = get_symbol_latest_data("AAPL", rows=400)
        >>> print(df.head())  # সবচেয়ে নতুন ডাটা দেখাবে
    """
    handler = HFStreamingHandler(repo_id=repo_id, token=token)
    return handler.get_symbol_latest_rows(symbol, rows=rows)

def get_symbol_latest_data_sorted(symbol, rows=400, repo_id=REPO_ID, token=HF_TOKEN):
    """
    নির্দিষ্ট সিম্বলের সর্বশেষ ডাটা রিটার্ন করে (নতুন থেকে পুরনো ক্রমে)
    
    Args:
        symbol (str): সিম্বল নাম
        rows (int): কতটি সর্বশেষ রো চাই
    
    Returns:
        pandas.DataFrame: নতুন থেকে পুরনো ক্রমে সাজানো ডাটা
    """
    handler = HFStreamingHandler(repo_id=repo_id, token=token)
    return handler.get_symbol_latest_by_date(symbol, rows=rows)

def get_multiple_symbols_latest_data(symbols_list, rows=400, repo_id=REPO_ID, token=HF_TOKEN):
    """
    একাধিক সিম্বলের সর্বশেষ ডাটা রিটার্ন করে
    
    Args:
        symbols_list (list): সিম্বলের লিস্ট
        rows (int): প্রতিটি সিম্বলের জন্য কতটি সর্বশেষ রো
        repo_id (str): HF রিপোজিটরি আইডি
        token (str): HF টোকেন
    
    Returns:
        dict: {symbol: dataframe} ফরম্যাটে ডাটা (প্রতিটি ডাটা নতুন থেকে পুরনো ক্রমে)
    """
    handler = HFStreamingHandler(repo_id=repo_id, token=token)
    results = {}

    for symbol in symbols_list:
        print(f"\n{'='*50}")
        df = handler.get_symbol_latest_rows(symbol, rows=rows)
        if df is not None and len(df) > 0:
            results[symbol] = df
            print(f"✅ {symbol}: {len(df)} টি সর্বশেষ রো প্রস্তুত")
        else:
            print(f"⚠️ {symbol} এর জন্য কোনো ডাটা পাওয়া যায়নি")

    return results

# ==================== স্মার্ট আপলোডার ====================

class SmartDatasetUploader:
    """স্মার্ট ডাটাসেট আপলোডার - শুধু পরিবর্তিত ফাইল আপলোড করে"""

    def __init__(self, repo_id=REPO_ID, token=HF_TOKEN):
        self.api = HfApi()
        self.repo_id = repo_id
        self.token = token
        self.metadata_file = ".dataset_metadata.json"
        self.stats = {
            'total_files': 0,
            'new_files': 0,
            'modified_files': 0,
            'unchanged_files': 0,
            'failed_files': 0
        }

    def get_file_hash(self, file_path):
        """ফাইলের MD5 হ্যাশ বের করে"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            print(f"⚠️ হ্যাশ গণনা ব্যর্থ: {e}")
            return None

    def get_remote_metadata(self):
        """HF থেকে মেটাডাটা ডাউনলোড"""
        try:
            temp_metadata = f"temp_{self.metadata_file}"
            hf_hub_download(
                repo_id=self.repo_id,
                filename=self.metadata_file,
                repo_type="dataset",
                token=self.token,
                local_path=temp_metadata
            )

            with open(temp_metadata, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            os.remove(temp_metadata)
            print(f"📋 মেটাডাটা পাওয়া গেছে: {len(metadata.get('files', {}))} টি ফাইলের তথ্য")
            return metadata

        except Exception as e:
            print(f"📋 কোন মেটাডাটা নেই। নতুন মেটাডাটা তৈরি করা হবে।")
            return {
                "files": {}, 
                "last_sync": None,
                "created_at": datetime.now().isoformat()
            }

    def upload_metadata(self, metadata):
        """মেটাডাটা HF-এ আপলোড"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            self.api.upload_file(
                path_or_fileobj=self.metadata_file,
                path_in_repo=self.metadata_file,
                repo_id=self.repo_id,
                repo_type="dataset",
                token=self.token
            )

            os.remove(self.metadata_file)
            print(f"📋 মেটাডাটা আপলোড সফল")
            return True

        except Exception as e:
            print(f"⚠️ মেটাডাটা আপলোড ব্যর্থ: {e}")
            return False

    def merge_csv_files(self, local_path, remote_filename, unique_columns=None):
        """দুই CSV ফাইল মার্জ করে"""
        temp_remote = f"temp_remote_{int(time.time())}.csv"

        try:
            local_df = pd.read_csv(local_path)
            print(f"   লোকাল ডাটা: {len(local_df)} রো")

            try:
                hf_hub_download(
                    repo_id=self.repo_id,
                    filename=remote_filename,
                    repo_type="dataset",
                    token=self.token,
                    local_path=temp_remote
                )

                remote_df = pd.read_csv(temp_remote)
                print(f"   রিমোট ডাটা: {len(remote_df)} রো")

                if unique_columns and all(col in remote_df.columns for col in unique_columns):
                    combined_df = pd.concat([remote_df, local_df], ignore_index=True)

                    if 'timestamp' in combined_df.columns:
                        combined_df = combined_df.sort_values('timestamp', ascending=False)

                    combined_df = combined_df.drop_duplicates(
                        subset=unique_columns, 
                        keep='first'
                    )
                    print(f"   ডুপ্লিকেট রিমুভের পর: {len(combined_df)} রো")

                else:
                    combined_df = pd.concat([remote_df, local_df], ignore_index=True)
                    combined_df = combined_df.drop_duplicates(keep='last')
                    print(f"   সব ডাটা মার্জ: {len(combined_df)} রো")

                if os.path.exists(temp_remote):
                    os.remove(temp_remote)

                return combined_df

            except Exception as e:
                print(f"   রিমোট ফাইল নেই, শুধু লোকাল ডাটা আপলোড হবে")
                if os.path.exists(temp_remote):
                    os.remove(temp_remote)
                return local_df

        except Exception as e:
            print(f"⚠️ মার্জিং ব্যর্থ: {e}")
            if os.path.exists(temp_remote):
                os.remove(temp_remote)
            return None

    def upload_file_with_retry(self, file_path, filename, retries=3, delay=2):
        """রিট্রাই সহ ফাইল আপলোড"""
        for attempt in range(1, retries + 1):
            try:
                self.api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=filename,
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    token=self.token
                )
                return True
            except Exception as e:
                print(f"   ⏳ আপলোড চেষ্টা {attempt} ব্যর্থ: {e}")
                if attempt < retries:
                    time.sleep(delay * attempt)
        return False

    def smart_upload(self, local_folder="./csv", unique_columns=None):
        """স্মার্ট আপলোড ফাংশন"""
        # ... (আগের কোড)
        pass

# ==================== সাধারণ ফাংশন ====================

def simple_upload(folder_path="./csv", repo_id=REPO_ID, token=HF_TOKEN, retries=3, delay=5):
    """সাধারণ ফোল্ডার আপলোড"""
    # ... (আগের কোড)
    pass

def download_from_hf(local_dir="./csv", repo_id=REPO_ID, token=HF_TOKEN):
    """HF থেকে ডাটাসেট ডাউনলোড"""
    # ... (আগের কোড)
    pass
