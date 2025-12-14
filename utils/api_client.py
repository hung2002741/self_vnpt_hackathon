import json
import requests
import hashlib
import os
import threading
import sys
import time
from datetime import datetime, timedelta

# --- CONSTANTS ---
URL_SMALL = "https://api.idg.vnpt.vn/data-service/v1/chat/completions/vnptai-hackathon-small"
URL_LARGE = "https://api.idg.vnpt.vn/data-service/v1/chat/completions/vnptai-hackathon-large"
URL_EMBED = "https://api.idg.vnpt.vn/data-service/vnptai-hackathon-embedding"

CACHE_FILE = "local_cache.json"
USAGE_FILE = "api_usage.json" # New file to track hourly usage

class APIClient:
    def __init__(self, config_path='api-keys.json'):
        self.configs = self._load_configs(config_path)
        self.cache = self._load_cache()
        self.usage = self._load_usage()
        self.lock = threading.Lock()

        # Define limits per hour
        self.limits = {
            "small": 60,
            "large": 40
        }

    def _load_configs(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            config_map = {}
            for item in data:
                name = item.get('llmApiName', '').lower()
                if 'small' in name: config_map['small'] = item
                elif 'large' in name: config_map['large'] = item
                elif 'embed' in name: config_map['embedding'] = item
            return config_map
        except FileNotFoundError:
            print(f"CRITICAL: {path} not found.")
            sys.exit(1)

    # --- CACHE MANAGEMENT ---
    def _load_cache(self):
        if not os.path.exists(CACHE_FILE): return {}
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f: return json.load(f)
        except: return {}

    def _save_cache(self):
        with self.lock:
            with open(CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)

    def _get_cache_key(self, model, content, params):
        payload = {"m": model, "c": content, "p": params}
        return hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    # --- USAGE TRACKING (NEW) ---
    def _load_usage(self):
        """Loads usage stats (hour and count) from disk."""
        if not os.path.exists(USAGE_FILE):
            return {"small": {"hour": -1, "count": 0}, "large": {"hour": -1, "count": 0}}
        try:
            with open(USAGE_FILE, 'r') as f: return json.load(f)
        except:
            return {"small": {"hour": -1, "count": 0}, "large": {"hour": -1, "count": 0}}

    def _save_usage(self):
        """Saves usage stats to disk to persist across script restarts."""
        with self.lock:
            with open(USAGE_FILE, 'w') as f:
                json.dump(self.usage, f)

    def _check_quota_and_sleep(self, model_type):
        """
        Checks if we have reached the limit for the current hour.
        If yes, sleeps until the next hour starts.
        """
        if model_type not in self.limits: return # Embeddings don't use this strict logic

        limit = self.limits[model_type]
        now = datetime.now()
        current_hour = now.hour

        # Update local memory from file in case another process updated it
        # (Optional but good safety if running multiple scripts)
        self.usage = self._load_usage()
        
        stats = self.usage.get(model_type, {"hour": -1, "count": 0})

        # 1. Reset if new hour
        if stats['hour'] != current_hour:
            print(f"[{model_type.upper()}] New hour detected ({current_hour}:00). Resetting quota.")
            stats['hour'] = current_hour
            stats['count'] = 0
            self.usage[model_type] = stats
            self._save_usage()

        # 2. Check Limit
        if stats['count'] >= limit:
            # Calculate seconds until next hour
            next_hour_time = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
            sleep_seconds = (next_hour_time - now).total_seconds() + 5 # +5s buffer
            
            print(f"\n⚠️ QUOTA HIT for {model_type.upper()} ({stats['count']}/{limit}).")
            print(f"💤 Sleeping {int(sleep_seconds)} seconds until {next_hour_time.strftime('%H:%M:%S')}...")
            
            time.sleep(sleep_seconds)
            
            # After waking up, reset logic
            self.usage[model_type]['hour'] = datetime.now().hour
            self.usage[model_type]['count'] = 0
            self._save_usage()
            print(f"⏰ Woke up! Resuming {model_type.upper()}...")

    def _increment_usage(self, model_type):
        """Increments the counter after a successful API call."""
        if model_type not in self.limits: return
        
        self.usage[model_type]['count'] += 1
        self._save_usage()
        print(f"[{model_type.upper()} Usage: {self.usage[model_type]['count']}/{self.limits[model_type]}]")


    # --- API CALLS ---
    def _get_headers(self, model_type):
        conf = self.configs.get(model_type)
        return {
            'Authorization': conf['authorization'],
            'Token-id': conf['tokenId'],
            'Token-key': conf['tokenKey'],
            'Content-Type': 'application/json'
        }

    def call_chat(self, messages, model_type="small", temperature=0.7, n=1, top_p=1.0, top_k=50):
        params = {"t": temperature, "n": n, "tp": top_p, "tk": top_k}
        cache_key = self._get_cache_key(model_type, messages, params)
        
        # 1. Check Cache
        if cache_key in self.cache:
            print(f"[{model_type.upper()}] Cache Hit ⚡")
            return self.cache[cache_key]

        # 2. Check Quota / Sleep if necessary
        self._check_quota_and_sleep(model_type)

        url = URL_SMALL if model_type == "small" else URL_LARGE
        model_name = "vnptai_hackathon_small" if model_type == "small" else "vnptai_hackathon_large"

        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "n": n
        }

        # 3. Network Call with Retry
        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = requests.post(url, headers=self._get_headers(model_type), json=payload, timeout=120)
                
                if resp.status_code == 200:
                    data = resp.json()
                    
                    # Update Cache
                    self.cache[cache_key] = data
                    self._save_cache()
                    
                    # Update Usage Quota
                    self._increment_usage(model_type)
                    
                    return data
                
                elif resp.status_code == 429:
                    print(f"⚠️ API 429 Rate Limit. Sleeping 60s...")
                    time.sleep(60)
                else:
                    print(f"API Error {resp.status_code}: {resp.text}")
                    return None
            except Exception as e:
                print(f"Connection Error: {e}")
                time.sleep(10)
        
        return None

    def get_embedding(self, text):
        cache_key = self._get_cache_key("embed", text, {})
        if cache_key in self.cache: return self.cache[cache_key]

        payload = {
            "model": "vnptai_hackathon_embedding",
            "input": text,
            "encoding_format": "float"
        }
        
        # Embeddings are usually fast (500 req/min), simple retry suffices
        try:
            resp = requests.post(URL_EMBED, headers=self._get_headers('embedding'), json=payload)
            if resp.status_code == 200:
                vec = resp.json()['data'][0]['embedding']
                self.cache[cache_key] = vec
                self._save_cache()
                return vec
            elif resp.status_code == 429:
                 time.sleep(5) # Short sleep for embedding limit
                 return self.get_embedding(text) # Retry once
        except Exception as e:
            print(f"Embed Error: {e}")
        return None