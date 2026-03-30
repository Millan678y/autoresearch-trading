"""
BINANCE DATA ENGINE — Historical & Real-Time Data from Binance

Downloads kline (candlestick) data directly from Binance public API.
No API key required for historical data.

Supports:
- Any timeframe: 1m, 3m, 5m, 15m, 1h, 4h, 1d
- Any trading pair: BTCUSDT, XAUUSDT, ETHUSDT, etc.
- Automatic pagination (Binance returns max 1000 candles per request)
- Incremental downloads (only fetch new data)
- Parquet caching for fast reloads
- Train/val/test splitting

Usage:
    from core.binance_data import BinanceDataLoader
    loader = BinanceDataLoader()
    loader.download("BTCUSDT", "5m")
    df = loader.load("BTCUSDT", "5m", split="train")
"""

import os
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
BINANCE_TICKER_URL = "https://api.binance.com/api/v3/ticker/24hr"

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autotrader", "binance")

# Default symbols for scalping
SCALP_SYMBOLS = ["BTCUSDT", "ETHUSDT", "XAUUSDT", "SOLUSDT"]

# Time splits for backtesting
# In-sample: Full year 2024 (backtest)
# Out-of-sample: 2025 data (forward test)
SPLITS = {
    "train": ("2024-01-01", "2024-12-31"),   # Full 2024 — in-sample backtest
    "val":   ("2025-01-01", "2025-06-30"),   # H1 2025 — out-of-sample forward test
    "test":  ("2025-07-01", "2025-12-31"),   # H2 2025 — final validation
}

# Binance interval constants (ms per candle)
INTERVAL_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}

MAX_CANDLES_PER_REQUEST = 1000
REQUEST_DELAY = 0.15  # Be nice to Binance API


# ─────────────────────────────────────────────────────────────────
# Core Download Functions
# ─────────────────────────────────────────────────────────────────

def _ts_to_ms(date_str: str) -> int:
    """Convert 'YYYY-MM-DD' to millisecond timestamp."""
    return int(pd.Timestamp(date_str, tz="UTC").timestamp() * 1000)


def _ms_to_str(ms: int) -> str:
    """Convert ms timestamp to readable string."""
    return datetime.utcfromtimestamp(ms / 1000).strftime("%Y-%m-%d %H:%M")


def download_klines(symbol: str, interval: str,
                    start_date: str, end_date: str,
                    verbose: bool = True) -> pd.DataFrame:
    """
    Download kline data from Binance public API.
    
    Automatically paginates — Binance returns max 1000 candles per request.
    No API key needed.
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT", "XAUUSDT")
        interval: Candle interval ("1m", "5m", "15m", "1h", etc.)
        start_date: Start date "YYYY-MM-DD"
        end_date: End date "YYYY-MM-DD"
    
    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume,
        quote_volume, trades, taker_buy_volume, taker_buy_quote_volume
    """
    start_ms = _ts_to_ms(start_date)
    end_ms = _ts_to_ms(end_date)
    interval_ms = INTERVAL_MS.get(interval, 300_000)
    
    all_candles = []
    current_start = start_ms
    total_expected = (end_ms - start_ms) // interval_ms
    
    if verbose:
        print(f"  {symbol} {interval}: downloading {_ms_to_str(start_ms)} → {_ms_to_str(end_ms)}")
        print(f"  Expected ~{total_expected:,} candles...")
    
    request_count = 0
    
    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": MAX_CANDLES_PER_REQUEST,
        }
        
        try:
            resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=15)
            
            if resp.status_code == 429:
                # Rate limited — wait and retry
                wait = int(resp.headers.get("Retry-After", 30))
                if verbose:
                    print(f"  Rate limited. Waiting {wait}s...")
                time.sleep(wait)
                continue
            
            resp.raise_for_status()
            data = resp.json()
            
            if not data:
                break
            
            for candle in data:
                all_candles.append({
                    "timestamp": int(candle[0]),
                    "open": float(candle[1]),
                    "high": float(candle[2]),
                    "low": float(candle[3]),
                    "close": float(candle[4]),
                    "volume": float(candle[5]),
                    "close_time": int(candle[6]),
                    "quote_volume": float(candle[7]),
                    "trades": int(candle[8]),
                    "taker_buy_volume": float(candle[9]),
                    "taker_buy_quote_volume": float(candle[10]),
                })
            
            # Move to next chunk
            last_ts = int(data[-1][0])
            if last_ts <= current_start:
                break  # No progress, avoid infinite loop
            current_start = last_ts + interval_ms
            
            request_count += 1
            if verbose and request_count % 20 == 0:
                pct = len(all_candles) / max(total_expected, 1) * 100
                print(f"  ... {len(all_candles):,} candles ({pct:.0f}%)")
            
            time.sleep(REQUEST_DELAY)
        
        except requests.exceptions.RequestException as e:
            if verbose:
                print(f"  Request error: {e}. Retrying in 5s...")
            time.sleep(5)
            continue
    
    if not all_candles:
        if verbose:
            print(f"  {symbol}: no data returned")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_candles)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    
    if verbose:
        print(f"  {symbol} {interval}: {len(df):,} candles downloaded")
    
    return df


# ─────────────────────────────────────────────────────────────────
# Binance Data Loader (with caching)
# ─────────────────────────────────────────────────────────────────

class BinanceDataLoader:
    """
    High-level data loader with caching and split management.
    
    Usage:
        loader = BinanceDataLoader()
        loader.download_all()  # Download all symbols
        train = loader.load("BTCUSDT", "5m", split="train")
        val = loader.load("BTCUSDT", "5m", split="val")
    """
    
    def __init__(self, cache_dir: str = CACHE_DIR,
                 symbols: List[str] = None,
                 interval: str = "5m"):
        self.cache_dir = cache_dir
        self.symbols = symbols or SCALP_SYMBOLS
        self.interval = interval
        os.makedirs(cache_dir, exist_ok=True)
    
    def _cache_path(self, symbol: str, interval: str) -> str:
        return os.path.join(self.cache_dir, f"{symbol}_{interval}.parquet")
    
    def download(self, symbol: str, interval: str = None,
                 force: bool = False) -> pd.DataFrame:
        """Download and cache data for a single symbol."""
        interval = interval or self.interval
        path = self._cache_path(symbol, interval)
        
        if os.path.exists(path) and not force:
            df = pd.read_parquet(path)
            print(f"  {symbol} {interval}: loaded {len(df):,} candles from cache")
            
            # Check if we need to update (fetch new candles)
            last_ts = df["timestamp"].max()
            now_ms = int(time.time() * 1000)
            gap_hours = (now_ms - last_ts) / 3_600_000
            
            if gap_hours > 24:
                # Fetch new candles and append
                new_start = datetime.utcfromtimestamp(last_ts / 1000).strftime("%Y-%m-%d")
                new_end = datetime.utcnow().strftime("%Y-%m-%d")
                print(f"  Updating: {gap_hours:.0f}h of new data...")
                
                new_df = download_klines(symbol, interval, new_start, new_end, verbose=False)
                if len(new_df) > 0:
                    df = pd.concat([df, new_df]).drop_duplicates(subset=["timestamp"])
                    df = df.sort_values("timestamp").reset_index(drop=True)
                    df.to_parquet(path, index=False)
                    print(f"  Updated: {len(df):,} total candles")
            
            return df
        
        # Full download
        # Get earliest available date from all splits
        all_starts = [v[0] for v in SPLITS.values()]
        all_ends = [v[1] for v in SPLITS.values()]
        start = min(all_starts)
        end = max(all_ends)
        
        # Also get up to today for forward testing
        today = datetime.utcnow().strftime("%Y-%m-%d")
        if today > end:
            end = today
        
        df = download_klines(symbol, interval, start, end)
        
        if len(df) > 0:
            df.to_parquet(path, index=False)
        
        return df
    
    def download_all(self, force: bool = False) -> Dict[str, pd.DataFrame]:
        """Download data for all configured symbols."""
        print("=" * 60)
        print(f"BINANCE DATA LOADER — {self.interval} timeframe")
        print(f"Symbols: {', '.join(self.symbols)}")
        print("=" * 60)
        
        result = {}
        for symbol in self.symbols:
            df = self.download(symbol, force=force)
            if len(df) > 0:
                result[symbol] = df
                hours = len(df) * INTERVAL_MS.get(self.interval, 300000) / 3_600_000
                print(f"  → {len(df):,} candles ({hours:.0f} hours of data)")
            print()
        
        self._print_summary(result)
        return result
    
    def load(self, symbol: str, interval: str = None,
             split: str = "train") -> pd.DataFrame:
        """Load data for a specific split."""
        interval = interval or self.interval
        path = self._cache_path(symbol, interval)
        
        if not os.path.exists(path):
            print(f"  No cached data for {symbol}. Downloading...")
            self.download(symbol, interval)
        
        if not os.path.exists(path):
            return pd.DataFrame()
        
        df = pd.read_parquet(path)
        
        if split not in SPLITS:
            return df
        
        start_str, end_str = SPLITS[split]
        start_ms = _ts_to_ms(start_str)
        end_ms = _ts_to_ms(end_str)
        
        mask = (df["timestamp"] >= start_ms) & (df["timestamp"] < end_ms)
        result = df[mask].reset_index(drop=True)
        
        return result
    
    def load_all(self, split: str = "train") -> Dict[str, pd.DataFrame]:
        """Load data for all symbols for a given split."""
        result = {}
        for symbol in self.symbols:
            df = self.load(symbol, split=split)
            if len(df) > 0:
                result[symbol] = df
        return result
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price from Binance ticker."""
        try:
            resp = requests.get(BINANCE_TICKER_URL,
                              params={"symbol": symbol}, timeout=5)
            data = resp.json()
            return float(data["lastPrice"])
        except:
            return None
    
    def _print_summary(self, data: Dict[str, pd.DataFrame]):
        """Print a summary of downloaded data."""
        print("\n" + "─" * 50)
        print("DATA SUMMARY")
        print("─" * 50)
        print(f"{'Symbol':<12} {'Candles':>10} {'From':>12} {'To':>12}")
        print("─" * 50)
        
        for symbol, df in data.items():
            n = len(df)
            t_min = _ms_to_str(df["timestamp"].min())[:10]
            t_max = _ms_to_str(df["timestamp"].max())[:10]
            print(f"{symbol:<12} {n:>10,} {t_min:>12} {t_max:>12}")
        
        # Split sizes
        print("\nSplit sizes:")
        for split_name, (s, e) in SPLITS.items():
            s_ms = _ts_to_ms(s)
            e_ms = _ts_to_ms(e)
            for symbol, df in data.items():
                n = ((df["timestamp"] >= s_ms) & (df["timestamp"] < e_ms)).sum()
                print(f"  {split_name:<8} {symbol:<12} {n:>8,} candles")


# ─────────────────────────────────────────────────────────────────
# Scalping-Specific Features
# ─────────────────────────────────────────────────────────────────

def compute_scalp_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute features optimized for 5-minute scalping.
    
    Includes microstructure features not useful on higher timeframes:
    - Taker buy ratio (aggressor imbalance)
    - Trade count spikes
    - Micro-volatility
    - Speed of price change
    - Bid/ask pressure proxy
    """
    df = df.copy()
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values
    v = df["volume"].values
    
    # ── Taker Buy Ratio (order flow proxy) ──
    # Ratio of taker buy volume to total volume
    if "taker_buy_volume" in df.columns:
        df["taker_buy_ratio"] = df["taker_buy_volume"] / df["volume"].clip(lower=1e-10)
        df["taker_sell_ratio"] = 1 - df["taker_buy_ratio"]
        # Imbalance: positive = more buying, negative = more selling
        df["taker_imbalance"] = df["taker_buy_ratio"] - 0.5
    
    # ── Trade Count Features ──
    if "trades" in df.columns:
        df["trades_sma"] = df["trades"].rolling(12).mean()
        df["trades_spike"] = df["trades"] / df["trades_sma"].clip(lower=1)
    
    # ── Micro Returns (1-6 candles) ──
    for period in [1, 2, 3, 6, 12]:
        df[f"ret_{period}"] = df["close"].pct_change(period)
    
    # ── Speed (rate of change per bar) ──
    df["speed_3"] = df["ret_3"] / 3
    df["speed_6"] = df["ret_6"] / 6
    df["acceleration"] = df["speed_3"] - df["speed_3"].shift(3)
    
    # ── Micro Volatility ──
    df["microvol_6"] = df["close"].pct_change().rolling(6).std()
    df["microvol_12"] = df["close"].pct_change().rolling(12).std()
    df["microvol_ratio"] = df["microvol_6"] / df["microvol_12"].clip(lower=1e-10)
    
    # ── Candle Body / Shadow Ratios ──
    candle_range = (h - l)
    candle_range = np.where(candle_range == 0, 1e-10, candle_range)
    body = np.abs(c - df["open"].values)
    df["body_ratio"] = body / candle_range
    df["upper_shadow_ratio"] = (h - np.maximum(c, df["open"].values)) / candle_range
    df["lower_shadow_ratio"] = (np.minimum(c, df["open"].values) - l) / candle_range
    
    # ── Volume Features ──
    df["vol_sma_12"] = df["volume"].rolling(12).mean()
    df["vol_sma_36"] = df["volume"].rolling(36).mean()
    df["vol_spike"] = df["volume"] / df["vol_sma_12"].clip(lower=1)
    df["vol_trend"] = df["vol_sma_12"] / df["vol_sma_36"].clip(lower=1)
    
    # ── Price Position ──
    for period in [12, 36, 72]:
        rolling_high = df["high"].rolling(period).max()
        rolling_low = df["low"].rolling(period).min()
        rng = (rolling_high - rolling_low).clip(lower=1e-10)
        df[f"price_pos_{period}"] = (df["close"] - rolling_low) / rng
    
    # ── ATR (fast for scalping) ──
    for period in [6, 12, 24]:
        tr = np.maximum(
            h[1:] - l[1:],
            np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1]))
        )
        atr_series = pd.Series(np.concatenate([[np.nan], tr])).rolling(period).mean()
        df[f"atr_{period}"] = atr_series.values
    
    # ── RSI (fast) ──
    for period in [6, 9, 14]:
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.clip(lower=1e-10)
        df[f"rsi_{period}"] = 100 - 100 / (1 + rs)
    
    # ── EMA ──
    for span in [5, 9, 21, 50]:
        df[f"ema_{span}"] = df["close"].ewm(span=span).mean()
    
    # ── VWAP (session-based proxy using rolling window) ──
    tp = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap_72"] = (tp * df["volume"]).rolling(72).sum() / df["volume"].rolling(72).sum().clip(lower=1)
    df["price_vs_vwap"] = (df["close"] - df["vwap_72"]) / df["vwap_72"].clip(lower=1e-10)
    
    return df


# ─────────────────────────────────────────────────────────────────
# Quick Download Script
# ─────────────────────────────────────────────────────────────────

def quick_download(symbols: List[str] = None, interval: str = "5m"):
    """One-liner to download everything needed for scalping backtests."""
    loader = BinanceDataLoader(symbols=symbols, interval=interval)
    return loader.download_all()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Binance historical data")
    parser.add_argument("--symbols", nargs="+", default=SCALP_SYMBOLS,
                       help="Trading pairs (e.g., BTCUSDT ETHUSDT)")
    parser.add_argument("--interval", default="5m",
                       help="Candle interval (1m, 5m, 15m, 1h, etc.)")
    parser.add_argument("--force", action="store_true",
                       help="Force re-download (ignore cache)")
    args = parser.parse_args()
    
    loader = BinanceDataLoader(symbols=args.symbols, interval=args.interval)
    loader.download_all(force=args.force)
