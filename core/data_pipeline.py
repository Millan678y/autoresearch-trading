"""
DATA PIPELINE — Real-Time & Historical Data Ingestion

Handles:
1. Historical OHLCV for BTC/USD and XAU/USD (yfinance + CryptoCompare)
2. News sentiment via free RSS feeds + VADER
3. Macroeconomic indicators via FRED API
4. Funding rates for BTC (Hyperliquid/Binance)
5. Train/val/test split management

All data cached to ~/.cache/autotrader/data/
"""

import os
import time
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autotrader", "data")
SENTIMENT_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "autotrader", "sentiment")

SYMBOLS = {
    "BTC": {"yf": "BTC-USD", "type": "crypto", "exchange": "crypto"},
    "XAU": {"yf": "GC=F", "type": "commodity", "exchange": "COMEX"},
}

# Date splits
TRAIN_START = "2024-01-01"
TRAIN_END = "2024-12-31"
VAL_START = "2025-01-01"
VAL_END = "2025-06-30"
TEST_START = "2025-07-01"
TEST_END = "2025-12-31"

# Macro indicators from FRED
MACRO_SERIES = {
    "DXY": "DTWEXBGS",        # Trade-weighted USD index
    "US10Y": "DGS10",          # 10-year Treasury yield
    "VIX": "VIXCLS",           # VIX volatility index
    "FED_RATE": "FEDFUNDS",    # Fed funds rate
    "CPI_YOY": "CPIAUCSL",    # CPI (compute YoY change)
    "GOLD_ETF_FLOW": "GLDM",  # Gold ETF as proxy
}

# News RSS feeds for sentiment
NEWS_FEEDS = {
    "crypto": [
        "https://cointelegraph.com/rss",
        "https://coindesk.com/arc/outboundfeeds/rss/",
    ],
    "gold": [
        "https://www.kitco.com/rss/gold.xml",
    ],
    "macro": [
        "https://feeds.reuters.com/reuters/businessNews",
    ],
}


# ─────────────────────────────────────────────────────────────────
# Historical OHLCV Download
# ─────────────────────────────────────────────────────────────────

def download_ohlcv(symbol: str, interval: str = "1h",
                   start: str = TRAIN_START, end: str = TEST_END,
                   force: bool = False) -> pd.DataFrame:
    """
    Download historical OHLCV data for a symbol.
    Uses yfinance for both BTC and XAU (Gold futures).
    Caches to parquet.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    cache_key = f"{symbol}_{interval}_{start}_{end}"
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.parquet")
    
    if os.path.exists(cache_path) and not force:
        df = pd.read_parquet(cache_path)
        print(f"  {symbol}: loaded {len(df)} bars from cache")
        return df
    
    try:
        import yfinance as yf
    except ImportError:
        print("  yfinance not installed. Run: pip install yfinance")
        return pd.DataFrame()
    
    sym_info = SYMBOLS.get(symbol, {"yf": symbol, "type": "unknown"})
    yf_ticker = sym_info["yf"]
    
    print(f"  {symbol}: downloading {interval} bars from yfinance ({yf_ticker})...")
    
    # yfinance limits 1h data to ~730 days, so chunk if needed
    all_dfs = []
    current_start = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    
    while current_start < end_ts:
        chunk_end = min(current_start + timedelta(days=700), end_ts)
        
        try:
            df = yf.download(
                yf_ticker,
                start=current_start.strftime("%Y-%m-%d"),
                end=chunk_end.strftime("%Y-%m-%d"),
                interval=interval,
                progress=False,
            )
            if len(df) > 0:
                all_dfs.append(df)
        except Exception as e:
            print(f"  {symbol}: download error for chunk: {e}")
        
        current_start = chunk_end
        time.sleep(0.5)
    
    if not all_dfs:
        print(f"  {symbol}: no data downloaded")
        return pd.DataFrame()
    
    df = pd.concat(all_dfs)
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()
    
    # Normalize columns
    result = pd.DataFrame({
        "timestamp": (df.index.astype(np.int64) // 10**6).astype(np.int64),
        "open": df["Open"].values.flatten(),
        "high": df["High"].values.flatten(),
        "low": df["Low"].values.flatten(),
        "close": df["Close"].values.flatten(),
        "volume": df["Volume"].values.flatten(),
    })
    
    # Add empty funding_rate column (filled later for crypto)
    result["funding_rate"] = 0.0
    
    result.to_parquet(cache_path, index=False)
    print(f"  {symbol}: saved {len(result)} bars to {cache_path}")
    
    return result


def download_all(symbols: List[str] = None, interval: str = "1h",
                 force: bool = False) -> Dict[str, pd.DataFrame]:
    """Download OHLCV for all target symbols."""
    if symbols is None:
        symbols = list(SYMBOLS.keys())
    
    result = {}
    for sym in symbols:
        df = download_ohlcv(sym, interval=interval, force=force)
        if len(df) > 0:
            result[sym] = df
    return result


def load_split(symbol: str, split: str, interval: str = "1h") -> pd.DataFrame:
    """Load data for a specific train/val/test split."""
    splits = {
        "train": (TRAIN_START, TRAIN_END),
        "val": (VAL_START, VAL_END),
        "test": (TEST_START, TEST_END),
    }
    assert split in splits, f"split must be one of {list(splits.keys())}"
    
    # Load full dataset
    cache_key = f"{symbol}_{interval}_{TRAIN_START}_{TEST_END}"
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.parquet")
    
    if not os.path.exists(cache_path):
        download_ohlcv(symbol, interval=interval)
    
    df = pd.read_parquet(cache_path)
    
    start_str, end_str = splits[split]
    start_ms = int(pd.Timestamp(start_str, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end_str, tz="UTC").timestamp() * 1000)
    
    mask = (df["timestamp"] >= start_ms) & (df["timestamp"] < end_ms)
    return df[mask].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────
# News Sentiment
# ─────────────────────────────────────────────────────────────────

def fetch_news_sentiment(category: str = "crypto") -> List[dict]:
    """
    Fetch recent news headlines from RSS feeds and compute sentiment.
    Uses VADER (rule-based, fast, no API key needed).
    
    Returns list of {title, published, sentiment, compound, source}
    """
    try:
        import feedparser
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        print("  Install: pip install feedparser vaderSentiment")
        return []
    
    os.makedirs(SENTIMENT_CACHE, exist_ok=True)
    analyzer = SentimentIntensityAnalyzer()
    
    feeds = NEWS_FEEDS.get(category, NEWS_FEEDS["macro"])
    results = []
    
    for feed_url in feeds:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:20]:  # Last 20 headlines
                title = entry.get("title", "")
                published = entry.get("published", "")
                scores = analyzer.polarity_scores(title)
                
                results.append({
                    "title": title,
                    "published": published,
                    "sentiment": "bullish" if scores["compound"] > 0.2 else
                                "bearish" if scores["compound"] < -0.2 else "neutral",
                    "compound": scores["compound"],
                    "source": feed_url.split("/")[2],
                })
        except Exception as e:
            print(f"  Feed error ({feed_url}): {e}")
    
    # Cache
    cache_file = os.path.join(SENTIMENT_CACHE, f"{category}_{int(time.time())}.json")
    with open(cache_file, "w") as f:
        json.dump(results, f)
    
    return results


def compute_sentiment_score(headlines: List[dict]) -> dict:
    """
    Aggregate headline sentiments into a single score.
    
    Returns {score: float [-1, 1], bullish_pct, bearish_pct, count}
    """
    if not headlines:
        return {"score": 0.0, "bullish_pct": 0, "bearish_pct": 0, "count": 0}
    
    compounds = [h["compound"] for h in headlines]
    bullish = sum(1 for c in compounds if c > 0.2)
    bearish = sum(1 for c in compounds if c < -0.2)
    
    return {
        "score": float(np.mean(compounds)),
        "bullish_pct": bullish / len(compounds) * 100,
        "bearish_pct": bearish / len(compounds) * 100,
        "count": len(compounds),
    }


# ─────────────────────────────────────────────────────────────────
# Macroeconomic Indicators
# ─────────────────────────────────────────────────────────────────

def fetch_macro_indicators(fred_api_key: Optional[str] = None) -> dict:
    """
    Fetch macroeconomic indicators from FRED.
    
    Set FRED_API_KEY env var or pass directly.
    Free API key from: https://fred.stlouisfed.org/docs/api/api_key.html
    
    Returns dict of indicator_name -> {value, date, trend}
    """
    api_key = fred_api_key or os.environ.get("FRED_API_KEY")
    if not api_key:
        return _get_macro_fallback()
    
    try:
        from fredapi import Fred
    except ImportError:
        print("  Install: pip install fredapi")
        return _get_macro_fallback()
    
    fred = Fred(api_key=api_key)
    results = {}
    
    for name, series_id in MACRO_SERIES.items():
        try:
            data = fred.get_series(series_id, observation_start="2024-01-01")
            if len(data) > 0:
                current = float(data.iloc[-1])
                prev = float(data.iloc[-2]) if len(data) > 1 else current
                
                # Simple trend detection
                if len(data) > 20:
                    sma20 = float(data.iloc[-20:].mean())
                    trend = "rising" if current > sma20 else "falling"
                else:
                    trend = "rising" if current > prev else "falling"
                
                results[name] = {
                    "value": current,
                    "previous": prev,
                    "change": current - prev,
                    "change_pct": (current - prev) / abs(prev) * 100 if prev != 0 else 0,
                    "trend": trend,
                    "date": str(data.index[-1].date()),
                }
        except Exception as e:
            print(f"  FRED error ({name}): {e}")
    
    return results


def _get_macro_fallback() -> dict:
    """
    Fallback macro indicators when FRED API key is unavailable.
    Uses yfinance for DXY proxy, VIX, and 10Y yield.
    """
    try:
        import yfinance as yf
    except ImportError:
        return {}
    
    indicators = {}
    proxies = {
        "DXY": "DX-Y.NYB",    # USD index
        "VIX": "^VIX",         # VIX
        "US10Y": "^TNX",       # 10-year yield
        "SPX": "^GSPC",        # S&P 500 (risk sentiment)
    }
    
    for name, ticker in proxies.items():
        try:
            data = yf.download(ticker, period="1mo", interval="1d", progress=False)
            if len(data) > 0:
                current = float(data["Close"].iloc[-1])
                prev = float(data["Close"].iloc[-2]) if len(data) > 1 else current
                indicators[name] = {
                    "value": current,
                    "previous": prev,
                    "change_pct": (current - prev) / abs(prev) * 100 if prev != 0 else 0,
                    "trend": "rising" if current > prev else "falling",
                }
        except:
            pass
    
    return indicators


def get_macro_context() -> dict:
    """
    Build a macro context summary for strategy generation.
    
    Returns {
        risk_environment: "risk_on" | "risk_off" | "neutral",
        dollar_trend: "strengthening" | "weakening",
        rate_environment: "hawkish" | "dovish" | "neutral",
        volatility: "low" | "normal" | "high" | "extreme",
        signals: {indicator: direction}
    }
    """
    macro = fetch_macro_indicators()
    if not macro:
        return {
            "risk_environment": "neutral",
            "dollar_trend": "neutral",
            "rate_environment": "neutral",
            "volatility": "normal",
            "signals": {},
        }
    
    signals = {}
    
    # Dollar
    dxy = macro.get("DXY", {})
    dollar_trend = dxy.get("trend", "neutral")
    signals["DXY"] = dollar_trend
    
    # Volatility
    vix = macro.get("VIX", {})
    vix_val = vix.get("value", 20)
    if vix_val < 15:
        vol_regime = "low"
    elif vix_val < 25:
        vol_regime = "normal"
    elif vix_val < 35:
        vol_regime = "high"
    else:
        vol_regime = "extreme"
    signals["VIX"] = vol_regime
    
    # Rates
    rate = macro.get("FED_RATE", macro.get("US10Y", {}))
    rate_trend = rate.get("trend", "neutral")
    rate_env = "hawkish" if rate_trend == "rising" else "dovish" if rate_trend == "falling" else "neutral"
    signals["RATES"] = rate_env
    
    # Risk sentiment
    spx = macro.get("SPX", {})
    spx_trend = spx.get("trend", "neutral")
    risk = "risk_on" if spx_trend == "rising" else "risk_off" if spx_trend == "falling" else "neutral"
    
    return {
        "risk_environment": risk,
        "dollar_trend": "weakening" if dollar_trend == "falling" else "strengthening",
        "rate_environment": rate_env,
        "volatility": vol_regime,
        "signals": signals,
    }


# ─────────────────────────────────────────────────────────────────
# Feature Engineering (for strategy signals)
# ─────────────────────────────────────────────────────────────────

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical and derived features from OHLCV data.
    Returns DataFrame with additional columns for strategy use.
    """
    df = df.copy()
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values
    v = df["volume"].values
    
    # Returns at various horizons
    for period in [1, 4, 8, 12, 24, 48]:
        df[f"ret_{period}"] = df["close"].pct_change(period)
    
    # Volatility
    df["vol_12"] = df["close"].pct_change().rolling(12).std()
    df["vol_24"] = df["close"].pct_change().rolling(24).std()
    df["vol_72"] = df["close"].pct_change().rolling(72).std()
    
    # Vol regime (ratio of short to long vol)
    df["vol_ratio"] = df["vol_12"] / df["vol_72"].clip(lower=1e-10)
    
    # ATR
    for period in [14, 24]:
        tr = np.maximum(h[1:] - l[1:],
                       np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
        atr = pd.Series(np.concatenate([[np.nan], tr])).rolling(period).mean()
        df[f"atr_{period}"] = atr.values
    
    # RSI
    for period in [8, 14]:
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.clip(lower=1e-10)
        df[f"rsi_{period}"] = 100 - 100 / (1 + rs)
    
    # MACD
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    
    # Bollinger Bands
    sma20 = df["close"].rolling(20).mean()
    std20 = df["close"].rolling(20).std()
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma20
    df["bb_pctb"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).clip(lower=1e-10)
    
    # Volume features
    df["vol_sma"] = df["volume"].rolling(24).mean()
    df["vol_ratio_24"] = df["volume"] / df["vol_sma"].clip(lower=1)
    
    # Price position relative to recent range
    for period in [24, 72]:
        rolling_high = df["high"].rolling(period).max()
        rolling_low = df["low"].rolling(period).min()
        df[f"price_position_{period}"] = (
            (df["close"] - rolling_low) / (rolling_high - rolling_low).clip(lower=1e-10)
        )
    
    return df


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def prepare_all(force: bool = False):
    """Download and prepare all data for backtesting."""
    print("=" * 50)
    print("DATA PIPELINE — Preparing all data")
    print("=" * 50)
    
    print("\n📊 Downloading OHLCV data...")
    data = download_all(force=force)
    
    for sym, df in data.items():
        print(f"  {sym}: {len(df)} bars ({df['timestamp'].min()} → {df['timestamp'].max()})")
    
    print("\n📰 Fetching news sentiment...")
    for category in ["crypto", "gold", "macro"]:
        headlines = fetch_news_sentiment(category)
        sentiment = compute_sentiment_score(headlines)
        print(f"  {category}: {sentiment['count']} headlines, "
              f"score={sentiment['score']:.3f} "
              f"(bull={sentiment['bullish_pct']:.0f}% bear={sentiment['bearish_pct']:.0f}%)")
    
    print("\n🏛️  Fetching macro indicators...")
    macro = get_macro_context()
    print(f"  Risk: {macro['risk_environment']}")
    print(f"  Dollar: {macro['dollar_trend']}")
    print(f"  Rates: {macro['rate_environment']}")
    print(f"  Volatility: {macro['volatility']}")
    
    print("\nDone! Data ready for backtesting.")
    return data


if __name__ == "__main__":
    prepare_all()
