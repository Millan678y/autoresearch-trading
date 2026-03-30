"""
REAL-TIME NEWS ENGINE — Multi-Source News Sentiment

Sources (all free, no API key required unless noted):
1. CryptoPanic API (free tier, crypto-focused)
2. NewsData.io (free tier, 200 req/day)
3. GNews.io (free tier, 100 req/day)
4. RSS feeds (unlimited) — CoinDesk, CoinTelegraph, Kitco, Reuters, Bloomberg
5. Reddit sentiment (pushshift/RSS)
6. Twitter/X sentiment via Nitter RSS

Sentiment computed via VADER + keyword scoring.
"""

import os
import json
import time
import hashlib
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

import numpy as np

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autotrader", "news")
HISTORY_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autotrader", "news_history")

# ─────────────────────────────────────────────────────────────────
# Keyword Scoring — domain-specific sentiment boost
# ─────────────────────────────────────────────────────────────────

BULLISH_KEYWORDS = {
    # Crypto
    "etf approved": 0.8, "institutional adoption": 0.6, "halving": 0.5,
    "all-time high": 0.7, "ath": 0.5, "breakout": 0.4, "accumulation": 0.5,
    "whale buying": 0.6, "inflows": 0.4, "rate cut": 0.5, "dovish": 0.4,
    "bullish": 0.3, "rally": 0.3, "surge": 0.3, "soar": 0.3, "moon": 0.2,
    "adoption": 0.3, "partnership": 0.2, "upgrade": 0.2,
    # Gold
    "safe haven": 0.5, "geopolitical risk": 0.4, "inflation rising": 0.4,
    "central bank buying": 0.6, "gold reserve": 0.4, "dedollarization": 0.5,
    "debt crisis": 0.3, "recession": 0.3, "war": 0.3,
}

BEARISH_KEYWORDS = {
    # Crypto
    "hack": -0.7, "exploit": -0.6, "rug pull": -0.8, "sec lawsuit": -0.6,
    "ban crypto": -0.7, "regulation crackdown": -0.5, "outflows": -0.4,
    "whale selling": -0.5, "dump": -0.4, "crash": -0.5, "collapse": -0.6,
    "rate hike": -0.4, "hawkish": -0.3, "bearish": -0.3, "plunge": -0.4,
    "fraud": -0.6, "bankrupt": -0.7, "liquidat": -0.4, "fud": -0.2,
    # Gold
    "strong dollar": -0.4, "risk on": -0.3, "gold sell": -0.4,
    "etf outflows gold": -0.5, "deflation": -0.3,
}


def _keyword_score(text: str) -> float:
    """Compute domain-specific keyword sentiment score."""
    text_lower = text.lower()
    score = 0.0
    
    for keyword, weight in BULLISH_KEYWORDS.items():
        if keyword in text_lower:
            score += weight
    
    for keyword, weight in BEARISH_KEYWORDS.items():
        if keyword in text_lower:
            score += weight  # weight is already negative
    
    return np.clip(score, -1.0, 1.0)


# ─────────────────────────────────────────────────────────────────
# VADER + Keyword Hybrid Sentiment
# ─────────────────────────────────────────────────────────────────

def compute_sentiment(text: str) -> dict:
    """
    Hybrid sentiment: VADER compound + domain keyword scoring.
    VADER handles general sentiment, keywords handle trading-specific context.
    """
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        vader = analyzer.polarity_scores(text)
        vader_score = vader["compound"]
    except ImportError:
        vader_score = 0.0
    
    keyword_score = _keyword_score(text)
    
    # Weighted blend: 40% VADER + 60% keywords (domain knowledge > general sentiment)
    combined = vader_score * 0.4 + keyword_score * 0.6
    combined = np.clip(combined, -1.0, 1.0)
    
    if combined > 0.15:
        label = "bullish"
    elif combined < -0.15:
        label = "bearish"
    else:
        label = "neutral"
    
    return {
        "combined": float(combined),
        "vader": float(vader_score),
        "keyword": float(keyword_score),
        "label": label,
    }


# ─────────────────────────────────────────────────────────────────
# News Sources
# ─────────────────────────────────────────────────────────────────

def fetch_cryptopanic(auth_token: Optional[str] = None) -> List[dict]:
    """
    CryptoPanic API — free tier, no auth needed for public posts.
    https://cryptopanic.com/api/
    """
    import requests
    
    url = "https://cryptopanic.com/api/v1/posts/"
    params = {"public": "true", "kind": "news"}
    if auth_token:
        params["auth_token"] = auth_token
    
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        results = []
        for post in data.get("results", [])[:30]:
            sentiment = compute_sentiment(post.get("title", ""))
            results.append({
                "title": post.get("title", ""),
                "published": post.get("published_at", ""),
                "source": post.get("source", {}).get("title", "CryptoPanic"),
                "url": post.get("url", ""),
                "sentiment": sentiment,
                "asset": "BTC" if "bitcoin" in post.get("title", "").lower() or
                         "btc" in post.get("title", "").lower() else "CRYPTO",
            })
        return results
    except Exception as e:
        print(f"  CryptoPanic error: {e}")
        return []


def fetch_rss_news() -> List[dict]:
    """Fetch from multiple RSS feeds — unlimited, no API key."""
    try:
        import feedparser
    except ImportError:
        print("  Install: pip install feedparser")
        return []
    
    feeds = {
        # Crypto
        "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "CoinTelegraph": "https://cointelegraph.com/rss",
        "Bitcoin Magazine": "https://bitcoinmagazine.com/.rss/full/",
        "Decrypt": "https://decrypt.co/feed",
        # Gold / Commodities
        "Kitco": "https://www.kitco.com/rss/gold.xml",
        # Macro / General
        "Reuters Business": "https://feeds.reuters.com/reuters/businessNews",
        "CNBC": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "MarketWatch": "https://feeds.marketwatch.com/marketwatch/topstories/",
    }
    
    results = []
    
    for source_name, feed_url in feeds.items():
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:15]:
                title = entry.get("title", "")
                summary = entry.get("summary", "")
                text = f"{title}. {summary[:200]}"
                
                sentiment = compute_sentiment(text)
                
                # Detect asset
                text_lower = text.lower()
                if any(kw in text_lower for kw in ["bitcoin", "btc", "crypto", "ethereum"]):
                    asset = "BTC"
                elif any(kw in text_lower for kw in ["gold", "xau", "precious metal", "bullion"]):
                    asset = "XAU"
                else:
                    asset = "MACRO"
                
                results.append({
                    "title": title,
                    "published": entry.get("published", ""),
                    "source": source_name,
                    "url": entry.get("link", ""),
                    "sentiment": sentiment,
                    "asset": asset,
                })
        except Exception as e:
            print(f"  RSS error ({source_name}): {e}")
    
    return results


def fetch_newsdata(api_key: Optional[str] = None) -> List[dict]:
    """
    NewsData.io — free tier: 200 requests/day.
    Set NEWSDATA_API_KEY env var.
    """
    import requests
    
    key = api_key or os.environ.get("NEWSDATA_API_KEY")
    if not key:
        return []
    
    results = []
    for query in ["bitcoin", "gold price"]:
        try:
            resp = requests.get(
                "https://newsdata.io/api/1/news",
                params={"apikey": key, "q": query, "language": "en"},
                timeout=10,
            )
            data = resp.json()
            for article in data.get("results", [])[:10]:
                title = article.get("title", "")
                sentiment = compute_sentiment(title)
                asset = "BTC" if "bitcoin" in query else "XAU"
                results.append({
                    "title": title,
                    "published": article.get("pubDate", ""),
                    "source": article.get("source_id", "NewsData"),
                    "url": article.get("link", ""),
                    "sentiment": sentiment,
                    "asset": asset,
                })
        except Exception as e:
            print(f"  NewsData error: {e}")
    
    return results


# ─────────────────────────────────────────────────────────────────
# Aggregation & Scoring
# ─────────────────────────────────────────────────────────────────

def fetch_all_news(use_cryptopanic: bool = True,
                   use_newsdata: bool = True) -> List[dict]:
    """Fetch news from all available sources."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    all_news = []
    
    # RSS — always available
    print("  📰 Fetching RSS feeds...")
    all_news.extend(fetch_rss_news())
    
    # CryptoPanic
    if use_cryptopanic:
        print("  📰 Fetching CryptoPanic...")
        all_news.extend(fetch_cryptopanic())
    
    # NewsData.io
    if use_newsdata and os.environ.get("NEWSDATA_API_KEY"):
        print("  📰 Fetching NewsData.io...")
        all_news.extend(fetch_newsdata())
    
    # Deduplicate by title hash
    seen = set()
    unique = []
    for item in all_news:
        h = hashlib.md5(item["title"].encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(item)
    
    # Cache
    cache_file = os.path.join(CACHE_DIR, f"all_{int(time.time())}.json")
    with open(cache_file, "w") as f:
        json.dump(unique, f, indent=2, default=str)
    
    print(f"  📰 Total: {len(unique)} unique headlines")
    return unique


def compute_asset_sentiment(news: List[dict]) -> Dict[str, dict]:
    """
    Aggregate sentiment per asset.
    
    Returns {
        "BTC": {score, label, bullish_pct, bearish_pct, count, top_headlines},
        "XAU": {...},
        "MACRO": {...},
    }
    """
    by_asset = defaultdict(list)
    
    for item in news:
        asset = item.get("asset", "MACRO")
        by_asset[asset].append(item)
    
    results = {}
    for asset, items in by_asset.items():
        scores = [i["sentiment"]["combined"] for i in items]
        bullish = sum(1 for s in scores if s > 0.15)
        bearish = sum(1 for s in scores if s < -0.15)
        
        # Top 3 most impactful headlines (by absolute score)
        sorted_items = sorted(items, key=lambda x: abs(x["sentiment"]["combined"]), reverse=True)
        top = [{"title": i["title"], "score": i["sentiment"]["combined"],
                "label": i["sentiment"]["label"]} for i in sorted_items[:3]]
        
        results[asset] = {
            "score": float(np.mean(scores)) if scores else 0.0,
            "label": "bullish" if np.mean(scores) > 0.15 else
                     "bearish" if np.mean(scores) < -0.15 else "neutral",
            "bullish_pct": bullish / max(len(scores), 1) * 100,
            "bearish_pct": bearish / max(len(scores), 1) * 100,
            "count": len(items),
            "top_headlines": top,
        }
    
    return results


def get_news_signal() -> dict:
    """
    Get actionable news signal for strategy generation.
    
    Returns {
        btc_sentiment: float [-1, 1],
        xau_sentiment: float [-1, 1],
        macro_sentiment: float [-1, 1],
        overall_risk: "risk_on" | "risk_off" | "neutral",
        alert: Optional[str]  # if extreme sentiment detected
    }
    """
    news = fetch_all_news()
    sentiment = compute_asset_sentiment(news)
    
    btc = sentiment.get("BTC", {}).get("score", 0)
    xau = sentiment.get("XAU", {}).get("score", 0)
    macro = sentiment.get("MACRO", {}).get("score", 0)
    
    # Overall risk assessment
    avg = (btc + macro) / 2
    if avg > 0.3:
        risk = "risk_on"
    elif avg < -0.3:
        risk = "risk_off"
    else:
        risk = "neutral"
    
    # Alert on extreme sentiment
    alert = None
    if abs(btc) > 0.6:
        alert = f"EXTREME BTC sentiment: {'BULLISH' if btc > 0 else 'BEARISH'} ({btc:.2f})"
    if abs(xau) > 0.6:
        alert = f"EXTREME XAU sentiment: {'BULLISH' if xau > 0 else 'BEARISH'} ({xau:.2f})"
    
    return {
        "btc_sentiment": float(btc),
        "xau_sentiment": float(xau),
        "macro_sentiment": float(macro),
        "overall_risk": risk,
        "alert": alert,
        "details": sentiment,
    }


# ─────────────────────────────────────────────────────────────────
# Sentiment History (for backtesting)
# ─────────────────────────────────────────────────────────────────

def build_sentiment_timeseries(lookback_days: int = 30) -> Dict[str, List[float]]:
    """
    Build a time series of sentiment scores from cached news files.
    Useful for incorporating sentiment into backtests.
    """
    os.makedirs(HISTORY_DIR, exist_ok=True)
    
    cutoff = time.time() - (lookback_days * 86400)
    daily_scores = defaultdict(lambda: defaultdict(list))
    
    for fname in os.listdir(CACHE_DIR):
        if not fname.startswith("all_"):
            continue
        try:
            ts = int(fname.split("_")[1].split(".")[0])
            if ts < cutoff:
                continue
            
            day = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
            
            with open(os.path.join(CACHE_DIR, fname)) as f:
                news = json.load(f)
            
            for item in news:
                asset = item.get("asset", "MACRO")
                score = item.get("sentiment", {}).get("combined", 0)
                daily_scores[asset][day].append(score)
        except:
            continue
    
    # Average per day
    result = {}
    for asset, days in daily_scores.items():
        sorted_days = sorted(days.keys())
        result[asset] = [float(np.mean(days[d])) for d in sorted_days]
    
    return result


if __name__ == "__main__":
    print("=" * 50)
    print("REAL-TIME NEWS ENGINE — Fetching all sources")
    print("=" * 50)
    
    signal = get_news_signal()
    
    print(f"\nBTC Sentiment: {signal['btc_sentiment']:.3f}")
    print(f"XAU Sentiment: {signal['xau_sentiment']:.3f}")
    print(f"Macro Sentiment: {signal['macro_sentiment']:.3f}")
    print(f"Overall Risk: {signal['overall_risk']}")
    if signal["alert"]:
        print(f"⚠️  ALERT: {signal['alert']}")
    
    print("\nDetails:")
    for asset, data in signal["details"].items():
        print(f"\n  {asset}:")
        print(f"    Score: {data['score']:.3f} ({data['label']})")
        print(f"    Headlines: {data['count']} (bull={data['bullish_pct']:.0f}% bear={data['bearish_pct']:.0f}%)")
        for h in data.get("top_headlines", []):
            print(f"    → [{h['label']}] {h['title'][:70]}")
