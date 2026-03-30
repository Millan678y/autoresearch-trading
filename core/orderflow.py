"""
ORDER FLOW ANALYSIS — Volume-Based Institutional Activity Detection

Implements:
1. Volume Profile (VPVR) — value area, POC, high/low volume nodes
2. Delta analysis — buying vs selling pressure proxy from OHLCV
3. CVD (Cumulative Volume Delta) — net buying/selling over time
4. Absorption detection — large volume with no price movement
5. Exhaustion detection — climactic volume at extremes
6. VWAP and anchored VWAP

Note: True order flow requires tick data. These are OHLCV proxies
that approximate institutional activity. Good enough for hourly timeframes.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────
# Volume Delta (Buy/Sell Pressure Proxy)
# ─────────────────────────────────────────────────────────────────

def estimate_delta(opens: np.ndarray, highs: np.ndarray,
                   lows: np.ndarray, closes: np.ndarray,
                   volumes: np.ndarray) -> np.ndarray:
    """
    Estimate buy/sell delta from OHLCV using the close position method.
    
    Delta = Volume * (Close - Low) / (High - Low) - Volume * (High - Close) / (High - Low)
    Simplified: Delta = Volume * (2 * Close - High - Low) / (High - Low)
    
    Positive delta = more buying pressure
    Negative delta = more selling pressure
    """
    range_hl = highs - lows
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    
    delta = volumes * (2 * closes - highs - lows) / range_hl
    return delta


def cumulative_volume_delta(opens: np.ndarray, highs: np.ndarray,
                            lows: np.ndarray, closes: np.ndarray,
                            volumes: np.ndarray) -> np.ndarray:
    """
    Cumulative Volume Delta (CVD).
    Running sum of estimated delta — shows net buying/selling over time.
    
    Rising CVD + Rising price = confirmed uptrend (strong)
    Rising CVD + Falling price = hidden bullish divergence
    Falling CVD + Rising price = hidden bearish divergence (distribution)
    Falling CVD + Falling price = confirmed downtrend
    """
    delta = estimate_delta(opens, highs, lows, closes, volumes)
    return np.cumsum(delta)


def cvd_divergence(closes: np.ndarray, cvd: np.ndarray,
                   lookback: int = 20) -> str:
    """
    Detect CVD divergence.
    
    Returns: "bullish_div", "bearish_div", "confirmed_bull",
             "confirmed_bear", "neutral"
    """
    if len(closes) < lookback or len(cvd) < lookback:
        return "neutral"
    
    price_change = closes[-1] - closes[-lookback]
    cvd_change = cvd[-1] - cvd[-lookback]
    
    price_up = price_change > 0
    cvd_up = cvd_change > 0
    
    if price_up and cvd_up:
        return "confirmed_bull"
    elif not price_up and not cvd_up:
        return "confirmed_bear"
    elif not price_up and cvd_up:
        return "bullish_div"  # Price falling but buying pressure increasing
    elif price_up and not cvd_up:
        return "bearish_div"  # Price rising but selling pressure increasing
    
    return "neutral"


# ─────────────────────────────────────────────────────────────────
# Volume Profile (VPVR)
# ─────────────────────────────────────────────────────────────────

@dataclass
class VolumeProfile:
    """Volume profile for a price range."""
    poc: float                    # Point of Control — highest volume price
    value_area_high: float        # Top of value area (70% of volume)
    value_area_low: float         # Bottom of value area
    high_volume_nodes: List[float]  # Price levels with unusually high volume
    low_volume_nodes: List[float]   # Price levels with low volume (LVN = fast moves)
    total_volume: float
    bins: np.ndarray              # Price bins
    profile: np.ndarray           # Volume at each bin


def compute_volume_profile(highs: np.ndarray, lows: np.ndarray,
                           closes: np.ndarray, volumes: np.ndarray,
                           n_bins: int = 50) -> VolumeProfile:
    """
    Compute Volume Profile (VPVR) for the given data.
    
    Distributes volume across price bins based on where price traded.
    """
    price_min = float(np.min(lows))
    price_max = float(np.max(highs))
    
    if price_max == price_min:
        price_max = price_min + 1
    
    bins = np.linspace(price_min, price_max, n_bins + 1)
    profile = np.zeros(n_bins)
    
    for i in range(len(closes)):
        # Distribute volume across bins that this candle covers
        candle_low = lows[i]
        candle_high = highs[i]
        candle_vol = volumes[i]
        
        for j in range(n_bins):
            bin_low = bins[j]
            bin_high = bins[j + 1]
            
            # Overlap between candle and bin
            overlap_low = max(candle_low, bin_low)
            overlap_high = min(candle_high, bin_high)
            
            if overlap_high > overlap_low:
                candle_range = candle_high - candle_low
                if candle_range > 0:
                    overlap_pct = (overlap_high - overlap_low) / candle_range
                    profile[j] += candle_vol * overlap_pct
    
    # Point of Control
    poc_idx = np.argmax(profile)
    poc = float((bins[poc_idx] + bins[poc_idx + 1]) / 2)
    
    # Value Area (70% of total volume, centered on POC)
    total_vol = np.sum(profile)
    target_vol = total_vol * 0.70
    
    # Expand from POC outward
    va_low_idx = poc_idx
    va_high_idx = poc_idx
    current_vol = profile[poc_idx]
    
    while current_vol < target_vol and (va_low_idx > 0 or va_high_idx < n_bins - 1):
        expand_low = profile[va_low_idx - 1] if va_low_idx > 0 else 0
        expand_high = profile[va_high_idx + 1] if va_high_idx < n_bins - 1 else 0
        
        if expand_low >= expand_high and va_low_idx > 0:
            va_low_idx -= 1
            current_vol += expand_low
        elif va_high_idx < n_bins - 1:
            va_high_idx += 1
            current_vol += expand_high
        else:
            va_low_idx -= 1
            current_vol += expand_low
    
    vah = float(bins[va_high_idx + 1])
    val = float(bins[va_low_idx])
    
    # High Volume Nodes (above 1.5x average)
    avg_profile = np.mean(profile[profile > 0]) if np.any(profile > 0) else 0
    hvn = []
    lvn = []
    
    for j in range(n_bins):
        price_mid = (bins[j] + bins[j + 1]) / 2
        if profile[j] > avg_profile * 1.5:
            hvn.append(float(price_mid))
        elif 0 < profile[j] < avg_profile * 0.3:
            lvn.append(float(price_mid))
    
    return VolumeProfile(
        poc=poc,
        value_area_high=vah,
        value_area_low=val,
        high_volume_nodes=hvn,
        low_volume_nodes=lvn,
        total_volume=float(total_vol),
        bins=bins,
        profile=profile,
    )


# ─────────────────────────────────────────────────────────────────
# Absorption & Exhaustion
# ─────────────────────────────────────────────────────────────────

def detect_absorption(opens: np.ndarray, highs: np.ndarray,
                      lows: np.ndarray, closes: np.ndarray,
                      volumes: np.ndarray,
                      vol_threshold: float = 2.0,
                      body_threshold: float = 0.3) -> List[dict]:
    """
    Detect absorption — high volume but small candle body.
    Indicates institutional players absorbing selling/buying pressure.
    
    Bullish absorption: High volume at support, small body, price holds
    Bearish absorption: High volume at resistance, small body, price holds
    """
    n = len(closes)
    if n < 20:
        return []
    
    avg_vol = np.mean(volumes[-20:])
    avg_range = np.mean(highs[-20:] - lows[-20:])
    
    absorptions = []
    
    for i in range(max(20, n - 50), n):
        vol_ratio = volumes[i] / max(avg_vol, 1)
        body = abs(closes[i] - opens[i])
        candle_range = highs[i] - lows[i]
        
        if candle_range == 0:
            continue
        
        body_ratio = body / candle_range
        
        # High volume + small body = absorption
        if vol_ratio > vol_threshold and body_ratio < body_threshold:
            # Determine direction based on position in range
            close_position = (closes[i] - lows[i]) / candle_range
            
            if close_position > 0.6:
                abs_type = "bullish_absorption"  # Closed near high despite selling
            elif close_position < 0.4:
                abs_type = "bearish_absorption"
            else:
                abs_type = "neutral_absorption"
            
            absorptions.append({
                "index": i,
                "type": abs_type,
                "vol_ratio": float(vol_ratio),
                "body_ratio": float(body_ratio),
                "price": float(closes[i]),
            })
    
    return absorptions


def detect_exhaustion(opens: np.ndarray, highs: np.ndarray,
                      lows: np.ndarray, closes: np.ndarray,
                      volumes: np.ndarray,
                      vol_threshold: float = 2.5,
                      lookback: int = 10) -> List[dict]:
    """
    Detect exhaustion — climactic volume at price extremes.
    Often signals the end of a trend.
    
    Buying exhaustion: Huge volume at highs, then reversal
    Selling exhaustion: Huge volume at lows, then reversal
    """
    n = len(closes)
    if n < lookback + 5:
        return []
    
    avg_vol = np.mean(volumes[-50:]) if n > 50 else np.mean(volumes)
    
    exhaustions = []
    
    for i in range(max(lookback, n - 30), n - 2):
        vol_ratio = volumes[i] / max(avg_vol, 1)
        
        if vol_ratio < vol_threshold:
            continue
        
        # Is this at a local extreme?
        is_local_high = highs[i] == np.max(highs[i - lookback:i + 1])
        is_local_low = lows[i] == np.min(lows[i - lookback:i + 1])
        
        # Check for reversal in next few bars
        if is_local_high and closes[i + 1] < closes[i] and closes[i + 2] < closes[i + 1]:
            exhaustions.append({
                "index": i,
                "type": "buying_exhaustion",
                "vol_ratio": float(vol_ratio),
                "price": float(highs[i]),
                "reversed": True,
            })
        
        elif is_local_low and closes[i + 1] > closes[i] and closes[i + 2] > closes[i + 1]:
            exhaustions.append({
                "index": i,
                "type": "selling_exhaustion",
                "vol_ratio": float(vol_ratio),
                "price": float(lows[i]),
                "reversed": True,
            })
    
    return exhaustions


# ─────────────────────────────────────────────────────────────────
# VWAP
# ─────────────────────────────────────────────────────────────────

def compute_vwap(highs: np.ndarray, lows: np.ndarray,
                 closes: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    """
    Volume Weighted Average Price.
    Institutional benchmark — price above VWAP = bullish, below = bearish.
    """
    typical_price = (highs + lows + closes) / 3
    cum_tp_vol = np.cumsum(typical_price * volumes)
    cum_vol = np.cumsum(volumes)
    
    vwap = cum_tp_vol / np.where(cum_vol > 0, cum_vol, 1)
    return vwap


def vwap_signal(closes: np.ndarray, vwap: np.ndarray) -> str:
    """
    Simple VWAP signal.
    Price above VWAP = institutional buying, below = selling.
    """
    if len(closes) < 2 or len(vwap) < 2:
        return "neutral"
    
    above = closes[-1] > vwap[-1]
    crossing_up = closes[-2] <= vwap[-2] and closes[-1] > vwap[-1]
    crossing_down = closes[-2] >= vwap[-2] and closes[-1] < vwap[-1]
    
    if crossing_up:
        return "bullish_cross"
    elif crossing_down:
        return "bearish_cross"
    elif above:
        return "above_vwap"
    else:
        return "below_vwap"


# ─────────────────────────────────────────────────────────────────
# Composite Order Flow Signal
# ─────────────────────────────────────────────────────────────────

def compute_orderflow_signal(opens: np.ndarray, highs: np.ndarray,
                             lows: np.ndarray, closes: np.ndarray,
                             volumes: np.ndarray) -> dict:
    """
    Composite order flow signal from all components.
    
    Returns {
        bias: "bullish" | "bearish" | "neutral",
        strength: float [0, 1],
        components: {cvd, vwap, absorption, exhaustion, volume_profile}
    }
    """
    n = len(closes)
    if n < 50:
        return {"bias": "neutral", "strength": 0.0, "components": {}}
    
    bull_score = 0.0
    bear_score = 0.0
    
    # ── CVD ──
    cvd = cumulative_volume_delta(opens, highs, lows, closes, volumes)
    div = cvd_divergence(closes, cvd, lookback=20)
    
    if div == "confirmed_bull":
        bull_score += 0.25
    elif div == "confirmed_bear":
        bear_score += 0.25
    elif div == "bullish_div":
        bull_score += 0.35  # Divergences are stronger signals
    elif div == "bearish_div":
        bear_score += 0.35
    
    # ── VWAP ──
    vwap = compute_vwap(highs, lows, closes, volumes)
    vwap_sig = vwap_signal(closes, vwap)
    
    if vwap_sig == "bullish_cross":
        bull_score += 0.2
    elif vwap_sig == "bearish_cross":
        bear_score += 0.2
    elif vwap_sig == "above_vwap":
        bull_score += 0.1
    elif vwap_sig == "below_vwap":
        bear_score += 0.1
    
    # ── Absorption ──
    absorptions = detect_absorption(opens, highs, lows, closes, volumes)
    recent_abs = [a for a in absorptions if a["index"] > n - 5]
    
    for a in recent_abs:
        if a["type"] == "bullish_absorption":
            bull_score += 0.2
        elif a["type"] == "bearish_absorption":
            bear_score += 0.2
    
    # ── Exhaustion ──
    exhaustions = detect_exhaustion(opens, highs, lows, closes, volumes)
    recent_exh = [e for e in exhaustions if e["index"] > n - 5]
    
    for e in recent_exh:
        if e["type"] == "selling_exhaustion":
            bull_score += 0.3  # Selling exhaustion = bottom forming
        elif e["type"] == "buying_exhaustion":
            bear_score += 0.3
    
    # ── Volume Profile ──
    vp = compute_volume_profile(highs[-100:], lows[-100:], closes[-100:], volumes[-100:])
    current_price = float(closes[-1])
    
    if current_price < vp.value_area_low:
        bull_score += 0.15  # Below value = discount
    elif current_price > vp.value_area_high:
        bear_score += 0.15  # Above value = premium
    
    # Near POC = potential magnet
    poc_distance = abs(current_price - vp.poc) / current_price
    
    # ── Composite ──
    total = bull_score + bear_score
    if total == 0:
        bias = "neutral"
        strength = 0.0
    elif bull_score > bear_score:
        bias = "bullish"
        strength = min(1.0, (bull_score - bear_score) / max(total, 1))
    else:
        bias = "bearish"
        strength = min(1.0, (bear_score - bull_score) / max(total, 1))
    
    return {
        "bias": bias,
        "strength": float(strength),
        "bull_score": float(bull_score),
        "bear_score": float(bear_score),
        "components": {
            "cvd_divergence": div,
            "vwap": vwap_sig,
            "absorptions": len(recent_abs),
            "exhaustions": len(recent_exh),
            "volume_profile": {
                "poc": vp.poc,
                "vah": vp.value_area_high,
                "val": vp.value_area_low,
                "price_vs_va": "above" if current_price > vp.value_area_high else
                              "below" if current_price < vp.value_area_low else "inside",
            },
        },
    }
