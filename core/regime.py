"""
REGIME DETECTION — Market State Classification

Identifies the current market regime and allows strategies to adapt.
A strategy that works in trends will die in chop, and vice versa.

Regimes:
1. Trend (strong directional move)
2. Range (sideways, mean-reverting)
3. High Volatility (expansion, often breakouts)
4. Low Volatility (compression, often before big moves)
5. Crisis (extreme moves, correlations spike)

Methods:
- Hurst exponent (trend vs mean-reversion)
- Volatility regime (realized vs implied proxy)
- ADX-based trend strength
- Markov regime switching (simple version)

Asset-specific adjustments:
- BTC: momentum-dominant, trend-following regimes more common
- XAU: mean-reversion dominant, session-driven, crisis-correlated
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class RegimeState:
    """Current market regime classification."""
    regime: str              # "trending_up", "trending_down", "ranging", "high_vol", "low_vol", "crisis"
    confidence: float        # 0-1
    trend_strength: float    # 0-1 (0 = pure range, 1 = strong trend)
    vol_regime: str          # "low", "normal", "high", "extreme"
    vol_percentile: float    # Current vol as percentile of history
    hurst: float             # Hurst exponent (>0.5 = trending, <0.5 = mean-reverting)
    recommended_approach: str # "momentum", "mean_reversion", "breakout", "stay_flat"


# ─────────────────────────────────────────────────────────────────
# Hurst Exponent
# ─────────────────────────────────────────────────────────────────

def compute_hurst(prices: np.ndarray, max_lag: int = 50) -> float:
    """
    Compute Hurst exponent using R/S analysis.
    
    H > 0.5: Trending (persistent) — use momentum
    H = 0.5: Random walk — no edge
    H < 0.5: Mean-reverting (anti-persistent) — use mean reversion
    
    This is THE most important regime indicator for strategy selection.
    """
    if len(prices) < max_lag * 2:
        return 0.5  # Not enough data
    
    log_returns = np.diff(np.log(prices))
    
    lags = range(2, max_lag)
    rs_values = []
    
    for lag in lags:
        # Split into chunks of size 'lag'
        n_chunks = len(log_returns) // lag
        if n_chunks < 1:
            continue
        
        rs_chunk = []
        for i in range(n_chunks):
            chunk = log_returns[i * lag:(i + 1) * lag]
            mean = np.mean(chunk)
            cumdev = np.cumsum(chunk - mean)
            R = np.max(cumdev) - np.min(cumdev)
            S = np.std(chunk, ddof=1) if np.std(chunk, ddof=1) > 0 else 1e-10
            rs_chunk.append(R / S)
        
        rs_values.append((lag, np.mean(rs_chunk)))
    
    if len(rs_values) < 5:
        return 0.5
    
    # Hurst = slope of log(R/S) vs log(lag)
    log_lags = np.log([v[0] for v in rs_values])
    log_rs = np.log([v[1] for v in rs_values])
    
    # Linear regression
    try:
        slope = np.polyfit(log_lags, log_rs, 1)[0]
        return float(np.clip(slope, 0.0, 1.0))
    except:
        return 0.5


# ─────────────────────────────────────────────────────────────────
# ADX Trend Strength
# ─────────────────────────────────────────────────────────────────

def compute_adx(highs: np.ndarray, lows: np.ndarray,
                closes: np.ndarray, period: int = 14) -> float:
    """
    Average Directional Index — trend strength indicator.
    
    ADX < 20: Weak trend / ranging
    ADX 20-40: Moderate trend
    ADX > 40: Strong trend
    ADX > 60: Extremely strong trend
    """
    n = len(closes)
    if n < period + 2:
        return 20.0  # Default: no strong trend
    
    # True Range
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1]))
    )
    
    # Directional Movement
    dm_plus = np.maximum(highs[1:] - highs[:-1], 0)
    dm_minus = np.maximum(lows[:-1] - lows[1:], 0)
    
    # Zero out when other direction is larger
    mask = dm_plus > dm_minus
    dm_plus = np.where(mask, dm_plus, 0)
    dm_minus = np.where(~mask, dm_minus, 0)
    
    # Smoothed averages (Wilder's smoothing)
    atr = np.zeros(len(tr))
    di_plus = np.zeros(len(tr))
    di_minus = np.zeros(len(tr))
    
    atr[period-1] = np.mean(tr[:period])
    di_plus[period-1] = np.mean(dm_plus[:period])
    di_minus[period-1] = np.mean(dm_minus[:period])
    
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        di_plus[i] = (di_plus[i-1] * (period - 1) + dm_plus[i]) / period
        di_minus[i] = (di_minus[i-1] * (period - 1) + dm_minus[i]) / period
    
    # DI+ and DI-
    di_plus_pct = np.where(atr > 0, di_plus / atr * 100, 0)
    di_minus_pct = np.where(atr > 0, di_minus / atr * 100, 0)
    
    # DX
    di_sum = di_plus_pct + di_minus_pct
    dx = np.where(di_sum > 0, np.abs(di_plus_pct - di_minus_pct) / di_sum * 100, 0)
    
    # ADX (smoothed DX)
    if len(dx) < period * 2:
        return float(dx[-1]) if len(dx) > 0 else 20.0
    
    adx = np.mean(dx[-period:])  # Simple average for the last period
    
    return float(adx)


# ─────────────────────────────────────────────────────────────────
# Volatility Regime
# ─────────────────────────────────────────────────────────────────

def classify_vol_regime(closes: np.ndarray,
                        short_window: int = 12,
                        long_window: int = 100) -> Tuple[str, float]:
    """
    Classify volatility regime.
    
    Returns (regime_name, vol_percentile)
    """
    if len(closes) < long_window + 1:
        return "normal", 50.0
    
    log_rets = np.diff(np.log(closes))
    
    # Current short-term vol
    current_vol = float(np.std(log_rets[-short_window:]))
    
    # Historical vol distribution
    vol_history = []
    for i in range(short_window, len(log_rets)):
        v = np.std(log_rets[i-short_window:i])
        vol_history.append(v)
    
    if not vol_history:
        return "normal", 50.0
    
    # Percentile
    percentile = float(np.sum(np.array(vol_history) <= current_vol) / len(vol_history) * 100)
    
    if percentile < 20:
        regime = "low"
    elif percentile < 40:
        regime = "below_normal"
    elif percentile < 60:
        regime = "normal"
    elif percentile < 80:
        regime = "above_normal"
    elif percentile < 95:
        regime = "high"
    else:
        regime = "extreme"
    
    return regime, percentile


# ─────────────────────────────────────────────────────────────────
# Composite Regime Detection
# ─────────────────────────────────────────────────────────────────

def detect_regime(opens: np.ndarray, highs: np.ndarray,
                  lows: np.ndarray, closes: np.ndarray,
                  asset_type: str = "crypto") -> RegimeState:
    """
    Comprehensive regime detection combining multiple methods.
    
    Args:
        asset_type: "crypto" or "commodity" — affects defaults
    """
    n = len(closes)
    if n < 60:
        return RegimeState(
            regime="unknown", confidence=0.0, trend_strength=0.0,
            vol_regime="normal", vol_percentile=50.0, hurst=0.5,
            recommended_approach="stay_flat"
        )
    
    # Hurst exponent
    hurst = compute_hurst(closes, max_lag=min(50, n // 4))
    
    # ADX
    adx = compute_adx(highs, lows, closes)
    trend_strength = min(1.0, adx / 60)
    
    # Vol regime
    vol_regime, vol_pctile = classify_vol_regime(closes)
    
    # Direction
    sma_20 = np.mean(closes[-20:])
    sma_50 = np.mean(closes[-min(50, n):])
    direction = 1 if closes[-1] > sma_20 > sma_50 else -1 if closes[-1] < sma_20 < sma_50 else 0
    
    # Classify regime
    if vol_regime in ("extreme", "high") and adx > 40:
        regime = "crisis" if direction == -1 else "high_vol"
        confidence = min(1.0, vol_pctile / 100)
    elif adx > 30 and hurst > 0.55:
        regime = "trending_up" if direction >= 0 else "trending_down"
        confidence = min(1.0, (adx - 20) / 40)
    elif adx < 20 and hurst < 0.45:
        regime = "ranging"
        confidence = min(1.0, (20 - adx) / 20)
    elif vol_pctile < 20:
        regime = "low_vol"
        confidence = min(1.0, (20 - vol_pctile) / 20)
    elif vol_pctile > 80:
        regime = "high_vol"
        confidence = min(1.0, (vol_pctile - 80) / 20)
    else:
        regime = "ranging" if hurst < 0.5 else "trending_up" if direction >= 0 else "trending_down"
        confidence = 0.4
    
    # Recommended approach based on regime + asset type
    if regime in ("trending_up", "trending_down"):
        recommended = "momentum"
    elif regime == "ranging":
        if asset_type == "commodity":
            recommended = "mean_reversion"  # Gold mean-reverts more
        else:
            recommended = "mean_reversion"
    elif regime == "high_vol":
        recommended = "breakout"
    elif regime == "low_vol":
        recommended = "breakout"  # Vol compression → breakout
    elif regime == "crisis":
        if asset_type == "commodity":
            recommended = "momentum"  # Gold goes up in crisis
        else:
            recommended = "stay_flat"  # Crypto crashes in crisis
    else:
        recommended = "stay_flat"
    
    return RegimeState(
        regime=regime,
        confidence=float(confidence),
        trend_strength=float(trend_strength),
        vol_regime=vol_regime,
        vol_percentile=float(vol_pctile),
        hurst=float(hurst),
        recommended_approach=recommended,
    )


# ─────────────────────────────────────────────────────────────────
# Asset-Specific Strategy Hints
# ─────────────────────────────────────────────────────────────────

ASSET_STRATEGY_HINTS = {
    "BTC": {
        "preferred_approaches": ["momentum", "breakout"],
        "avoid_approaches": ["mean_reversion"],  # BTC trends, rarely mean-reverts cleanly
        "typical_hurst": 0.55,  # Slightly trending
        "session_bias": None,   # 24/7
        "vol_profile": "high",
        "notes": "Strong momentum. Weekend vol lower. Halving cycles matter.",
    },
    "BTCUSDT": {
        "preferred_approaches": ["momentum", "breakout"],
        "avoid_approaches": ["mean_reversion"],
        "typical_hurst": 0.55,
        "session_bias": None,
        "vol_profile": "high",
        "notes": "Same as BTC. Funding rates add carry component.",
    },
    "XAU": {
        "preferred_approaches": ["mean_reversion", "session_breakout"],
        "avoid_approaches": ["pure_momentum"],
        "typical_hurst": 0.45,  # Slightly mean-reverting
        "session_bias": "london",  # Most volume in London session
        "vol_profile": "medium",
        "notes": "Mean-reverts in range. Trends during crisis. London session key.",
    },
    "XAUUSD": {
        "preferred_approaches": ["mean_reversion", "session_breakout"],
        "avoid_approaches": ["pure_momentum"],
        "typical_hurst": 0.45,
        "session_bias": "london",
        "vol_profile": "medium",
        "notes": "Session-based. London open breakout. DXY inverse correlation.",
    },
    "ETH": {
        "preferred_approaches": ["momentum", "btc_correlation"],
        "avoid_approaches": [],
        "typical_hurst": 0.52,
        "session_bias": None,
        "vol_profile": "very_high",
        "notes": "Follows BTC with higher beta. Gas fees affect on-chain activity.",
    },
    "SOL": {
        "preferred_approaches": ["momentum", "breakout"],
        "avoid_approaches": ["mean_reversion"],
        "typical_hurst": 0.53,
        "session_bias": None,
        "vol_profile": "very_high",
        "notes": "High beta to BTC. More volatile. Faster trends.",
    },
}


def get_asset_hints(symbol: str) -> dict:
    """Get asset-specific strategy hints."""
    # Try exact match
    if symbol in ASSET_STRATEGY_HINTS:
        return ASSET_STRATEGY_HINTS[symbol]
    
    # Try prefix match
    for key, hints in ASSET_STRATEGY_HINTS.items():
        if symbol.startswith(key):
            return hints
    
    return {
        "preferred_approaches": ["momentum", "mean_reversion"],
        "avoid_approaches": [],
        "typical_hurst": 0.50,
        "session_bias": None,
        "vol_profile": "medium",
        "notes": "No specific hints for this asset.",
    }
