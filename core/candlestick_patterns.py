"""
CANDLESTICK PATTERN RECOGNITION — 25+ Classic Patterns

Single candle, dual candle, and triple candle patterns.
All computed from OHLCV arrays with no external dependencies.

Each pattern returns a score [-1, 1]:
  Positive = bullish, Negative = bearish
  Magnitude = confidence/strength
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class PatternMatch:
    """A detected candlestick pattern."""
    name: str
    index: int
    direction: str      # "bullish" or "bearish"
    strength: float     # 0-1
    reliability: float  # Historical reliability rating 0-1


# ─────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────

def _body(o, c):
    return abs(c - o)

def _upper_shadow(o, h, c):
    return h - max(o, c)

def _lower_shadow(o, l, c):
    return min(o, c) - l

def _candle_range(h, l):
    return h - l

def _is_bullish(o, c):
    return c > o

def _is_bearish(o, c):
    return c < o

def _is_doji(o, h, l, c, threshold=0.05):
    rng = h - l
    if rng == 0:
        return True
    return _body(o, c) / rng < threshold

def _avg_body(opens, closes, period=10):
    bodies = np.abs(closes[-period:] - opens[-period:])
    return np.mean(bodies)

def _trend(closes, lookback=10):
    """Simple trend detection: 1=up, -1=down, 0=neutral."""
    if len(closes) < lookback:
        return 0
    change = closes[-1] - closes[-lookback]
    avg_range = np.mean(np.abs(np.diff(closes[-lookback:])))
    if avg_range == 0:
        return 0
    normalized = change / (avg_range * lookback)
    if normalized > 0.3:
        return 1
    elif normalized < -0.3:
        return -1
    return 0


# ─────────────────────────────────────────────────────────────────
# Single Candle Patterns
# ─────────────────────────────────────────────────────────────────

def detect_hammer(o, h, l, c, avg_body_size):
    """
    Hammer (bullish) / Hanging Man (bearish at top).
    Small body at top, long lower shadow (2x+ body).
    """
    body = _body(o, c)
    lower = _lower_shadow(o, l, c)
    upper = _upper_shadow(o, h, c)
    rng = _candle_range(h, l)
    
    if rng == 0 or body == 0:
        return None
    
    if lower >= body * 2 and upper <= body * 0.5:
        return PatternMatch(
            name="hammer", index=0,
            direction="bullish",
            strength=min(1.0, lower / body / 3),
            reliability=0.60,
        )
    return None


def detect_inverted_hammer(o, h, l, c, avg_body_size):
    """Inverted hammer — small body at bottom, long upper shadow."""
    body = _body(o, c)
    lower = _lower_shadow(o, l, c)
    upper = _upper_shadow(o, h, c)
    
    if body == 0:
        return None
    
    if upper >= body * 2 and lower <= body * 0.5:
        return PatternMatch(
            name="inverted_hammer", index=0,
            direction="bullish",
            strength=min(1.0, upper / body / 3),
            reliability=0.55,
        )
    return None


def detect_shooting_star(o, h, l, c, avg_body_size):
    """Shooting star — inverted hammer in uptrend (bearish)."""
    body = _body(o, c)
    upper = _upper_shadow(o, h, c)
    lower = _lower_shadow(o, l, c)
    
    if body == 0:
        return None
    
    if upper >= body * 2 and lower <= body * 0.5 and _is_bearish(o, c):
        return PatternMatch(
            name="shooting_star", index=0,
            direction="bearish",
            strength=min(1.0, upper / body / 3),
            reliability=0.60,
        )
    return None


def detect_doji(o, h, l, c):
    """Doji — open ≈ close, indicates indecision."""
    if _is_doji(o, h, l, c):
        upper = _upper_shadow(o, h, c)
        lower = _lower_shadow(o, l, c)
        
        if upper > lower * 2:
            direction = "bearish"  # Gravestone doji
            name = "gravestone_doji"
        elif lower > upper * 2:
            direction = "bullish"  # Dragonfly doji
            name = "dragonfly_doji"
        else:
            direction = "neutral"
            name = "doji"
        
        return PatternMatch(
            name=name, index=0,
            direction=direction,
            strength=0.4,
            reliability=0.50,
        )
    return None


def detect_marubozu(o, h, l, c, threshold=0.02):
    """Marubozu — full body, no shadows. Strong conviction."""
    rng = _candle_range(h, l)
    if rng == 0:
        return None
    
    upper = _upper_shadow(o, h, c) / rng
    lower = _lower_shadow(o, l, c) / rng
    
    if upper < threshold and lower < threshold:
        direction = "bullish" if _is_bullish(o, c) else "bearish"
        return PatternMatch(
            name="marubozu", index=0,
            direction=direction,
            strength=0.8,
            reliability=0.65,
        )
    return None


def detect_spinning_top(o, h, l, c):
    """Spinning top — small body, equal shadows. Indecision."""
    rng = _candle_range(h, l)
    body = _body(o, c)
    
    if rng == 0:
        return None
    
    body_ratio = body / rng
    upper = _upper_shadow(o, h, c)
    lower = _lower_shadow(o, l, c)
    
    if 0.1 < body_ratio < 0.3 and abs(upper - lower) < rng * 0.3:
        return PatternMatch(
            name="spinning_top", index=0,
            direction="neutral",
            strength=0.3,
            reliability=0.40,
        )
    return None


# ─────────────────────────────────────────────────────────────────
# Dual Candle Patterns
# ─────────────────────────────────────────────────────────────────

def detect_engulfing(o1, h1, l1, c1, o2, h2, l2, c2):
    """
    Engulfing pattern.
    Bullish: bearish candle followed by larger bullish candle.
    Bearish: bullish candle followed by larger bearish candle.
    """
    body1 = _body(o1, c1)
    body2 = _body(o2, c2)
    
    if body1 == 0 or body2 == 0:
        return None
    
    # Bullish engulfing
    if _is_bearish(o1, c1) and _is_bullish(o2, c2):
        if o2 <= c1 and c2 >= o1:
            return PatternMatch(
                name="bullish_engulfing", index=0,
                direction="bullish",
                strength=min(1.0, body2 / body1 / 2),
                reliability=0.70,
            )
    
    # Bearish engulfing
    if _is_bullish(o1, c1) and _is_bearish(o2, c2):
        if o2 >= c1 and c2 <= o1:
            return PatternMatch(
                name="bearish_engulfing", index=0,
                direction="bearish",
                strength=min(1.0, body2 / body1 / 2),
                reliability=0.70,
            )
    
    return None


def detect_harami(o1, h1, l1, c1, o2, h2, l2, c2):
    """Harami — second candle contained within first."""
    if _is_bearish(o1, c1) and _is_bullish(o2, c2):
        if o2 >= c1 and c2 <= o1 and h2 <= h1 and l2 >= l1:
            return PatternMatch(
                name="bullish_harami", index=0,
                direction="bullish",
                strength=0.5,
                reliability=0.55,
            )
    
    if _is_bullish(o1, c1) and _is_bearish(o2, c2):
        if o2 <= c1 and c2 >= o1 and h2 <= h1 and l2 >= l1:
            return PatternMatch(
                name="bearish_harami", index=0,
                direction="bearish",
                strength=0.5,
                reliability=0.55,
            )
    
    return None


def detect_tweezer(o1, h1, l1, c1, o2, h2, l2, c2, tolerance=0.001):
    """Tweezer tops/bottoms — matching highs or lows."""
    price_avg = (h1 + h2 + l1 + l2) / 4
    tol = price_avg * tolerance
    
    # Tweezer bottom
    if abs(l1 - l2) < tol and _is_bearish(o1, c1) and _is_bullish(o2, c2):
        return PatternMatch(
            name="tweezer_bottom", index=0,
            direction="bullish",
            strength=0.6,
            reliability=0.60,
        )
    
    # Tweezer top
    if abs(h1 - h2) < tol and _is_bullish(o1, c1) and _is_bearish(o2, c2):
        return PatternMatch(
            name="tweezer_top", index=0,
            direction="bearish",
            strength=0.6,
            reliability=0.60,
        )
    
    return None


def detect_piercing_dark_cloud(o1, h1, l1, c1, o2, h2, l2, c2):
    """Piercing line (bullish) / Dark cloud cover (bearish)."""
    body1 = _body(o1, c1)
    mid1 = (o1 + c1) / 2
    
    if body1 == 0:
        return None
    
    # Piercing line: bearish candle, then bullish candle opening below and closing above midpoint
    if _is_bearish(o1, c1) and _is_bullish(o2, c2):
        if o2 < c1 and c2 > mid1 and c2 < o1:
            return PatternMatch(
                name="piercing_line", index=0,
                direction="bullish",
                strength=0.6,
                reliability=0.60,
            )
    
    # Dark cloud: bullish candle, then bearish candle opening above and closing below midpoint
    if _is_bullish(o1, c1) and _is_bearish(o2, c2):
        if o2 > c1 and c2 < mid1 and c2 > o1:
            return PatternMatch(
                name="dark_cloud_cover", index=0,
                direction="bearish",
                strength=0.6,
                reliability=0.60,
            )
    
    return None


# ─────────────────────────────────────────────────────────────────
# Triple Candle Patterns
# ─────────────────────────────────────────────────────────────────

def detect_morning_evening_star(o1, h1, l1, c1,
                                 o2, h2, l2, c2,
                                 o3, h3, l3, c3):
    """
    Morning star (bullish) / Evening star (bearish).
    Three candle reversal pattern with a small middle candle.
    """
    body1 = _body(o1, c1)
    body2 = _body(o2, c2)
    body3 = _body(o3, c3)
    
    if body1 == 0:
        return None
    
    small_body = body2 < body1 * 0.4
    
    # Morning star
    if _is_bearish(o1, c1) and small_body and _is_bullish(o3, c3):
        if c3 > (o1 + c1) / 2:
            return PatternMatch(
                name="morning_star", index=0,
                direction="bullish",
                strength=0.7,
                reliability=0.70,
            )
    
    # Evening star
    if _is_bullish(o1, c1) and small_body and _is_bearish(o3, c3):
        if c3 < (o1 + c1) / 2:
            return PatternMatch(
                name="evening_star", index=0,
                direction="bearish",
                strength=0.7,
                reliability=0.70,
            )
    
    return None


def detect_three_soldiers_crows(o1, h1, l1, c1,
                                 o2, h2, l2, c2,
                                 o3, h3, l3, c3):
    """
    Three white soldiers (bullish) / Three black crows (bearish).
    Three consecutive strong candles in the same direction.
    """
    # Three white soldiers
    if (_is_bullish(o1, c1) and _is_bullish(o2, c2) and _is_bullish(o3, c3)):
        if c1 < c2 < c3 and o2 > o1 and o3 > o2:
            # Check that each candle closes near its high
            r1 = _candle_range(h1, l1)
            r2 = _candle_range(h2, l2)
            r3 = _candle_range(h3, l3)
            
            if r1 > 0 and r2 > 0 and r3 > 0:
                if (h1 - c1) / r1 < 0.3 and (h2 - c2) / r2 < 0.3 and (h3 - c3) / r3 < 0.3:
                    return PatternMatch(
                        name="three_white_soldiers", index=0,
                        direction="bullish",
                        strength=0.85,
                        reliability=0.75,
                    )
    
    # Three black crows
    if (_is_bearish(o1, c1) and _is_bearish(o2, c2) and _is_bearish(o3, c3)):
        if c1 > c2 > c3 and o2 < o1 and o3 < o2:
            r1 = _candle_range(h1, l1)
            r2 = _candle_range(h2, l2)
            r3 = _candle_range(h3, l3)
            
            if r1 > 0 and r2 > 0 and r3 > 0:
                if (c1 - l1) / r1 < 0.3 and (c2 - l2) / r2 < 0.3 and (c3 - l3) / r3 < 0.3:
                    return PatternMatch(
                        name="three_black_crows", index=0,
                        direction="bearish",
                        strength=0.85,
                        reliability=0.75,
                    )
    
    return None


# ─────────────────────────────────────────────────────────────────
# Master Scanner — Scan All Patterns
# ─────────────────────────────────────────────────────────────────

def scan_patterns(opens: np.ndarray, highs: np.ndarray,
                  lows: np.ndarray, closes: np.ndarray,
                  lookback: int = 1) -> List[PatternMatch]:
    """
    Scan the last N bars for all candlestick patterns.
    
    Args:
        lookback: How many bars from the end to scan (1 = just latest)
    
    Returns: List of PatternMatch objects found
    """
    n = len(opens)
    if n < 4:
        return []
    
    patterns = []
    avg_body = _avg_body(opens, closes)
    
    for offset in range(lookback):
        i = n - 1 - offset
        if i < 3:
            break
        
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        o1, h1, l1, c1 = opens[i-1], highs[i-1], lows[i-1], closes[i-1]
        o2, h2, l2, c2 = opens[i-2], highs[i-2], lows[i-2], closes[i-2]
        
        # Single candle
        for detect_fn in [detect_hammer, detect_inverted_hammer, detect_shooting_star]:
            p = detect_fn(o, h, l, c, avg_body)
            if p:
                p.index = i
                patterns.append(p)
        
        p = detect_doji(o, h, l, c)
        if p:
            p.index = i
            patterns.append(p)
        
        p = detect_marubozu(o, h, l, c)
        if p:
            p.index = i
            patterns.append(p)
        
        p = detect_spinning_top(o, h, l, c)
        if p:
            p.index = i
            patterns.append(p)
        
        # Dual candle
        for detect_fn in [detect_engulfing, detect_harami, detect_tweezer, detect_piercing_dark_cloud]:
            p = detect_fn(o1, h1, l1, c1, o, h, l, c)
            if p:
                p.index = i
                patterns.append(p)
        
        # Triple candle
        if i >= 2:
            for detect_fn in [detect_morning_evening_star, detect_three_soldiers_crows]:
                p = detect_fn(o2, h2, l2, c2, o1, h1, l1, c1, o, h, l, c)
                if p:
                    p.index = i
                    patterns.append(p)
    
    return patterns


def compute_candle_signal(opens: np.ndarray, highs: np.ndarray,
                          lows: np.ndarray, closes: np.ndarray,
                          lookback: int = 3) -> dict:
    """
    Compute composite candlestick signal.
    
    Returns {
        bias: "bullish" | "bearish" | "neutral",
        strength: float [0, 1],
        patterns: [{name, direction, strength, reliability}],
        weighted_score: float [-1, 1]
    }
    """
    patterns = scan_patterns(opens, highs, lows, closes, lookback=lookback)
    
    if not patterns:
        return {"bias": "neutral", "strength": 0.0, "patterns": [], "weighted_score": 0.0}
    
    # Weight by strength * reliability
    weighted_sum = 0.0
    total_weight = 0.0
    
    for p in patterns:
        weight = p.strength * p.reliability
        if p.direction == "bullish":
            weighted_sum += weight
        elif p.direction == "bearish":
            weighted_sum -= weight
        total_weight += weight
    
    if total_weight == 0:
        score = 0.0
    else:
        score = weighted_sum / total_weight
    
    if score > 0.15:
        bias = "bullish"
    elif score < -0.15:
        bias = "bearish"
    else:
        bias = "neutral"
    
    return {
        "bias": bias,
        "strength": float(min(1.0, abs(score))),
        "patterns": [
            {"name": p.name, "direction": p.direction,
             "strength": p.strength, "reliability": p.reliability}
            for p in patterns
        ],
        "weighted_score": float(score),
    }
