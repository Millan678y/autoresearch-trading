"""
TECHNICAL ANALYSIS — Complete Indicator Library

30+ indicators across all major categories. Pure numpy, no external deps.

Categories:
1. Trend: Ichimoku, Supertrend, Parabolic SAR, ADX/DMI, Donchian, HMA
2. Momentum: Williams %R, CCI, Awesome Oscillator, Elder Ray, TRIX, ROC
3. Volume: OBV, MFI, Chaikin Money Flow, VWMA, A/D Line
4. Volatility: Keltner Channels, Donchian Width, Historical Vol, ATR%
5. Support/Resistance: Fibonacci, Pivot Points, Linear Regression Channel

Each indicator returns numpy arrays or scalar values.
Composite signals available for strategy integration.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# ═══════════════════════════════════════════════════════════════════
# 1. TREND INDICATORS
# ═══════════════════════════════════════════════════════════════════

def ichimoku(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
             tenkan: int = 9, kijun: int = 26, senkou_b: int = 52) -> dict:
    """
    Ichimoku Cloud — the king of trend indicators.
    
    Returns dict with:
    - tenkan_sen: Conversion line (fast)
    - kijun_sen: Base line (slow)
    - senkou_a: Leading span A (cloud top/bottom)
    - senkou_b: Leading span B (cloud top/bottom)
    - chikou: Lagging span
    - signal: "bullish_above_cloud", "bearish_below_cloud", "inside_cloud",
              "tk_cross_bull", "tk_cross_bear"
    """
    n = len(closes)
    
    def mid(arr, period):
        result = np.full(n, np.nan)
        for i in range(period - 1, n):
            result[i] = (np.max(arr[i - period + 1:i + 1]) + np.min(arr[i - period + 1:i + 1])) / 2
        return result
    
    tenkan_sen = mid(np.column_stack([highs, lows]).max(axis=1), tenkan)
    # Recalc properly
    tenkan_sen = np.full(n, np.nan)
    kijun_sen = np.full(n, np.nan)
    senkou_b_line = np.full(n, np.nan)
    
    for i in range(n):
        if i >= tenkan - 1:
            tenkan_sen[i] = (np.max(highs[i-tenkan+1:i+1]) + np.min(lows[i-tenkan+1:i+1])) / 2
        if i >= kijun - 1:
            kijun_sen[i] = (np.max(highs[i-kijun+1:i+1]) + np.min(lows[i-kijun+1:i+1])) / 2
        if i >= senkou_b - 1:
            senkou_b_line[i] = (np.max(highs[i-senkou_b+1:i+1]) + np.min(lows[i-senkou_b+1:i+1])) / 2
    
    # Senkou A = (tenkan + kijun) / 2, shifted forward 26 periods
    senkou_a = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(tenkan_sen[i]) and not np.isnan(kijun_sen[i]):
            target = i + kijun
            if target < n:
                senkou_a[target] = (tenkan_sen[i] + kijun_sen[i]) / 2
    
    # Senkou B shifted forward
    senkou_b_shifted = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(senkou_b_line[i]):
            target = i + kijun
            if target < n:
                senkou_b_shifted[target] = senkou_b_line[i]
    
    # Chikou = close shifted back 26 periods
    chikou = np.full(n, np.nan)
    for i in range(kijun, n):
        chikou[i - kijun] = closes[i]
    
    # Signal
    sig = "neutral"
    if n > kijun + 2:
        cloud_top = max(senkou_a[-1] if not np.isnan(senkou_a[-1]) else 0,
                       senkou_b_shifted[-1] if not np.isnan(senkou_b_shifted[-1]) else 0)
        cloud_bot = min(senkou_a[-1] if not np.isnan(senkou_a[-1]) else float('inf'),
                       senkou_b_shifted[-1] if not np.isnan(senkou_b_shifted[-1]) else float('inf'))
        
        price = closes[-1]
        
        if price > cloud_top and cloud_top > 0:
            sig = "bullish_above_cloud"
        elif price < cloud_bot and cloud_bot < float('inf'):
            sig = "bearish_below_cloud"
        else:
            sig = "inside_cloud"
        
        # TK cross
        if (not np.isnan(tenkan_sen[-1]) and not np.isnan(kijun_sen[-1]) and
            not np.isnan(tenkan_sen[-2]) and not np.isnan(kijun_sen[-2])):
            if tenkan_sen[-2] <= kijun_sen[-2] and tenkan_sen[-1] > kijun_sen[-1]:
                sig = "tk_cross_bull"
            elif tenkan_sen[-2] >= kijun_sen[-2] and tenkan_sen[-1] < kijun_sen[-1]:
                sig = "tk_cross_bear"
    
    return {
        "tenkan_sen": tenkan_sen,
        "kijun_sen": kijun_sen,
        "senkou_a": senkou_a,
        "senkou_b": senkou_b_shifted,
        "chikou": chikou,
        "signal": sig,
    }


def supertrend(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
               period: int = 10, multiplier: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Supertrend — trend-following indicator.
    
    Returns (supertrend_line, direction)
    direction: 1 = bullish, -1 = bearish
    """
    n = len(closes)
    atr = np.zeros(n)
    
    # ATR
    for i in range(1, n):
        tr = max(highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1]))
        if i < period:
            atr[i] = tr
        else:
            atr[i] = (atr[i-1] * (period - 1) + tr) / period
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    st = np.zeros(n)
    direction = np.ones(n)
    
    for i in range(period, n):
        hl2 = (highs[i] + lows[i]) / 2
        upper[i] = hl2 + multiplier * atr[i]
        lower[i] = hl2 - multiplier * atr[i]
        
        # Adjust bands
        if lower[i] > lower[i-1] or closes[i-1] < lower[i-1]:
            pass
        else:
            lower[i] = lower[i-1]
        
        if upper[i] < upper[i-1] or closes[i-1] > upper[i-1]:
            pass
        else:
            upper[i] = upper[i-1]
        
        # Direction
        if closes[i] > upper[i-1] if direction[i-1] == -1 else closes[i] > lower[i]:
            direction[i] = 1
        elif closes[i] < lower[i-1] if direction[i-1] == 1 else closes[i] < upper[i]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
        
        st[i] = lower[i] if direction[i] == 1 else upper[i]
    
    return st, direction


def parabolic_sar(highs: np.ndarray, lows: np.ndarray,
                  af_start: float = 0.02, af_step: float = 0.02,
                  af_max: float = 0.20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parabolic SAR — trailing stop and trend indicator.
    
    Returns (sar_values, direction)  direction: 1=long, -1=short
    """
    n = len(highs)
    sar = np.zeros(n)
    direction = np.ones(n)
    
    af = af_start
    ep = highs[0]  # Extreme point
    sar[0] = lows[0]
    
    for i in range(1, n):
        prev_sar = sar[i-1]
        
        if direction[i-1] == 1:  # Long
            sar[i] = prev_sar + af * (ep - prev_sar)
            sar[i] = min(sar[i], lows[i-1])
            if i >= 2:
                sar[i] = min(sar[i], lows[i-2])
            
            if lows[i] < sar[i]:
                direction[i] = -1
                sar[i] = ep
                ep = lows[i]
                af = af_start
            else:
                direction[i] = 1
                if highs[i] > ep:
                    ep = highs[i]
                    af = min(af + af_step, af_max)
        else:  # Short
            sar[i] = prev_sar + af * (ep - prev_sar)
            sar[i] = max(sar[i], highs[i-1])
            if i >= 2:
                sar[i] = max(sar[i], highs[i-2])
            
            if highs[i] > sar[i]:
                direction[i] = 1
                sar[i] = ep
                ep = highs[i]
                af = af_start
            else:
                direction[i] = -1
                if lows[i] < ep:
                    ep = lows[i]
                    af = min(af + af_step, af_max)
    
    return sar, direction


def donchian_channels(highs: np.ndarray, lows: np.ndarray,
                      period: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Donchian Channels — N-period high/low breakout system.
    
    Returns (upper, middle, lower)
    """
    n = len(highs)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(highs[i - period + 1:i + 1])
        lower[i] = np.min(lows[i - period + 1:i + 1])
    
    middle = (upper + lower) / 2
    return upper, middle, lower


def hull_moving_average(closes: np.ndarray, period: int = 16) -> np.ndarray:
    """
    Hull Moving Average — fast, smooth, low lag.
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    def wma(arr, p):
        result = np.full(len(arr), np.nan)
        weights = np.arange(1, p + 1, dtype=float)
        w_sum = weights.sum()
        for i in range(p - 1, len(arr)):
            result[i] = np.sum(arr[i - p + 1:i + 1] * weights) / w_sum
        return result
    
    half_period = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    
    wma_half = wma(closes, half_period)
    wma_full = wma(closes, period)
    
    diff = 2 * wma_half - wma_full
    
    # Remove NaNs for final WMA
    valid = ~np.isnan(diff)
    if valid.sum() < sqrt_period:
        return np.full(len(closes), np.nan)
    
    hma = wma(np.where(np.isnan(diff), closes, diff), sqrt_period)
    return hma


# ═══════════════════════════════════════════════════════════════════
# 2. MOMENTUM INDICATORS
# ═══════════════════════════════════════════════════════════════════

def williams_r(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
               period: int = 14) -> np.ndarray:
    """
    Williams %R — momentum oscillator (-100 to 0).
    < -80 = oversold, > -20 = overbought
    """
    n = len(closes)
    wr = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        hh = np.max(highs[i - period + 1:i + 1])
        ll = np.min(lows[i - period + 1:i + 1])
        if hh != ll:
            wr[i] = -100 * (hh - closes[i]) / (hh - ll)
    
    return wr


def cci(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
        period: int = 20) -> np.ndarray:
    """
    Commodity Channel Index — identifies cyclical trends.
    > 100 = overbought/strong trend, < -100 = oversold/strong downtrend
    """
    tp = (highs + lows + closes) / 3
    n = len(tp)
    result = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        window = tp[i - period + 1:i + 1]
        sma = np.mean(window)
        mad = np.mean(np.abs(window - sma))
        if mad > 0:
            result[i] = (tp[i] - sma) / (0.015 * mad)
    
    return result


def awesome_oscillator(highs: np.ndarray, lows: np.ndarray,
                       fast: int = 5, slow: int = 34) -> np.ndarray:
    """
    Awesome Oscillator — momentum using median price SMAs.
    AO = SMA(median, 5) - SMA(median, 34)
    Positive = bullish momentum, negative = bearish
    """
    median = (highs + lows) / 2
    n = len(median)
    
    sma_fast = np.full(n, np.nan)
    sma_slow = np.full(n, np.nan)
    
    for i in range(fast - 1, n):
        sma_fast[i] = np.mean(median[i - fast + 1:i + 1])
    for i in range(slow - 1, n):
        sma_slow[i] = np.mean(median[i - slow + 1:i + 1])
    
    return sma_fast - sma_slow


def elder_ray(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
              period: int = 13) -> Tuple[np.ndarray, np.ndarray]:
    """
    Elder Ray — Bull Power and Bear Power.
    Bull Power = High - EMA(close, 13)
    Bear Power = Low - EMA(close, 13)
    
    Returns (bull_power, bear_power)
    """
    ema_val = _ema(closes, period)
    bull_power = highs - ema_val
    bear_power = lows - ema_val
    return bull_power, bear_power


def trix(closes: np.ndarray, period: int = 15) -> np.ndarray:
    """
    TRIX — Triple exponential smoothed rate of change.
    Filters noise aggressively. Good for trend confirmation.
    """
    ema1 = _ema(closes, period)
    ema2 = _ema(ema1, period)
    ema3 = _ema(ema2, period)
    
    result = np.full(len(closes), np.nan)
    for i in range(1, len(ema3)):
        if ema3[i-1] != 0:
            result[i] = (ema3[i] - ema3[i-1]) / ema3[i-1] * 10000
    
    return result


def roc(closes: np.ndarray, period: int = 12) -> np.ndarray:
    """Rate of Change — simple momentum."""
    n = len(closes)
    result = np.full(n, np.nan)
    for i in range(period, n):
        if closes[i - period] != 0:
            result[i] = (closes[i] - closes[i - period]) / closes[i - period] * 100
    return result


# ═══════════════════════════════════════════════════════════════════
# 3. VOLUME INDICATORS
# ═══════════════════════════════════════════════════════════════════

def obv(closes: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    """
    On Balance Volume — cumulative volume flow.
    Rising OBV confirms uptrend, divergence = warning.
    """
    n = len(closes)
    result = np.zeros(n)
    
    for i in range(1, n):
        if closes[i] > closes[i-1]:
            result[i] = result[i-1] + volumes[i]
        elif closes[i] < closes[i-1]:
            result[i] = result[i-1] - volumes[i]
        else:
            result[i] = result[i-1]
    
    return result


def mfi(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
        volumes: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Money Flow Index — volume-weighted RSI.
    > 80 = overbought, < 20 = oversold
    """
    tp = (highs + lows + closes) / 3
    mf = tp * volumes
    n = len(closes)
    result = np.full(n, np.nan)
    
    for i in range(period, n):
        pos_mf = 0
        neg_mf = 0
        for j in range(i - period + 1, i + 1):
            if tp[j] > tp[j-1]:
                pos_mf += mf[j]
            else:
                neg_mf += mf[j]
        
        if neg_mf > 0:
            ratio = pos_mf / neg_mf
            result[i] = 100 - 100 / (1 + ratio)
        else:
            result[i] = 100
    
    return result


def chaikin_money_flow(highs: np.ndarray, lows: np.ndarray,
                       closes: np.ndarray, volumes: np.ndarray,
                       period: int = 20) -> np.ndarray:
    """
    Chaikin Money Flow — measures buying/selling pressure.
    Positive = accumulation (bullish), Negative = distribution (bearish)
    """
    hl_range = highs - lows
    clv = np.where(hl_range > 0, ((closes - lows) - (highs - closes)) / hl_range, 0)
    mfv = clv * volumes
    
    n = len(closes)
    result = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        vol_sum = np.sum(volumes[i - period + 1:i + 1])
        if vol_sum > 0:
            result[i] = np.sum(mfv[i - period + 1:i + 1]) / vol_sum
    
    return result


def vwma(closes: np.ndarray, volumes: np.ndarray, period: int = 20) -> np.ndarray:
    """Volume Weighted Moving Average."""
    n = len(closes)
    result = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        vol_sum = np.sum(volumes[i - period + 1:i + 1])
        if vol_sum > 0:
            result[i] = np.sum(closes[i - period + 1:i + 1] * volumes[i - period + 1:i + 1]) / vol_sum
    
    return result


def ad_line(highs: np.ndarray, lows: np.ndarray,
            closes: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    """Accumulation/Distribution Line."""
    hl_range = highs - lows
    clv = np.where(hl_range > 0, ((closes - lows) - (highs - closes)) / hl_range, 0)
    return np.cumsum(clv * volumes)


# ═══════════════════════════════════════════════════════════════════
# 4. VOLATILITY INDICATORS
# ═══════════════════════════════════════════════════════════════════

def keltner_channels(highs: np.ndarray, lows: np.ndarray,
                     closes: np.ndarray, ema_period: int = 20,
                     atr_period: int = 10, multiplier: float = 1.5
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Keltner Channels — EMA ± ATR multiplier.
    Squeeze with Bollinger Bands = breakout setup.
    
    Returns (upper, middle, lower)
    """
    middle = _ema(closes, ema_period)
    
    n = len(closes)
    atr = np.zeros(n)
    for i in range(1, n):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        if i < atr_period:
            atr[i] = tr
        else:
            atr[i] = (atr[i-1] * (atr_period - 1) + tr) / atr_period
    
    upper = middle + multiplier * atr
    lower = middle - multiplier * atr
    
    return upper, middle, lower


def historical_volatility(closes: np.ndarray, period: int = 20,
                          annualize: float = 252) -> np.ndarray:
    """
    Historical (realized) volatility — annualized std of log returns.
    """
    n = len(closes)
    result = np.full(n, np.nan)
    log_rets = np.diff(np.log(closes))
    
    for i in range(period, n):
        result[i] = np.std(log_rets[i - period:i]) * np.sqrt(annualize)
    
    return result


def atr_percent(highs: np.ndarray, lows: np.ndarray,
                closes: np.ndarray, period: int = 14) -> np.ndarray:
    """ATR as percentage of price — normalized volatility."""
    n = len(closes)
    atr = np.zeros(n)
    
    for i in range(1, n):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        if i < period:
            atr[i] = tr
        else:
            atr[i] = (atr[i-1] * (period - 1) + tr) / period
    
    return np.where(closes > 0, atr / closes * 100, 0)


# ═══════════════════════════════════════════════════════════════════
# 5. SUPPORT / RESISTANCE
# ═══════════════════════════════════════════════════════════════════

def fibonacci_retracements(high: float, low: float) -> dict:
    """
    Fibonacci retracement levels from a swing high/low.
    
    Returns dict of level_name -> price
    """
    diff = high - low
    return {
        "0.0": high,
        "0.236": high - 0.236 * diff,
        "0.382": high - 0.382 * diff,
        "0.5": high - 0.5 * diff,
        "0.618": high - 0.618 * diff,
        "0.786": high - 0.786 * diff,
        "1.0": low,
        "1.272": low - 0.272 * diff,
        "1.618": low - 0.618 * diff,
    }


def pivot_points(high: float, low: float, close: float,
                 method: str = "standard") -> dict:
    """
    Pivot Points — classic support/resistance levels.
    
    Methods: "standard", "fibonacci", "camarilla"
    """
    pivot = (high + low + close) / 3
    
    if method == "standard":
        return {
            "R3": high + 2 * (pivot - low),
            "R2": pivot + (high - low),
            "R1": 2 * pivot - low,
            "P": pivot,
            "S1": 2 * pivot - high,
            "S2": pivot - (high - low),
            "S3": low - 2 * (high - pivot),
        }
    elif method == "fibonacci":
        diff = high - low
        return {
            "R3": pivot + 1.000 * diff,
            "R2": pivot + 0.618 * diff,
            "R1": pivot + 0.382 * diff,
            "P": pivot,
            "S1": pivot - 0.382 * diff,
            "S2": pivot - 0.618 * diff,
            "S3": pivot - 1.000 * diff,
        }
    elif method == "camarilla":
        diff = high - low
        return {
            "R4": close + diff * 1.1 / 2,
            "R3": close + diff * 1.1 / 4,
            "R2": close + diff * 1.1 / 6,
            "R1": close + diff * 1.1 / 12,
            "P": pivot,
            "S1": close - diff * 1.1 / 12,
            "S2": close - diff * 1.1 / 6,
            "S3": close - diff * 1.1 / 4,
            "S4": close - diff * 1.1 / 2,
        }
    return {"P": pivot}


def linear_regression_channel(closes: np.ndarray, period: int = 50,
                               deviations: float = 2.0
                               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Linear Regression Channel — trendline with deviation bands.
    
    Returns (upper, middle, lower)
    """
    n = len(closes)
    upper = np.full(n, np.nan)
    middle = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        y = closes[i - period + 1:i + 1]
        x = np.arange(period)
        
        # Linear regression
        coeffs = np.polyfit(x, y, 1)
        fitted = np.polyval(coeffs, x)
        
        # Standard deviation of residuals
        residuals = y - fitted
        std = np.std(residuals)
        
        middle[i] = fitted[-1]
        upper[i] = fitted[-1] + deviations * std
        lower[i] = fitted[-1] - deviations * std
    
    return upper, middle, lower


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════

def _ema(values: np.ndarray, span: int) -> np.ndarray:
    """Exponential Moving Average."""
    alpha = 2.0 / (span + 1)
    result = np.empty(len(values), dtype=float)
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
    return result


# ═══════════════════════════════════════════════════════════════════
# COMPOSITE TA SIGNAL
# ═══════════════════════════════════════════════════════════════════

def compute_ta_signal(opens: np.ndarray, highs: np.ndarray,
                      lows: np.ndarray, closes: np.ndarray,
                      volumes: np.ndarray) -> dict:
    """
    Compute a composite technical analysis signal from multiple indicators.
    
    Returns {
        bias: "bullish" | "bearish" | "neutral",
        strength: float [0, 1],
        indicators: {name: {value, signal}}
    }
    """
    n = len(closes)
    if n < 60:
        return {"bias": "neutral", "strength": 0.0, "indicators": {}}
    
    bull = 0
    bear = 0
    indicators = {}
    
    # Ichimoku
    ichi = ichimoku(highs, lows, closes)
    if "bull" in ichi["signal"]:
        bull += 1
    elif "bear" in ichi["signal"]:
        bear += 1
    indicators["ichimoku"] = {"signal": ichi["signal"]}
    
    # Supertrend
    st, st_dir = supertrend(highs, lows, closes)
    if st_dir[-1] == 1:
        bull += 1
    else:
        bear += 1
    indicators["supertrend"] = {"direction": float(st_dir[-1])}
    
    # Parabolic SAR
    sar, sar_dir = parabolic_sar(highs, lows)
    if sar_dir[-1] == 1:
        bull += 1
    else:
        bear += 1
    indicators["parabolic_sar"] = {"direction": float(sar_dir[-1])}
    
    # Williams %R
    wr = williams_r(highs, lows, closes)
    wr_val = wr[-1] if not np.isnan(wr[-1]) else -50
    if wr_val > -20:
        bear += 1  # Overbought
    elif wr_val < -80:
        bull += 1  # Oversold
    indicators["williams_r"] = {"value": float(wr_val)}
    
    # CCI
    cci_val = cci(highs, lows, closes)
    c = cci_val[-1] if not np.isnan(cci_val[-1]) else 0
    if c > 100:
        bull += 1
    elif c < -100:
        bear += 1
    indicators["cci"] = {"value": float(c)}
    
    # MFI
    mfi_val = mfi(highs, lows, closes, volumes)
    m = mfi_val[-1] if not np.isnan(mfi_val[-1]) else 50
    if m > 80:
        bear += 1  # Overbought
    elif m < 20:
        bull += 1  # Oversold
    indicators["mfi"] = {"value": float(m)}
    
    # Awesome Oscillator
    ao = awesome_oscillator(highs, lows)
    ao_val = ao[-1] if not np.isnan(ao[-1]) else 0
    if ao_val > 0:
        bull += 1
    else:
        bear += 1
    indicators["awesome_oscillator"] = {"value": float(ao_val)}
    
    # OBV trend
    obv_val = obv(closes, volumes)
    obv_sma = np.mean(obv_val[-20:]) if n >= 20 else obv_val[-1]
    if obv_val[-1] > obv_sma:
        bull += 1
    else:
        bear += 1
    indicators["obv"] = {"trend": "up" if obv_val[-1] > obv_sma else "down"}
    
    # Chaikin MF
    cmf = chaikin_money_flow(highs, lows, closes, volumes)
    cmf_val = cmf[-1] if not np.isnan(cmf[-1]) else 0
    if cmf_val > 0.05:
        bull += 1
    elif cmf_val < -0.05:
        bear += 1
    indicators["chaikin_mf"] = {"value": float(cmf_val)}
    
    # TRIX
    trix_val = trix(closes)
    t = trix_val[-1] if not np.isnan(trix_val[-1]) else 0
    if t > 0:
        bull += 1
    else:
        bear += 1
    indicators["trix"] = {"value": float(t)}
    
    # Composite
    total = bull + bear
    if total == 0:
        return {"bias": "neutral", "strength": 0.0, "indicators": indicators}
    
    if bull > bear:
        bias = "bullish"
        strength = (bull - bear) / total
    elif bear > bull:
        bias = "bearish"
        strength = (bear - bull) / total
    else:
        bias = "neutral"
        strength = 0.0
    
    return {
        "bias": bias,
        "strength": float(min(1.0, strength)),
        "bull_count": bull,
        "bear_count": bear,
        "total_indicators": total,
        "indicators": indicators,
    }
