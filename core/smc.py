"""
SMART MONEY CONCEPTS (SMC) — Institutional Price Action Analysis

Implements:
1. Order Blocks (OB) — institutional supply/demand zones
2. Fair Value Gaps (FVG) — imbalance zones / liquidity voids
3. Break of Structure (BOS) / Change of Character (CHoCH)
4. Liquidity sweeps — stop hunts above/below key levels
5. Premium/Discount zones — relative to dealing range
6. Inducement — fake breakouts trapping retail traders

All computed on OHLCV arrays. No external dependencies beyond numpy.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class OrderBlock:
    """Institutional supply/demand zone."""
    index: int              # Bar index where OB formed
    type: str               # "bullish" or "bearish"
    high: float             # Top of the zone
    low: float              # Bottom of the zone
    volume: float           # Volume at formation
    mitigated: bool = False # Has price returned and filled it?
    strength: float = 0.0   # 0-1 strength score


@dataclass
class FairValueGap:
    """Imbalance / liquidity void between candles."""
    index: int
    type: str               # "bullish" or "bearish"
    high: float             # Top of gap
    low: float              # Bottom of gap
    filled: bool = False
    size_pct: float = 0.0   # Gap size as % of price


@dataclass
class StructureBreak:
    """Break of Structure or Change of Character."""
    index: int
    type: str               # "bos_bullish", "bos_bearish", "choch_bullish", "choch_bearish"
    level: float            # Price level of the break
    prior_trend: str        # Trend before the break


@dataclass
class LiquiditySweep:
    """Stop hunt / liquidity grab."""
    index: int
    type: str               # "buy_side" (swept highs) or "sell_side" (swept lows)
    level: float            # Level that was swept
    reversal: bool = False  # Did price reverse after sweep?


# ─────────────────────────────────────────────────────────────────
# Swing Point Detection
# ─────────────────────────────────────────────────────────────────

def find_swing_highs(highs: np.ndarray, lookback: int = 5) -> List[Tuple[int, float]]:
    """Find swing highs — local maxima with N bars on each side."""
    swings = []
    for i in range(lookback, len(highs) - lookback):
        if highs[i] == np.max(highs[i - lookback:i + lookback + 1]):
            swings.append((i, float(highs[i])))
    return swings


def find_swing_lows(lows: np.ndarray, lookback: int = 5) -> List[Tuple[int, float]]:
    """Find swing lows — local minima with N bars on each side."""
    swings = []
    for i in range(lookback, len(lows) - lookback):
        if lows[i] == np.min(lows[i - lookback:i + lookback + 1]):
            swings.append((i, float(lows[i])))
    return swings


# ─────────────────────────────────────────────────────────────────
# Order Blocks
# ─────────────────────────────────────────────────────────────────

def detect_order_blocks(opens: np.ndarray, highs: np.ndarray,
                        lows: np.ndarray, closes: np.ndarray,
                        volumes: np.ndarray,
                        min_move_pct: float = 0.005,
                        lookback: int = 3) -> List[OrderBlock]:
    """
    Detect order blocks — the last opposing candle before an impulsive move.
    
    Bullish OB: Last bearish candle before a strong bullish move
    Bearish OB: Last bullish candle before a strong bearish move
    
    Args:
        min_move_pct: Minimum impulsive move size (as % of price)
        lookback: Number of candles to look back for the opposing candle
    """
    n = len(closes)
    order_blocks = []
    
    for i in range(lookback + 1, n - 1):
        # Check for impulsive bullish move
        move = (closes[i] - closes[i - 1]) / closes[i - 1]
        
        if move > min_move_pct:
            # Find last bearish candle before this move
            for j in range(i - 1, max(i - lookback - 1, 0), -1):
                if closes[j] < opens[j]:  # Bearish candle
                    # Volume confirmation
                    avg_vol = np.mean(volumes[max(0, j-10):j]) if j > 10 else volumes[j]
                    vol_ratio = volumes[j] / max(avg_vol, 1)
                    
                    strength = min(1.0, abs(move) / min_move_pct * 0.5 + vol_ratio * 0.3)
                    
                    order_blocks.append(OrderBlock(
                        index=j,
                        type="bullish",
                        high=float(highs[j]),
                        low=float(lows[j]),
                        volume=float(volumes[j]),
                        strength=float(strength),
                    ))
                    break
        
        elif move < -min_move_pct:
            # Find last bullish candle before this move
            for j in range(i - 1, max(i - lookback - 1, 0), -1):
                if closes[j] > opens[j]:  # Bullish candle
                    avg_vol = np.mean(volumes[max(0, j-10):j]) if j > 10 else volumes[j]
                    vol_ratio = volumes[j] / max(avg_vol, 1)
                    
                    strength = min(1.0, abs(move) / min_move_pct * 0.5 + vol_ratio * 0.3)
                    
                    order_blocks.append(OrderBlock(
                        index=j,
                        type="bearish",
                        high=float(highs[j]),
                        low=float(lows[j]),
                        volume=float(volumes[j]),
                        strength=float(strength),
                    ))
                    break
    
    # Check mitigation
    for ob in order_blocks:
        for k in range(ob.index + 1, n):
            if ob.type == "bullish" and lows[k] <= ob.low:
                ob.mitigated = True
                break
            elif ob.type == "bearish" and highs[k] >= ob.high:
                ob.mitigated = True
                break
    
    return order_blocks


def get_active_order_blocks(opens, highs, lows, closes, volumes,
                            current_price: float, max_age: int = 100) -> List[OrderBlock]:
    """Get unmitigated order blocks near current price."""
    obs = detect_order_blocks(opens, highs, lows, closes, volumes)
    n = len(closes)
    
    active = []
    for ob in obs:
        if ob.mitigated:
            continue
        if n - ob.index > max_age:
            continue
        # Is it near current price? (within 5%)
        ob_mid = (ob.high + ob.low) / 2
        distance = abs(current_price - ob_mid) / current_price
        if distance < 0.05:
            active.append(ob)
    
    return active


# ─────────────────────────────────────────────────────────────────
# Fair Value Gaps (FVG)
# ─────────────────────────────────────────────────────────────────

def detect_fvg(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
               min_gap_pct: float = 0.001) -> List[FairValueGap]:
    """
    Detect Fair Value Gaps — price imbalances between 3 candles.
    
    Bullish FVG: Low of candle 3 > High of candle 1 (gap up)
    Bearish FVG: High of candle 3 < Low of candle 1 (gap down)
    """
    n = len(highs)
    gaps = []
    
    for i in range(2, n):
        # Bullish FVG: candle 3's low is above candle 1's high
        if lows[i] > highs[i - 2]:
            gap_size = lows[i] - highs[i - 2]
            gap_pct = gap_size / closes[i - 1]
            
            if gap_pct >= min_gap_pct:
                gaps.append(FairValueGap(
                    index=i - 1,  # The middle candle
                    type="bullish",
                    high=float(lows[i]),
                    low=float(highs[i - 2]),
                    size_pct=float(gap_pct),
                ))
        
        # Bearish FVG: candle 3's high is below candle 1's low
        if highs[i] < lows[i - 2]:
            gap_size = lows[i - 2] - highs[i]
            gap_pct = gap_size / closes[i - 1]
            
            if gap_pct >= min_gap_pct:
                gaps.append(FairValueGap(
                    index=i - 1,
                    type="bearish",
                    high=float(lows[i - 2]),
                    low=float(highs[i]),
                    size_pct=float(gap_pct),
                ))
    
    # Check if FVGs have been filled
    for gap in gaps:
        for k in range(gap.index + 2, n):
            if gap.type == "bullish" and lows[k] <= gap.low:
                gap.filled = True
                break
            elif gap.type == "bearish" and highs[k] >= gap.high:
                gap.filled = True
                break
    
    return gaps


# ─────────────────────────────────────────────────────────────────
# Break of Structure (BOS) / Change of Character (CHoCH)
# ─────────────────────────────────────────────────────────────────

def detect_structure_breaks(highs: np.ndarray, lows: np.ndarray,
                            closes: np.ndarray,
                            swing_lookback: int = 5) -> List[StructureBreak]:
    """
    Detect BOS and CHoCH.
    
    BOS (Break of Structure): Price breaks a swing high/low in the direction of trend.
    CHoCH (Change of Character): Price breaks a swing high/low AGAINST the trend.
    
    Higher highs + higher lows = uptrend
    Lower highs + lower lows = downtrend
    """
    swing_highs = find_swing_highs(highs, swing_lookback)
    swing_lows = find_swing_lows(lows, swing_lookback)
    
    breaks = []
    
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return breaks
    
    # Determine trend at each point
    for i in range(1, len(swing_highs)):
        prev_idx, prev_high = swing_highs[i - 1]
        curr_idx, curr_high = swing_highs[i]
        
        # Find corresponding swing lows
        nearby_lows = [sl for sl in swing_lows if prev_idx <= sl[0] <= curr_idx]
        
        if not nearby_lows:
            continue
        
        # Current trend based on swing structure
        higher_high = curr_high > prev_high
        
        # Check for breaks after this swing high
        for j in range(curr_idx + 1, min(curr_idx + swing_lookback * 3, len(closes))):
            if closes[j] > curr_high:
                # Broke above swing high
                if higher_high:
                    break_type = "bos_bullish"
                    prior = "uptrend"
                else:
                    break_type = "choch_bullish"
                    prior = "downtrend"
                
                breaks.append(StructureBreak(
                    index=j, type=break_type,
                    level=float(curr_high), prior_trend=prior
                ))
                break
    
    for i in range(1, len(swing_lows)):
        prev_idx, prev_low = swing_lows[i - 1]
        curr_idx, curr_low = swing_lows[i]
        
        lower_low = curr_low < prev_low
        
        for j in range(curr_idx + 1, min(curr_idx + swing_lookback * 3, len(closes))):
            if closes[j] < curr_low:
                if lower_low:
                    break_type = "bos_bearish"
                    prior = "downtrend"
                else:
                    break_type = "choch_bearish"
                    prior = "uptrend"
                
                breaks.append(StructureBreak(
                    index=j, type=break_type,
                    level=float(curr_low), prior_trend=prior
                ))
                break
    
    breaks.sort(key=lambda x: x.index)
    return breaks


# ─────────────────────────────────────────────────────────────────
# Liquidity Sweeps
# ─────────────────────────────────────────────────────────────────

def detect_liquidity_sweeps(highs: np.ndarray, lows: np.ndarray,
                            closes: np.ndarray,
                            swing_lookback: int = 5,
                            reversal_bars: int = 3) -> List[LiquiditySweep]:
    """
    Detect liquidity sweeps — price briefly breaks a key level then reverses.
    
    Buy-side sweep: Price spikes above swing high then closes below it
    Sell-side sweep: Price dips below swing low then closes above it
    """
    swing_highs = find_swing_highs(highs, swing_lookback)
    swing_lows = find_swing_lows(lows, swing_lookback)
    
    sweeps = []
    n = len(closes)
    
    # Check for buy-side liquidity sweeps (stop hunts above highs)
    for sh_idx, sh_level in swing_highs:
        for i in range(sh_idx + swing_lookback, min(sh_idx + 50, n)):
            if highs[i] > sh_level and closes[i] < sh_level:
                # Swept above then closed below — classic stop hunt
                # Check for reversal
                reversal = False
                if i + reversal_bars < n:
                    future_closes = closes[i + 1:i + reversal_bars + 1]
                    if len(future_closes) > 0 and np.all(future_closes < sh_level):
                        reversal = True
                
                sweeps.append(LiquiditySweep(
                    index=i, type="buy_side",
                    level=float(sh_level), reversal=reversal
                ))
                break
    
    # Check for sell-side liquidity sweeps (stop hunts below lows)
    for sl_idx, sl_level in swing_lows:
        for i in range(sl_idx + swing_lookback, min(sl_idx + 50, n)):
            if lows[i] < sl_level and closes[i] > sl_level:
                reversal = False
                if i + reversal_bars < n:
                    future_closes = closes[i + 1:i + reversal_bars + 1]
                    if len(future_closes) > 0 and np.all(future_closes > sl_level):
                        reversal = True
                
                sweeps.append(LiquiditySweep(
                    index=i, type="sell_side",
                    level=float(sl_level), reversal=reversal
                ))
                break
    
    sweeps.sort(key=lambda x: x.index)
    return sweeps


# ─────────────────────────────────────────────────────────────────
# Premium / Discount Zones
# ─────────────────────────────────────────────────────────────────

def compute_premium_discount(highs: np.ndarray, lows: np.ndarray,
                             closes: np.ndarray,
                             lookback: int = 50) -> dict:
    """
    Compute where price is relative to the dealing range.
    
    Premium zone: Above 50% of range (expensive, look to sell)
    Discount zone: Below 50% of range (cheap, look to buy)
    Equilibrium: Around 50%
    """
    if len(closes) < lookback:
        return {"zone": "equilibrium", "position_pct": 50.0, "range_high": 0, "range_low": 0}
    
    range_high = float(np.max(highs[-lookback:]))
    range_low = float(np.min(lows[-lookback:]))
    current = float(closes[-1])
    
    if range_high == range_low:
        return {"zone": "equilibrium", "position_pct": 50.0,
                "range_high": range_high, "range_low": range_low}
    
    position = (current - range_low) / (range_high - range_low) * 100
    
    if position > 70:
        zone = "premium"
    elif position < 30:
        zone = "discount"
    elif position > 55:
        zone = "slight_premium"
    elif position < 45:
        zone = "slight_discount"
    else:
        zone = "equilibrium"
    
    return {
        "zone": zone,
        "position_pct": float(position),
        "range_high": range_high,
        "range_low": range_low,
        "equilibrium": float((range_high + range_low) / 2),
    }


# ─────────────────────────────────────────────────────────────────
# SMC Composite Signal
# ─────────────────────────────────────────────────────────────────

def compute_smc_signal(opens: np.ndarray, highs: np.ndarray,
                       lows: np.ndarray, closes: np.ndarray,
                       volumes: np.ndarray) -> dict:
    """
    Compute a composite SMC signal from all components.
    
    Returns {
        bias: "bullish" | "bearish" | "neutral",
        strength: float [0, 1],
        components: {
            order_blocks, fvg, structure, liquidity, premium_discount
        }
    }
    """
    n = len(closes)
    if n < 50:
        return {"bias": "neutral", "strength": 0.0, "components": {}}
    
    current_price = float(closes[-1])
    bull_score = 0.0
    bear_score = 0.0
    
    # ── Order Blocks ──
    active_obs = get_active_order_blocks(opens, highs, lows, closes, volumes, current_price)
    bullish_obs = [ob for ob in active_obs if ob.type == "bullish"]
    bearish_obs = [ob for ob in active_obs if ob.type == "bearish"]
    
    # Price near bullish OB = bullish
    for ob in bullish_obs:
        if current_price >= ob.low and current_price <= ob.high * 1.01:
            bull_score += 0.3 * ob.strength
    
    for ob in bearish_obs:
        if current_price <= ob.high and current_price >= ob.low * 0.99:
            bear_score += 0.3 * ob.strength
    
    # ── Fair Value Gaps ──
    fvgs = detect_fvg(highs, lows, closes)
    unfilled_bull_fvg = [g for g in fvgs if g.type == "bullish" and not g.filled and g.index > n - 50]
    unfilled_bear_fvg = [g for g in fvgs if g.type == "bearish" and not g.filled and g.index > n - 50]
    
    # Unfilled FVGs below price = bullish support
    for gap in unfilled_bull_fvg:
        if gap.high < current_price:
            bull_score += 0.15
    for gap in unfilled_bear_fvg:
        if gap.low > current_price:
            bear_score += 0.15
    
    # ── Structure ──
    breaks = detect_structure_breaks(highs, lows, closes)
    recent_breaks = [b for b in breaks if b.index > n - 20]
    
    for brk in recent_breaks:
        if brk.type == "bos_bullish":
            bull_score += 0.25
        elif brk.type == "bos_bearish":
            bear_score += 0.25
        elif brk.type == "choch_bullish":
            bull_score += 0.35  # CHoCH is stronger signal
        elif brk.type == "choch_bearish":
            bear_score += 0.35
    
    # ── Liquidity Sweeps ──
    sweeps = detect_liquidity_sweeps(highs, lows, closes)
    recent_sweeps = [s for s in sweeps if s.index > n - 10]
    
    for sweep in recent_sweeps:
        if sweep.type == "sell_side" and sweep.reversal:
            bull_score += 0.3  # Sell-side swept + reversed = bullish
        elif sweep.type == "buy_side" and sweep.reversal:
            bear_score += 0.3
    
    # ── Premium/Discount ──
    pd_zone = compute_premium_discount(highs, lows, closes)
    if pd_zone["zone"] in ("discount", "slight_discount"):
        bull_score += 0.15
    elif pd_zone["zone"] in ("premium", "slight_premium"):
        bear_score += 0.15
    
    # ── Composite ──
    total = bull_score + bear_score
    if total == 0:
        return {"bias": "neutral", "strength": 0.0, "components": {
            "order_blocks": {"bullish": len(bullish_obs), "bearish": len(bearish_obs)},
            "fvg": {"bullish_unfilled": len(unfilled_bull_fvg), "bearish_unfilled": len(unfilled_bear_fvg)},
            "structure": {"recent_breaks": [b.type for b in recent_breaks]},
            "liquidity": {"recent_sweeps": [(s.type, s.reversal) for s in recent_sweeps]},
            "premium_discount": pd_zone,
        }}
    
    if bull_score > bear_score:
        bias = "bullish"
        strength = min(1.0, (bull_score - bear_score) / max(total, 1))
    elif bear_score > bull_score:
        bias = "bearish"
        strength = min(1.0, (bear_score - bull_score) / max(total, 1))
    else:
        bias = "neutral"
        strength = 0.0
    
    return {
        "bias": bias,
        "strength": float(strength),
        "bull_score": float(bull_score),
        "bear_score": float(bear_score),
        "components": {
            "order_blocks": {"bullish": len(bullish_obs), "bearish": len(bearish_obs)},
            "fvg": {"bullish_unfilled": len(unfilled_bull_fvg), "bearish_unfilled": len(unfilled_bear_fvg)},
            "structure": {"recent_breaks": [b.type for b in recent_breaks]},
            "liquidity": {"recent_sweeps": [(s.type, s.reversal) for s in recent_sweeps]},
            "premium_discount": pd_zone,
        },
    }
