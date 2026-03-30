"""
GENESIS ENGINE — Generative Strategy Factory

Generates new strategy variants through:
1. Parameter mutation (±10-50% of parent params)
2. Signal combination (pick N from signal library)
3. Template instantiation (proven patterns with random params)
4. Crossover (merge two parent strategies)

No LLM required — pure programmatic generation.
"""

import hashlib
import random
import copy
import itertools
import numpy as np
from typing import List, Optional
from .models import StrategyRecord, save_strategy, load_strategies, log_event

# ─────────────────────────────────────────────────────────────────
# Signal Library — building blocks for strategy assembly
# ─────────────────────────────────────────────────────────────────

SIGNAL_LIBRARY = {
    "momentum": {
        "code": '''
    ret = (closes[-1] - closes[-{lookback}]) / closes[-{lookback}]
    bull = ret > {threshold}
    bear = ret < -{threshold}''',
        "params": {"lookback": (4, 48), "threshold": (0.005, 0.03)},
    },
    "ema_cross": {
        "code": '''
    ema_f = _ema(closes, {fast})
    ema_s = _ema(closes, {slow})
    bull = ema_f[-1] > ema_s[-1]
    bear = ema_f[-1] < ema_s[-1]''',
        "params": {"fast": (5, 15), "slow": (18, 50)},
    },
    "rsi": {
        "code": '''
    rsi_val = _calc_rsi(closes, {period})
    bull = rsi_val > {bull_thresh}
    bear = rsi_val < {bear_thresh}''',
        "params": {"period": (6, 20), "bull_thresh": (45, 60), "bear_thresh": (40, 55)},
    },
    "macd": {
        "code": '''
    macd_hist = _calc_macd(closes, {fast}, {slow}, {signal})
    bull = macd_hist > 0
    bear = macd_hist < 0''',
        "params": {"fast": (8, 16), "slow": (18, 30), "signal": (7, 12)},
    },
    "bb_compression": {
        "code": '''
    bb_pct = _bb_width_percentile(closes, {period})
    bull = bb_pct < {threshold}
    bear = bb_pct < {threshold}''',
        "params": {"period": (5, 20), "threshold": (50, 95)},
    },
    "funding_carry": {
        "code": '''
    avg_fr = np.mean(funding_rates[-{lookback}:]) if len(funding_rates) >= {lookback} else 0
    bull = avg_fr < -{carry_thresh}
    bear = avg_fr > {carry_thresh}''',
        "params": {"lookback": (8, 48), "carry_thresh": (0.0001, 0.001)},
    },
    "volatility_breakout": {
        "code": '''
    vol = np.std(np.diff(np.log(closes[-{lookback}:])))
    vol_sma = np.mean([np.std(np.diff(np.log(closes[i-{lookback}:i]))) for i in range(-{sma_period}, 0)])
    bull = vol > vol_sma * {mult} and closes[-1] > closes[-2]
    bear = vol > vol_sma * {mult} and closes[-1] < closes[-2]''',
        "params": {"lookback": (12, 48), "sma_period": (5, 20), "mult": (1.1, 2.0)},
    },
    "mean_reversion": {
        "code": '''
    sma = np.mean(closes[-{period}:])
    zscore = (closes[-1] - sma) / max(np.std(closes[-{period}:]), 1e-10)
    bull = zscore < -{entry_z}
    bear = zscore > {entry_z}''',
        "params": {"period": (12, 72), "entry_z": (1.5, 3.0)},
    },
    "stochastic": {
        "code": '''
    h = np.max(highs[-{period}:])
    l = np.min(lows[-{period}:])
    k = (closes[-1] - l) / max(h - l, 1e-10) * 100
    bull = k < {oversold}
    bear = k > {overbought}''',
        "params": {"period": (8, 21), "oversold": (15, 30), "overbought": (70, 85)},
    },
    "atr_breakout": {
        "code": '''
    atr = _calc_atr(highs, lows, closes, {period})
    bull = closes[-1] > closes[-2] + atr * {mult}
    bear = closes[-1] < closes[-2] - atr * {mult}''',
        "params": {"period": (10, 30), "mult": (1.0, 3.0)},
    },
    # ── SMC / Order Flow / Candlestick Signals ──
    "order_block": {
        "code": '''
    from core.smc import get_active_order_blocks
    active_obs = get_active_order_blocks(opens, highs, lows, closes, volumes, closes[-1], max_age={max_age})
    bull_obs = [ob for ob in active_obs if ob.type == "bullish" and closes[-1] >= ob.low and closes[-1] <= ob.high * 1.01]
    bear_obs = [ob for ob in active_obs if ob.type == "bearish" and closes[-1] <= ob.high and closes[-1] >= ob.low * 0.99]
    bull = len(bull_obs) > 0
    bear = len(bear_obs) > 0''',
        "params": {"max_age": (50, 200)},
    },
    "fvg": {
        "code": '''
    from core.smc import detect_fvg
    fvgs = detect_fvg(highs, lows, closes, min_gap_pct={min_gap})
    recent_bull = [g for g in fvgs if g.type == "bullish" and not g.filled and g.index > len(closes) - {recency}]
    recent_bear = [g for g in fvgs if g.type == "bearish" and not g.filled and g.index > len(closes) - {recency}]
    bull = len(recent_bull) > 0 and closes[-1] > closes[-2]
    bear = len(recent_bear) > 0 and closes[-1] < closes[-2]''',
        "params": {"min_gap": (0.001, 0.005), "recency": (10, 50)},
    },
    "structure_break": {
        "code": '''
    from core.smc import detect_structure_breaks
    breaks = detect_structure_breaks(highs, lows, closes, swing_lookback={swing_lb})
    recent = [b for b in breaks if b.index > len(closes) - 10]
    bull = any(b.type in ("bos_bullish", "choch_bullish") for b in recent)
    bear = any(b.type in ("bos_bearish", "choch_bearish") for b in recent)''',
        "params": {"swing_lb": (3, 10)},
    },
    "liquidity_sweep": {
        "code": '''
    from core.smc import detect_liquidity_sweeps
    sweeps = detect_liquidity_sweeps(highs, lows, closes, swing_lookback={swing_lb})
    recent = [s for s in sweeps if s.index > len(closes) - 5 and s.reversal]
    bull = any(s.type == "sell_side" for s in recent)
    bear = any(s.type == "buy_side" for s in recent)''',
        "params": {"swing_lb": (3, 8)},
    },
    "premium_discount": {
        "code": '''
    from core.smc import compute_premium_discount
    pd_zone = compute_premium_discount(highs, lows, closes, lookback={lookback})
    bull = pd_zone["zone"] in ("discount", "slight_discount")
    bear = pd_zone["zone"] in ("premium", "slight_premium")''',
        "params": {"lookback": (24, 100)},
    },
    "cvd_divergence": {
        "code": '''
    from core.orderflow import cumulative_volume_delta, cvd_divergence
    cvd = cumulative_volume_delta(opens, highs, lows, closes, volumes)
    div = cvd_divergence(closes, cvd, lookback={lookback})
    bull = div in ("confirmed_bull", "bullish_div")
    bear = div in ("confirmed_bear", "bearish_div")''',
        "params": {"lookback": (10, 40)},
    },
    "vwap": {
        "code": '''
    from core.orderflow import compute_vwap, vwap_signal
    vwap_arr = compute_vwap(highs, lows, closes, volumes)
    sig = vwap_signal(closes, vwap_arr)
    bull = sig in ("bullish_cross", "above_vwap")
    bear = sig in ("bearish_cross", "below_vwap")''',
        "params": {},
    },
    "absorption": {
        "code": '''
    from core.orderflow import detect_absorption
    absorptions = detect_absorption(opens, highs, lows, closes, volumes, vol_threshold={vol_thresh})
    recent = [a for a in absorptions if a["index"] > len(closes) - 5]
    bull = any(a["type"] == "bullish_absorption" for a in recent)
    bear = any(a["type"] == "bearish_absorption" for a in recent)''',
        "params": {"vol_thresh": (1.5, 3.0)},
    },
    "exhaustion": {
        "code": '''
    from core.orderflow import detect_exhaustion
    exhs = detect_exhaustion(opens, highs, lows, closes, volumes, vol_threshold={vol_thresh})
    recent = [e for e in exhs if e["index"] > len(closes) - 5]
    bull = any(e["type"] == "selling_exhaustion" for e in recent)
    bear = any(e["type"] == "buying_exhaustion" for e in recent)''',
        "params": {"vol_thresh": (2.0, 4.0)},
    },
    "candlestick_pattern": {
        "code": '''
    from core.candlestick_patterns import compute_candle_signal
    candle_sig = compute_candle_signal(opens, highs, lows, closes, lookback={lookback})
    bull = candle_sig["bias"] == "bullish" and candle_sig["strength"] > {min_strength}
    bear = candle_sig["bias"] == "bearish" and candle_sig["strength"] > {min_strength}''',
        "params": {"lookback": (1, 5), "min_strength": (0.2, 0.6)},
    },
    "engulfing": {
        "code": '''
    from core.candlestick_patterns import detect_engulfing
    if len(opens) >= 2:
        p = detect_engulfing(opens[-2], highs[-2], lows[-2], closes[-2], opens[-1], highs[-1], lows[-1], closes[-1])
        bull = p is not None and p.direction == "bullish"
        bear = p is not None and p.direction == "bearish"
    else:
        bull = False
        bear = False''',
        "params": {},
    },
    "volume_profile": {
        "code": '''
    from core.orderflow import compute_volume_profile
    vp = compute_volume_profile(highs[-{lookback}:], lows[-{lookback}:], closes[-{lookback}:], volumes[-{lookback}:])
    bull = closes[-1] < vp.value_area_low
    bear = closes[-1] > vp.value_area_high''',
        "params": {"lookback": (50, 200)},
    },
    # ── Technical Analysis Signals ──
    "ichimoku": {
        "code": '''
    from core.technical_analysis import ichimoku
    ichi = ichimoku(highs, lows, closes, tenkan={tenkan}, kijun={kijun})
    bull = "bull" in ichi["signal"]
    bear = "bear" in ichi["signal"]''',
        "params": {"tenkan": (7, 12), "kijun": (20, 30)},
    },
    "supertrend": {
        "code": '''
    from core.technical_analysis import supertrend
    st, st_dir = supertrend(highs, lows, closes, period={period}, multiplier={mult})
    bull = st_dir[-1] == 1
    bear = st_dir[-1] == -1''',
        "params": {"period": (7, 14), "mult": (2.0, 4.0)},
    },
    "parabolic_sar": {
        "code": '''
    from core.technical_analysis import parabolic_sar
    sar, sar_dir = parabolic_sar(highs, lows, af_start={af_start}, af_max={af_max})
    bull = sar_dir[-1] == 1
    bear = sar_dir[-1] == -1''',
        "params": {"af_start": (0.01, 0.03), "af_max": (0.15, 0.25)},
    },
    "williams_r": {
        "code": '''
    from core.technical_analysis import williams_r
    wr = williams_r(highs, lows, closes, period={period})
    wr_val = wr[-1] if not np.isnan(wr[-1]) else -50
    bull = wr_val < -{oversold}
    bear = wr_val > -{overbought}''',
        "params": {"period": (10, 21), "oversold": (75, 85), "overbought": (15, 25)},
    },
    "cci_signal": {
        "code": '''
    from core.technical_analysis import cci
    cci_arr = cci(highs, lows, closes, period={period})
    c = cci_arr[-1] if not np.isnan(cci_arr[-1]) else 0
    bull = c > {bull_thresh}
    bear = c < -{bear_thresh}''',
        "params": {"period": (14, 30), "bull_thresh": (80, 150), "bear_thresh": (80, 150)},
    },
    "mfi_signal": {
        "code": '''
    from core.technical_analysis import mfi
    mfi_arr = mfi(highs, lows, closes, volumes, period={period})
    m = mfi_arr[-1] if not np.isnan(mfi_arr[-1]) else 50
    bull = m < {oversold}
    bear = m > {overbought}''',
        "params": {"period": (10, 20), "oversold": (15, 30), "overbought": (70, 85)},
    },
    "obv_trend": {
        "code": '''
    from core.technical_analysis import obv
    obv_arr = obv(closes, volumes)
    obv_sma = np.mean(obv_arr[-{period}:])
    bull = obv_arr[-1] > obv_sma
    bear = obv_arr[-1] < obv_sma''',
        "params": {"period": (10, 30)},
    },
    "hull_ma": {
        "code": '''
    from core.technical_analysis import hull_moving_average
    hma = hull_moving_average(closes, period={period})
    bull = not np.isnan(hma[-1]) and closes[-1] > hma[-1] and hma[-1] > hma[-2]
    bear = not np.isnan(hma[-1]) and closes[-1] < hma[-1] and hma[-1] < hma[-2]''',
        "params": {"period": (12, 24)},
    },
    "donchian_breakout": {
        "code": '''
    from core.technical_analysis import donchian_channels
    dc_upper, dc_mid, dc_lower = donchian_channels(highs, lows, period={period})
    bull = not np.isnan(dc_upper[-2]) and closes[-1] > dc_upper[-2]
    bear = not np.isnan(dc_lower[-2]) and closes[-1] < dc_lower[-2]''',
        "params": {"period": (15, 40)},
    },
    "keltner_squeeze": {
        "code": '''
    from core.technical_analysis import keltner_channels
    kc_upper, kc_mid, kc_lower = keltner_channels(highs, lows, closes, ema_period={period}, multiplier={mult})
    bull = closes[-1] > kc_upper[-1]
    bear = closes[-1] < kc_lower[-1]''',
        "params": {"period": (15, 25), "mult": (1.0, 2.5)},
    },
    "chaikin_mf": {
        "code": '''
    from core.technical_analysis import chaikin_money_flow
    cmf = chaikin_money_flow(highs, lows, closes, volumes, period={period})
    cmf_val = cmf[-1] if not np.isnan(cmf[-1]) else 0
    bull = cmf_val > {threshold}
    bear = cmf_val < -{threshold}''',
        "params": {"period": (15, 25), "threshold": (0.03, 0.10)},
    },
    "trix_signal": {
        "code": '''
    from core.technical_analysis import trix
    trix_arr = trix(closes, period={period})
    t = trix_arr[-1] if not np.isnan(trix_arr[-1]) else 0
    bull = t > 0
    bear = t < 0''',
        "params": {"period": (10, 20)},
    },
}

# ─────────────────────────────────────────────────────────────────
# Risk Management Templates
# ─────────────────────────────────────────────────────────────────

STOP_TEMPLATES = {
    "atr_trailing": {
        "params": {"atr_period": (14, 30), "atr_mult": (3.0, 7.0)},
    },
    "pct_trailing": {
        "params": {"trail_pct": (0.02, 0.08)},
    },
    "vol_adaptive": {
        "params": {"vol_lookback": (24, 72), "base_stop": (0.02, 0.05), "vol_scale": (0.5, 2.0)},
    },
}

EXIT_TEMPLATES = {
    "rsi_exit": {
        "params": {"overbought": (65, 80), "oversold": (20, 35)},
    },
    "time_exit": {
        "params": {"max_bars": (24, 168)},
    },
    "profit_target": {
        "params": {"target_pct": (0.03, 0.15)},
    },
}


def _random_param(param_range: tuple) -> float:
    """Generate random parameter within range."""
    lo, hi = param_range
    if isinstance(lo, int) and isinstance(hi, int):
        return random.randint(lo, hi)
    return round(random.uniform(lo, hi), 6)


def _mutate_param(value, param_range: tuple, intensity: float = 0.2) -> float:
    """Mutate a parameter by ±intensity%."""
    lo, hi = param_range
    delta = (hi - lo) * intensity * random.uniform(-1, 1)
    new_val = value + delta
    new_val = max(lo, min(hi, new_val))
    if isinstance(lo, int) and isinstance(hi, int):
        return int(round(new_val))
    return round(new_val, 6)


def _strategy_hash(signals: list, params: dict) -> str:
    """Deterministic hash for deduplication."""
    key = str(sorted(signals)) + str(sorted(params.items()))
    return hashlib.sha256(key.encode()).hexdigest()[:12]


# ─────────────────────────────────────────────────────────────────
# Generation Methods
# ─────────────────────────────────────────────────────────────────

def generate_random(
    n_signals: int = None,
    symbols: list = None,
    min_votes: int = None,
) -> StrategyRecord:
    """Generate a completely random strategy from the signal library."""
    if symbols is None:
        symbols = ["BTC", "ETH", "SOL"]
    if n_signals is None:
        n_signals = random.randint(3, 6)
    
    # Pick random signals
    chosen_signals = random.sample(list(SIGNAL_LIBRARY.keys()), min(n_signals, len(SIGNAL_LIBRARY)))
    
    if min_votes is None:
        min_votes = max(2, n_signals - 2)
    
    # Generate random parameters for each signal
    params = {}
    for sig_name in chosen_signals:
        sig_def = SIGNAL_LIBRARY[sig_name]
        for p_name, p_range in sig_def["params"].items():
            params[f"{sig_name}__{p_name}"] = _random_param(p_range)
    
    # Risk params
    stop_type = random.choice(list(STOP_TEMPLATES.keys()))
    for p_name, p_range in STOP_TEMPLATES[stop_type]["params"].items():
        params[f"stop__{p_name}"] = _random_param(p_range)
    
    exit_type = random.choice(list(EXIT_TEMPLATES.keys()))
    for p_name, p_range in EXIT_TEMPLATES[exit_type]["params"].items():
        params[f"exit__{p_name}"] = _random_param(p_range)
    
    # Position sizing
    params["position_pct"] = round(random.uniform(0.04, 0.15), 3)
    params["cooldown_bars"] = random.randint(1, 5)
    params["min_votes"] = min_votes
    params["stop_type"] = stop_type
    params["exit_type"] = exit_type
    
    # Symbol weights
    for sym in symbols:
        params[f"weight_{sym}"] = round(random.uniform(0.15, 0.50), 3)
    # Normalize weights
    total_w = sum(params[f"weight_{sym}"] for sym in symbols)
    for sym in symbols:
        params[f"weight_{sym}"] = round(params[f"weight_{sym}"] / total_w, 3)
    
    sid = _strategy_hash(chosen_signals, params)
    name = f"gen_{'-'.join(s[:3] for s in chosen_signals)}_{sid[:6]}"
    
    # Generate the actual strategy code
    code = _build_strategy_code(chosen_signals, params, symbols)
    
    rec = StrategyRecord(
        id=sid,
        name=name,
        code=code,
        params=params,
        signals_used=chosen_signals,
    )
    return rec


def mutate_strategy(parent: StrategyRecord, intensity: float = 0.2) -> StrategyRecord:
    """
    Mutate an existing strategy's parameters.
    Small intensity = fine-tuning, large = exploration.
    """
    new_params = copy.deepcopy(parent.params)
    
    # Mutate 30-70% of numeric params
    numeric_keys = [k for k, v in new_params.items() if isinstance(v, (int, float))]
    n_mutate = max(1, int(len(numeric_keys) * random.uniform(0.3, 0.7)))
    keys_to_mutate = random.sample(numeric_keys, n_mutate)
    
    for key in keys_to_mutate:
        # Find the param range
        parts = key.split("__")
        if len(parts) == 2:
            group, param = parts
            if group in SIGNAL_LIBRARY and param in SIGNAL_LIBRARY[group]["params"]:
                p_range = SIGNAL_LIBRARY[group]["params"][param]
            elif group == "stop" and new_params.get("stop_type") in STOP_TEMPLATES:
                p_range = STOP_TEMPLATES[new_params["stop_type"]]["params"].get(param, (0, 1))
            elif group == "exit" and new_params.get("exit_type") in EXIT_TEMPLATES:
                p_range = EXIT_TEMPLATES[new_params["exit_type"]]["params"].get(param, (0, 1))
            else:
                continue
            new_params[key] = _mutate_param(new_params[key], p_range, intensity)
        elif key == "position_pct":
            new_params[key] = _mutate_param(new_params[key], (0.04, 0.15), intensity)
        elif key == "cooldown_bars":
            new_params[key] = _mutate_param(new_params[key], (1, 8), intensity)
        elif key == "min_votes":
            n_sig = len(parent.signals_used)
            new_params[key] = _mutate_param(new_params[key], (2, n_sig), intensity)
    
    signals = parent.signals_used[:]
    code = _build_strategy_code(signals, new_params, ["BTC", "ETH", "SOL"])
    
    sid = _strategy_hash(signals, new_params)
    rec = StrategyRecord(
        id=sid,
        name=f"mut_{parent.name[:20]}_{sid[:6]}",
        code=code,
        params=new_params,
        signals_used=signals,
        parent_id=parent.id,
        generation=parent.generation + 1,
    )
    return rec


def crossover(parent_a: StrategyRecord, parent_b: StrategyRecord) -> StrategyRecord:
    """
    Combine signals from two parents.
    Takes signals from A and risk management from B (or vice versa).
    """
    # Combine unique signals
    all_signals = list(set(parent_a.signals_used + parent_b.signals_used))
    n_pick = random.randint(3, min(6, len(all_signals)))
    chosen = random.sample(all_signals, n_pick)
    
    # Blend params: take each param from whichever parent has it,
    # with 50/50 chance when both have it
    new_params = {}
    for key in set(list(parent_a.params.keys()) + list(parent_b.params.keys())):
        a_val = parent_a.params.get(key)
        b_val = parent_b.params.get(key)
        if a_val is not None and b_val is not None:
            if isinstance(a_val, (int, float)) and isinstance(b_val, (int, float)):
                # Blend
                alpha = random.uniform(0.3, 0.7)
                blended = a_val * alpha + b_val * (1 - alpha)
                new_params[key] = int(round(blended)) if isinstance(a_val, int) else round(blended, 6)
            else:
                new_params[key] = random.choice([a_val, b_val])
        elif a_val is not None:
            new_params[key] = a_val
        else:
            new_params[key] = b_val
    
    new_params["min_votes"] = max(2, n_pick - 2)
    
    code = _build_strategy_code(chosen, new_params, ["BTC", "ETH", "SOL"])
    sid = _strategy_hash(chosen, new_params)
    
    rec = StrategyRecord(
        id=sid,
        name=f"cross_{sid[:6]}",
        code=code,
        params=new_params,
        signals_used=chosen,
        parent_id=f"{parent_a.id}+{parent_b.id}",
        generation=max(parent_a.generation, parent_b.generation) + 1,
    )
    return rec


# ─────────────────────────────────────────────────────────────────
# Code Builder — assembles strategy.py from components
# ─────────────────────────────────────────────────────────────────

def _build_strategy_code(signals: list, params: dict, symbols: list) -> str:
    """Build a complete strategy.py from signal names and parameters."""
    
    code = '''"""
Auto-generated strategy: {name}
Signals: {signals}
"""

import numpy as np
from prepare import Signal, PortfolioState, BarData

ACTIVE_SYMBOLS = {symbols}
MIN_VOTES = {min_votes}
POSITION_PCT = {position_pct}
COOLDOWN_BARS = {cooldown_bars}
SYMBOL_WEIGHTS = {weights}


def _ema(values, span):
    alpha = 2.0 / (span + 1)
    result = np.empty(len(values), dtype=float)
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
    return result


def _calc_rsi(closes, period):
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes[-(period+1):])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    rs = avg_gain / max(avg_loss, 1e-10)
    return 100 - 100 / (1 + rs)


def _calc_macd(closes, fast, slow, signal):
    if len(closes) < slow + signal + 5:
        return 0.0
    f = _ema(closes[-(slow + signal + 5):], fast)
    s = _ema(closes[-(slow + signal + 5):], slow)
    macd_line = f - s
    sig_line = _ema(macd_line, signal)
    return macd_line[-1] - sig_line[-1]


def _bb_width_percentile(closes, period):
    if len(closes) < period * 3:
        return 50.0
    widths = []
    for i in range(period * 2, len(closes)):
        window = closes[i-period:i]
        sma = np.mean(window)
        std = np.std(window)
        widths.append((2 * std) / sma if sma > 0 else 0)
    if len(widths) < 2:
        return 50.0
    return 100 * np.sum(np.array(widths) <= widths[-1]) / len(widths)


def _calc_atr(highs, lows, closes, lookback):
    if len(highs) < lookback + 1:
        return None
    h = highs[-lookback:]
    l = lows[-lookback:]
    c = closes[-(lookback+1):-1]
    tr = np.maximum(h - l, np.maximum(np.abs(h - c), np.abs(l - c)))
    return np.mean(tr)


class Strategy:
    def __init__(self):
        self.entry_prices = {{}}
        self.peak_prices = {{}}
        self.atr_at_entry = {{}}
        self.exit_bar = {{}}
        self.bar_count = 0
        self.peak_equity = 100000.0

    def on_bar(self, bar_data, portfolio):
        signals = []
        equity = portfolio.equity if portfolio.equity > 0 else portfolio.cash
        self.bar_count += 1
        self.peak_equity = max(self.peak_equity, equity)

        for symbol in ACTIVE_SYMBOLS:
            if symbol not in bar_data:
                continue
            bd = bar_data[symbol]
            if len(bd.history) < 80:
                continue

            closes = bd.history["close"].values.astype(float)
            highs = bd.history["high"].values.astype(float)
            lows = bd.history["low"].values.astype(float)
            funding_rates = bd.history["funding_rate"].values.astype(float)
            mid = bd.close

            # ── Compute signals ──
            bull_votes = 0
            bear_votes = 0
{signal_blocks}

            # ── Entry / exit logic ──
            current_pos = portfolio.positions.get(symbol, 0.0)
            in_cooldown = (self.bar_count - self.exit_bar.get(symbol, -999)) < COOLDOWN_BARS
            weight = SYMBOL_WEIGHTS.get(symbol, 0.33)
            size = equity * POSITION_PCT * weight

            target = current_pos
            bullish = bull_votes >= MIN_VOTES
            bearish = bear_votes >= MIN_VOTES

            if current_pos == 0:
                if not in_cooldown:
                    if bullish:
                        target = size
                    elif bearish:
                        target = -size
            else:
{exit_block}
                # Signal flip
                if current_pos > 0 and bearish and not in_cooldown:
                    target = -size
                elif current_pos < 0 and bullish and not in_cooldown:
                    target = size

            if abs(target - current_pos) > 1.0:
                signals.append(Signal(symbol=symbol, target_position=target))
                if target != 0 and current_pos == 0:
                    self.entry_prices[symbol] = mid
                    self.peak_prices[symbol] = mid
                    self.atr_at_entry[symbol] = _calc_atr(highs, lows, closes, 24) or mid * 0.02
                elif target == 0:
                    self.entry_prices.pop(symbol, None)
                    self.peak_prices.pop(symbol, None)
                    self.exit_bar[symbol] = self.bar_count
                elif (target > 0 and current_pos < 0) or (target < 0 and current_pos > 0):
                    self.entry_prices[symbol] = mid
                    self.peak_prices[symbol] = mid
                    self.atr_at_entry[symbol] = _calc_atr(highs, lows, closes, 24) or mid * 0.02

        return signals
'''
    
    # Build signal blocks
    signal_blocks = []
    for sig_name in signals:
        if sig_name not in SIGNAL_LIBRARY:
            continue
        sig_def = SIGNAL_LIBRARY[sig_name]
        # Format the code template with actual params
        sig_code = sig_def["code"]
        for p_name in sig_def["params"]:
            key = f"{sig_name}__{p_name}"
            val = params.get(key, _random_param(sig_def["params"][p_name]))
            sig_code = sig_code.replace(f"{{{p_name}}}", str(val))
        
        block = f"\n            # Signal: {sig_name}"
        block += sig_code
        block += f"\n            if bull: bull_votes += 1"
        block += f"\n            if bear: bear_votes += 1"
        signal_blocks.append(block)
    
    signal_block_str = "\n".join(signal_blocks) if signal_blocks else "            pass"
    
    # Build exit block based on stop type
    stop_type = params.get("stop_type", "atr_trailing")
    if stop_type == "atr_trailing":
        atr_mult = params.get("stop__atr_mult", 5.5)
        atr_period = params.get("stop__atr_period", 24)
        exit_block = f"""                # ATR trailing stop
                atr = _calc_atr(highs, lows, closes, {atr_period})
                if atr is None:
                    atr = self.atr_at_entry.get(symbol, mid * 0.02)
                if symbol not in self.peak_prices:
                    self.peak_prices[symbol] = mid
                if current_pos > 0:
                    self.peak_prices[symbol] = max(self.peak_prices[symbol], mid)
                    if mid < self.peak_prices[symbol] - {atr_mult} * atr:
                        target = 0.0
                else:
                    self.peak_prices[symbol] = min(self.peak_prices[symbol], mid)
                    if mid > self.peak_prices[symbol] + {atr_mult} * atr:
                        target = 0.0"""
    elif stop_type == "pct_trailing":
        trail_pct = params.get("stop__trail_pct", 0.04)
        exit_block = f"""                # Percentage trailing stop
                if symbol not in self.peak_prices:
                    self.peak_prices[symbol] = mid
                if current_pos > 0:
                    self.peak_prices[symbol] = max(self.peak_prices[symbol], mid)
                    if mid < self.peak_prices[symbol] * (1 - {trail_pct}):
                        target = 0.0
                else:
                    self.peak_prices[symbol] = min(self.peak_prices[symbol], mid)
                    if mid > self.peak_prices[symbol] * (1 + {trail_pct}):
                        target = 0.0"""
    else:  # vol_adaptive
        vol_lb = params.get("stop__vol_lookback", 36)
        base_stop = params.get("stop__base_stop", 0.03)
        vol_scale = params.get("stop__vol_scale", 1.0)
        exit_block = f"""                # Vol-adaptive trailing stop
                vol = np.std(np.diff(np.log(closes[-{vol_lb}:]))) if len(closes) > {vol_lb} else 0.02
                stop_dist = {base_stop} * (1 + vol * {vol_scale})
                if symbol not in self.peak_prices:
                    self.peak_prices[symbol] = mid
                if current_pos > 0:
                    self.peak_prices[symbol] = max(self.peak_prices[symbol], mid)
                    if mid < self.peak_prices[symbol] * (1 - stop_dist):
                        target = 0.0
                else:
                    self.peak_prices[symbol] = min(self.peak_prices[symbol], mid)
                    if mid > self.peak_prices[symbol] * (1 + stop_dist):
                        target = 0.0"""
    
    # Build weights dict
    weights = {sym: params.get(f"weight_{sym}", round(1/len(symbols), 3)) for sym in symbols}
    
    formatted = code.format(
        name=f"{'_'.join(signals)}",
        signals=signals,
        symbols=symbols,
        min_votes=params.get("min_votes", 4),
        position_pct=params.get("position_pct", 0.08),
        cooldown_bars=params.get("cooldown_bars", 2),
        weights=weights,
        signal_blocks=signal_block_str,
        exit_block=exit_block,
    )
    
    return formatted


# ─────────────────────────────────────────────────────────────────
# Batch Generation
# ─────────────────────────────────────────────────────────────────

def generate_batch(
    n_random: int = 5,
    n_mutations: int = 5,
    n_crossovers: int = 3,
    mutation_intensity: float = 0.2,
) -> List[StrategyRecord]:
    """
    Generate a batch of new strategies.
    Uses existing top performers as parents for mutations and crossovers.
    """
    strategies = []
    
    # Random new strategies
    for _ in range(n_random):
        strategies.append(generate_random())
    
    # Mutations of top performers
    top = load_strategies(status="passed_oos", limit=10)
    if not top:
        top = load_strategies(status="passed_is", limit=10)
    if not top:
        # No history yet — generate more random
        for _ in range(n_mutations):
            strategies.append(generate_random())
    else:
        for _ in range(n_mutations):
            parent_dict = random.choice(top)
            parent = StrategyRecord(
                id=parent_dict["id"],
                name=parent_dict["name"],
                code=parent_dict["code"],
                params=json.loads(parent_dict["params"]) if isinstance(parent_dict["params"], str) else parent_dict["params"],
                signals_used=json.loads(parent_dict["signals_used"]) if isinstance(parent_dict["signals_used"], str) else parent_dict["signals_used"],
                generation=parent_dict.get("generation", 0),
            )
            strategies.append(mutate_strategy(parent, mutation_intensity))
    
    # Crossovers
    if len(top) >= 2:
        for _ in range(n_crossovers):
            a_dict, b_dict = random.sample(top[:10], 2)
            a = StrategyRecord(id=a_dict["id"], name=a_dict["name"], code=a_dict["code"],
                             params=json.loads(a_dict["params"]) if isinstance(a_dict["params"], str) else a_dict["params"],
                             signals_used=json.loads(a_dict["signals_used"]) if isinstance(a_dict["signals_used"], str) else a_dict["signals_used"])
            b = StrategyRecord(id=b_dict["id"], name=b_dict["name"], code=b_dict["code"],
                             params=json.loads(b_dict["params"]) if isinstance(b_dict["params"], str) else b_dict["params"],
                             signals_used=json.loads(b_dict["signals_used"]) if isinstance(b_dict["signals_used"], str) else b_dict["signals_used"])
            strategies.append(crossover(a, b))
    
    return strategies


import json  # needed at module level for load_strategies parsing
