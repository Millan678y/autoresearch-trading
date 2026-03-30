"""
HEALER MODULE — Self-Healing & Error Correction

Detects and fixes:
1. Runtime errors (syntax, import, attribute errors)
2. Overfitting (IS >> OOS performance)
3. Degenerate strategies (no trades, all-in, stuck positions)
4. Parameter drift (values hitting bounds)
"""

import re
import traceback
import numpy as np
from typing import Optional, Tuple, List

from .models import StrategyRecord, StrategyStatus, save_strategy, log_event


# ─────────────────────────────────────────────────────────────────
# Error Pattern Detection
# ─────────────────────────────────────────────────────────────────

ERROR_PATTERNS = {
    "index_out_of_range": {
        "pattern": r"IndexError|index.*out of range",
        "diagnosis": "Array access with insufficient data",
        "fix": "increase_lookback_guard",
    },
    "division_by_zero": {
        "pattern": r"ZeroDivisionError|division by zero",
        "diagnosis": "Missing zero-check in calculation",
        "fix": "add_epsilon_guards",
    },
    "nan_values": {
        "pattern": r"NaN|nan|invalid value",
        "diagnosis": "NaN propagation from empty windows or log(0)",
        "fix": "add_nan_guards",
    },
    "memory_error": {
        "pattern": r"MemoryError|OOM|out of memory",
        "diagnosis": "Array too large for available RAM",
        "fix": "reduce_lookbacks",
    },
    "timeout": {
        "pattern": r"TIMEOUT|exceeded time budget",
        "diagnosis": "Strategy computation too slow",
        "fix": "simplify_computation",
    },
    "no_trades": {
        "pattern": r"num_trades.*0|<10 trades",
        "diagnosis": "Entry conditions too restrictive",
        "fix": "relax_entry_conditions",
    },
}


def diagnose_error(error_msg: str) -> Optional[dict]:
    """Match an error to a known pattern and suggest a fix."""
    for name, pattern_info in ERROR_PATTERNS.items():
        if re.search(pattern_info["pattern"], error_msg, re.IGNORECASE):
            return {
                "error_type": name,
                "diagnosis": pattern_info["diagnosis"],
                "fix_type": pattern_info["fix"],
            }
    return None


# ─────────────────────────────────────────────────────────────────
# Automatic Fixes
# ─────────────────────────────────────────────────────────────────

def heal_strategy(rec: StrategyRecord, error_msg: str = "") -> Optional[StrategyRecord]:
    """
    Attempt to automatically fix a broken strategy.
    Returns a new StrategyRecord with fixes applied, or None if unfixable.
    """
    diagnosis = diagnose_error(error_msg or rec.kill_reason)
    if not diagnosis:
        return None
    
    fix_type = diagnosis["fix_type"]
    code = rec.code
    params = rec.params.copy()
    fixed = False
    fix_desc = []
    
    if fix_type == "increase_lookback_guard":
        # Add/increase the minimum history check
        old_guard = re.search(r'if len\(bd\.history\) < (\d+)', code)
        if old_guard:
            current = int(old_guard.group(1))
            new_val = max(current, 100)
            code = code.replace(f"len(bd.history) < {current}", f"len(bd.history) < {new_val}")
            fixed = True
            fix_desc.append(f"Increased lookback guard from {current} to {new_val}")
        
        # Also reduce any lookback params that might be too large
        for key in list(params.keys()):
            if "lookback" in key and isinstance(params[key], (int, float)):
                if params[key] > 100:
                    params[key] = min(params[key], 72)
                    fixed = True
                    fix_desc.append(f"Reduced {key} to {params[key]}")
    
    elif fix_type == "add_epsilon_guards":
        # Add epsilon to divisions
        code = re.sub(
            r'/ (np\.std\([^)]+\))',
            r'/ max(\1, 1e-10)',
            code
        )
        fixed = True
        fix_desc.append("Added epsilon guards to standard deviation divisions")
    
    elif fix_type == "add_nan_guards":
        # Add NaN checks after calculations
        code = code.replace(
            "bull_votes += 1",
            "bull_votes += 1 if not (isinstance(bull, float) and np.isnan(bull)) else 0"
        )
        fixed = True
        fix_desc.append("Added NaN guards to vote counting")
    
    elif fix_type == "reduce_lookbacks":
        # Halve all lookback periods
        for key in list(params.keys()):
            if "lookback" in key or "period" in key or "slow" in key:
                if isinstance(params[key], (int, float)):
                    params[key] = max(4, params[key] // 2)
                    fixed = True
        fix_desc.append("Halved all lookback periods to reduce memory")
    
    elif fix_type == "simplify_computation":
        # Remove expensive computations (BB percentile is often the culprit)
        if "bb_compression" in rec.signals_used:
            new_signals = [s for s in rec.signals_used if s != "bb_compression"]
            if len(new_signals) >= 3:
                rec.signals_used = new_signals
                params["min_votes"] = max(2, params.get("min_votes", 4) - 1)
                fixed = True
                fix_desc.append("Removed BB compression (too slow)")
    
    elif fix_type == "relax_entry_conditions":
        # Lower min_votes
        current_votes = params.get("min_votes", 4)
        if current_votes > 2:
            params["min_votes"] = current_votes - 1
            fixed = True
            fix_desc.append(f"Reduced min_votes from {current_votes} to {current_votes - 1}")
        
        # Widen threshold
        for key in list(params.keys()):
            if "threshold" in key and isinstance(params[key], (int, float)):
                params[key] = params[key] * 0.7
                fixed = True
                fix_desc.append(f"Relaxed {key} by 30%")
    
    if not fixed:
        return None
    
    # Build healed strategy
    from .genesis import _build_strategy_code, _strategy_hash
    
    new_code = _build_strategy_code(rec.signals_used, params, ["BTC", "ETH", "SOL"])
    sid = _strategy_hash(rec.signals_used, params)
    
    healed = StrategyRecord(
        id=sid,
        name=f"healed_{rec.name[:20]}_{sid[:6]}",
        code=new_code,
        params=params,
        signals_used=rec.signals_used,
        parent_id=rec.id,
        generation=rec.generation + 1,
    )
    
    log_event(rec.id, "healed", f"Applied fixes: {'; '.join(fix_desc)}")
    log_event(healed.id, "created", f"Healed from {rec.id}: {'; '.join(fix_desc)}")
    
    return healed


# ─────────────────────────────────────────────────────────────────
# Overfit Detection & Correction
# ─────────────────────────────────────────────────────────────────

def detect_overfit(rec: StrategyRecord) -> Tuple[bool, str]:
    """
    Detect if a strategy is overfit based on IS vs OOS metrics.
    Returns (is_overfit, explanation).
    """
    if rec.is_sharpe <= 0 or rec.oos_sharpe <= 0:
        return False, "Insufficient data for overfit detection"
    
    sharpe_decay = rec.oos_sharpe / rec.is_sharpe
    
    issues = []
    
    # Sharpe decay
    if sharpe_decay < 0.3:
        issues.append(f"Severe Sharpe decay: {sharpe_decay:.2f}x (IS={rec.is_sharpe:.2f} → OOS={rec.oos_sharpe:.2f})")
    elif sharpe_decay < 0.5:
        issues.append(f"Moderate Sharpe decay: {sharpe_decay:.2f}x")
    
    # Return decay
    if rec.is_return_pct > 0 and rec.oos_return_pct > 0:
        return_decay = rec.oos_return_pct / rec.is_return_pct
        if return_decay < 0.2:
            issues.append(f"Return collapsed: IS={rec.is_return_pct:.1f}% → OOS={rec.oos_return_pct:.1f}%")
    
    # Drawdown expansion
    if rec.oos_max_dd_pct > rec.is_max_dd_pct * 2:
        issues.append(f"Drawdown doubled: IS={rec.is_max_dd_pct:.1f}% → OOS={rec.oos_max_dd_pct:.1f}%")
    
    # Trade count collapse
    if rec.is_num_trades > 0 and rec.oos_num_trades / rec.is_num_trades < 0.3:
        issues.append(f"Trade count collapsed: IS={rec.is_num_trades} → OOS={rec.oos_num_trades}")
    
    if issues:
        return True, "; ".join(issues)
    return False, "No overfit detected"


def fix_overfit(rec: StrategyRecord) -> Optional[StrategyRecord]:
    """
    Attempt to fix an overfit strategy by:
    1. Reducing complexity (fewer signals)
    2. Widening parameters (less curve-fit)
    3. Increasing regularization (wider stops, smaller positions)
    """
    params = rec.params.copy()
    fix_desc = []
    
    # Reduce position size (less exposed = less overfit damage)
    if params.get("position_pct", 0.08) > 0.05:
        params["position_pct"] = max(0.04, params["position_pct"] * 0.7)
        fix_desc.append(f"Reduced position size to {params['position_pct']:.3f}")
    
    # Widen stops (overfit strategies often have tight stops tuned to IS noise)
    for key in params:
        if "atr_mult" in key:
            params[key] = min(params[key] * 1.3, 8.0)
            fix_desc.append(f"Widened stop to {params[key]:.1f}x ATR")
        if "trail_pct" in key:
            params[key] = min(params[key] * 1.3, 0.10)
            fix_desc.append(f"Widened trail to {params[key]:.3f}")
    
    # Increase cooldown (reduce overtrading)
    params["cooldown_bars"] = min(params.get("cooldown_bars", 2) + 1, 6)
    fix_desc.append(f"Increased cooldown to {params['cooldown_bars']}")
    
    # If too many signals, drop the weakest
    if len(rec.signals_used) > 4:
        # Drop a random signal that isn't momentum or ema_cross (core signals)
        droppable = [s for s in rec.signals_used if s not in ("momentum", "ema_cross", "rsi")]
        if droppable:
            drop = droppable[0]
            new_signals = [s for s in rec.signals_used if s != drop]
            fix_desc.append(f"Dropped signal: {drop}")
        else:
            new_signals = rec.signals_used
    else:
        new_signals = rec.signals_used
    
    params["min_votes"] = max(2, len(new_signals) - 2)
    
    from .genesis import _build_strategy_code, _strategy_hash
    
    new_code = _build_strategy_code(new_signals, params, ["BTC", "ETH", "SOL"])
    sid = _strategy_hash(new_signals, params)
    
    fixed = StrategyRecord(
        id=sid,
        name=f"deoverfit_{rec.name[:15]}_{sid[:6]}",
        code=new_code,
        params=params,
        signals_used=new_signals,
        parent_id=rec.id,
        generation=rec.generation + 1,
    )
    
    log_event(rec.id, "deoverfit_attempt", "; ".join(fix_desc))
    
    return fixed
