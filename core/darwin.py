"""
DARWIN MODULE — Strategy Evaluation, Ranking, and Pruning

The ruthless culling engine. Tests strategies against in-sample and
out-of-sample data, ranks them, kills the weak, explains the decisions.
"""

import json
import time
import os
import sys
import traceback
import importlib.util
import tempfile
from typing import Optional, Tuple

import numpy as np

from .models import (
    StrategyRecord, StrategyStatus, save_strategy, load_strategies,
    log_event, init_db
)

# ─────────────────────────────────────────────────────────────────
# Thresholds — strategies must clear these or die
# ─────────────────────────────────────────────────────────────────

# In-sample thresholds (lenient — let candidates through)
IS_MIN_SHARPE = 0.5
IS_MIN_TRADES = 20
IS_MAX_DRAWDOWN = 30.0      # percent
IS_MIN_RETURN = -10.0        # percent — allow slightly negative

# Out-of-sample thresholds (strict — only the real ones survive)
OOS_MIN_SHARPE = 1.0
OOS_MIN_TRADES = 15
OOS_MAX_DRAWDOWN = 25.0
OOS_MIN_RETURN = 0.0         # must be profitable
OOS_SHARPE_DECAY_MAX = 0.6   # OOS sharpe must be >= 60% of IS sharpe

# Overfitting detection
OVERFIT_SHARPE_RATIO = 0.5   # if OOS/IS sharpe < 0.5, it's overfit
OVERFIT_RETURN_RATIO = 0.3   # if OOS/IS return < 0.3, it's overfit


# ─────────────────────────────────────────────────────────────────
# Backtest Runner
# ─────────────────────────────────────────────────────────────────

def _run_backtest_for_strategy(strategy_code: str, split: str = "val") -> dict:
    """
    Run a strategy through the existing backtest engine.
    Writes strategy code to a temp file, imports it, runs backtest.
    
    Returns dict with all metrics or {"error": "..."} on failure.
    """
    # We need to import prepare from the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from prepare import load_data, run_backtest, compute_score
    
    # Write strategy to temp file
    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, "strategy.py")
    
    try:
        with open(tmp_path, "w") as f:
            f.write(strategy_code)
        
        # Import the strategy
        spec = importlib.util.spec_from_file_location("temp_strategy", tmp_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        
        strategy_obj = mod.Strategy()
        
        # Determine split
        data = load_data(split)
        if not data:
            return {"error": f"No data for split '{split}'"}
        
        # Run backtest
        result = run_backtest(strategy_obj, data)
        score = compute_score(result)
        
        # Compute Sortino ratio
        eq = np.array(result.equity_curve)
        if len(eq) > 1:
            returns = np.diff(eq) / eq[:-1]
            downside = returns[returns < 0]
            downside_std = np.std(downside) if len(downside) > 0 else 1e-10
            sortino = (np.mean(returns) / downside_std) * np.sqrt(8760) if downside_std > 0 else 0
        else:
            sortino = 0.0
        
        return {
            "score": score,
            "sharpe": result.sharpe,
            "return_pct": result.total_return_pct,
            "max_dd_pct": result.max_drawdown_pct,
            "win_rate": result.win_rate_pct,
            "num_trades": result.num_trades,
            "profit_factor": result.profit_factor,
            "sortino": sortino,
            "annual_turnover": result.annual_turnover,
        }
    
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)}", "traceback": traceback.format_exc()}
    
    finally:
        # Cleanup
        try:
            os.unlink(tmp_path)
            os.rmdir(tmp_dir)
        except:
            pass


# ─────────────────────────────────────────────────────────────────
# Evaluation Pipeline
# ─────────────────────────────────────────────────────────────────

def evaluate_strategy(rec: StrategyRecord) -> StrategyRecord:
    """
    Full evaluation pipeline:
    1. In-sample backtest (training period)
    2. If passes → out-of-sample backtest
    3. Overfit detection
    4. Status update + explanation
    """
    rec.status = StrategyStatus.BACKTESTING.value
    save_strategy(rec)
    log_event(rec.id, "backtest_start", f"Starting IS evaluation")
    
    # ── Step 1: In-sample backtest ──
    is_result = _run_backtest_for_strategy(rec.code, split="train")
    
    if "error" in is_result:
        rec.status = StrategyStatus.ERROR.value
        rec.kill_reason = f"IS backtest error: {is_result['error']}"
        save_strategy(rec)
        log_event(rec.id, "error", rec.kill_reason)
        return rec
    
    # Store IS metrics
    rec.is_sharpe = is_result["sharpe"]
    rec.is_return_pct = is_result["return_pct"]
    rec.is_max_dd_pct = is_result["max_dd_pct"]
    rec.is_win_rate = is_result["win_rate"]
    rec.is_num_trades = is_result["num_trades"]
    rec.is_profit_factor = is_result["profit_factor"]
    rec.is_sortino = is_result["sortino"]
    rec.is_score = is_result["score"]
    
    # ── Step 2: IS threshold check ──
    kill_reasons = []
    
    if rec.is_sharpe < IS_MIN_SHARPE:
        kill_reasons.append(f"IS Sharpe {rec.is_sharpe:.2f} < {IS_MIN_SHARPE}")
    if rec.is_num_trades < IS_MIN_TRADES:
        kill_reasons.append(f"IS trades {rec.is_num_trades} < {IS_MIN_TRADES}")
    if rec.is_max_dd_pct > IS_MAX_DRAWDOWN:
        kill_reasons.append(f"IS drawdown {rec.is_max_dd_pct:.1f}% > {IS_MAX_DRAWDOWN}%")
    if rec.is_return_pct < IS_MIN_RETURN:
        kill_reasons.append(f"IS return {rec.is_return_pct:.1f}% < {IS_MIN_RETURN}%")
    
    if kill_reasons:
        rec.status = StrategyStatus.KILLED.value
        rec.kill_reason = "Failed IS: " + "; ".join(kill_reasons)
        save_strategy(rec)
        log_event(rec.id, "killed_is", rec.kill_reason)
        return rec
    
    rec.status = StrategyStatus.PASSED_IS.value
    save_strategy(rec)
    log_event(rec.id, "passed_is", f"IS score={rec.is_score:.3f} sharpe={rec.is_sharpe:.3f}")
    
    # ── Step 3: Out-of-sample backtest ──
    oos_result = _run_backtest_for_strategy(rec.code, split="val")
    
    if "error" in oos_result:
        rec.status = StrategyStatus.ERROR.value
        rec.kill_reason = f"OOS backtest error: {oos_result['error']}"
        save_strategy(rec)
        log_event(rec.id, "error", rec.kill_reason)
        return rec
    
    # Store OOS metrics
    rec.oos_sharpe = oos_result["sharpe"]
    rec.oos_return_pct = oos_result["return_pct"]
    rec.oos_max_dd_pct = oos_result["max_dd_pct"]
    rec.oos_win_rate = oos_result["win_rate"]
    rec.oos_num_trades = oos_result["num_trades"]
    rec.oos_score = oos_result["score"]
    
    # ── Step 4: OOS threshold check ──
    kill_reasons = []
    
    if rec.oos_sharpe < OOS_MIN_SHARPE:
        kill_reasons.append(f"OOS Sharpe {rec.oos_sharpe:.2f} < {OOS_MIN_SHARPE}")
    if rec.oos_num_trades < OOS_MIN_TRADES:
        kill_reasons.append(f"OOS trades {rec.oos_num_trades} < {OOS_MIN_TRADES}")
    if rec.oos_max_dd_pct > OOS_MAX_DRAWDOWN:
        kill_reasons.append(f"OOS drawdown {rec.oos_max_dd_pct:.1f}% > {OOS_MAX_DRAWDOWN}%")
    if rec.oos_return_pct < OOS_MIN_RETURN:
        kill_reasons.append(f"OOS return {rec.oos_return_pct:.1f}% — not profitable")
    
    # ── Step 5: Overfit detection ──
    if rec.is_sharpe > 0:
        sharpe_ratio = rec.oos_sharpe / rec.is_sharpe
        if sharpe_ratio < OVERFIT_SHARPE_RATIO:
            kill_reasons.append(
                f"Overfit: OOS/IS Sharpe ratio {sharpe_ratio:.2f} < {OVERFIT_SHARPE_RATIO} "
                f"(IS={rec.is_sharpe:.2f}, OOS={rec.oos_sharpe:.2f})"
            )
    
    if rec.is_return_pct > 0:
        return_ratio = rec.oos_return_pct / rec.is_return_pct
        if return_ratio < OVERFIT_RETURN_RATIO:
            kill_reasons.append(
                f"Overfit: OOS/IS return ratio {return_ratio:.2f} < {OVERFIT_RETURN_RATIO}"
            )
    
    if kill_reasons:
        rec.status = StrategyStatus.KILLED.value
        rec.kill_reason = "Failed OOS: " + "; ".join(kill_reasons)
        save_strategy(rec)
        log_event(rec.id, "killed_oos", rec.kill_reason)
        return rec
    
    # ── Survived! ──
    rec.status = StrategyStatus.PASSED_OOS.value
    rec.keep_reason = _generate_keep_reason(rec)
    save_strategy(rec)
    log_event(rec.id, "passed_oos", rec.keep_reason)
    
    return rec


# ─────────────────────────────────────────────────────────────────
# Explanation Generator
# ─────────────────────────────────────────────────────────────────

def _generate_keep_reason(rec: StrategyRecord) -> str:
    """Generate a concise analytical explanation of why a strategy works."""
    reasons = []
    
    # Characterize by signal composition
    signals = rec.signals_used
    if "momentum" in signals and "ema_cross" in signals:
        reasons.append("trend-following with dual confirmation")
    elif "mean_reversion" in signals:
        reasons.append("mean-reversion approach")
    elif "momentum" in signals:
        reasons.append("momentum-driven")
    
    if "funding_carry" in signals:
        reasons.append("with funding carry overlay")
    if "bb_compression" in signals:
        reasons.append("using vol compression for entry timing")
    if "volatility_breakout" in signals:
        reasons.append("capturing volatility expansion")
    
    # Characterize by metrics
    if rec.oos_sharpe > 5:
        reasons.append(f"exceptional risk-adjusted returns (Sharpe {rec.oos_sharpe:.1f})")
    elif rec.oos_sharpe > 2:
        reasons.append(f"strong risk-adjusted returns (Sharpe {rec.oos_sharpe:.1f})")
    
    if rec.oos_max_dd_pct < 3:
        reasons.append(f"very tight risk control ({rec.oos_max_dd_pct:.1f}% max DD)")
    
    if rec.oos_win_rate > 55:
        reasons.append(f"good win rate ({rec.oos_win_rate:.0f}%)")
    
    # Stability check
    if rec.is_sharpe > 0:
        decay = rec.oos_sharpe / rec.is_sharpe
        if decay > 0.8:
            reasons.append("excellent IS→OOS stability")
        elif decay > 0.6:
            reasons.append("acceptable IS→OOS stability")
    
    return "; ".join(reasons) if reasons else "passed all thresholds"


def generate_kill_report(rec: StrategyRecord) -> str:
    """Generate a detailed report on why a strategy was killed."""
    report = f"""
═══════════════════════════════════════════════
STRATEGY REPORT: {rec.name} ({rec.id})
Status: {rec.status.upper()}
Signals: {', '.join(rec.signals_used)}
Generation: {rec.generation}
Parent: {rec.parent_id or 'None (random)'}
═══════════════════════════════════════════════

IN-SAMPLE METRICS:
  Sharpe:       {rec.is_sharpe:>8.3f}
  Return:       {rec.is_return_pct:>8.1f}%
  Max Drawdown: {rec.is_max_dd_pct:>8.1f}%
  Win Rate:     {rec.is_win_rate:>8.1f}%
  Trades:       {rec.is_num_trades:>8d}
  Sortino:      {rec.is_sortino:>8.3f}
  Score:        {rec.is_score:>8.3f}

OUT-OF-SAMPLE METRICS:
  Sharpe:       {rec.oos_sharpe:>8.3f}
  Return:       {rec.oos_return_pct:>8.1f}%
  Max Drawdown: {rec.oos_max_dd_pct:>8.1f}%
  Win Rate:     {rec.oos_win_rate:>8.1f}%
  Trades:       {rec.oos_num_trades:>8d}
  Score:        {rec.oos_score:>8.3f}
"""
    
    if rec.is_sharpe > 0 and rec.oos_sharpe > 0:
        report += f"""
STABILITY:
  Sharpe decay:  {rec.oos_sharpe/rec.is_sharpe:.2f}x (OOS/IS)
"""
    
    if rec.status == StrategyStatus.KILLED.value:
        report += f"""
VERDICT: KILLED ❌
Reason: {rec.kill_reason}
"""
    else:
        report += f"""
VERDICT: KEPT ✅
Reason: {rec.keep_reason}
"""
    
    return report


# ─────────────────────────────────────────────────────────────────
# Ranking
# ─────────────────────────────────────────────────────────────────

def get_leaderboard(top_n: int = 20) -> list:
    """
    Get the top N strategies ranked by OOS score.
    Returns list of dicts with all metrics.
    """
    survivors = load_strategies(status="passed_oos", limit=top_n)
    # Sort by OOS score descending
    survivors.sort(key=lambda x: x.get("oos_score", -999), reverse=True)
    return survivors


def get_statistics() -> dict:
    """Get overall system statistics."""
    import sqlite3
    from .models import DB_PATH
    
    conn = sqlite3.connect(DB_PATH)
    
    total = conn.execute("SELECT COUNT(*) FROM strategies").fetchone()[0]
    by_status = {}
    for row in conn.execute("SELECT status, COUNT(*) FROM strategies GROUP BY status"):
        by_status[row[0]] = row[1]
    
    best = conn.execute(
        "SELECT name, oos_score, oos_sharpe FROM strategies WHERE status='passed_oos' ORDER BY oos_score DESC LIMIT 1"
    ).fetchone()
    
    avg_gen = conn.execute(
        "SELECT AVG(generation) FROM strategies WHERE status='passed_oos'"
    ).fetchone()[0]
    
    conn.close()
    
    return {
        "total_strategies": total,
        "by_status": by_status,
        "best_strategy": {"name": best[0], "score": best[1], "sharpe": best[2]} if best else None,
        "avg_survivor_generation": avg_gen or 0,
        "kill_rate": by_status.get("killed", 0) / max(total, 1) * 100,
    }
