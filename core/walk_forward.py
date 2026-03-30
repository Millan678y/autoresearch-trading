"""
WALK-FORWARD VALIDATION — Rolling & Expanding Window Backtesting

Prevents overfitting by testing strategies on truly unseen data across
multiple time periods. If a strategy only works in one window, it's dead.

Methods:
1. Rolling window: Train on N months, test on next M months, slide forward
2. Expanding window: Train on all data up to T, test on T+1
3. Combinatorial purged cross-validation (advanced)

Each window produces independent metrics. Final score = aggregate across all windows.
A strategy must be consistently profitable across ALL windows to survive.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from .models import log_event


@dataclass
class WindowResult:
    """Result from a single walk-forward window."""
    window_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_bars: int = 0
    test_bars: int = 0
    
    # Test metrics (what matters)
    sharpe: float = 0.0
    sortino: float = 0.0
    total_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    num_trades: int = 0
    win_rate_pct: float = 0.0
    profit_factor: float = 0.0
    
    # Train metrics (for comparison / overfit detection)
    train_sharpe: float = 0.0
    train_return_pct: float = 0.0


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward validation result."""
    windows: List[WindowResult] = field(default_factory=list)
    
    # Aggregate metrics
    avg_sharpe: float = 0.0
    min_sharpe: float = 0.0
    max_sharpe: float = 0.0
    sharpe_std: float = 0.0
    
    avg_return: float = 0.0
    avg_max_dd: float = 0.0
    avg_win_rate: float = 0.0
    avg_profit_factor: float = 0.0
    
    # Consistency
    profitable_windows_pct: float = 0.0
    sharpe_positive_pct: float = 0.0
    consistency_score: float = 0.0
    
    # Overfit metrics
    avg_sharpe_decay: float = 0.0  # train vs test sharpe decay
    
    # Final verdict
    wf_score: float = -999.0
    passed: bool = False


# ─────────────────────────────────────────────────────────────────
# Window Generation
# ─────────────────────────────────────────────────────────────────

def generate_rolling_windows(
    start_date: str = "2024-01-01",
    end_date: str = "2025-06-30",
    train_months: int = 4,
    test_months: int = 2,
    step_months: int = 2,
) -> List[dict]:
    """
    Generate rolling train/test windows.
    
    Example with train=4mo, test=2mo, step=2mo:
        Window 1: Train Jan-Apr 2024, Test May-Jun 2024
        Window 2: Train Mar-Jun 2024, Test Jul-Aug 2024
        Window 3: Train May-Aug 2024, Test Sep-Oct 2024
        ...
    """
    windows = []
    current = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    window_id = 0
    
    while True:
        train_start = current
        train_end = train_start + pd.DateOffset(months=train_months)
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=test_months)
        
        if test_end > end:
            break
        
        windows.append({
            "id": window_id,
            "train_start": train_start.strftime("%Y-%m-%d"),
            "train_end": train_end.strftime("%Y-%m-%d"),
            "test_start": test_start.strftime("%Y-%m-%d"),
            "test_end": test_end.strftime("%Y-%m-%d"),
        })
        
        window_id += 1
        current += pd.DateOffset(months=step_months)
    
    return windows


def generate_expanding_windows(
    start_date: str = "2024-01-01",
    end_date: str = "2025-06-30",
    min_train_months: int = 3,
    test_months: int = 2,
    step_months: int = 2,
) -> List[dict]:
    """
    Generate expanding train windows (train grows, test slides).
    
    Example:
        Window 1: Train Jan-Mar 2024, Test Apr-May 2024
        Window 2: Train Jan-May 2024, Test Jun-Jul 2024
        Window 3: Train Jan-Jul 2024, Test Aug-Sep 2024
        ...
    """
    windows = []
    train_start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    window_id = 0
    
    current_train_end = train_start + pd.DateOffset(months=min_train_months)
    
    while True:
        test_start = current_train_end
        test_end = test_start + pd.DateOffset(months=test_months)
        
        if test_end > end:
            break
        
        windows.append({
            "id": window_id,
            "train_start": train_start.strftime("%Y-%m-%d"),
            "train_end": current_train_end.strftime("%Y-%m-%d"),
            "test_start": test_start.strftime("%Y-%m-%d"),
            "test_end": test_end.strftime("%Y-%m-%d"),
        })
        
        window_id += 1
        current_train_end += pd.DateOffset(months=step_months)
    
    return windows


# ─────────────────────────────────────────────────────────────────
# Walk-Forward Executor
# ─────────────────────────────────────────────────────────────────

def run_walk_forward(
    strategy_code: str,
    data: Dict[str, pd.DataFrame],
    windows: List[dict],
    backtest_fn,  # callable(strategy_code, data_slice) -> result dict
    verbose: bool = True,
) -> WalkForwardResult:
    """
    Run walk-forward validation across multiple windows.
    
    Args:
        strategy_code: Strategy source code
        data: {symbol: DataFrame} with 'timestamp' column
        windows: List of window dicts from generate_rolling/expanding_windows
        backtest_fn: Function that takes (strategy_code, data_dict) and returns
                     dict with keys: sharpe, total_return_pct, max_drawdown_pct,
                     num_trades, win_rate_pct, profit_factor, sortino
    
    Returns:
        WalkForwardResult with per-window and aggregate metrics
    """
    wf_result = WalkForwardResult()
    
    if verbose:
        print(f"  Walk-forward: {len(windows)} windows")
    
    for window in windows:
        wid = window["id"]
        train_start_ms = int(pd.Timestamp(window["train_start"], tz="UTC").timestamp() * 1000)
        train_end_ms = int(pd.Timestamp(window["train_end"], tz="UTC").timestamp() * 1000)
        test_start_ms = int(pd.Timestamp(window["test_start"], tz="UTC").timestamp() * 1000)
        test_end_ms = int(pd.Timestamp(window["test_end"], tz="UTC").timestamp() * 1000)
        
        # Slice data for train and test
        train_data = {}
        test_data = {}
        
        for symbol, df in data.items():
            train_mask = (df["timestamp"] >= train_start_ms) & (df["timestamp"] < train_end_ms)
            test_mask = (df["timestamp"] >= test_start_ms) & (df["timestamp"] < test_end_ms)
            
            train_df = df[train_mask].reset_index(drop=True)
            test_df = df[test_mask].reset_index(drop=True)
            
            if len(train_df) > 50:
                train_data[symbol] = train_df
            if len(test_df) > 20:
                test_data[symbol] = test_df
        
        if not train_data or not test_data:
            continue
        
        # Run backtest on train
        try:
            train_result = backtest_fn(strategy_code, train_data)
        except Exception:
            train_result = {"sharpe": 0, "total_return_pct": 0}
        
        # Run backtest on test
        try:
            test_result = backtest_fn(strategy_code, test_data)
        except Exception:
            test_result = {"sharpe": 0, "total_return_pct": 0, "max_drawdown_pct": 100,
                          "num_trades": 0, "win_rate_pct": 0, "profit_factor": 0, "sortino": 0}
        
        wr = WindowResult(
            window_id=wid,
            train_start=window["train_start"],
            train_end=window["train_end"],
            test_start=window["test_start"],
            test_end=window["test_end"],
            train_bars=sum(len(df) for df in train_data.values()),
            test_bars=sum(len(df) for df in test_data.values()),
            sharpe=test_result.get("sharpe", 0),
            sortino=test_result.get("sortino", 0),
            total_return_pct=test_result.get("total_return_pct", 0),
            max_drawdown_pct=test_result.get("max_drawdown_pct", 100),
            num_trades=test_result.get("num_trades", 0),
            win_rate_pct=test_result.get("win_rate_pct", 0),
            profit_factor=test_result.get("profit_factor", 0),
            train_sharpe=train_result.get("sharpe", 0),
            train_return_pct=train_result.get("total_return_pct", 0),
        )
        
        wf_result.windows.append(wr)
        
        if verbose:
            decay = wr.sharpe / wr.train_sharpe if wr.train_sharpe > 0 else 0
            status = "✅" if wr.sharpe > 0 and wr.total_return_pct > 0 else "❌"
            print(f"    W{wid}: {wr.test_start}→{wr.test_end} | "
                  f"Sharpe={wr.sharpe:.2f} (train={wr.train_sharpe:.2f}, "
                  f"decay={decay:.2f}) | Ret={wr.total_return_pct:.1f}% | "
                  f"DD={wr.max_drawdown_pct:.1f}% {status}")
    
    # Compute aggregates
    _compute_aggregates(wf_result)
    
    return wf_result


def _compute_aggregates(wf: WalkForwardResult):
    """Compute aggregate metrics across all windows."""
    if not wf.windows:
        wf.passed = False
        return
    
    sharpes = [w.sharpe for w in wf.windows]
    returns = [w.total_return_pct for w in wf.windows]
    dds = [w.max_drawdown_pct for w in wf.windows]
    win_rates = [w.win_rate_pct for w in wf.windows]
    pfs = [w.profit_factor for w in wf.windows]
    
    wf.avg_sharpe = float(np.mean(sharpes))
    wf.min_sharpe = float(np.min(sharpes))
    wf.max_sharpe = float(np.max(sharpes))
    wf.sharpe_std = float(np.std(sharpes))
    wf.avg_return = float(np.mean(returns))
    wf.avg_max_dd = float(np.mean(dds))
    wf.avg_win_rate = float(np.mean(win_rates))
    wf.avg_profit_factor = float(np.mean(pfs))
    
    # Consistency
    profitable = sum(1 for r in returns if r > 0)
    positive_sharpe = sum(1 for s in sharpes if s > 0)
    n = len(wf.windows)
    
    wf.profitable_windows_pct = profitable / n * 100
    wf.sharpe_positive_pct = positive_sharpe / n * 100
    
    # Consistency score: combination of win frequency and Sharpe stability
    if wf.sharpe_std > 0:
        sharpe_cv = wf.avg_sharpe / wf.sharpe_std  # Coefficient of variation (inverted)
    else:
        sharpe_cv = 0
    
    wf.consistency_score = float(
        (wf.profitable_windows_pct / 100) * 0.4 +
        min(sharpe_cv, 3) / 3 * 0.3 +
        (wf.sharpe_positive_pct / 100) * 0.3
    )
    
    # Overfit detection: average train vs test sharpe decay
    decays = []
    for w in wf.windows:
        if w.train_sharpe > 0:
            decays.append(w.sharpe / w.train_sharpe)
    wf.avg_sharpe_decay = float(np.mean(decays)) if decays else 0
    
    # Walk-forward score
    # Penalizes: inconsistency, negative windows, high drawdowns, overfit
    wf.wf_score = (
        wf.avg_sharpe * 0.3 +
        wf.consistency_score * 5.0 +
        min(wf.avg_sharpe_decay, 1.0) * 2.0 -
        max(0, wf.avg_max_dd - 10) * 0.1 -
        max(0, 50 - wf.profitable_windows_pct) * 0.05
    )
    
    # Pass criteria
    wf.passed = (
        wf.avg_sharpe > 0.5 and
        wf.profitable_windows_pct >= 60 and
        wf.avg_max_dd < 30 and
        wf.min_sharpe > -2 and
        wf.avg_sharpe_decay > 0.3  # No more than 70% decay
    )


# ─────────────────────────────────────────────────────────────────
# Monte Carlo Simulation
# ─────────────────────────────────────────────────────────────────

def monte_carlo_simulation(
    trade_pnls: List[float],
    n_simulations: int = 1000,
    n_trades: int = None,
    initial_capital: float = 10000.0,
) -> dict:
    """
    Monte Carlo simulation on trade P&L distribution.
    
    Randomly resamples trades to estimate:
    - Probability of profit
    - Expected drawdown distribution
    - Confidence intervals for returns
    - Worst-case scenarios
    
    This is the gold standard for testing strategy robustness.
    If Monte Carlo shows >30% chance of ruin, the strategy is garbage.
    """
    if not trade_pnls or len(trade_pnls) < 10:
        return {"error": "Need at least 10 trades for Monte Carlo"}
    
    pnls = np.array(trade_pnls)
    if n_trades is None:
        n_trades = len(pnls)
    
    final_equities = []
    max_drawdowns = []
    max_consecutive_losses_list = []
    
    for _ in range(n_simulations):
        # Resample trades with replacement
        sampled = np.random.choice(pnls, size=n_trades, replace=True)
        
        # Build equity curve
        equity = initial_capital
        peak = equity
        max_dd = 0
        consec_loss = 0
        max_consec = 0
        
        for pnl in sampled:
            equity += pnl
            peak = max(peak, equity)
            dd = (peak - equity) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
            
            if pnl < 0:
                consec_loss += 1
                max_consec = max(max_consec, consec_loss)
            else:
                consec_loss = 0
        
        final_equities.append(equity)
        max_drawdowns.append(max_dd * 100)
        max_consecutive_losses_list.append(max_consec)
    
    finals = np.array(final_equities)
    dds = np.array(max_drawdowns)
    
    return {
        "n_simulations": n_simulations,
        "n_trades": n_trades,
        
        # Return distribution
        "mean_return_pct": float((np.mean(finals) - initial_capital) / initial_capital * 100),
        "median_return_pct": float((np.median(finals) - initial_capital) / initial_capital * 100),
        "return_p5": float((np.percentile(finals, 5) - initial_capital) / initial_capital * 100),
        "return_p25": float((np.percentile(finals, 25) - initial_capital) / initial_capital * 100),
        "return_p75": float((np.percentile(finals, 75) - initial_capital) / initial_capital * 100),
        "return_p95": float((np.percentile(finals, 95) - initial_capital) / initial_capital * 100),
        
        # Risk metrics
        "prob_profit": float(np.mean(finals > initial_capital) * 100),
        "prob_loss_gt_20pct": float(np.mean(finals < initial_capital * 0.8) * 100),
        "prob_loss_gt_50pct": float(np.mean(finals < initial_capital * 0.5) * 100),
        
        # Drawdown distribution
        "mean_max_dd": float(np.mean(dds)),
        "median_max_dd": float(np.median(dds)),
        "dd_p95": float(np.percentile(dds, 95)),
        "dd_p99": float(np.percentile(dds, 99)),
        
        # Consecutive losses
        "mean_max_consec_loss": float(np.mean(max_consecutive_losses_list)),
        "p95_max_consec_loss": float(np.percentile(max_consecutive_losses_list, 95)),
        
        # Verdict
        "robust": bool(
            np.mean(finals > initial_capital) > 0.65 and
            np.percentile(dds, 95) < 40 and
            np.mean(finals < initial_capital * 0.5) < 5
        ),
    }


# ─────────────────────────────────────────────────────────────────
# Combined Validation Pipeline
# ─────────────────────────────────────────────────────────────────

def full_validation(
    strategy_code: str,
    data: Dict[str, pd.DataFrame],
    backtest_fn,
    trade_pnls: List[float] = None,
    verbose: bool = True,
) -> dict:
    """
    Full validation pipeline:
    1. Rolling walk-forward (4-month train, 2-month test)
    2. Expanding walk-forward
    3. Monte Carlo (if trade PnLs provided)
    4. Combined robustness score
    """
    results = {}
    
    if verbose:
        print("\n  📐 Rolling Walk-Forward Validation")
    
    rolling_windows = generate_rolling_windows(
        train_months=4, test_months=2, step_months=2
    )
    rolling_wf = run_walk_forward(strategy_code, data, rolling_windows, backtest_fn, verbose)
    results["rolling"] = rolling_wf
    
    if verbose:
        print(f"\n  📐 Expanding Walk-Forward Validation")
    
    expanding_windows = generate_expanding_windows(
        min_train_months=3, test_months=2, step_months=2
    )
    expanding_wf = run_walk_forward(strategy_code, data, expanding_windows, backtest_fn, verbose)
    results["expanding"] = expanding_wf
    
    # Monte Carlo
    if trade_pnls and len(trade_pnls) >= 10:
        if verbose:
            print(f"\n  🎲 Monte Carlo Simulation (1000 paths)")
        
        mc = monte_carlo_simulation(trade_pnls)
        results["monte_carlo"] = mc
        
        if verbose:
            print(f"    Prob profit: {mc['prob_profit']:.1f}%")
            print(f"    Mean return: {mc['mean_return_pct']:.1f}%")
            print(f"    5th pctile return: {mc['return_p5']:.1f}%")
            print(f"    95th pctile DD: {mc['dd_p95']:.1f}%")
            print(f"    Robust: {'✅' if mc['robust'] else '❌'}")
    
    # Combined robustness score
    rolling_score = rolling_wf.wf_score if rolling_wf.passed else rolling_wf.wf_score * 0.3
    expanding_score = expanding_wf.wf_score if expanding_wf.passed else expanding_wf.wf_score * 0.3
    mc_bonus = 1.0 if trade_pnls and results.get("monte_carlo", {}).get("robust") else 0.0
    
    robustness = (rolling_score * 0.4 + expanding_score * 0.4 + mc_bonus * 0.2)
    
    results["robustness_score"] = float(robustness)
    results["passed"] = rolling_wf.passed and expanding_wf.passed
    
    if verbose:
        print(f"\n  🎯 Robustness Score: {robustness:.3f}")
        print(f"  Rolling: {'PASS' if rolling_wf.passed else 'FAIL'} (score={rolling_wf.wf_score:.3f})")
        print(f"  Expanding: {'PASS' if expanding_wf.passed else 'FAIL'} (score={expanding_wf.wf_score:.3f})")
        print(f"  Overall: {'✅ ROBUST' if results['passed'] else '❌ NOT ROBUST'}")
    
    return results
