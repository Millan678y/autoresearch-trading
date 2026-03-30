"""
EVALUATOR — Multi-Factor Strategy Scoring with Anti-Overfit Penalties

Replaces naive Sharpe optimization with a comprehensive scoring system.

Metrics:
- Sharpe ratio (risk-adjusted return)
- Sortino ratio (downside-risk adjusted)
- Max drawdown (tail risk)
- Profit factor (gross profit / gross loss)
- Win rate + avg win/loss ratio (expectancy)
- Trade consistency across time periods
- Calmar ratio (return / max drawdown)

Penalties:
- Too few trades (insufficient sample)
- Too many trades (overtrading / curve-fit)
- High parameter count (complexity penalty)
- Sharpe decay IS → OOS (overfit penalty)
- Inconsistent performance across periods
- Excessive drawdown
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class EvaluationResult:
    """Comprehensive strategy evaluation."""
    # Core metrics
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    
    # Trade metrics
    num_trades: int = 0
    win_rate_pct: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    win_loss_ratio: float = 0.0
    expectancy: float = 0.0        # Expected $ per trade
    
    # Consistency
    monthly_returns: list = None
    profitable_months_pct: float = 0.0
    return_std: float = 0.0
    
    # Penalties
    complexity_penalty: float = 0.0
    overfit_penalty: float = 0.0
    trade_count_penalty: float = 0.0
    drawdown_penalty: float = 0.0
    consistency_penalty: float = 0.0
    turnover_penalty: float = 0.0
    
    # Final score
    raw_score: float = 0.0
    total_penalty: float = 0.0
    final_score: float = -999.0
    
    # Metadata
    bars_per_year: float = 8760    # Default hourly


def compute_full_evaluation(
    equity_curve: List[float],
    trade_pnls: List[float] = None,
    trade_log: List[dict] = None,
    initial_capital: float = 10000.0,
    bars_per_year: float = 8760,
    # Penalty parameters
    n_signals: int = 0,
    n_parameters: int = 0,
    is_sharpe: float = 0.0,       # In-sample Sharpe for overfit detection
    annual_turnover: float = 0.0,
    total_volume: float = 0.0,
) -> EvaluationResult:
    """
    Compute comprehensive evaluation with penalties.
    
    This is the REAL scoring function that determines if a strategy lives or dies.
    """
    result = EvaluationResult(bars_per_year=bars_per_year)
    
    eq = np.array(equity_curve) if equity_curve else np.array([initial_capital])
    
    if len(eq) < 10:
        return result
    
    # ── Core Metrics ──
    
    # Returns
    returns = np.diff(eq) / eq[:-1]
    returns = returns[np.isfinite(returns)]
    
    if len(returns) == 0:
        return result
    
    # Sharpe
    if np.std(returns) > 0:
        result.sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(bars_per_year))
    
    # Sortino
    downside = returns[returns < 0]
    if len(downside) > 0 and np.std(downside) > 0:
        result.sortino = float(np.mean(returns) / np.std(downside) * np.sqrt(bars_per_year))
    
    # Returns
    result.total_return_pct = float((eq[-1] - initial_capital) / initial_capital * 100)
    n_years = len(returns) / bars_per_year
    if n_years > 0 and eq[-1] > 0 and initial_capital > 0:
        result.annualized_return_pct = float(
            ((eq[-1] / initial_capital) ** (1 / max(n_years, 0.01)) - 1) * 100
        )
    
    # Max Drawdown
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / np.where(peak > 0, peak, 1)
    result.max_drawdown_pct = float(dd.max() * 100)
    
    # Calmar
    if result.max_drawdown_pct > 0:
        result.calmar = float(result.annualized_return_pct / result.max_drawdown_pct)
    
    # ── Trade Metrics ──
    
    if trade_pnls:
        pnls = np.array(trade_pnls)
        result.num_trades = len(pnls)
        
        winners = pnls[pnls > 0]
        losers = pnls[pnls < 0]
        
        result.win_rate_pct = float(len(winners) / len(pnls) * 100) if len(pnls) > 0 else 0
        result.avg_win = float(np.mean(winners)) if len(winners) > 0 else 0
        result.avg_loss = float(np.mean(losers)) if len(losers) > 0 else 0
        
        if result.avg_loss != 0:
            result.win_loss_ratio = float(abs(result.avg_win / result.avg_loss))
        
        gross_profit = float(np.sum(winners)) if len(winners) > 0 else 0
        gross_loss = float(abs(np.sum(losers))) if len(losers) > 0 else 1e-10
        result.profit_factor = float(gross_profit / gross_loss)
        
        # Expectancy: avg PnL per trade
        result.expectancy = float(np.mean(pnls))
    
    elif trade_log:
        result.num_trades = len(trade_log)
        pnls = [t.get("pnl", 0) for t in trade_log if "pnl" in t]
        if pnls:
            winners = [p for p in pnls if p > 0]
            losers = [p for p in pnls if p < 0]
            result.win_rate_pct = float(len(winners) / len(pnls) * 100)
            result.avg_win = float(np.mean(winners)) if winners else 0
            result.avg_loss = float(np.mean(losers)) if losers else 0
            result.expectancy = float(np.mean(pnls))
    
    # ── Consistency ──
    
    # Split equity curve into monthly chunks and compute per-month returns
    bars_per_month = int(bars_per_year / 12)
    if len(eq) > bars_per_month * 2:
        monthly_returns = []
        for i in range(0, len(eq) - 1, bars_per_month):
            chunk = eq[i:min(i + bars_per_month + 1, len(eq))]
            if len(chunk) > 1 and chunk[0] > 0:
                monthly_returns.append((chunk[-1] - chunk[0]) / chunk[0] * 100)
        
        result.monthly_returns = monthly_returns
        if monthly_returns:
            profitable = sum(1 for r in monthly_returns if r > 0)
            result.profitable_months_pct = float(profitable / len(monthly_returns) * 100)
            result.return_std = float(np.std(monthly_returns))
    
    # ── Penalties ──
    
    # 1. Complexity penalty (more signals/params = more overfit risk)
    if n_signals > 0:
        # Penalty starts at 5 signals, increases quadratically
        excess = max(0, n_signals - 5)
        result.complexity_penalty = float(excess ** 1.5 * 0.1)
    
    if n_parameters > 0:
        # Penalty for too many free parameters
        excess = max(0, n_parameters - 10)
        result.complexity_penalty += float(excess * 0.05)
    
    # 2. Overfit penalty (IS → OOS Sharpe decay)
    if is_sharpe > 0 and result.sharpe > 0:
        decay = result.sharpe / is_sharpe
        if decay < 0.5:
            result.overfit_penalty = float((0.5 - decay) * 5.0)
        elif decay < 0.7:
            result.overfit_penalty = float((0.7 - decay) * 2.0)
    elif is_sharpe > 2 and result.sharpe <= 0:
        # Massive overfit: great IS, terrible OOS
        result.overfit_penalty = 5.0
    
    # 3. Trade count penalty
    if result.num_trades < 30:
        result.trade_count_penalty = float((30 - result.num_trades) / 30 * 3.0)
    elif result.num_trades > 5000 and bars_per_year < 20000:
        # Excessive trading on hourly (not scalping)
        excess = (result.num_trades - 5000) / 5000
        result.trade_count_penalty = float(excess * 1.0)
    
    # 4. Drawdown penalty (progressive)
    if result.max_drawdown_pct > 10:
        result.drawdown_penalty = float((result.max_drawdown_pct - 10) * 0.08)
    if result.max_drawdown_pct > 25:
        result.drawdown_penalty += float((result.max_drawdown_pct - 25) * 0.15)
    
    # 5. Consistency penalty
    if result.monthly_returns and len(result.monthly_returns) >= 3:
        if result.profitable_months_pct < 50:
            result.consistency_penalty = float((50 - result.profitable_months_pct) / 50 * 2.0)
    
    # 6. Turnover penalty
    if annual_turnover > 0 and initial_capital > 0:
        turnover_ratio = annual_turnover / initial_capital
        if turnover_ratio > 500:
            result.turnover_penalty = float((turnover_ratio - 500) * 0.001)
    
    # ── Final Score ──
    
    # Raw score: weighted combination of positive metrics
    result.raw_score = (
        result.sharpe * 0.30 +
        result.sortino * 0.15 +
        result.calmar * 0.10 +
        result.profit_factor * 0.15 +
        (result.win_rate_pct / 100) * 0.10 +
        (result.profitable_months_pct / 100) * 0.10 +
        min(result.win_loss_ratio, 3) / 3 * 0.10
    )
    
    # Total penalty
    result.total_penalty = (
        result.complexity_penalty +
        result.overfit_penalty +
        result.trade_count_penalty +
        result.drawdown_penalty +
        result.consistency_penalty +
        result.turnover_penalty
    )
    
    # Final score
    result.final_score = result.raw_score - result.total_penalty
    
    # Hard kills
    if result.num_trades < 10:
        result.final_score = -999.0
    if result.max_drawdown_pct > 50:
        result.final_score = -999.0
    if eq[-1] < initial_capital * 0.5:
        result.final_score = -999.0
    
    return result


def print_evaluation(result: EvaluationResult, name: str = "Strategy"):
    """Print a detailed evaluation report."""
    print(f"\n{'═' * 60}")
    print(f"📊 EVALUATION: {name}")
    print(f"{'═' * 60}")
    
    print(f"\n  Core Metrics:")
    print(f"    Sharpe:           {result.sharpe:>8.3f}")
    print(f"    Sortino:          {result.sortino:>8.3f}")
    print(f"    Calmar:           {result.calmar:>8.3f}")
    print(f"    Total Return:     {result.total_return_pct:>8.1f}%")
    print(f"    Annual Return:    {result.annualized_return_pct:>8.1f}%")
    print(f"    Max Drawdown:     {result.max_drawdown_pct:>8.1f}%")
    
    print(f"\n  Trade Metrics:")
    print(f"    Trades:           {result.num_trades:>8d}")
    print(f"    Win Rate:         {result.win_rate_pct:>8.1f}%")
    print(f"    Profit Factor:    {result.profit_factor:>8.2f}")
    print(f"    Win/Loss Ratio:   {result.win_loss_ratio:>8.2f}")
    print(f"    Avg Win:         ${result.avg_win:>8.2f}")
    print(f"    Avg Loss:        ${result.avg_loss:>8.2f}")
    print(f"    Expectancy:      ${result.expectancy:>8.2f}")
    
    if result.monthly_returns:
        print(f"\n  Consistency:")
        print(f"    Profitable Months: {result.profitable_months_pct:>6.1f}%")
        print(f"    Monthly Ret Std:   {result.return_std:>6.2f}%")
    
    print(f"\n  Penalties:")
    print(f"    Complexity:       {result.complexity_penalty:>8.3f}")
    print(f"    Overfit:          {result.overfit_penalty:>8.3f}")
    print(f"    Trade Count:      {result.trade_count_penalty:>8.3f}")
    print(f"    Drawdown:         {result.drawdown_penalty:>8.3f}")
    print(f"    Consistency:      {result.consistency_penalty:>8.3f}")
    print(f"    Turnover:         {result.turnover_penalty:>8.3f}")
    print(f"    TOTAL PENALTY:    {result.total_penalty:>8.3f}")
    
    print(f"\n  {'─' * 40}")
    print(f"    Raw Score:        {result.raw_score:>8.3f}")
    print(f"    Final Score:      {result.final_score:>8.3f}")
    
    verdict = "✅ VIABLE" if result.final_score > 0 else "❌ REJECTED"
    print(f"\n  Verdict: {verdict}")
