"""
PROP FIRM CHALLENGE SIMULATOR — Two-Step Evaluation Challenge

Simulates prop firm funding challenges with strict rules:

Step 1:
  - Balance: $1,000
  - Profit target: $50 (5%)
  - Max daily loss: $50
  - Max total loss: $100 (account blown)
  - Minimum trading days: 2
  
Step 2:
  - Balance: $1,000 (reset)
  - Profit target: $150 (15%)
  - Max daily loss: $100
  - Max total loss: $150 (account blown)
  - Minimum trading days: 4

A strategy must pass BOTH steps sequentially to be considered viable.
The engine runs backtests with these exact constraints enforced.
"""

import time
import json
import os
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class PropFirmRules:
    """Rules for a prop firm challenge step."""
    name: str
    balance: float
    profit_target: float       # $ amount to hit
    max_daily_loss: float      # $ max loss per day
    max_total_loss: float      # $ max total loss (blown)
    min_trading_days: int      # Minimum days with at least 1 trade
    max_calendar_days: int = 30  # Max days to complete


@dataclass 
class DayResult:
    """Single trading day result."""
    date: str
    starting_equity: float
    ending_equity: float
    pnl: float
    trades: int
    high_water: float         # Highest equity during day
    low_water: float          # Lowest equity during day
    max_intraday_dd: float    # Worst drawdown during day


@dataclass
class ChallengeResult:
    """Result of a challenge step."""
    step: str
    passed: bool
    fail_reason: str = ""
    
    # Metrics
    final_equity: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    total_trades: int = 0
    trading_days: int = 0
    calendar_days: int = 0
    
    # Risk
    max_daily_loss_hit: float = 0.0
    max_drawdown: float = 0.0
    worst_day_pnl: float = 0.0
    best_day_pnl: float = 0.0
    
    # Per-day breakdown
    daily_results: List[DayResult] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────
# Challenge Definitions
# ─────────────────────────────────────────────────────────────────

STEP_1 = PropFirmRules(
    name="Step 1",
    balance=1000.0,
    profit_target=50.0,        # Hit $1,050
    max_daily_loss=50.0,       # Lose $50 in a day = fail
    max_total_loss=100.0,      # Equity drops to $900 = blown
    min_trading_days=2,
    max_calendar_days=30,
)

STEP_2 = PropFirmRules(
    name="Step 2",
    balance=1000.0,
    profit_target=150.0,       # Hit $1,150
    max_daily_loss=100.0,      # Lose $100 in a day = fail
    max_total_loss=150.0,      # Equity drops to $850 = blown
    min_trading_days=4,
    max_calendar_days=60,
)


# ─────────────────────────────────────────────────────────────────
# Challenge Simulator
# ─────────────────────────────────────────────────────────────────

class PropFirmSimulator:
    """
    Simulates a prop firm challenge using backtest equity curves.
    
    Takes an equity curve (from backtesting) and checks it against
    prop firm rules bar-by-bar. Tracks daily PnL, intraday drawdown,
    and enforces all limits in real-time.
    """
    
    def __init__(self, bars_per_day: int = 288):
        """
        Args:
            bars_per_day: How many bars per trading day
                          288 = 5-minute bars (24h crypto)
                          24 = 1-hour bars
        """
        self.bars_per_day = bars_per_day
    
    def run_challenge(self, equity_curve: List[float],
                      trade_log: List[dict],
                      rules: PropFirmRules) -> ChallengeResult:
        """
        Run a challenge step against an equity curve.
        
        Args:
            equity_curve: List of equity values (one per bar)
            trade_log: List of trade dicts with at least {pnl, duration_bars}
            rules: PropFirmRules for this step
        
        Returns:
            ChallengeResult with pass/fail and detailed breakdown
        """
        result = ChallengeResult(
            step=rules.name,
            passed=False,
        )
        
        if not equity_curve or len(equity_curve) < 2:
            result.fail_reason = "No equity data"
            return result
        
        # Scale equity curve to start at the challenge balance
        scale = rules.balance / equity_curve[0] if equity_curve[0] > 0 else 1.0
        scaled_eq = [e * scale for e in equity_curve]
        
        # Track state
        balance = rules.balance
        peak_equity = balance
        blown = False
        target_hit = False
        fail_reason = ""
        
        # Daily tracking
        daily_results = []
        current_day_start_equity = balance
        current_day_high = balance
        current_day_low = balance
        current_day_trades = 0
        day_count = 0
        trading_days = 0
        
        # Track which bars had trades
        trade_bars = set()
        for t in trade_log:
            if "entry_bar" in t:
                trade_bars.add(t["entry_bar"])
        
        # Process bar by bar
        for i, equity in enumerate(scaled_eq):
            bar_in_day = i % self.bars_per_day
            
            # New day
            if bar_in_day == 0 and i > 0:
                # End previous day
                day_pnl = equity - current_day_start_equity
                max_intraday_dd = current_day_start_equity - current_day_low
                
                day_result = DayResult(
                    date=f"Day {day_count}",
                    starting_equity=current_day_start_equity,
                    ending_equity=equity,
                    pnl=day_pnl,
                    trades=current_day_trades,
                    high_water=current_day_high,
                    low_water=current_day_low,
                    max_intraday_dd=max_intraday_dd,
                )
                daily_results.append(day_result)
                
                if current_day_trades > 0:
                    trading_days += 1
                
                # Check daily loss limit
                if day_pnl < -rules.max_daily_loss:
                    blown = True
                    fail_reason = f"Daily loss ${abs(day_pnl):.2f} exceeded limit ${rules.max_daily_loss:.2f} on Day {day_count}"
                    break
                
                # Start new day
                day_count += 1
                current_day_start_equity = equity
                current_day_high = equity
                current_day_low = equity
                current_day_trades = 0
                
                # Calendar limit
                if day_count > rules.max_calendar_days:
                    fail_reason = f"Exceeded {rules.max_calendar_days} calendar day limit"
                    break
            
            # Update intraday tracking
            current_day_high = max(current_day_high, equity)
            current_day_low = min(current_day_low, equity)
            
            # Check if this bar had a trade
            if i in trade_bars:
                current_day_trades += 1
            
            # Check total loss (account blown)
            total_loss = rules.balance - equity
            if total_loss >= rules.max_total_loss:
                blown = True
                fail_reason = f"Account blown: equity ${equity:.2f}, lost ${total_loss:.2f} (max ${rules.max_total_loss:.2f})"
                break
            
            # Check profit target
            total_profit = equity - rules.balance
            if total_profit >= rules.profit_target and not target_hit:
                target_hit = True
                # Don't break — still need to check min trading days
            
            # Update peak
            peak_equity = max(peak_equity, equity)
        
        # Handle last day
        if not blown and len(scaled_eq) > 0:
            last_equity = scaled_eq[-1]
            day_pnl = last_equity - current_day_start_equity
            
            if current_day_trades > 0:
                trading_days += 1
            
            daily_results.append(DayResult(
                date=f"Day {day_count}",
                starting_equity=current_day_start_equity,
                ending_equity=last_equity,
                pnl=day_pnl,
                trades=current_day_trades,
                high_water=current_day_high,
                low_water=current_day_low,
                max_intraday_dd=current_day_start_equity - current_day_low,
            ))
            day_count += 1
        
        # Determine pass/fail
        if blown:
            result.passed = False
            result.fail_reason = fail_reason
        elif not target_hit:
            result.passed = False
            final_pnl = scaled_eq[-1] - rules.balance if scaled_eq else 0
            result.fail_reason = f"Profit target not reached: ${final_pnl:.2f} / ${rules.profit_target:.2f}"
        elif trading_days < rules.min_trading_days:
            result.passed = False
            result.fail_reason = f"Only {trading_days} trading days (need {rules.min_trading_days})"
        else:
            result.passed = True
        
        # Fill metrics
        result.final_equity = scaled_eq[-1] if scaled_eq else rules.balance
        result.total_pnl = result.final_equity - rules.balance
        result.total_pnl_pct = result.total_pnl / rules.balance * 100
        result.total_trades = sum(d.trades for d in daily_results)
        result.trading_days = trading_days
        result.calendar_days = day_count
        result.daily_results = daily_results
        result.equity_curve = scaled_eq
        
        if daily_results:
            result.worst_day_pnl = min(d.pnl for d in daily_results)
            result.best_day_pnl = max(d.pnl for d in daily_results)
            result.max_daily_loss_hit = abs(min(d.pnl for d in daily_results))
        
        result.max_drawdown = max(0, peak_equity - min(scaled_eq)) if scaled_eq else 0
        
        return result
    
    def run_two_step_challenge(self, equity_curve: List[float],
                                trade_log: List[dict]) -> dict:
        """
        Run the full two-step prop firm challenge.
        
        Returns {
            step1: ChallengeResult,
            step2: ChallengeResult,
            overall_passed: bool,
            summary: str
        }
        """
        # Step 1
        step1 = self.run_challenge(equity_curve, trade_log, STEP_1)
        
        if not step1.passed:
            return {
                "step1": step1,
                "step2": None,
                "overall_passed": False,
                "summary": f"Failed Step 1: {step1.fail_reason}",
            }
        
        # Step 2 uses the same equity curve but with Step 2 rules
        # In reality you'd have separate periods, but for backtesting
        # we split the equity curve in half
        mid = len(equity_curve) // 2
        step2_curve = equity_curve[mid:]
        step2_trades = [t for t in trade_log 
                       if t.get("entry_bar", 0) >= mid or 
                          t.get("index", 0) >= mid]
        
        # Adjust trade bar indices for the new curve
        adjusted_trades = []
        for t in step2_trades:
            adj = dict(t)
            if "entry_bar" in adj:
                adj["entry_bar"] = adj["entry_bar"] - mid
            adjusted_trades.append(adj)
        
        step2 = self.run_challenge(step2_curve, adjusted_trades, STEP_2)
        
        overall = step1.passed and step2.passed
        
        if overall:
            summary = (f"✅ FUNDED! Step 1: +${step1.total_pnl:.2f} in {step1.trading_days} days | "
                      f"Step 2: +${step2.total_pnl:.2f} in {step2.trading_days} days")
        elif step1.passed:
            summary = f"Step 1 passed, Step 2 failed: {step2.fail_reason}"
        else:
            summary = f"Failed Step 1: {step1.fail_reason}"
        
        return {
            "step1": step1,
            "step2": step2,
            "overall_passed": overall,
            "summary": summary,
        }


def print_challenge_result(result: ChallengeResult):
    """Print detailed challenge result."""
    status = "✅ PASSED" if result.passed else "❌ FAILED"
    
    print(f"\n{'═' * 50}")
    print(f"  {result.step}: {status}")
    if not result.passed:
        print(f"  Reason: {result.fail_reason}")
    print(f"{'═' * 50}")
    
    print(f"  Final Equity:    ${result.final_equity:.2f}")
    print(f"  Total PnL:       ${result.total_pnl:+.2f} ({result.total_pnl_pct:+.1f}%)")
    print(f"  Trades:          {result.total_trades}")
    print(f"  Trading Days:    {result.trading_days}")
    print(f"  Calendar Days:   {result.calendar_days}")
    print(f"  Max Drawdown:    ${result.max_drawdown:.2f}")
    print(f"  Worst Day:       ${result.worst_day_pnl:.2f}")
    print(f"  Best Day:        ${result.best_day_pnl:+.2f}")
    
    if result.daily_results:
        print(f"\n  Daily Breakdown:")
        print(f"  {'Day':<8} {'PnL':>10} {'Equity':>10} {'Trades':>8} {'DD':>10}")
        print(f"  {'─' * 48}")
        for d in result.daily_results[:30]:  # First 30 days
            dd_str = f"${d.max_intraday_dd:.2f}" if d.max_intraday_dd > 0 else "-"
            print(f"  {d.date:<8} ${d.pnl:>+9.2f} ${d.ending_equity:>9.2f} {d.trades:>8} {dd_str:>10}")


def print_two_step_result(result: dict):
    """Print full two-step challenge result."""
    print(f"\n{'═' * 60}")
    print(f"  🏛️  PROP FIRM CHALLENGE EVALUATION")
    print(f"{'═' * 60}")
    
    print_challenge_result(result["step1"])
    
    if result["step2"]:
        print_challenge_result(result["step2"])
    
    print(f"\n{'═' * 60}")
    status = "🏆 FUNDED!" if result["overall_passed"] else "❌ NOT FUNDED"
    print(f"  {status}")
    print(f"  {result['summary']}")
    print(f"{'═' * 60}")
