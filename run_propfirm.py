#!/usr/bin/env python3
"""
PROP FIRM CHALLENGE MODE — Swing trading strategies optimized to pass funding challenges.

Two-step prop firm challenge:
  Step 1: $1,000 balance → Hit $50 profit, max $50/day loss, max $100 total loss, min 2 trading days
  Step 2: $1,000 balance → Hit $150 profit, max $100/day loss, max $150 total loss, min 4 trading days

Only strategies that pass BOTH steps get exported to MQL5.

Usage:
    python run_propfirm.py                          # Default
    python run_propfirm.py --batch-size 20          # More strategies per gen
    python run_propfirm.py --symbols BTCUSDT XAUUSD # Specific pairs
    python run_propfirm.py --evaluate-only          # Evaluate existing top strategies
"""

import os
import sys
import time
import json
import signal
import argparse
import hashlib
import random

import numpy as np

from core.binance_data import BinanceDataLoader
from core.scalp_engine import ScalpBacktester, ScalpResult
from core.propfirm import (
    PropFirmSimulator, STEP_1, STEP_2,
    print_challenge_result, print_two_step_result
)
from core.models import (
    StrategyRecord, init_db, save_strategy, load_strategies, log_event
)


# Import scalp strategy generator
from run_scalper import generate_scalp_strategy, SCALP_SIGNALS


class PropFirmOrchestrator:
    """
    Autonomous prop firm challenge optimizer.
    
    Generates strategies, backtests them, then runs the equity curve
    through the prop firm simulator with exact challenge rules.
    Only strategies that pass both steps survive.
    """
    
    def __init__(self, symbols, interval, batch_size, max_generations):
        self.symbols = symbols
        self.interval = interval
        self.batch_size = batch_size
        self.max_generations = max_generations
        
        # Use $1,000 balance for prop firm
        self.backtester = ScalpBacktester(
            symbols=symbols, interval=interval,
            initial_capital=1000.0, max_leverage=10
        )
        
        # Prop firm simulator
        # 5m = 288 bars/day for crypto (24h), ~156 for gold (13h session)
        bars_per_day = 288 if interval == "5m" else 24 if interval == "1h" else 288
        self.prop_sim = PropFirmSimulator(bars_per_day=bars_per_day)
        
        self.running = True
        self.best_score = -999.0
        self.generation = 0
        self.total_tested = 0
        self.total_step1_passed = 0
        self.total_step2_passed = 0
        self.total_funded = 0
        
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)
    
    def _shutdown(self, signum, frame):
        print("\n⚡ Shutting down...")
        self.running = False
    
    def run(self):
        init_db()
        
        print("=" * 60)
        print("🏛️  PROP FIRM CHALLENGE OPTIMIZER")
        print("=" * 60)
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Timeframe: {self.interval}")
        print(f"Account: $1,000")
        print()
        print(f"Step 1: +$50 target | $50/day max loss | $100 total max | 2+ trading days")
        print(f"Step 2: +$150 target | $100/day max loss | $150 total max | 4+ trading days")
        print("=" * 60)
        
        print("\n📥 Downloading Binance data...")
        self.backtester.loader.download_all()
        
        print("\n🚀 Starting autonomous loop...\n")
        
        while self.running and self.generation < self.max_generations:
            self.generation += 1
            self._run_generation()
        
        self._final_report()
    
    def _run_generation(self):
        t_start = time.time()
        
        print(f"\n{'─' * 50}")
        print(f"🔬 GENERATION {self.generation}")
        print(f"{'─' * 50}")
        
        for i in range(self.batch_size):
            if not self.running:
                break
            
            strat = generate_scalp_strategy()
            self.total_tested += 1
            
            print(f"\n  [{i+1}/{self.batch_size}] {strat['name']}")
            
            # Run full backtest on training data
            result = self.backtester.run(strat["code"], split="train")
            
            if result.score <= 0 or result.num_trades < 30:
                print(f"    ❌ Backtest failed: score={result.score:.2f} trades={result.num_trades}")
                continue
            
            print(f"    Backtest: Sharpe={result.sharpe:.2f} ret={result.total_return_pct:.1f}% "
                  f"DD={result.max_drawdown_pct:.1f}% trades={result.num_trades}")
            
            # ── Run Prop Firm Challenge ──
            challenge = self.prop_sim.run_two_step_challenge(
                result.equity_curve, result.trade_log
            )
            
            step1 = challenge["step1"]
            step2 = challenge["step2"]
            
            # Step 1
            if not step1.passed:
                print(f"    ❌ Step 1 FAILED: {step1.fail_reason}")
                continue
            
            self.total_step1_passed += 1
            print(f"    ✅ Step 1 PASSED: +${step1.total_pnl:.2f} in {step1.trading_days} days "
                  f"(worst day: ${step1.worst_day_pnl:.2f})")
            
            # Step 2
            if step2 is None or not step2.passed:
                reason = step2.fail_reason if step2 else "No step 2 data"
                print(f"    ❌ Step 2 FAILED: {reason}")
                continue
            
            self.total_step2_passed += 1
            self.total_funded += 1
            
            print(f"    ✅ Step 2 PASSED: +${step2.total_pnl:.2f} in {step2.trading_days} days")
            print(f"    🏆 FUNDED! {challenge['summary']}")
            
            # Now validate on OOS data
            oos_result = self.backtester.run(strat["code"], split="val")
            
            if oos_result.score > 0:
                print(f"    OOS: Sharpe={oos_result.sharpe:.2f} ret={oos_result.total_return_pct:.1f}%")
                
                # Run prop challenge on OOS too
                oos_challenge = self.prop_sim.run_two_step_challenge(
                    oos_result.equity_curve, oos_result.trade_log
                )
                
                if oos_challenge["overall_passed"]:
                    print(f"    🏆🏆 FUNDED ON OOS TOO! Robust strategy.")
                else:
                    print(f"    ⚠️  OOS challenge: {oos_challenge['summary']}")
            
            # Compute combined score
            # Prop firm score: heavier weight on consistency and drawdown control
            prop_score = (
                result.sharpe * 0.2 +
                (step1.total_pnl / STEP_1.profit_target) * 2.0 +
                (step2.total_pnl / STEP_2.profit_target) * 3.0 -
                (step1.max_drawdown / STEP_1.max_total_loss) * 1.0 -
                (step2.max_drawdown / STEP_2.max_total_loss) * 1.5
            )
            
            # Save
            rec = StrategyRecord(
                id=strat["id"], name=strat["name"], code=strat["code"],
                params=strat["params"], signals_used=strat["signals"],
                is_sharpe=result.sharpe, is_return_pct=result.total_return_pct,
                is_max_dd_pct=result.max_drawdown_pct, is_win_rate=result.win_rate_pct,
                is_num_trades=result.num_trades, is_score=prop_score,
                oos_sharpe=oos_result.sharpe if oos_result.score > 0 else 0,
                oos_return_pct=oos_result.total_return_pct if oos_result.score > 0 else 0,
                oos_max_dd_pct=oos_result.max_drawdown_pct if oos_result.score > 0 else 100,
                oos_score=oos_result.score if oos_result.score > 0 else -999,
                status="passed_oos",
                keep_reason=challenge["summary"],
            )
            save_strategy(rec)
            log_event(strat["id"], "prop_funded",
                     f"Step1: +${step1.total_pnl:.2f} | Step2: +${step2.total_pnl:.2f}")
            
            if prop_score > self.best_score:
                self.best_score = prop_score
                self._save_funded_strategy(strat, result, challenge)
        
        elapsed = time.time() - t_start
        print(f"\n  Gen {self.generation}: tested={self.batch_size} "
              f"funded={self.total_funded} best={self.best_score:.3f} {elapsed:.0f}s")
    
    def _save_funded_strategy(self, strat, result, challenge):
        """Save funded strategy with full details."""
        os.makedirs("reports/propfirm", exist_ok=True)
        
        with open("reports/propfirm/best_funded_strategy.py", "w") as f:
            f.write(strat["code"])
        
        step1 = challenge["step1"]
        step2 = challenge["step2"]
        
        with open("reports/propfirm/best_funded_result.json", "w") as f:
            json.dump({
                "name": strat["name"],
                "signals": strat["signals"],
                "params": strat["params"],
                "backtest": {
                    "sharpe": result.sharpe,
                    "return_pct": result.total_return_pct,
                    "max_dd_pct": result.max_drawdown_pct,
                    "trades": result.num_trades,
                    "win_rate": result.win_rate_pct,
                },
                "step1": {
                    "passed": step1.passed,
                    "pnl": step1.total_pnl,
                    "trading_days": step1.trading_days,
                    "worst_day": step1.worst_day_pnl,
                    "max_dd": step1.max_drawdown,
                },
                "step2": {
                    "passed": step2.passed if step2 else False,
                    "pnl": step2.total_pnl if step2 else 0,
                    "trading_days": step2.trading_days if step2 else 0,
                    "worst_day": step2.worst_day_pnl if step2 else 0,
                    "max_dd": step2.max_drawdown if step2 else 0,
                },
                "summary": challenge["summary"],
            }, f, indent=2)
        
        # Export to MQL5
        try:
            from core.mql5_converter import convert_to_mql5_from_params
            mql5 = convert_to_mql5_from_params(
                strat["signals"], strat["params"],
                name=f"PropFirm_{strat['name']}", mode="scalp"
            )
            with open("reports/propfirm/best_funded_ea.mq5", "w") as f:
                f.write(mql5)
            print(f"    📄 MQL5 EA saved: reports/propfirm/best_funded_ea.mq5")
        except Exception as e:
            print(f"    ⚠️  MQL5 export failed: {e}")
        
        print(f"    💾 Saved to reports/propfirm/")
    
    def _final_report(self):
        print(f"\n\n{'═' * 60}")
        print("🏛️  PROP FIRM CHALLENGE — FINAL REPORT")
        print(f"{'═' * 60}")
        print(f"Generations:     {self.generation}")
        print(f"Tested:          {self.total_tested}")
        print(f"Step 1 passed:   {self.total_step1_passed} ({self.total_step1_passed/max(self.total_tested,1)*100:.1f}%)")
        print(f"Step 2 passed:   {self.total_step2_passed} ({self.total_step2_passed/max(self.total_tested,1)*100:.1f}%)")
        print(f"FUNDED:          {self.total_funded} ({self.total_funded/max(self.total_tested,1)*100:.1f}%)")
        print(f"Best score:      {self.best_score:.3f}")
        
        if self.total_funded > 0:
            print(f"\n🏆 Funded strategies exported to:")
            print(f"   reports/propfirm/best_funded_strategy.py")
            print(f"   reports/propfirm/best_funded_ea.mq5")
            print(f"   reports/propfirm/best_funded_result.json")
        
        # Export all funded to MQL5
        try:
            from core.mql5_converter import export_top_strategies
            export_top_strategies(output_dir="mql5_experts", top_n=10, mode="scalp")
        except:
            pass


def evaluate_existing():
    """Evaluate existing top strategies against prop firm challenge."""
    init_db()
    top = load_strategies(status="passed_oos", limit=20)
    
    if not top:
        print("No strategies found. Run the engine first.")
        return
    
    sim = PropFirmSimulator(bars_per_day=288)
    bt = ScalpBacktester(symbols=["BTCUSDT"], interval="5m", initial_capital=1000.0)
    
    print(f"\n{'═' * 60}")
    print(f"🏛️  EVALUATING {len(top)} STRATEGIES AGAINST PROP FIRM CHALLENGE")
    print(f"{'═' * 60}")
    
    funded = 0
    for i, s in enumerate(top):
        name = s.get("name", f"strategy_{i}")
        code = s.get("code", "")
        
        if not code:
            continue
        
        result = bt.run(code, split="train")
        if result.score <= 0:
            continue
        
        challenge = sim.run_two_step_challenge(result.equity_curve, result.trade_log)
        
        status = "🏆 FUNDED" if challenge["overall_passed"] else "❌"
        print(f"  {i+1}. {name[:30]} | {status} | {challenge['summary'][:50]}")
        
        if challenge["overall_passed"]:
            funded += 1
    
    print(f"\nFunded: {funded}/{len(top)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prop Firm Challenge Optimizer")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "XAUUSDT"],
                       help="Trading pairs")
    parser.add_argument("--interval", default="5m", help="Candle interval")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-generations", type=int, default=500)
    parser.add_argument("--evaluate-only", action="store_true",
                       help="Evaluate existing strategies against prop rules")
    args = parser.parse_args()
    
    if args.evaluate_only:
        evaluate_existing()
    else:
        engine = PropFirmOrchestrator(
            symbols=args.symbols,
            interval=args.interval,
            batch_size=args.batch_size,
            max_generations=args.max_generations,
        )
        engine.run()
