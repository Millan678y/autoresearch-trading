#!/usr/bin/env python3
"""
SCALPING MODE — Autonomous scalping strategy research on 5-minute data.

Downloads data from Binance, generates scalping strategies,
backtests them, and evolves the population.

Usage:
    python run_scalper.py                              # Default settings
    python run_scalper.py --symbols BTCUSDT ETHUSDT    # Specific pairs
    python run_scalper.py --interval 1m                # 1-minute scalping
    python run_scalper.py --batch-size 20              # More strategies per gen
    python run_scalper.py --download-only              # Just download data

Press Ctrl+C to stop gracefully.
"""

import os
import sys
import time
import json
import signal
import argparse
import hashlib
import random
import math
from typing import List

import numpy as np

from core.binance_data import BinanceDataLoader, SCALP_SYMBOLS
from core.scalp_engine import ScalpBacktester, ScalpResult, SCALP_STRATEGY_TEMPLATE
from core.models import (
    StrategyRecord, StrategyStatus, init_db, save_strategy,
    load_strategies, log_event
)
from core.learner import compile_insights, get_generation_hints

# ─────────────────────────────────────────────────────────────────
# Scalp-Specific Signal Library
# ─────────────────────────────────────────────────────────────────

# ── INSTITUTIONAL SIGNALS (v3) — Chart Fanatics ──────────────
# Based on 33 verified trader strategies from chartfanatics.com
# Each signal fires on ~1-5% of bars (events, not states).
SCALP_SIGNALS = {
    # Tier 1: Core institutional concepts
    "liquidity_sweep": {
        "params": {"lookback": (24, 72), "wick_ratio": (0.5, 0.75)},
    },
    "break_retest": {
        "params": {"lookback": (24, 72), "retest_window": (6, 18)},
    },
    "mean_reversion": {
        "params": {"consec_bars": (4, 7), "speed_pct": (0.02, 0.05)},
    },
    "amd_model": {
        "params": {"range_bars": (18, 48), "manip_pct": (0.001, 0.005)},
    },
    "volume_node": {
        "params": {"lookback": (36, 72), "bins": (15, 30)},
    },
    "equilibrium_50": {
        "params": {"lookback": (36, 72), "zone_pct": (0.03, 0.08)},
    },
    # Tier 2: Session + trend
    "session_killzone": {
        "params": {"min_move": (0.001, 0.004)},
    },
    "ema_stack": {
        "params": {"fast": (4, 7), "mid1": (8, 11), "mid2": (12, 15), "slow": (18, 25)},
    },
    "regime_filter": {
        "params": {"period": (15, 25), "std": (1.8, 2.5)},
    },
    "fvg": {
        "params": {"min_gap_pct": (0.001, 0.004)},
    },
    # Tier 3: Classic confirmations
    "engulfing": {
        "params": {},
    },
    "wick_rejection": {
        "params": {"min_ratio": (1.5, 3.0)},
    },
    "vwap_reclaim": {
        "params": {},
    },
    "range_breakout": {
        "params": {"period": (12, 48)},
    },
    "smt_divergence": {
        "params": {"lookback": (8, 20)},
    },
    "macd_cross": {
        "params": {"fast": (8, 14), "slow": (20, 30), "signal": (6, 12)},
    },
    "rsi_reversal": {
        "params": {"period": (6, 14), "oversold": (25, 35), "overbought": (65, 75)},
    },
    "ema_cross": {
        "params": {"fast": (5, 12), "slow": (15, 50)},
    },
}


# ─────────────────────────────────────────────────────────────────
# Strategy Code Generator
# ─────────────────────────────────────────────────────────────────

def _random_param(param_range):
    if isinstance(param_range, list):
        return random.choice(param_range)
    lo, hi = param_range
    if isinstance(lo, int) and isinstance(hi, int):
        return random.randint(lo, hi)
    return round(random.uniform(lo, hi), 6)


ALL_SCALP_SIGNAL_NAMES = list(SCALP_SIGNALS.keys())

# ─────────────────────────────────────────────────────────────────
# MUTATION & CROSSOVER — Evolve from best strategies
# ─────────────────────────────────────────────────────────────────

def mutate_strategy(parent_params: dict, mutation_rate: float = 0.3) -> dict:
    """Mutate a parent strategy's params slightly."""
    params = dict(parent_params)
    signals = list(params.get("signals", []))
    
    # Maybe swap one signal
    if random.random() < mutation_rate and len(signals) >= 2:
        idx = random.randint(0, len(signals) - 1)
        available = [s for s in ALL_SCALP_SIGNAL_NAMES if s not in signals]
        if available:
            signals[idx] = random.choice(available)
            params["signals"] = signals
    
    # Mutate numeric params by ±20%
    for key, val in list(params.items()):
        if key in ("signals", "min_votes"):
            continue
        if isinstance(val, (int, float)) and random.random() < mutation_rate:
            noise = random.uniform(0.8, 1.2)
            new_val = val * noise
            if isinstance(val, int):
                new_val = max(1, int(new_val))
            else:
                new_val = round(new_val, 6)
            params[key] = new_val
    
    # Rebuild signals-specific params for any new signals
    for sig in params.get("signals", []):
        if sig in SCALP_SIGNALS:
            for p_name, p_range in SCALP_SIGNALS[sig]["params"].items():
                full_key = f"{sig}__{p_name}"
                if full_key not in params:
                    params[full_key] = _random_param(p_range)
    
    # Regenerate ID
    param_str = json.dumps(params, sort_keys=True)
    sid = hashlib.sha256(param_str.encode()).hexdigest()[:12]
    signals = params.get("signals", [])
    
    return {
        "id": sid,
        "name": f"mut_{'_'.join(s[:3] for s in signals)}_{sid[:6]}",
        "code": "",
        "params": params,
        "signals": signals,
    }


def crossover_strategies(parent_a: dict, parent_b: dict) -> dict:
    """Cross two parent strategies to create a child."""
    sig_a = parent_a.get("signals", [])
    sig_b = parent_b.get("signals", [])
    
    # Take some signals from each parent
    n = random.randint(2, min(5, len(sig_a) + len(sig_b)))
    combined = list(set(sig_a + sig_b))
    random.shuffle(combined)
    child_signals = combined[:n]
    
    # Mix params — prefer parent_a for shared params
    params = {}
    for sig in child_signals:
        if sig in SCALP_SIGNALS:
            for p_name, p_range in SCALP_SIGNALS[sig]["params"].items():
                full_key = f"{sig}__{p_name}"
                if full_key in parent_a:
                    params[full_key] = parent_a[full_key]
                elif full_key in parent_b:
                    params[full_key] = parent_b[full_key]
                else:
                    params[full_key] = _random_param(p_range)
    
    # Strategy-level params: average of parents
    for key in ("min_votes", "cooldown", "tp_mult", "sl_mult", "size_pct", "max_hold"):
        a_val = parent_a.get(key, parent_b.get(key, 2))
        b_val = parent_b.get(key, parent_a.get(key, 2))
        if isinstance(a_val, int):
            params[key] = (a_val + b_val) // 2
        else:
            params[key] = round((a_val + b_val) / 2, 4)
    
    params["signals"] = child_signals
    params["min_votes"] = min(params.get("min_votes", 2), len(child_signals) - 1)
    params["min_votes"] = max(2, params["min_votes"])
    
    param_str = json.dumps(params, sort_keys=True)
    sid = hashlib.sha256(param_str.encode()).hexdigest()[:12]
    
    return {
        "id": sid,
        "name": f"xov_{'_'.join(s[:3] for s in child_signals)}_{sid[:6]}",
        "code": "",
        "params": params,
        "signals": child_signals,
    }


def generate_scalp_strategy(hints: dict = None) -> dict:
    """Generate a scalping strategy config, guided by past learnings if available."""
    
    # If we have hints from past runs, bias signal selection
    preferred = hints.get("preferred_signals", []) if hints else []
    
    n_signals = random.randint(3, min(6, len(ALL_SCALP_SIGNAL_NAMES)))
    
    if preferred and random.random() < 0.6:
        # 60% chance: include at least 1-2 preferred signals
        n_preferred = min(2, len(preferred), n_signals - 1)
        valid_preferred = [s for s in preferred if s in ALL_SCALP_SIGNAL_NAMES]
        if valid_preferred:
            forced = random.sample(valid_preferred, min(n_preferred, len(valid_preferred)))
            remaining = [s for s in ALL_SCALP_SIGNAL_NAMES if s not in forced]
            chosen = forced + random.sample(remaining, n_signals - len(forced))
        else:
            chosen = random.sample(ALL_SCALP_SIGNAL_NAMES, n_signals)
    else:
        chosen = random.sample(ALL_SCALP_SIGNAL_NAMES, n_signals)
    
    # Generate params
    params = {}
    for sig in chosen:
        for p_name, p_range in SCALP_SIGNALS[sig]["params"].items():
            params[f"{sig}__{p_name}"] = _random_param(p_range)
    
    # Strategy-level params — event signals are rare, so 2 votes is enough
    min_votes = random.randint(2, min(3, n_signals - 1))
    cooldown = random.randint(6, 36)  # Wait 30min-3hr between trades
    tp_mult = round(random.uniform(1.5, 4.0), 2)  # Higher TP for better RR
    sl_mult = round(random.uniform(0.8, 2.0), 2)   # Tighter SL
    size_pct = round(random.uniform(0.05, 0.25), 2)  # Small positions, protect capital
    
    params.update({
        "min_votes": min_votes,
        "cooldown": cooldown,
        "tp_mult": tp_mult,
        "sl_mult": sl_mult,
        "size_pct": size_pct,
    })
    
    max_hold = random.randint(24, 144)
    
    params["signals"] = chosen
    params["max_hold"] = max_hold
    
    # No code generation — strategy is parametric
    strategy_code = ""  # Not used anymore
    
    # Hash the params to get a unique ID (not the empty code string!)
    param_str = json.dumps(params, sort_keys=True)
    sid = hashlib.sha256(param_str.encode()).hexdigest()[:12]
    
    return {
        "id": sid,
        "name": f"scalp_{'_'.join(s[:3] for s in chosen)}_{sid[:6]}",
        "code": strategy_code,
        "params": params,
        "signals": chosen,
    }


# ─────────────────────────────────────────────────────────────────
# Main Scalping Loop
# ─────────────────────────────────────────────────────────────────

class ScalpOrchestrator:
    """Autonomous scalping strategy research engine."""
    
    def __init__(self, symbols, interval, batch_size, max_generations):
        self.symbols = symbols
        self.interval = interval
        self.batch_size = batch_size
        self.max_generations = max_generations
        self.backtester = ScalpBacktester(symbols=symbols, interval=interval)
        self.running = True
        self.best_score = -999.0
        self.generation = 0
        self.total_tested = 0
        self.total_survived = 0
        
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)
    
    def _shutdown(self, signum, frame):
        print("\n⚡ Shutting down gracefully...")
        self.running = False
    
    def run(self):
        init_db()
        
        # Download data first
        print("=" * 60)
        print("🔪 SCALPING STRATEGY RESEARCH ENGINE")
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Timeframe: {self.interval}")
        print(f"Batch size: {self.batch_size}")
        print("=" * 60)
        
        print("\n📥 Downloading data from Binance...")
        self.backtester.loader.download_all()
        
        print("\n🚀 Starting autonomous loop...\n")
        
        self.hints = {}
        self.elite_pool = []  # Top strategies for mutation/crossover
        self.all_tested = []  # Track all tested for learning
        
        while self.running and self.generation < self.max_generations:
            self.generation += 1
            
            # Every 5 generations, learn from past results
            if self.generation % 5 == 1:
                try:
                    compile_insights()
                    self.hints = get_generation_hints()
                    if self.hints.get("preferred_signals"):
                        print(f"  🧠 Learned: prefer {self.hints['preferred_signals'][:4]}")
                except:
                    self.hints = {}
            
            self._run_generation()
        
        self._final_report()
    
    def _run_generation(self):
        t_start = time.time()
        gen_survived = 0
        
        print(f"\n{'─' * 50}")
        print(f"🔬 GENERATION {self.generation}")
        print(f"{'─' * 50}")
        
        for i in range(self.batch_size):
            if not self.running:
                break
            
            # Strategy creation: 40% random, 30% mutation, 30% crossover
            strat = None
            if self.elite_pool and random.random() < 0.6:
                if random.random() < 0.5 and len(self.elite_pool) >= 2:
                    # Crossover from two elite parents
                    parents = random.sample(self.elite_pool, 2)
                    strat = crossover_strategies(parents[0]["params"], parents[1]["params"])
                else:
                    # Mutate a random elite
                    parent = random.choice(self.elite_pool)
                    strat = mutate_strategy(parent["params"])
            
            if strat is None:
                strat = generate_scalp_strategy(hints=self.hints)
            
            self.total_tested += 1
            
            print(f"  [{i+1}/{self.batch_size}] {strat['name']}")
            print(f"    Signals: {', '.join(strat['signals'])}")
            
            # In-sample test (using params, not code)
            is_result = self.backtester.run(None, split="train", params=strat["params"])
            
            if is_result.score <= 0:
                print(f"    ❌ IS failed: score={is_result.score:.2f} "
                      f"trades={is_result.num_trades} dd={is_result.max_drawdown_pct:.1f}%")
                
                # Save killed strategy to DB for dashboard
                kill_reason = f"IS fail: score={is_result.score:.2f} trades={is_result.num_trades} dd={is_result.max_drawdown_pct:.1f}%"
                rec = StrategyRecord(
                    id=strat["id"], name=strat["name"], code="",
                    params=strat["params"], signals_used=strat["signals"],
                    is_sharpe=is_result.sharpe, is_return_pct=is_result.total_return_pct,
                    is_max_dd_pct=is_result.max_drawdown_pct, is_num_trades=is_result.num_trades,
                    is_score=is_result.score, status="killed", kill_reason=kill_reason,
                )
                save_strategy(rec)
                log_event(strat["id"], "killed_is", kill_reason)
                
                # Track near-misses for mutation (score > -50 = promising)
                if is_result.score > -50 and is_result.num_trades >= 20:
                    self.elite_pool.append(strat)
                    # Keep elite pool manageable
                    if len(self.elite_pool) > 20:
                        # Sort by IS score, keep top 20
                        self.elite_pool.sort(key=lambda s: s.get("_is_score", -999), reverse=True)
                        self.elite_pool = self.elite_pool[:20]
                    strat["_is_score"] = is_result.score
                
                continue
            
            print(f"    IS: score={is_result.score:.2f} sharpe={is_result.sharpe:.2f} "
                  f"ret={is_result.total_return_pct:.1f}% dd={is_result.max_drawdown_pct:.1f}% "
                  f"trades={is_result.num_trades} wr={is_result.win_rate_pct:.0f}%")
            
            # Out-of-sample test
            oos_result = self.backtester.run(None, split="val", params=strat["params"])
            
            if oos_result.score <= 0:
                print(f"    ❌ OOS failed: score={oos_result.score:.2f}")
                continue
            
            # Overfit check
            if is_result.sharpe > 0:
                decay = oos_result.sharpe / is_result.sharpe
                if decay < 0.4:
                    print(f"    ❌ Overfit: sharpe decay {decay:.2f}")
                    continue
            
            gen_survived += 1
            self.total_survived += 1
            
            print(f"    ✅ OOS: score={oos_result.score:.2f} sharpe={oos_result.sharpe:.2f} "
                  f"ret={oos_result.total_return_pct:.1f}% dd={oos_result.max_drawdown_pct:.1f}%")
            
            # WALK-FORWARD: Also test on hidden test split (2025 forward data)
            test_result = self.backtester.run(None, split="test", params=strat["params"])
            
            if test_result.score <= 0:
                print(f"    ❌ Walk-forward FAILED: test_score={test_result.score:.2f}")
                rec = StrategyRecord(
                    id=strat["id"], name=strat["name"], code="",
                    params=strat["params"], signals_used=strat["signals"],
                    is_sharpe=is_result.sharpe, is_return_pct=is_result.total_return_pct,
                    is_max_dd_pct=is_result.max_drawdown_pct, is_num_trades=is_result.num_trades,
                    is_score=is_result.score, oos_sharpe=oos_result.sharpe,
                    oos_return_pct=oos_result.total_return_pct,
                    oos_max_dd_pct=oos_result.max_drawdown_pct,
                    oos_num_trades=oos_result.num_trades, oos_score=oos_result.score,
                    status="killed", kill_reason=f"Walk-forward fail: test={test_result.score:.2f}",
                )
                save_strategy(rec)
                continue
            
            print(f"    ✅ Walk-forward PASSED: test={test_result.score:.2f} "
                  f"sharpe={test_result.sharpe:.2f}")
            
            # Save to DB — strategy passed ALL THREE splits
            rec = StrategyRecord(
                id=strat["id"], name=strat["name"], code=strat["code"],
                params=strat["params"], signals_used=strat["signals"],
                is_sharpe=is_result.sharpe, is_return_pct=is_result.total_return_pct,
                is_max_dd_pct=is_result.max_drawdown_pct, is_win_rate=is_result.win_rate_pct,
                is_num_trades=is_result.num_trades, is_score=is_result.score,
                oos_sharpe=oos_result.sharpe, oos_return_pct=oos_result.total_return_pct,
                oos_max_dd_pct=oos_result.max_drawdown_pct, oos_win_rate=oos_result.win_rate_pct,
                oos_num_trades=oos_result.num_trades, oos_score=oos_result.score,
                status="passed_oos",
            )
            save_strategy(rec)
            
            # Add to elite pool (top priority — passed all 3 splits)
            strat["_is_score"] = is_result.score
            self.elite_pool.append(strat)
            if len(self.elite_pool) > 20:
                self.elite_pool.sort(key=lambda s: s.get("_is_score", -999), reverse=True)
                self.elite_pool = self.elite_pool[:20]
            
            if oos_result.score > self.best_score:
                self.best_score = oos_result.score
                print(f"    🏆 NEW BEST! Score: {self.best_score:.3f}")
                
                # Save best
                os.makedirs("reports", exist_ok=True)
                with open("reports/best_scalp_strategy.py", "w") as f:
                    f.write(strat["code"])
                with open("reports/best_scalp_result.json", "w") as f:
                    json.dump({
                        "name": strat["name"],
                        "signals": strat["signals"],
                        "params": strat["params"],
                        "is_score": is_result.score,
                        "oos_score": oos_result.score,
                        "oos_sharpe": oos_result.sharpe,
                        "oos_return": oos_result.total_return_pct,
                        "oos_max_dd": oos_result.max_drawdown_pct,
                        "oos_trades": oos_result.num_trades,
                        "oos_win_rate": oos_result.win_rate_pct,
                        "avg_duration_bars": oos_result.avg_trade_duration_bars,
                    }, f, indent=2)
        
        elapsed = time.time() - t_start
        print(f"\n  Gen {self.generation}: {gen_survived}/{self.batch_size} survived, "
              f"best={self.best_score:.3f}, {elapsed:.0f}s")
    
    def _final_report(self):
        print(f"\n\n{'═' * 60}")
        print("📊 FINAL REPORT")
        print(f"{'═' * 60}")
        print(f"Generations:  {self.generation}")
        print(f"Tested:       {self.total_tested}")
        print(f"Survived:     {self.total_survived} ({self.total_survived/max(self.total_tested,1)*100:.1f}%)")
        print(f"Best Score:   {self.best_score:.3f}")
        
        top = load_strategies(status="passed_oos", limit=10)
        if top:
            print(f"\n🏆 Top Strategies:")
            for i, s in enumerate(top):
                print(f"  {i+1}. {s['name'][:30]} | OOS={s.get('oos_score',0):.3f} "
                      f"Sharpe={s.get('oos_sharpe',0):.2f} "
                      f"DD={s.get('oos_max_dd_pct',0):.1f}%")
        
        # Auto-export to MQL5
        if self.total_survived > 0:
            print(f"\n🔄 Exporting top strategies to MQL5...")
            try:
                from core.mql5_converter import export_top_strategies
                export_top_strategies(output_dir="mql5_experts", top_n=10, mode="scalp")
            except Exception as e:
                print(f"  MQL5 export error: {e}")
        
        # Note: Scalping mode uses $100 balance
        print(f"\n💰 Scalping mode: $100 account balance")
        print(f"   All results are based on $100 starting capital")


# ─────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scalping Strategy Research Engine")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"],
                       help="Binance trading pairs")
    parser.add_argument("--interval", default="5m", help="Candle interval")
    parser.add_argument("--batch-size", type=int, default=10, help="Strategies per generation")
    parser.add_argument("--max-generations", type=int, default=500, help="Max generations")
    parser.add_argument("--download-only", action="store_true", help="Just download data")
    args = parser.parse_args()
    
    if args.download_only:
        loader = BinanceDataLoader(symbols=args.symbols, interval=args.interval)
        loader.download_all()
    else:
        engine = ScalpOrchestrator(
            symbols=args.symbols,
            interval=args.interval,
            batch_size=args.batch_size,
            max_generations=args.max_generations,
        )
        engine.run()
