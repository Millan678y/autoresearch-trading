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

SCALP_SIGNALS = {
    "ema_cross": {
        "code": """
        ema_f = h["ema_{fast}"].iloc[-1] if "ema_{fast}" in h.columns else np.mean(closes[-{fast}:])
        ema_s = h["ema_{slow}"].iloc[-1] if "ema_{slow}" in h.columns else np.mean(closes[-{slow}:])
        s_bull = ema_f > ema_s
        s_bear = ema_f < ema_s""",
        "params": {"fast": (5, 12), "slow": (15, 50)},
    },
    "rsi_scalp": {
        "code": """
        rsi_col = "rsi_{period}"
        if rsi_col in h.columns:
            rsi_val = h[rsi_col].iloc[-1]
        else:
            delta = np.diff(closes[-{period}-1:])
            gains = np.where(delta > 0, delta, 0).mean()
            losses = np.where(delta < 0, -delta, 0).mean()
            rsi_val = 100 - 100 / (1 + gains / max(losses, 1e-10))
        s_bull = rsi_val > {bull_thresh} and rsi_val < 70
        s_bear = rsi_val < {bear_thresh} and rsi_val > 30""",
        "params": {"period": (6, 14), "bull_thresh": (48, 58), "bear_thresh": (42, 52)},
    },
    "micro_momentum": {
        "code": """
        ret = (closes[-1] - closes[-{lookback}]) / closes[-{lookback}]
        s_bull = ret > {threshold}
        s_bear = ret < -{threshold}""",
        "params": {"lookback": (2, 12), "threshold": (0.001, 0.005)},
    },
    "taker_imbalance": {
        "code": """
        taker = bar.taker_buy_ratio
        s_bull = taker > {bull_thresh}
        s_bear = taker < {bear_thresh}""",
        "params": {"bull_thresh": (0.55, 0.70), "bear_thresh": (0.30, 0.45)},
    },
    "vol_spike": {
        "code": """
        if "vol_spike" in h.columns:
            vs = h["vol_spike"].iloc[-1]
        else:
            vs = bar.volume / max(np.mean(h["volume"].values[-12:]), 1)
        speed = (closes[-1] - closes[-3]) / closes[-3]
        s_bull = vs > {thresh} and speed > 0
        s_bear = vs > {thresh} and speed < 0""",
        "params": {"thresh": (1.5, 3.0)},
    },
    "vwap_position": {
        "code": """
        if "price_vs_vwap" in h.columns:
            pvw = h["price_vs_vwap"].iloc[-1]
        else:
            pvw = 0
        s_bull = pvw > {bull_thresh}
        s_bear = pvw < {bear_thresh}""",
        "params": {"bull_thresh": (0.0005, 0.003), "bear_thresh": (-0.003, -0.0005)},
    },
    "candle_body": {
        "code": """
        if "body_ratio" in h.columns:
            br = h["body_ratio"].iloc[-1]
        else:
            rng = bar.high - bar.low
            br = abs(bar.close - bar.open) / max(rng, 1e-10)
        bullish_candle = bar.close > bar.open
        s_bull = br > {min_body} and bullish_candle
        s_bear = br > {min_body} and not bullish_candle""",
        "params": {"min_body": (0.5, 0.8)},
    },
    "price_position": {
        "code": """
        col = "price_pos_{lookback}"
        if col in h.columns:
            pp = h[col].iloc[-1]
        else:
            rh = np.max(highs[-{lookback}:])
            rl = np.min(lows[-{lookback}:])
            pp = (closes[-1] - rl) / max(rh - rl, 1e-10)
        s_bull = pp < {oversold}
        s_bear = pp > {overbought}""",
        "params": {"lookback": (12, 72), "oversold": (0.15, 0.35), "overbought": (0.65, 0.85)},
    },
    "speed_acceleration": {
        "code": """
        if "acceleration" in h.columns:
            acc = h["acceleration"].iloc[-1]
        else:
            speed_now = (closes[-1] - closes[-4]) / 3
            speed_prev = (closes[-4] - closes[-7]) / 3
            acc = speed_now - speed_prev
        s_bull = acc > {threshold}
        s_bear = acc < -{threshold}""",
        "params": {"threshold": (0.0001, 0.001)},
    },
    "session_filter": {
        "code": """
        good_session = bar.session in ("london", "ny")
        s_bull = good_session and closes[-1] > closes[-3]
        s_bear = good_session and closes[-1] < closes[-3]""",
        "params": {},
    },
    "microvol_regime": {
        "code": """
        if "microvol_ratio" in h.columns:
            mvr = h["microvol_ratio"].iloc[-1]
        else:
            mv6 = np.std(np.diff(closes[-7:])) if len(closes) > 7 else 0.01
            mv12 = np.std(np.diff(closes[-13:])) if len(closes) > 13 else 0.01
            mvr = mv6 / max(mv12, 1e-10)
        expanding = mvr > {expand_thresh}
        s_bull = expanding and closes[-1] > closes[-2]
        s_bear = expanding and closes[-1] < closes[-2]""",
        "params": {"expand_thresh": (1.1, 2.0)},
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


SCALP_SIGNALS["macd_fast"] = {
    "code": "",
    "params": {"fast": (8, 14), "slow": (16, 26), "signal": (5, 9)},
}
SCALP_SIGNALS["bb_squeeze"] = {
    "code": "",
    "params": {"period": (10, 20)},
}
SCALP_SIGNALS["obv_trend"] = {
    "code": "",
    "params": {"period": (10, 25)},
}

ALL_SCALP_SIGNAL_NAMES = list(SCALP_SIGNALS.keys())

def generate_scalp_strategy() -> dict:
    """Generate a random scalping strategy config (no code generation)."""
    
    # Pick 3-6 signals
    n_signals = random.randint(3, min(6, len(ALL_SCALP_SIGNAL_NAMES)))
    chosen = random.sample(ALL_SCALP_SIGNAL_NAMES, n_signals)
    
    # Generate params
    params = {}
    for sig in chosen:
        for p_name, p_range in SCALP_SIGNALS[sig]["params"].items():
            params[f"{sig}__{p_name}"] = _random_param(p_range)
    
    # Strategy-level params
    min_votes = random.randint(2, max(2, n_signals - 2))
    cooldown = random.randint(3, 12)
    tp_mult = round(random.uniform(1.0, 3.0), 2)
    sl_mult = round(random.uniform(1.0, 3.0), 2)
    size_pct = round(random.uniform(0.3, 0.8), 2)
    
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
    
    sid = hashlib.sha256(strategy_code.encode()).hexdigest()[:12]
    
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
        
        while self.running and self.generation < self.max_generations:
            self.generation += 1
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
            
            strat = generate_scalp_strategy()
            self.total_tested += 1
            
            print(f"  [{i+1}/{self.batch_size}] {strat['name']}")
            print(f"    Signals: {', '.join(strat['signals'])}")
            
            # In-sample test (using params, not code)
            is_result = self.backtester.run(None, split="train", params=strat["params"])
            
            if is_result.score <= 0:
                print(f"    ❌ IS failed: score={is_result.score:.2f} "
                      f"trades={is_result.num_trades} dd={is_result.max_drawdown_pct:.1f}%")
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
            
            # Save to DB
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
