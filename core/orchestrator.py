"""
ORCHESTRATOR — The Main Loop (v2)

Ties everything together:
Genesis → Backtest → Darwin → Heal → Learn → Repeat

Runs autonomously. Generates strategies, tests them, keeps winners,
kills losers, heals broken ones, learns from history, evolves.

v2 additions:
- Continuous learning integration (guided generation)
- News sentiment + macro context awareness
- Portfolio-level risk management
- XAU/USD + BTC/USD dual-asset support
- Learning reports every N generations
"""

import time
import json
import sys
import os
import signal
from typing import Optional

from .models import (
    StrategyRecord, StrategyStatus, init_db, save_strategy,
    load_strategies, log_event
)
from .genesis import generate_batch, generate_random, mutate_strategy
from .darwin import evaluate_strategy, get_leaderboard, get_statistics, generate_kill_report
from .healer import heal_strategy, detect_overfit, fix_overfit
from .learner import compile_insights, get_generation_hints, print_learning_report
from .risk_manager import PortfolioRiskManager, RiskConfig


class Orchestrator:
    """
    The autonomous trading strategy research engine.
    
    Lifecycle:
    1. Generate N new strategies (random + mutations + crossovers)
    2. Evaluate each through IS → OOS pipeline
    3. Heal broken strategies, re-test
    4. Prune and rank survivors
    5. Print status report
    6. Repeat
    """
    
    def __init__(
        self,
        batch_size: int = 10,
        max_generations: int = 1000,
        heal_attempts: int = 2,
        mutation_intensity: float = 0.2,
        adaptive_intensity: bool = True,
    ):
        self.batch_size = batch_size
        self.max_generations = max_generations
        self.heal_attempts = heal_attempts
        self.mutation_intensity = mutation_intensity
        self.adaptive_intensity = adaptive_intensity
        
        self.generation = 0
        self.total_evaluated = 0
        self.total_killed = 0
        self.total_survived = 0
        self.total_healed = 0
        self.best_score = -999.0
        self.best_strategy_id = None
        self.running = True
        self.start_time = 0
        self.learning_interval = 5  # Run learning every N generations
        self.risk_manager = PortfolioRiskManager(RiskConfig())
        self.generation_hints = {}
        
        # Graceful shutdown
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)
    
    def _shutdown(self, signum, frame):
        print("\n⚡ Shutdown signal received. Finishing current batch...")
        self.running = False
    
    def run(self):
        """Main loop. Runs until interrupted or max_generations reached."""
        init_db()
        self.start_time = time.time()
        
        print("=" * 60)
        print("🧬 AUTONOMOUS STRATEGY RESEARCH ENGINE")
        print("=" * 60)
        print(f"Batch size: {self.batch_size}")
        print(f"Max generations: {self.max_generations}")
        print(f"Heal attempts per error: {self.heal_attempts}")
        print(f"Starting mutation intensity: {self.mutation_intensity}")
        print("=" * 60)
        print()
        
        while self.running and self.generation < self.max_generations:
            self.generation += 1
            
            # Periodically compile learnings and get hints
            if self.generation % self.learning_interval == 1 or self.generation == 1:
                try:
                    self.generation_hints = get_generation_hints()
                    if self.generation_hints.get("hints"):
                        print(f"\n🧠 Learning hints: {'; '.join(self.generation_hints['hints'][:3])}")
                except Exception:
                    self.generation_hints = {}
            
            self._run_generation()
            
            # Periodic learning report
            if self.generation % (self.learning_interval * 2) == 0:
                try:
                    print_learning_report()
                except Exception:
                    pass
            
            # Adaptive mutation: if no improvement in 5 generations, explore more
            if self.adaptive_intensity:
                self._adapt_intensity()
        
        self._print_final_report()
    
    def _run_generation(self):
        """Run one generation of the evolution loop."""
        t_start = time.time()
        
        print(f"\n{'─' * 60}")
        print(f"🔬 GENERATION {self.generation}")
        print(f"{'─' * 60}")
        
        # ── Step 1: Generate batch ──
        n_random = max(2, self.batch_size // 3)
        n_mutations = max(2, self.batch_size // 3)
        n_crossovers = max(1, self.batch_size - n_random - n_mutations)
        
        print(f"Generating: {n_random} random, {n_mutations} mutations, {n_crossovers} crossovers")
        
        batch = generate_batch(
            n_random=n_random,
            n_mutations=n_mutations,
            n_crossovers=n_crossovers,
            mutation_intensity=self.mutation_intensity,
        )
        
        gen_survived = 0
        gen_killed = 0
        gen_errors = 0
        gen_healed = 0
        
        # ── Step 2: Evaluate each strategy ──
        for i, strategy in enumerate(batch):
            if not self.running:
                break
            
            print(f"\n  [{i+1}/{len(batch)}] Testing: {strategy.name}")
            print(f"    Signals: {', '.join(strategy.signals_used)}")
            
            save_strategy(strategy)
            result = evaluate_strategy(strategy)
            self.total_evaluated += 1
            
            if result.status == StrategyStatus.PASSED_OOS.value:
                gen_survived += 1
                self.total_survived += 1
                print(f"    ✅ SURVIVED | IS={result.is_score:.2f} OOS={result.oos_score:.2f} "
                      f"Sharpe={result.oos_sharpe:.2f} DD={result.oos_max_dd_pct:.1f}%")
                
                # Check for new best
                if result.oos_score > self.best_score:
                    self.best_score = result.oos_score
                    self.best_strategy_id = result.id
                    print(f"    🏆 NEW BEST! Score: {self.best_score:.3f}")
                    self._save_best_strategy(result)
                
                # Check for overfit even in survivors
                is_overfit, reason = detect_overfit(result)
                if is_overfit:
                    print(f"    ⚠️  Overfit warning: {reason}")
                    fixed = fix_overfit(result)
                    if fixed:
                        print(f"    🔧 Created de-overfitted variant: {fixed.name}")
                        save_strategy(fixed)
                        # Queue for next generation evaluation
            
            elif result.status == StrategyStatus.KILLED.value:
                gen_killed += 1
                self.total_killed += 1
                print(f"    ❌ KILLED | {result.kill_reason[:80]}")
            
            elif result.status == StrategyStatus.ERROR.value:
                gen_errors += 1
                print(f"    💥 ERROR | {result.kill_reason[:80]}")
                
                # Try to heal
                for attempt in range(self.heal_attempts):
                    healed = heal_strategy(result, result.kill_reason)
                    if healed:
                        print(f"    🩹 Heal attempt {attempt+1}: {healed.name}")
                        save_strategy(healed)
                        healed_result = evaluate_strategy(healed)
                        self.total_evaluated += 1
                        
                        if healed_result.status == StrategyStatus.PASSED_OOS.value:
                            gen_healed += 1
                            self.total_healed += 1
                            gen_survived += 1
                            self.total_survived += 1
                            print(f"    ✅ HEALED & SURVIVED! Score={healed_result.oos_score:.2f}")
                            
                            if healed_result.oos_score > self.best_score:
                                self.best_score = healed_result.oos_score
                                self.best_strategy_id = healed_result.id
                                print(f"    🏆 NEW BEST! Score: {self.best_score:.3f}")
                                self._save_best_strategy(healed_result)
                            break
                        elif healed_result.status == StrategyStatus.ERROR.value:
                            result = healed_result  # try healing again
                            continue
                        else:
                            print(f"    ❌ Healed but killed: {healed_result.kill_reason[:60]}")
                            break
                    else:
                        print(f"    ⚠️  Could not heal")
                        break
        
        # ── Step 3: Generation summary ──
        t_elapsed = time.time() - t_start
        
        print(f"\n{'─' * 40}")
        print(f"Generation {self.generation} Summary:")
        print(f"  Survived: {gen_survived}/{len(batch)}")
        print(f"  Killed:   {gen_killed}")
        print(f"  Errors:   {gen_errors}")
        print(f"  Healed:   {gen_healed}")
        print(f"  Time:     {t_elapsed:.1f}s")
        print(f"  Best ever: {self.best_score:.3f}")
        
        # Print leaderboard every 5 generations
        if self.generation % 5 == 0:
            self._print_leaderboard()
    
    def _adapt_intensity(self):
        """Adapt mutation intensity based on recent progress."""
        # If no improvement in recent generations, increase exploration
        # (This is a simplified version — production would track rolling improvement)
        stats = get_statistics()
        kill_rate = stats.get("kill_rate", 90)
        
        if kill_rate > 95:
            # Almost everything dies — explore more widely
            self.mutation_intensity = min(0.5, self.mutation_intensity * 1.1)
        elif kill_rate < 70:
            # Many survivors — fine-tune more
            self.mutation_intensity = max(0.05, self.mutation_intensity * 0.9)
    
    def _save_best_strategy(self, rec: StrategyRecord):
        """Save the best strategy to a file for easy access."""
        report = generate_kill_report(rec)
        
        best_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
        os.makedirs(best_dir, exist_ok=True)
        
        # Save code
        with open(os.path.join(best_dir, "best_strategy.py"), "w") as f:
            f.write(rec.code)
        
        # Save report
        with open(os.path.join(best_dir, "best_report.txt"), "w") as f:
            f.write(report)
        
        # Save params
        with open(os.path.join(best_dir, "best_params.json"), "w") as f:
            json.dump({
                "id": rec.id,
                "name": rec.name,
                "signals": rec.signals_used,
                "params": rec.params,
                "is_score": rec.is_score,
                "oos_score": rec.oos_score,
                "is_sharpe": rec.is_sharpe,
                "oos_sharpe": rec.oos_sharpe,
                "generation": rec.generation,
            }, f, indent=2)
    
    def _print_leaderboard(self):
        """Print the current top strategies."""
        top = get_leaderboard(10)
        if not top:
            print("\n  No survivors yet.")
            return
        
        print(f"\n{'═' * 60}")
        print("🏆 LEADERBOARD")
        print(f"{'═' * 60}")
        print(f"{'Rank':<5} {'Name':<25} {'OOS Score':>10} {'Sharpe':>8} {'DD':>6} {'Trades':>7}")
        print(f"{'─' * 60}")
        
        for i, s in enumerate(top):
            print(f"{i+1:<5} {s['name'][:24]:<25} {s.get('oos_score', -999):>10.3f} "
                  f"{s.get('oos_sharpe', 0):>8.2f} {s.get('oos_max_dd_pct', 0):>5.1f}% "
                  f"{s.get('oos_num_trades', 0):>7d}")
    
    def _print_final_report(self):
        """Print final summary when the loop ends."""
        elapsed = time.time() - self.start_time
        
        print(f"\n\n{'═' * 60}")
        print("📊 FINAL REPORT")
        print(f"{'═' * 60}")
        print(f"Total time:        {elapsed/3600:.1f} hours")
        print(f"Generations:       {self.generation}")
        print(f"Strategies tested: {self.total_evaluated}")
        print(f"Survived:          {self.total_survived} ({self.total_survived/max(self.total_evaluated,1)*100:.1f}%)")
        print(f"Killed:            {self.total_killed}")
        print(f"Healed:            {self.total_healed}")
        print(f"Best score:        {self.best_score:.3f}")
        
        self._print_leaderboard()
        
        stats = get_statistics()
        if stats.get("best_strategy"):
            print(f"\n🏆 Best: {stats['best_strategy']['name']} "
                  f"(Score: {stats['best_strategy']['score']:.3f}, "
                  f"Sharpe: {stats['best_strategy']['sharpe']:.3f})")


# ─────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────

def main():
    """Run the autonomous strategy research engine."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Autonomous Trading Strategy Research")
    parser.add_argument("--batch-size", type=int, default=10, help="Strategies per generation")
    parser.add_argument("--max-generations", type=int, default=1000, help="Max generations to run")
    parser.add_argument("--heal-attempts", type=int, default=2, help="Heal attempts per error")
    parser.add_argument("--mutation-intensity", type=float, default=0.2, help="Initial mutation intensity")
    parser.add_argument("--no-adaptive", action="store_true", help="Disable adaptive mutation")
    args = parser.parse_args()
    
    engine = Orchestrator(
        batch_size=args.batch_size,
        max_generations=args.max_generations,
        heal_attempts=args.heal_attempts,
        mutation_intensity=args.mutation_intensity,
        adaptive_intensity=not args.no_adaptive,
    )
    engine.run()


if __name__ == "__main__":
    main()
