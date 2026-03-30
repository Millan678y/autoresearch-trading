"""
CONTINUOUS LEARNING MODULE — Learns from Execution History

Analyzes:
1. Which signal combinations work in which market regimes
2. Common failure patterns and how to avoid them
3. Optimal parameter ranges based on historical winners
4. Entry/exit timing mistakes

Feeds insights back into Genesis for smarter generation.
"""

import json
import os
import time
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np

from .models import (
    StrategyRecord, load_strategies, log_event, DB_PATH
)

INSIGHTS_PATH = os.path.join(os.path.expanduser("~"), ".cache", "autotrader", "insights.json")


# ─────────────────────────────────────────────────────────────────
# Signal Performance Tracking
# ─────────────────────────────────────────────────────────────────

def analyze_signal_performance() -> Dict[str, dict]:
    """
    Analyze which signals appear most in winners vs losers.
    
    Returns {signal_name: {
        win_count, lose_count, win_rate,
        avg_sharpe_when_present, avg_sharpe_when_absent
    }}
    """
    import sqlite3
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    all_strats = conn.execute(
        "SELECT signals_used, status, oos_sharpe, oos_score FROM strategies "
        "WHERE status IN ('passed_oos', 'killed') AND oos_sharpe != 0"
    ).fetchall()
    conn.close()
    
    if not all_strats:
        return {}
    
    signal_stats = defaultdict(lambda: {
        "win_count": 0, "lose_count": 0,
        "sharpes_present": [], "sharpes_absent": []
    })
    
    # Track all unique signals
    all_signals = set()
    for row in all_strats:
        signals = json.loads(row["signals_used"]) if row["signals_used"] else []
        all_signals.update(signals)
    
    for row in all_strats:
        signals = set(json.loads(row["signals_used"]) if row["signals_used"] else [])
        is_winner = row["status"] == "passed_oos"
        sharpe = row["oos_sharpe"] or 0
        
        for sig in all_signals:
            if sig in signals:
                if is_winner:
                    signal_stats[sig]["win_count"] += 1
                else:
                    signal_stats[sig]["lose_count"] += 1
                signal_stats[sig]["sharpes_present"].append(sharpe)
            else:
                signal_stats[sig]["sharpes_absent"].append(sharpe)
    
    # Compute derived metrics
    results = {}
    for sig, stats in signal_stats.items():
        total = stats["win_count"] + stats["lose_count"]
        results[sig] = {
            "win_count": stats["win_count"],
            "lose_count": stats["lose_count"],
            "win_rate": stats["win_count"] / max(total, 1) * 100,
            "total_appearances": total,
            "avg_sharpe_present": float(np.mean(stats["sharpes_present"])) if stats["sharpes_present"] else 0,
            "avg_sharpe_absent": float(np.mean(stats["sharpes_absent"])) if stats["sharpes_absent"] else 0,
        }
    
    return results


# ─────────────────────────────────────────────────────────────────
# Parameter Range Learning
# ─────────────────────────────────────────────────────────────────

def learn_optimal_params() -> Dict[str, dict]:
    """
    Analyze surviving strategies to find optimal parameter ranges.
    
    Returns {param_name: {
        mean, median, std, min, max, p25, p75,
        recommended_range: (low, high)
    }}
    """
    import sqlite3
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    winners = conn.execute(
        "SELECT params, oos_score FROM strategies "
        "WHERE status = 'passed_oos' ORDER BY oos_score DESC LIMIT 50"
    ).fetchall()
    conn.close()
    
    if not winners:
        return {}
    
    param_values = defaultdict(list)
    
    for row in winners:
        params = json.loads(row["params"]) if row["params"] else {}
        for key, val in params.items():
            if isinstance(val, (int, float)):
                param_values[key].append(val)
    
    results = {}
    for param, values in param_values.items():
        if len(values) < 3:
            continue
        arr = np.array(values)
        p25, p75 = np.percentile(arr, [25, 75])
        iqr = p75 - p25
        
        results[param] = {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "p25": float(p25),
            "p75": float(p75),
            "recommended_range": (
                float(max(p25 - 0.5 * iqr, np.min(arr))),
                float(min(p75 + 0.5 * iqr, np.max(arr)))
            ),
            "n_samples": len(values),
        }
    
    return results


# ─────────────────────────────────────────────────────────────────
# Failure Pattern Analysis
# ─────────────────────────────────────────────────────────────────

def analyze_failure_patterns() -> List[dict]:
    """
    Identify common failure patterns from killed strategies.
    
    Returns list of {pattern, count, description, recommendation}
    """
    import sqlite3
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    killed = conn.execute(
        "SELECT kill_reason, signals_used, params, is_sharpe, oos_sharpe, "
        "is_max_dd_pct, oos_max_dd_pct, is_num_trades, oos_num_trades "
        "FROM strategies WHERE status = 'killed'"
    ).fetchall()
    conn.close()
    
    if not killed:
        return []
    
    patterns = defaultdict(lambda: {"count": 0, "examples": []})
    
    for row in killed:
        reason = row["kill_reason"] or ""
        
        # Categorize failure
        if "Sharpe" in reason and "IS" in reason:
            patterns["low_is_sharpe"]["count"] += 1
            patterns["low_is_sharpe"]["examples"].append(reason[:100])
        
        elif "Sharpe" in reason and "OOS" in reason:
            patterns["low_oos_sharpe"]["count"] += 1
            patterns["low_oos_sharpe"]["examples"].append(reason[:100])
        
        elif "Overfit" in reason:
            patterns["overfit"]["count"] += 1
            patterns["overfit"]["examples"].append(reason[:100])
            
        elif "drawdown" in reason.lower():
            patterns["excessive_drawdown"]["count"] += 1
            patterns["excessive_drawdown"]["examples"].append(reason[:100])
        
        elif "trades" in reason.lower():
            patterns["insufficient_trades"]["count"] += 1
            patterns["insufficient_trades"]["examples"].append(reason[:100])
        
        elif "return" in reason.lower() or "profitable" in reason.lower():
            patterns["unprofitable"]["count"] += 1
            patterns["unprofitable"]["examples"].append(reason[:100])
        
        elif "error" in reason.lower() or "Error" in reason:
            patterns["runtime_error"]["count"] += 1
            patterns["runtime_error"]["examples"].append(reason[:100])
        
        else:
            patterns["other"]["count"] += 1
    
    # Build recommendations
    recommendations = {
        "low_is_sharpe": "Signals too weak for the training period. Try different signal combos or tighter entry conditions.",
        "low_oos_sharpe": "Strategy doesn't generalize. Reduce complexity, use simpler signals.",
        "overfit": "IS performance doesn't hold OOS. Reduce parameters, widen stops, increase regularization.",
        "excessive_drawdown": "Risk management insufficient. Tighter stops, smaller positions, or vol-adaptive sizing.",
        "insufficient_trades": "Entry conditions too restrictive. Lower min_votes or widen thresholds.",
        "unprofitable": "Core signal doesn't capture edge. Try different signal combinations.",
        "runtime_error": "Code bugs. Check array bounds, add NaN guards, validate lookback periods.",
        "other": "Misc failures. Review specific kill reasons.",
    }
    
    results = []
    for pattern, data in sorted(patterns.items(), key=lambda x: -x[1]["count"]):
        results.append({
            "pattern": pattern,
            "count": data["count"],
            "pct": data["count"] / len(killed) * 100,
            "description": recommendations.get(pattern, "Unknown pattern"),
            "example": data["examples"][0] if data["examples"] else "",
        })
    
    return results


# ─────────────────────────────────────────────────────────────────
# Regime Analysis
# ─────────────────────────────────────────────────────────────────

def analyze_regime_performance(equity_curves: Dict[str, List[float]],
                               market_data: Dict[str, np.ndarray]) -> dict:
    """
    Analyze how strategies perform in different market regimes.
    
    Regimes: trending_up, trending_down, ranging, high_vol, low_vol
    """
    if not market_data or "close" not in market_data:
        return {}
    
    closes = market_data["close"]
    n = len(closes)
    
    # Define regimes per period
    regime_labels = []
    window = 24  # 24h for regime detection
    
    for i in range(n):
        if i < window:
            regime_labels.append("unknown")
            continue
        
        segment = closes[i - window:i]
        ret = (segment[-1] - segment[0]) / segment[0]
        vol = np.std(np.diff(np.log(segment)))
        
        if vol > 0.02:  # High vol threshold
            regime_labels.append("high_vol")
        elif vol < 0.005:  # Low vol
            regime_labels.append("low_vol")
        elif ret > 0.02:
            regime_labels.append("trending_up")
        elif ret < -0.02:
            regime_labels.append("trending_down")
        else:
            regime_labels.append("ranging")
    
    # Analyze each strategy's returns per regime
    results = {}
    for strat_name, equity in equity_curves.items():
        if len(equity) != n + 1:  # equity has one more point than bars
            continue
        
        returns = np.diff(equity) / np.array(equity[:-1])
        
        regime_returns = defaultdict(list)
        for i, regime in enumerate(regime_labels):
            if i < len(returns):
                regime_returns[regime].append(returns[i])
        
        results[strat_name] = {}
        for regime, rets in regime_returns.items():
            if not rets:
                continue
            arr = np.array(rets)
            results[strat_name][regime] = {
                "mean_return": float(np.mean(arr)),
                "sharpe": float(np.mean(arr) / max(np.std(arr), 1e-10) * np.sqrt(8760)),
                "win_rate": float(np.sum(arr > 0) / len(arr) * 100),
                "n_bars": len(arr),
            }
    
    return results


# ─────────────────────────────────────────────────────────────────
# Insight Compilation
# ─────────────────────────────────────────────────────────────────

def compile_insights() -> dict:
    """
    Compile all learnings into a single insights object.
    This is fed back into Genesis to guide strategy generation.
    """
    insights = {
        "timestamp": time.time(),
        "signal_performance": analyze_signal_performance(),
        "optimal_params": learn_optimal_params(),
        "failure_patterns": analyze_failure_patterns(),
    }
    
    # Compute recommended signals (ranked by win rate)
    sig_perf = insights["signal_performance"]
    if sig_perf:
        ranked = sorted(sig_perf.items(), key=lambda x: -x[1]["win_rate"])
        insights["recommended_signals"] = [
            {"signal": name, "win_rate": stats["win_rate"], "avg_sharpe": stats["avg_sharpe_present"]}
            for name, stats in ranked[:6]
        ]
    else:
        insights["recommended_signals"] = []
    
    # Compute top failure to avoid
    if insights["failure_patterns"]:
        top_failure = insights["failure_patterns"][0]
        insights["top_failure"] = top_failure["pattern"]
        insights["top_failure_pct"] = top_failure["pct"]
    
    # Save
    os.makedirs(os.path.dirname(INSIGHTS_PATH), exist_ok=True)
    with open(INSIGHTS_PATH, "w") as f:
        json.dump(insights, f, indent=2, default=str)
    
    return insights


def get_generation_hints() -> dict:
    """
    Get hints for Genesis based on accumulated learning.
    Returns actionable guidance for strategy generation.
    """
    if not os.path.exists(INSIGHTS_PATH):
        return {"mode": "explore", "hints": []}
    
    with open(INSIGHTS_PATH) as f:
        insights = json.load(f)
    
    hints = []
    
    # Signal hints
    recommended = insights.get("recommended_signals", [])
    if recommended:
        top_signals = [s["signal"] for s in recommended[:4]]
        hints.append(f"Prefer signals: {', '.join(top_signals)}")
    
    # Param hints
    params = insights.get("optimal_params", {})
    for key in ["position_pct", "cooldown_bars", "min_votes"]:
        if key in params:
            r = params[key]["recommended_range"]
            hints.append(f"Keep {key} in [{r[0]:.3f}, {r[1]:.3f}]")
    
    # Failure avoidance
    failures = insights.get("failure_patterns", [])
    if failures:
        top = failures[0]
        hints.append(f"Top failure: {top['pattern']} ({top['pct']:.0f}%) — {top['description']}")
    
    return {
        "mode": "guided" if recommended else "explore",
        "hints": hints,
        "preferred_signals": [s["signal"] for s in recommended[:4]] if recommended else [],
        "param_ranges": {
            k: v["recommended_range"]
            for k, v in params.items()
            if "recommended_range" in v
        },
    }


def print_learning_report():
    """Print a human-readable learning report."""
    insights = compile_insights()
    
    print("\n" + "=" * 60)
    print("🧠 CONTINUOUS LEARNING REPORT")
    print("=" * 60)
    
    # Signal performance
    print("\n📊 Signal Performance Ranking:")
    sig_perf = insights.get("signal_performance", {})
    if sig_perf:
        ranked = sorted(sig_perf.items(), key=lambda x: -x[1]["win_rate"])
        for name, stats in ranked:
            bar = "█" * int(stats["win_rate"] / 5)
            print(f"  {name:<20} Win: {stats['win_rate']:>5.1f}% {bar} "
                  f"(Sharpe: {stats['avg_sharpe_present']:>6.2f}, n={stats['total_appearances']})")
    else:
        print("  No data yet.")
    
    # Optimal params
    print("\n⚙️  Learned Optimal Parameters:")
    params = insights.get("optimal_params", {})
    if params:
        for key in sorted(params.keys()):
            p = params[key]
            print(f"  {key:<30} [{p['recommended_range'][0]:.4f} - {p['recommended_range'][1]:.4f}] "
                  f"(median={p['median']:.4f}, n={p['n_samples']})")
    else:
        print("  No data yet.")
    
    # Failure patterns
    print("\n💀 Failure Patterns:")
    failures = insights.get("failure_patterns", [])
    if failures:
        for f in failures[:5]:
            print(f"  {f['pattern']:<25} {f['count']:>4} ({f['pct']:.1f}%) — {f['description'][:60]}")
    else:
        print("  No data yet.")
    
    print()
