#!/usr/bin/env python3
"""
Export winning strategies to MQL5 Expert Advisors for MetaTrader 5.

Usage:
    python export_mql5.py                          # Export top 10
    python export_mql5.py --top 5                  # Top 5 only
    python export_mql5.py --mode scalp             # Force scalping mode
    python export_mql5.py --output my_eas/         # Custom output dir
    python export_mql5.py --strategy-id abc123     # Export specific strategy

Output: .mq5 files ready to compile in MetaEditor and run on MT5.
"""

import argparse
import json
import os
import sys

from core.mql5_converter import export_top_strategies, convert_to_mql5
from core.models import load_strategies, init_db


def export_single(strategy_id: str, output_dir: str, mode: str):
    """Export a single strategy by ID."""
    import sqlite3
    from core.models import DB_PATH
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM strategies WHERE id = ?", (strategy_id,)).fetchone()
    conn.close()
    
    if not row:
        print(f"Strategy '{strategy_id}' not found.")
        return
    
    strat = dict(row)
    mql5_code = convert_to_mql5(strat, mode=mode)
    
    os.makedirs(output_dir, exist_ok=True)
    name = strat.get("name", strategy_id)[:40]
    clean = "".join(c for c in name if c.isalnum() or c == "_")
    filepath = os.path.join(output_dir, f"{clean}.mq5")
    
    with open(filepath, "w") as f:
        f.write(mql5_code)
    
    print(f"✅ Exported: {filepath}")
    print(f"   Score: {strat.get('oos_score', 'N/A')}")
    print(f"   Sharpe: {strat.get('oos_sharpe', 'N/A')}")


def main():
    parser = argparse.ArgumentParser(description="Export strategies to MQL5 Expert Advisors")
    parser.add_argument("--output", default="mql5_experts", help="Output directory")
    parser.add_argument("--top", type=int, default=10, help="Number of top strategies to export")
    parser.add_argument("--mode", default="auto", choices=["auto", "scalp", "swing"],
                       help="Trading mode (auto-detects if not specified)")
    parser.add_argument("--strategy-id", help="Export a specific strategy by ID")
    parser.add_argument("--list", action="store_true", help="List available strategies")
    args = parser.parse_args()
    
    init_db()
    
    if args.list:
        top = load_strategies(status="passed_oos", limit=30)
        if not top:
            print("No surviving strategies found. Run the engine first.")
            return
        
        print(f"\n{'ID':<14} {'Name':<30} {'Score':>8} {'Sharpe':>8} {'DD':>6} {'Trades':>7}")
        print("─" * 75)
        for s in top:
            print(f"{s['id']:<14} {s['name'][:29]:<30} "
                  f"{s.get('oos_score', 0):>8.3f} "
                  f"{s.get('oos_sharpe', 0):>8.2f} "
                  f"{s.get('oos_max_dd_pct', 0):>5.1f}% "
                  f"{s.get('oos_num_trades', 0):>7d}")
        return
    
    if args.strategy_id:
        export_single(args.strategy_id, args.output, args.mode)
    else:
        export_top_strategies(output_dir=args.output, top_n=args.top, mode=args.mode)


if __name__ == "__main__":
    main()
