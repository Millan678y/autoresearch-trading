#!/usr/bin/env python3
"""
Entry point for the autonomous strategy research engine.

Usage:
    uv run run_engine.py                          # Default: 10 strategies/gen, 1000 gens
    uv run run_engine.py --batch-size 20          # Larger batches
    uv run run_engine.py --max-generations 50     # Quick test
    uv run run_engine.py --mutation-intensity 0.4  # More exploration

Press Ctrl+C to gracefully stop. The engine will finish the current strategy
and print a final report.
"""

from core.orchestrator import main

if __name__ == "__main__":
    main()
