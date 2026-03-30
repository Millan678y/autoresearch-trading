# 🤖 Autoresearch Trading

**A fully autonomous, self-healing algorithmic trading system that generates, tests, and evolves its own strategies.**

No static rules. No manual tuning. The system creates strategies, backtests them ruthlessly, kills the weak, heals the broken, learns from mistakes, and evolves the survivors.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                      ORCHESTRATOR (run_engine.py)              │
│                    Main autonomous loop                        │
├──────────┬───────────┬──────────┬───────────┬────────────────┤
│ GENESIS  │ BACKTEST  │ DARWIN   │ HEALER    │ LEARNER        │
│ Engine   │ Engine    │ Module   │ Module    │ Module         │
│          │           │          │           │                │
│ Generate │ In-sample │ Rank     │ Fix bugs  │ Signal stats   │
│ Random   │ 2024      │ Score    │ Fix       │ Param ranges   │
│ Mutate   │ Out-of-   │ Prune    │ overfit   │ Failure        │
│ Crossover│ sample    │ Explain  │ Heal      │ patterns       │
│          │ 2025      │ Archive  │ errors    │ Regime aware   │
├──────────┴───────────┴──────────┴───────────┴────────────────┤
│                    DATA PIPELINE                              │
│  OHLCV (BTC + XAU) • News Sentiment • Macro Indicators       │
├──────────────────────────────────────────────────────────────┤
│                    RISK MANAGER                               │
│  Hybrid trailing stops • Vol-adaptive sizing • Circuit        │
│  breakers • Kelly criterion • Correlation caps                │
└──────────────────────────────────────────────────────────────┘
```

## Core Modules

| Module | File | Purpose |
|--------|------|---------|
| **Genesis** | `core/genesis.py` | Generates new strategies from 10 signal types, 3 stop-loss templates, 3 exit templates. Methods: random, mutation, crossover |
| **Darwin** | `core/darwin.py` | Two-stage evaluation (IS → OOS), overfit detection, ranking, pruning with analytical kill/keep reports |
| **Healer** | `core/healer.py` | Self-healing error correction. Pattern-matches 6 error types, auto-fixes code and parameters |
| **Learner** | `core/learner.py` | Continuous learning. Tracks signal win rates, optimal param ranges, failure patterns. Feeds hints back to Genesis |
| **Risk Manager** | `core/risk_manager.py` | Hybrid trailing stops (ATR + vol + time-decay), drawdown circuit breakers, vol-regime sizing, Kelly criterion |
| **Data Pipeline** | `core/data_pipeline.py` | Historical OHLCV for BTC/XAU, news sentiment (VADER), macro indicators (FRED/yfinance), feature engineering |
| **Orchestrator** | `core/orchestrator.py` | Main loop tying everything together. Adaptive mutation, learning integration, graceful shutdown |
| **Models** | `core/models.py` | SQLite-backed strategy database, experiment logging |

## Target Assets

- **BTC/USD** — Bitcoin (24/7 crypto market)
- **XAU/USD** — Gold (COMEX futures)

## Signal Library

| Signal | Description |
|--------|-------------|
| Momentum | N-period return threshold |
| EMA Cross | Fast/slow EMA crossover |
| RSI | Relative Strength Index (overbought/oversold) |
| MACD | Moving Average Convergence Divergence histogram |
| BB Compression | Bollinger Band width percentile (volatility squeeze) |
| Funding Carry | Crypto funding rate mean-reversion |
| Vol Breakout | Realized vol breaking above SMA |
| Mean Reversion | Z-score from rolling mean |
| Stochastic | Stochastic oscillator (K%D) |
| ATR Breakout | Price breaking N × ATR from prior close |

## Risk Management

### Hybrid Trailing Stop
Three mechanisms, **tightest wins**:
1. **ATR trailing** — adapts to asset volatility
2. **Vol-adaptive %** — wider in high vol, tighter in low vol
3. **Time-decay** — the longer you hold, the tighter the stop

### Drawdown Circuit Breaker
| Level | Trigger | Action |
|-------|---------|--------|
| 🟢 Green | DD < 5% | Normal trading |
| 🟡 Yellow | DD 5-10% | Warning, reduce sizes 25% |
| 🟠 Orange | DD 10-20% | Cut sizes 50% |
| 🔴 Red | DD > 20% | Halt all trading, 24-bar cooldown |

### Position Sizing
- **Inverse volatility targeting** — constant risk per trade
- **Kelly criterion** (quarter-Kelly) — mathematically optimal sizing
- **Correlation caps** — limit exposure to correlated assets

## Self-Healing

The system detects and auto-fixes:
- `IndexError` → Increase lookback guards
- `ZeroDivisionError` → Add epsilon guards  
- `NaN propagation` → Add NaN checks
- `MemoryError` → Reduce lookback periods
- `Timeout` → Simplify computation (drop expensive signals)
- `No trades` → Relax entry conditions

## Continuous Learning

After each generation, the Learner module:
1. Ranks signals by win rate across all evaluated strategies
2. Identifies optimal parameter ranges from survivors
3. Catalogs failure patterns with recommendations
4. Feeds guided hints to Genesis for smarter generation

## Quick Start

### Install Dependencies

```bash
pip install torch numpy pandas yfinance flask tqdm tiktoken matplotlib
pip install feedparser vaderSentiment  # For news sentiment
pip install fredapi                    # For macro indicators (optional)
```

### Download Data

```bash
python -c "from core.data_pipeline import prepare_all; prepare_all()"
```

### Run the Engine

```bash
python run_engine.py                              # Default: 10/batch, 1000 generations
python run_engine.py --batch-size 20              # Larger batches
python run_engine.py --max-generations 50         # Quick test
python run_engine.py --mutation-intensity 0.4     # More exploration
```

Press `Ctrl+C` to stop gracefully. The engine finishes the current strategy and prints a final report.

### View Dashboard

```bash
python dashboard.py
# Open http://localhost:5000
```

## Project Structure

```
autoresearch-trading/
├── core/
│   ├── __init__.py
│   ├── models.py           # Strategy database (SQLite)
│   ├── genesis.py          # Strategy generation engine
│   ├── darwin.py           # Evaluation, ranking, pruning
│   ├── healer.py           # Self-healing error correction
│   ├── learner.py          # Continuous learning module
│   ├── risk_manager.py     # Advanced risk management
│   ├── data_pipeline.py    # Data ingestion (OHLCV, sentiment, macro)
│   └── orchestrator.py     # Main autonomous loop
├── reports/                # Best strategy outputs
├── run_engine.py           # Entry point
├── dashboard.py            # Web dashboard
├── config.py               # Configuration
├── train.py                # Original training script
├── prepare.py              # Original data preparation
├── program.md              # Agent instructions
└── README.md               # This file
```

## How It Works

```
LOOP FOREVER:
  1. Genesis creates N strategies (random + mutated + crossed)
  2. Each strategy backtested on 2024 data (in-sample)
  3. Survivors forward-tested on 2025 data (out-of-sample)
  4. Overfit detection: IS vs OOS metric decay
  5. Darwin kills failures, explains why
  6. Healer tries to fix broken strategies
  7. Learner compiles insights for next generation
  8. Adaptive mutation: explore more if stuck, fine-tune if progressing
  9. Best strategy saved to reports/
  10. GOTO 1
```

---

*Built with zero human intervention in mind. Set it running and come back to a population of battle-tested strategies.*
