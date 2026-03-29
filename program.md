# Autoresearch Trading Agent

You are managing a dual-mode trading research system running on an Android phone (12GB RAM).

## Modes

### Mode A: LLM Training (`MODE=llm`)
- Goal: Minimize `val_bpb` (validation bits per byte) on price sequences
- Edit: `train.py` TradingTransformer class (architecture changes)
- Constraints: Keep parameters < 5M (mobile RAM limit)
- Good ideas: Try different attention mechanisms, RoPE, layer counts, activation functions

### Mode B: Strategy Evolution (`MODE=strategy`)
- Goal: Maximize Sharpe ratio in `strategies/base_strategy.py`
- Edit: Indicator logic, entry/exit rules, risk management
- Must maintain function signature: `def strategy(df): return position (-1, 0, 1)`
- Good ideas: RSI, MACD, Bollinger Bands, volatility targeting, stop losses

## Workflow
1. Check current mode in `config.py`
2. Read previous results in `results/` to see what worked
3. Make ONE meaningful change
4. Run `python3 train.py` (5-min time budget enforced)
5. Check `results/exp_*.json` for metrics
6. If improved, keep change. If not, revert.
7. Alternate modes every 3 experiments for diversification

## Mobile Constraints
- DO NOT increase batch size above 8 (OOM risk)
- DO NOT increase seq_len above 512
- If OOM occurs, reduce DEPTH or DIM immediately
- Flask dashboard runs at localhost:5000 - check it to see rankings
