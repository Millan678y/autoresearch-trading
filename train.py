import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import json
import os
import sys
import importlib.util
from pathlib import Path

from config import *

# ==================== LLM MODE ====================
class TradingTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, DIM)
        self.pos = nn.Embedding(MAX_SEQ_LEN, DIM)
        
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(DIM, NUM_HEADS, DIM*4, dropout=0.1, batch_first=True)
            for _ in range(DEPTH)
        ])
        
        self.ln_f = nn.LayerNorm(DIM)
        self.head = nn.Linear(DIM, VOCAB_SIZE, bias=False)
        
    def forward(self, x):
        b, t = x.size()
        pos = torch.arange(0, t, device=x.device).unsqueeze(0)
        
        x = self.emb(x) + self.pos(pos)
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        return self.head(x)

def train_llm():
    """Standard LLM training on price sequences"""
    # Load data
    with open("data/train.txt", "r") as f:
        text = f.read()
    
    # Simple char-level/token-level encoding for demo
    # In production, use the discretized bins
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    
    def encode(s): return [stoi.get(c, 0) for c in s]
    
    data = torch.tensor(encode(text), dtype=torch.long)
    
    model = TradingTransformer().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    start_time = time.time()
    best_loss = float('inf')
    
    while time.time() - start_time < TIME_BUDGET:
        # Sample batch
        ix = torch.randint(len(data) - MAX_SEQ_LEN, (DEVICE_BATCH_SIZE,))
        x = torch.stack([data[i:i+MAX_SEQ_LEN] for i in ix])
        y = torch.stack([data[i+1:i+MAX_SEQ_LEN+1] for i in ix])
        
        x, y = x.to(DEVICE), y.to(DEVICE)
        
        logits = model(x)
        loss = nn.functional.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), "best_model.pt")
    
    return {"val_bpb": best_loss, "mode": "llm"}

# ==================== STRATEGY MODE ====================
def backtest_strategy(strategy_code, df):
    """Vectorized backtest with Sharpe calculation"""
    try:
        # Write temporary strategy file
        with open("strategies/temp_strategy.py", "w") as f:
            f.write(strategy_code)
        
        # Import dynamically
        spec = importlib.util.spec_from_file_location("temp_strategy", "strategies/temp_strategy.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        
        # Vectorized backtest
        positions = df.apply(lambda row: mod.strategy(df.loc[:row.name]), axis=1)
        returns = df['Close'].pct_change().shift(-1) * positions
        
        sharpe = returns.mean() / (returns.std() + 1e-9) * np.sqrt(252)
        total_return = (1 + returns).prod() - 1
        
        return {
            "sharpe": float(sharpe),
            "return": float(total_return),
            "trades": int((positions.diff() != 0).sum())
        }
    except Exception as e:
        return {"sharpe": -999, "return": -1, "error": str(e)}

def train_strategy():
    """Generate and evaluate strategy variations"""
    df = pd.read_csv("data/prices.csv", index_col=0, parse_dates=True)
    
    # Load current best strategy
    with open("strategies/base_strategy.py", "r") as f:
        base_code = f.read()
    
    # Evaluate baseline
    baseline_metrics = backtest_strategy(base_code, df)
    
    # Agent would modify strategy here - for demo we just mutate parameters
    # In autoresearch, the LLM edits this file directly
    results = {
        "baseline": baseline_metrics,
        "experiment_time": TIME_BUDGET,
        "mode": "strategy"
    }
    
    return results

# ==================== MAIN ====================
if __name__ == "__main__":
    if MODE == "llm":
        results = train_llm()
    else:
        results = train_strategy()
    
    # Save results for dashboard
    exp_id = int(time.time())
    with open(f"results/exp_{exp_id}.json", "w") as f:
        json.dump(results, f)
    
    print(json.dumps(results))
