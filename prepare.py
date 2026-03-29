import yfinance as yf
import numpy as np
import json
import os
from pathlib import Path
import pickle

def download_data(symbol="BTC-USD", period="1y", interval="1h"):
    """Download OHLCV data from Yahoo Finance"""
    print(f"Downloading {symbol}...")
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    
    # Create text representation for LLM mode
    text_data = []
    for idx, row in df.iterrows():
        # Format: "OPEN:42050 HIGH:42120 LOW:41980 CLOSE:42080 VOL:15000000"
        bar = f"O:{row['Open']:.2f} H:{row['High']:.2f} L:{row['Low']:.2f} C:{row['Close']:.2f} V:{row['Volume']:.0f}\n"
        text_data.append(bar)
    
    # Save raw for LLM
    with open("data/train.txt", "w") as f:
        f.write("".join(text_data))
    
    # Save structured for strategy mode
    df.to_csv("data/prices.csv")
    
    # Simple price discretization tokenizer (0-1023 bins)
    returns = df['Close'].pct_change().dropna().values
    bins = np.linspace(returns.min(), returns.max(), VOCAB_SIZE-256)  # Reserve 256 for special tokens
    np.save("data/price_bins.npy", bins)
    
    return df

def prepare():
    os.makedirs("data", exist_ok=True)
    os.makedirs("strategies", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    if not os.path.exists("data/train.txt"):
        download_data()
    
    # Create strategy template
    if not os.path.exists("strategies/base_strategy.py"):
        with open("strategies/base_strategy.py", "w") as f:
            f.write("""import pandas as pd
import numpy as np

def strategy(df):
    '''
    Returns: position (1=long, -1=short, 0=flat)
    '''
    # Default: Simple MA crossover
    df['ma_fast'] = df['Close'].rolling(10).mean()
    df['ma_slow'] = df['Close'].rolling(30).mean()
    
    if df['ma_fast'].iloc[-1] > df['ma_slow'].iloc[-1]:
        return 1
    elif df['ma_fast'].iloc[-1] < df['ma_slow'].iloc[-1]:
        return -1
    return 0
""")

if __name__ == "__main__":
    from config import VOCAB_SIZE
    prepare()
