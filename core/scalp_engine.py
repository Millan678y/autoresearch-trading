"""
SCALPING BACKTEST ENGINE — Optimized for 5-Minute Timeframe

Key differences from the hourly engine:
1. Tighter execution model (spread, slippage, partial fills)
2. Faster indicator computations (6-bar RSI, micro-vol)
3. Scalp-specific metrics (avg trade duration, profit per trade)
4. Taker buy/sell ratio as first-class signal
5. Session-aware (Asian, London, NY sessions)
6. Binance fee structure (0.1% spot / 0.02-0.04% futures)

Usage:
    from core.scalp_engine import ScalpBacktester
    bt = ScalpBacktester(symbols=["BTCUSDT"], interval="5m")
    result = bt.run(strategy_code, split="train")
"""

import os
import sys
import time
import math
import importlib
import importlib.util
import tempfile
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .binance_data import BinanceDataLoader, compute_scalp_features, INTERVAL_MS


# ─────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────

INITIAL_CAPITAL = 100.0        # $100 scalping account
MAKER_FEE = 0.0002             # 0.02% futures maker
TAKER_FEE = 0.0004             # 0.04% futures taker
SLIPPAGE_BPS = 2.0             # 2 bps slippage (tighter on 5m)
MAX_LEVERAGE = 10              # Conservative for scalping
LOOKBACK_BARS = 200            # History buffer for strategy
TIME_BUDGET = 180              # 3 minutes max per backtest

BARS_PER_YEAR_5M = 365.25 * 24 * 12  # ~105,120 five-minute bars/year

# Session definitions (UTC hours)
SESSIONS = {
    "asian":  (0, 8),    # 00:00 - 08:00 UTC
    "london": (8, 16),   # 08:00 - 16:00 UTC
    "ny":     (13, 21),  # 13:00 - 21:00 UTC (overlap 13-16)
}


# ─────────────────────────────────────────────────────────────────
# Data Types
# ─────────────────────────────────────────────────────────────────

@dataclass
class ScalpBar:
    """Single bar data provided to scalping strategy."""
    symbol: str
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    trades: int
    taker_buy_ratio: float    # 0-1, >0.5 = more buying
    session: str              # "asian", "london", "ny"
    history: pd.DataFrame     # Last LOOKBACK_BARS with computed features


@dataclass
class ScalpSignal:
    """Trading signal from scalping strategy."""
    symbol: str
    action: str              # "long", "short", "close", "none"
    size_pct: float = 1.0    # Position size as fraction of max (0-1)
    take_profit: float = 0.0 # TP price (0 = no TP)
    stop_loss: float = 0.0   # SL price (0 = no SL)
    reason: str = ""


@dataclass
class ScalpPosition:
    """Active position."""
    symbol: str
    direction: int           # 1 = long, -1 = short
    entry_price: float
    size_usd: float
    entry_bar: int
    take_profit: float = 0.0
    stop_loss: float = 0.0


@dataclass
class ScalpResult:
    """Backtest results."""
    score: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    total_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    num_trades: int = 0
    win_rate_pct: float = 0.0
    profit_factor: float = 0.0
    avg_trade_pnl: float = 0.0
    avg_trade_duration_bars: float = 0.0
    avg_winner: float = 0.0
    avg_loser: float = 0.0
    max_consecutive_losses: int = 0
    profit_per_trade_bps: float = 0.0
    annual_turnover: float = 0.0
    backtest_seconds: float = 0.0
    equity_curve: list = field(default_factory=list)
    trade_log: list = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────
# Session Detection
# ─────────────────────────────────────────────────────────────────

def _get_session(timestamp_ms: int) -> str:
    """Determine trading session from timestamp."""
    hour = (timestamp_ms // 3_600_000) % 24
    
    # Check overlapping sessions (NY gets priority during overlap)
    if 13 <= hour < 21:
        return "ny"
    elif 8 <= hour < 16:
        return "london"
    else:
        return "asian"


# ─────────────────────────────────────────────────────────────────
# Scalping Backtester
# ─────────────────────────────────────────────────────────────────

class ScalpBacktester:
    """
    Backtesting engine optimized for 5-minute scalping.
    
    Usage:
        bt = ScalpBacktester(symbols=["BTCUSDT"])
        result = bt.run(strategy_code, split="train")
    """
    
    def __init__(self, symbols: List[str] = None, interval: str = "5m",
                 initial_capital: float = INITIAL_CAPITAL,
                 max_leverage: int = MAX_LEVERAGE):
        self.symbols = symbols or ["BTCUSDT"]
        self.interval = interval
        self.initial_capital = initial_capital
        self.max_leverage = max_leverage
        self.loader = BinanceDataLoader(symbols=self.symbols, interval=interval)
    
    def run(self, strategy_code: str, split: str = "train") -> ScalpResult:
        """
        Run a backtest with the given strategy code.
        
        The strategy must implement:
            class ScalpStrategy:
                def __init__(self): ...
                def on_bar(self, bar: ScalpBar, position: Optional[ScalpPosition],
                          equity: float) -> ScalpSignal: ...
        """
        t_start = time.time()
        
        # Load data
        data = {}
        for symbol in self.symbols:
            df = self.loader.load(symbol, split=split)
            if len(df) > 0:
                df = compute_scalp_features(df)
                data[symbol] = df
        
        if not data:
            return ScalpResult(score=-999)
        
        # Load strategy
        try:
            strategy = self._load_strategy(strategy_code)
        except Exception as e:
            return ScalpResult(score=-999, 
                             trade_log=[{"error": f"Strategy load error: {e}"}])
        
        # Run simulation
        return self._simulate(strategy, data, t_start)
    
    def _load_strategy(self, code: str):
        """Load strategy from code string."""
        tmp_dir = tempfile.mkdtemp()
        tmp_path = os.path.join(tmp_dir, "scalp_strategy.py")
        
        with open(tmp_path, "w") as f:
            f.write(code)
        
        spec = importlib.util.spec_from_file_location("scalp_strategy", tmp_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        
        strategy = mod.ScalpStrategy()
        
        # Cleanup
        try:
            os.unlink(tmp_path)
            os.rmdir(tmp_dir)
        except:
            pass
        
        return strategy
    
    def _simulate(self, strategy, data: Dict[str, pd.DataFrame],
                  t_start: float) -> ScalpResult:
        """Core simulation loop."""
        
        # Build unified timeline
        all_timestamps = set()
        for symbol, df in data.items():
            all_timestamps.update(df["timestamp"].tolist())
        timestamps = sorted(all_timestamps)
        
        if not timestamps:
            return ScalpResult(score=-999)
        
        # Index data
        indexed = {}
        for symbol, df in data.items():
            indexed[symbol] = df.set_index("timestamp")
        
        # State
        cash = self.initial_capital
        positions: Dict[str, ScalpPosition] = {}
        equity_curve = [self.initial_capital]
        trade_log = []
        total_volume = 0.0
        prev_equity = self.initial_capital
        bar_returns = []
        
        # History buffers
        history_bufs = {symbol: [] for symbol in data}
        bar_count = 0
        
        for ts in timestamps:
            # Time budget check
            if time.time() - t_start > TIME_BUDGET:
                break
            
            bar_count += 1
            session = _get_session(ts)
            
            for symbol in data:
                if symbol not in indexed or ts not in indexed[symbol].index:
                    continue
                
                row = indexed[symbol].loc[ts]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                
                # Build history
                bar_dict = row.to_dict()
                bar_dict["timestamp"] = ts
                history_bufs[symbol].append(bar_dict)
                if len(history_bufs[symbol]) > LOOKBACK_BARS:
                    history_bufs[symbol] = history_bufs[symbol][-LOOKBACK_BARS:]
                
                if len(history_bufs[symbol]) < 50:
                    continue
                
                hist_df = pd.DataFrame(history_bufs[symbol])
                
                # Build ScalpBar
                taker_buy_ratio = float(row.get("taker_buy_ratio", 0.5)) if "taker_buy_ratio" in row.index else 0.5
                
                bar = ScalpBar(
                    symbol=symbol,
                    timestamp=ts,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                    trades=int(row.get("trades", 0)),
                    taker_buy_ratio=taker_buy_ratio,
                    session=session,
                    history=hist_df,
                )
                
                current_pos = positions.get(symbol)
                current_price = bar.close
                
                # ── Check TP/SL on existing positions ──
                if current_pos:
                    hit_tp = False
                    hit_sl = False
                    
                    if current_pos.direction == 1:  # Long
                        if current_pos.take_profit > 0 and bar.high >= current_pos.take_profit:
                            hit_tp = True
                        if current_pos.stop_loss > 0 and bar.low <= current_pos.stop_loss:
                            hit_sl = True
                    else:  # Short
                        if current_pos.take_profit > 0 and bar.low <= current_pos.take_profit:
                            hit_tp = True
                        if current_pos.stop_loss > 0 and bar.high >= current_pos.stop_loss:
                            hit_sl = True
                    
                    if hit_sl:
                        # SL hit (worse execution assumed)
                        exit_price = current_pos.stop_loss
                        pnl = self._calc_pnl(current_pos, exit_price)
                        cash += current_pos.size_usd + pnl - abs(current_pos.size_usd) * TAKER_FEE
                        total_volume += abs(current_pos.size_usd)
                        
                        duration = bar_count - current_pos.entry_bar
                        trade_log.append({
                            "symbol": symbol, "direction": current_pos.direction,
                            "entry": current_pos.entry_price, "exit": exit_price,
                            "pnl": pnl, "reason": "stop_loss",
                            "duration_bars": duration,
                        })
                        del positions[symbol]
                        current_pos = None
                    
                    elif hit_tp:
                        exit_price = current_pos.take_profit
                        pnl = self._calc_pnl(current_pos, exit_price)
                        cash += current_pos.size_usd + pnl - abs(current_pos.size_usd) * MAKER_FEE
                        total_volume += abs(current_pos.size_usd)
                        
                        duration = bar_count - current_pos.entry_bar
                        trade_log.append({
                            "symbol": symbol, "direction": current_pos.direction,
                            "entry": current_pos.entry_price, "exit": exit_price,
                            "pnl": pnl, "reason": "take_profit",
                            "duration_bars": duration,
                        })
                        del positions[symbol]
                        current_pos = None
                
                # ── Get strategy signal ──
                equity = self._calc_equity(cash, positions, {s: indexed[s].loc[ts]["close"] 
                                           for s in data if s in indexed and ts in indexed[s].index})
                
                try:
                    signal = strategy.on_bar(bar, current_pos, equity)
                except Exception:
                    signal = ScalpSignal(symbol=symbol, action="none")
                
                if signal is None or signal.action == "none":
                    continue
                
                # ── Execute signal ──
                if signal.action == "close" and current_pos:
                    exit_price = current_price * (1 - SLIPPAGE_BPS / 10000 * current_pos.direction)
                    pnl = self._calc_pnl(current_pos, exit_price)
                    cash += current_pos.size_usd + pnl - abs(current_pos.size_usd) * TAKER_FEE
                    total_volume += abs(current_pos.size_usd)
                    
                    duration = bar_count - current_pos.entry_bar
                    trade_log.append({
                        "symbol": symbol, "direction": current_pos.direction,
                        "entry": current_pos.entry_price, "exit": exit_price,
                        "pnl": pnl, "reason": signal.reason or "signal_close",
                        "duration_bars": duration,
                    })
                    del positions[symbol]
                
                elif signal.action in ("long", "short") and not current_pos:
                    direction = 1 if signal.action == "long" else -1
                    
                    # Position sizing
                    max_size = equity * self.max_leverage * 0.1  # 10% of max leverage
                    size = max_size * np.clip(signal.size_pct, 0.1, 1.0)
                    size = min(size, cash * 0.95)  # Keep some cash
                    
                    if size < 5:  # Minimum $5 position
                        continue
                    
                    entry_price = current_price * (1 + SLIPPAGE_BPS / 10000 * direction)
                    fee = size * TAKER_FEE
                    cash -= size + fee
                    total_volume += size
                    
                    positions[symbol] = ScalpPosition(
                        symbol=symbol,
                        direction=direction,
                        entry_price=entry_price,
                        size_usd=size,
                        entry_bar=bar_count,
                        take_profit=signal.take_profit,
                        stop_loss=signal.stop_loss,
                    )
                
                elif signal.action in ("long", "short") and current_pos:
                    new_dir = 1 if signal.action == "long" else -1
                    if new_dir != current_pos.direction:
                        # Flip position: close current, open new
                        exit_price = current_price * (1 - SLIPPAGE_BPS / 10000 * current_pos.direction)
                        pnl = self._calc_pnl(current_pos, exit_price)
                        cash += current_pos.size_usd + pnl - abs(current_pos.size_usd) * TAKER_FEE
                        total_volume += abs(current_pos.size_usd)
                        
                        duration = bar_count - current_pos.entry_bar
                        trade_log.append({
                            "symbol": symbol, "direction": current_pos.direction,
                            "entry": current_pos.entry_price, "exit": exit_price,
                            "pnl": pnl, "reason": "flip",
                            "duration_bars": duration,
                        })
                        
                        # Open reverse
                        equity = self._calc_equity(cash, {}, {})
                        max_size = equity * self.max_leverage * 0.1
                        size = max_size * np.clip(signal.size_pct, 0.1, 1.0)
                        size = min(size, cash * 0.95)
                        
                        if size >= 5:
                            entry_price = current_price * (1 + SLIPPAGE_BPS / 10000 * new_dir)
                            cash -= size + size * TAKER_FEE
                            total_volume += size
                            
                            positions[symbol] = ScalpPosition(
                                symbol=symbol, direction=new_dir,
                                entry_price=entry_price, size_usd=size,
                                entry_bar=bar_count,
                                take_profit=signal.take_profit,
                                stop_loss=signal.stop_loss,
                            )
                        else:
                            positions.pop(symbol, None)
            
            # ── Update equity curve ──
            prices = {}
            for s in data:
                if s in indexed and ts in indexed[s].index:
                    prices[s] = float(indexed[s].loc[ts]["close"])
            
            current_equity = self._calc_equity(cash, positions, prices)
            equity_curve.append(current_equity)
            
            if prev_equity > 0:
                bar_returns.append((current_equity - prev_equity) / prev_equity)
            prev_equity = current_equity
            
            # Liquidation check
            if current_equity < self.initial_capital * 0.05:
                break
        
        t_end = time.time()
        
        # ── Compute metrics ──
        return self._compute_metrics(
            equity_curve, bar_returns, trade_log,
            total_volume, timestamps, t_end - t_start
        )
    
    def _calc_pnl(self, pos: ScalpPosition, exit_price: float) -> float:
        """Calculate PnL for a position."""
        price_change = (exit_price - pos.entry_price) / pos.entry_price
        return pos.size_usd * price_change * pos.direction
    
    def _calc_equity(self, cash: float, positions: Dict[str, ScalpPosition],
                     prices: Dict[str, float]) -> float:
        """Calculate total equity."""
        equity = cash
        for symbol, pos in positions.items():
            price = prices.get(symbol, pos.entry_price)
            pnl = self._calc_pnl(pos, price)
            equity += pos.size_usd + pnl
        return equity
    
    def _compute_metrics(self, equity_curve, bar_returns, trade_log,
                         total_volume, timestamps, elapsed) -> ScalpResult:
        """Compute all scalping metrics."""
        returns = np.array(bar_returns) if bar_returns else np.array([0.0])
        eq = np.array(equity_curve)
        
        # Sharpe (annualized from 5-min bars)
        if returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(BARS_PER_YEAR_5M)
        else:
            sharpe = 0.0
        
        # Sortino
        downside = returns[returns < 0]
        if len(downside) > 0 and np.std(downside) > 0:
            sortino = (returns.mean() / np.std(downside)) * np.sqrt(BARS_PER_YEAR_5M)
        else:
            sortino = 0.0
        
        # Total return
        final = eq[-1] if len(eq) > 0 else self.initial_capital
        total_return_pct = (final - self.initial_capital) / self.initial_capital * 100
        
        # Max drawdown
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / np.where(peak > 0, peak, 1)
        max_dd_pct = float(dd.max() * 100)
        
        # Trade metrics
        pnls = [t["pnl"] for t in trade_log]
        num_trades = len(trade_log)
        
        if pnls:
            winners = [p for p in pnls if p > 0]
            losers = [p for p in pnls if p < 0]
            
            win_rate = len(winners) / len(pnls) * 100
            avg_pnl = np.mean(pnls)
            avg_winner = np.mean(winners) if winners else 0
            avg_loser = np.mean(losers) if losers else 0
            
            gross_profit = sum(winners) if winners else 0
            gross_loss = abs(sum(losers)) if losers else 1e-10
            profit_factor = gross_profit / gross_loss
            
            # Profit per trade in bps
            entries = [t.get("entry", 0) for t in trade_log]
            if entries and np.mean(entries) > 0:
                profit_per_trade_bps = avg_pnl / np.mean(entries) * 10000
            else:
                profit_per_trade_bps = 0
            
            # Average trade duration
            durations = [t.get("duration_bars", 0) for t in trade_log]
            avg_duration = np.mean(durations) if durations else 0
            
            # Max consecutive losses
            max_consec = 0
            current_consec = 0
            for p in pnls:
                if p < 0:
                    current_consec += 1
                    max_consec = max(max_consec, current_consec)
                else:
                    current_consec = 0
        else:
            win_rate = avg_pnl = avg_winner = avg_loser = 0
            profit_factor = profit_per_trade_bps = avg_duration = 0
            max_consec = 0
        
        # Turnover
        data_bars = len(timestamps)
        if data_bars > 0:
            annual_turnover = total_volume * (BARS_PER_YEAR_5M / data_bars)
        else:
            annual_turnover = 0
        
        # ── Score ──
        score = self._compute_score(
            sharpe, num_trades, max_dd_pct, total_return_pct,
            annual_turnover, final
        )
        
        return ScalpResult(
            score=score,
            sharpe=sharpe,
            sortino=sortino,
            total_return_pct=total_return_pct,
            max_drawdown_pct=max_dd_pct,
            num_trades=num_trades,
            win_rate_pct=win_rate,
            profit_factor=profit_factor,
            avg_trade_pnl=avg_pnl,
            avg_trade_duration_bars=avg_duration,
            avg_winner=avg_winner,
            avg_loser=avg_loser,
            max_consecutive_losses=max_consec,
            profit_per_trade_bps=profit_per_trade_bps,
            annual_turnover=annual_turnover,
            backtest_seconds=elapsed,
            equity_curve=equity_curve,
            trade_log=trade_log,
        )
    
    def _compute_score(self, sharpe, num_trades, max_dd_pct,
                       total_return_pct, annual_turnover, final_equity) -> float:
        """
        Scalping-specific scoring.
        
        Scalping needs MORE trades than swing trading, so:
        - Full credit at 200+ trades (vs 50 for hourly)
        - Tighter drawdown tolerance (10% vs 15%)
        - Turnover penalty threshold higher (scalping = high turnover by nature)
        """
        # Hard cutoffs
        if num_trades < 30:
            return -999.0
        if max_dd_pct > 40.0:
            return -999.0
        if final_equity < self.initial_capital * 0.6:
            return -999.0
        
        # Trade count factor (full credit at 200+)
        trade_factor = min(num_trades / 200.0, 1.0)
        
        # Drawdown penalty (stricter for scalping)
        dd_penalty = max(0, max_dd_pct - 10.0) * 0.08
        
        # Turnover penalty (higher threshold for scalping)
        turnover_ratio = annual_turnover / self.initial_capital if self.initial_capital > 0 else 0
        turnover_penalty = max(0, turnover_ratio - 2000) * 0.0005
        
        score = sharpe * math.sqrt(trade_factor) - dd_penalty - turnover_penalty
        return score


# ─────────────────────────────────────────────────────────────────
# Strategy Template (for Genesis to generate)
# ─────────────────────────────────────────────────────────────────

SCALP_STRATEGY_TEMPLATE = '''
"""Auto-generated scalping strategy."""

import numpy as np

class ScalpStrategy:
    def __init__(self):
        self.bar_count = 0
        self.last_trade_bar = -999
    
    def on_bar(self, bar, position, equity):
        """
        Args:
            bar: ScalpBar with .close, .high, .low, .volume, .taker_buy_ratio,
                 .session, .history (DataFrame with computed features)
            position: ScalpPosition or None
            equity: Current equity float
        
        Returns:
            ScalpSignal(symbol, action, size_pct, take_profit, stop_loss, reason)
        """
        from core.scalp_engine import ScalpSignal
        
        self.bar_count += 1
        h = bar.history
        
        if len(h) < 50:
            return ScalpSignal(symbol=bar.symbol, action="none")
        
        # Cooldown
        if self.bar_count - self.last_trade_bar < {cooldown}:
            return ScalpSignal(symbol=bar.symbol, action="none")
        
        closes = h["close"].values.astype(float)
        highs = h["high"].values.astype(float)
        lows = h["low"].values.astype(float)
        
        # === SIGNALS ===
        {signal_code}
        
        # === POSITION MANAGEMENT ===
        if position is not None:
            # Exit logic
            {exit_code}
            return ScalpSignal(symbol=bar.symbol, action="none")
        
        # === ENTRY LOGIC ===
        atr = np.mean(np.maximum(highs[-12:] - lows[-12:],
                      np.maximum(np.abs(highs[-12:] - closes[-13:-1]),
                                np.abs(lows[-12:] - closes[-13:-1]))))
        
        if bull_votes >= {min_votes}:
            tp = bar.close + atr * {tp_mult}
            sl = bar.close - atr * {sl_mult}
            self.last_trade_bar = self.bar_count
            return ScalpSignal(
                symbol=bar.symbol, action="long",
                size_pct={size_pct},
                take_profit=tp, stop_loss=sl,
                reason="bull_signal"
            )
        
        elif bear_votes >= {min_votes}:
            tp = bar.close - atr * {tp_mult}
            sl = bar.close + atr * {sl_mult}
            self.last_trade_bar = self.bar_count
            return ScalpSignal(
                symbol=bar.symbol, action="short",
                size_pct={size_pct},
                take_profit=tp, stop_loss=sl,
                reason="bear_signal"
            )
        
        return ScalpSignal(symbol=bar.symbol, action="none")
'''


if __name__ == "__main__":
    # Quick test
    bt = ScalpBacktester(symbols=["BTCUSDT"], interval="5m")
    
    test_strategy = '''
import numpy as np
from core.scalp_engine import ScalpSignal

class ScalpStrategy:
    def __init__(self):
        self.bar_count = 0
        self.last_trade_bar = -999
    
    def on_bar(self, bar, position, equity):
        self.bar_count += 1
        h = bar.history
        if len(h) < 50:
            return ScalpSignal(symbol=bar.symbol, action="none")
        
        closes = h["close"].values.astype(float)
        
        # Simple EMA cross
        ema_fast = closes[-9:].mean()
        ema_slow = closes[-21:].mean()
        rsi = 50  # placeholder
        
        if self.bar_count - self.last_trade_bar < 6:
            return ScalpSignal(symbol=bar.symbol, action="none")
        
        if position is not None:
            return ScalpSignal(symbol=bar.symbol, action="none")
        
        if ema_fast > ema_slow:
            self.last_trade_bar = self.bar_count
            return ScalpSignal(symbol=bar.symbol, action="long", size_pct=0.5,
                             take_profit=bar.close * 1.003, stop_loss=bar.close * 0.998)
        elif ema_fast < ema_slow:
            self.last_trade_bar = self.bar_count
            return ScalpSignal(symbol=bar.symbol, action="short", size_pct=0.5,
                             take_profit=bar.close * 0.997, stop_loss=bar.close * 1.002)
        
        return ScalpSignal(symbol=bar.symbol, action="none")
'''
    
    print("Running scalp backtest...")
    result = bt.run(test_strategy, split="train")
    
    print(f"\nScore:        {result.score:.4f}")
    print(f"Sharpe:       {result.sharpe:.4f}")
    print(f"Return:       {result.total_return_pct:.2f}%")
    print(f"Max DD:       {result.max_drawdown_pct:.2f}%")
    print(f"Trades:       {result.num_trades}")
    print(f"Win Rate:     {result.win_rate_pct:.1f}%")
    print(f"Avg Duration: {result.avg_trade_duration_bars:.1f} bars")
    print(f"Time:         {result.backtest_seconds:.1f}s")
