"""
ADVANCED RISK MANAGER — Adaptive Hybrid Stop-Loss System

Implements:
1. Hybrid trailing stops (ATR + volatility + time-decay)
2. Portfolio-level risk limits (max exposure, correlation caps)
3. Drawdown circuit breakers (reduce/halt trading on DD)
4. Vol-regime adaptive position sizing
5. Kelly criterion position sizing
6. Intraday risk checks

This module is used BY strategies (injected into generated code)
and BY the orchestrator (portfolio-level checks).
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────

@dataclass
class RiskConfig:
    """Risk management configuration. Strategies can override defaults."""
    # Position sizing
    max_position_pct: float = 0.15          # Max single position as % of equity
    max_total_exposure_pct: float = 0.60    # Max total exposure as % of equity
    min_position_usd: float = 100.0         # Minimum position size
    
    # Stop-loss
    atr_stop_mult: float = 5.5             # ATR multiplier for trailing stop
    vol_stop_base: float = 0.03            # Base stop distance for vol-adaptive
    time_decay_start: int = 48             # Hours before time-decay tightens stop
    time_decay_factor: float = 0.95        # Stop tightens by this factor per bar after start
    
    # Circuit breakers
    dd_warning_pct: float = 5.0            # Warning threshold
    dd_reduce_pct: float = 10.0            # Reduce position sizes by 50%
    dd_halt_pct: float = 20.0              # Halt all trading
    dd_recovery_bars: int = 24             # Bars to wait after halt before resuming
    
    # Vol regime
    vol_lookback: int = 36                 # Bars for vol calculation
    target_vol: float = 0.015              # Target annualized vol
    vol_scale_min: float = 0.3             # Minimum vol scale factor
    vol_scale_max: float = 2.0             # Maximum vol scale factor
    
    # Correlation
    max_correlated_exposure: float = 0.40  # Max exposure to correlated assets
    correlation_threshold: float = 0.7     # Assets above this are "correlated"


# ─────────────────────────────────────────────────────────────────
# Hybrid Trailing Stop
# ─────────────────────────────────────────────────────────────────

class HybridTrailingStop:
    """
    Combines three stop-loss mechanisms:
    1. ATR-based trailing stop (adapts to asset volatility)
    2. Percentage trailing from peak/trough
    3. Time-decay tightening (the longer you hold, the tighter the stop)
    
    The TIGHTEST of the three wins at any given moment.
    """
    
    def __init__(self, config: RiskConfig = None):
        self.config = config or RiskConfig()
        self.positions: Dict[str, dict] = {}
    
    def open_position(self, symbol: str, direction: int, entry_price: float,
                      atr: float, current_vol: float):
        """Register a new position for stop tracking."""
        self.positions[symbol] = {
            "direction": direction,  # 1 = long, -1 = short
            "entry_price": entry_price,
            "peak_price": entry_price,
            "trough_price": entry_price,
            "atr_at_entry": atr,
            "vol_at_entry": current_vol,
            "bars_held": 0,
            "initial_stop_distance": atr * self.config.atr_stop_mult,
        }
    
    def update(self, symbol: str, current_price: float,
               current_atr: float, current_vol: float) -> Tuple[bool, float, str]:
        """
        Update stop for a position.
        
        Returns: (should_exit, stop_price, reason)
        """
        if symbol not in self.positions:
            return False, 0.0, ""
        
        pos = self.positions[symbol]
        pos["bars_held"] += 1
        direction = pos["direction"]
        
        # Update peak/trough
        if direction == 1:  # Long
            pos["peak_price"] = max(pos["peak_price"], current_price)
            reference = pos["peak_price"]
        else:  # Short
            pos["trough_price"] = min(pos["trough_price"], current_price)
            reference = pos["trough_price"]
        
        # ── Stop 1: ATR trailing ──
        atr_distance = current_atr * self.config.atr_stop_mult
        if direction == 1:
            atr_stop = reference - atr_distance
        else:
            atr_stop = reference + atr_distance
        
        # ── Stop 2: Volatility-adaptive percentage ──
        vol_ratio = current_vol / max(self.config.target_vol, 1e-10)
        # In high vol: wider stops. In low vol: tighter.
        vol_pct = self.config.vol_stop_base * np.clip(vol_ratio, 0.5, 3.0)
        if direction == 1:
            vol_stop = reference * (1 - vol_pct)
        else:
            vol_stop = reference * (1 + vol_pct)
        
        # ── Stop 3: Time-decay tightening ──
        bars = pos["bars_held"]
        if bars > self.config.time_decay_start:
            decay_bars = bars - self.config.time_decay_start
            decay_mult = self.config.time_decay_factor ** decay_bars
            initial_dist = pos["initial_stop_distance"]
            decayed_dist = initial_dist * max(decay_mult, 0.3)  # Floor at 30%
            
            if direction == 1:
                time_stop = reference - decayed_dist
            else:
                time_stop = reference + decayed_dist
        else:
            time_stop = atr_stop  # No time decay yet
        
        # ── Pick the TIGHTEST stop ──
        if direction == 1:
            # For longs, tightest = highest stop price
            stop_price = max(atr_stop, vol_stop, time_stop)
            triggered = current_price <= stop_price
        else:
            # For shorts, tightest = lowest stop price
            stop_price = min(atr_stop, vol_stop, time_stop)
            triggered = current_price >= stop_price
        
        # Determine which stop triggered
        reason = ""
        if triggered:
            if direction == 1:
                if current_price <= time_stop and bars > self.config.time_decay_start:
                    reason = f"time_decay (held {bars} bars)"
                elif current_price <= vol_stop:
                    reason = f"vol_adaptive (vol={current_vol:.4f})"
                else:
                    reason = f"atr_trailing (atr={current_atr:.2f})"
            else:
                if current_price >= time_stop and bars > self.config.time_decay_start:
                    reason = f"time_decay (held {bars} bars)"
                elif current_price >= vol_stop:
                    reason = f"vol_adaptive (vol={current_vol:.4f})"
                else:
                    reason = f"atr_trailing (atr={current_atr:.2f})"
            
            del self.positions[symbol]
        
        return triggered, stop_price, reason
    
    def close_position(self, symbol: str):
        """Remove position from tracking."""
        self.positions.pop(symbol, None)


# ─────────────────────────────────────────────────────────────────
# Drawdown Circuit Breaker
# ─────────────────────────────────────────────────────────────────

class DrawdownCircuitBreaker:
    """
    Monitors portfolio drawdown and enforces risk limits.
    
    Levels:
    - Green: Normal trading
    - Yellow (warning): Log warning, reduce new position sizes
    - Orange (reduce): Cut all positions by 50%, reduce new sizes
    - Red (halt): Close all positions, halt trading for N bars
    """
    
    def __init__(self, config: RiskConfig = None):
        self.config = config or RiskConfig()
        self.peak_equity = 0.0
        self.halted = False
        self.halt_bar = 0
        self.current_bar = 0
    
    def update(self, equity: float) -> dict:
        """
        Update with current equity. Returns risk state.
        
        Returns {
            level: "green" | "yellow" | "orange" | "red",
            drawdown_pct: float,
            size_multiplier: float,  # 0.0 = halt, 0.5 = reduce, 1.0 = normal
            message: str
        }
        """
        self.current_bar += 1
        self.peak_equity = max(self.peak_equity, equity)
        
        if self.peak_equity <= 0:
            return {"level": "green", "drawdown_pct": 0, "size_multiplier": 1.0, "message": ""}
        
        dd_pct = (self.peak_equity - equity) / self.peak_equity * 100
        
        # Check if we're recovering from halt
        if self.halted:
            bars_since_halt = self.current_bar - self.halt_bar
            if bars_since_halt < self.config.dd_recovery_bars:
                return {
                    "level": "red",
                    "drawdown_pct": dd_pct,
                    "size_multiplier": 0.0,
                    "message": f"HALTED — recovery period ({bars_since_halt}/{self.config.dd_recovery_bars} bars)",
                }
            else:
                self.halted = False
                # Resume at reduced size
                return {
                    "level": "orange",
                    "drawdown_pct": dd_pct,
                    "size_multiplier": 0.5,
                    "message": "Resuming from halt at 50% size",
                }
        
        if dd_pct >= self.config.dd_halt_pct:
            self.halted = True
            self.halt_bar = self.current_bar
            return {
                "level": "red",
                "drawdown_pct": dd_pct,
                "size_multiplier": 0.0,
                "message": f"HALT: Drawdown {dd_pct:.1f}% exceeds {self.config.dd_halt_pct}%",
            }
        
        elif dd_pct >= self.config.dd_reduce_pct:
            return {
                "level": "orange",
                "drawdown_pct": dd_pct,
                "size_multiplier": 0.5,
                "message": f"REDUCE: Drawdown {dd_pct:.1f}% exceeds {self.config.dd_reduce_pct}%",
            }
        
        elif dd_pct >= self.config.dd_warning_pct:
            return {
                "level": "yellow",
                "drawdown_pct": dd_pct,
                "size_multiplier": 0.75,
                "message": f"WARNING: Drawdown {dd_pct:.1f}%",
            }
        
        return {
            "level": "green",
            "drawdown_pct": dd_pct,
            "size_multiplier": 1.0,
            "message": "",
        }


# ─────────────────────────────────────────────────────────────────
# Vol-Regime Position Sizer
# ─────────────────────────────────────────────────────────────────

class VolRegimePositionSizer:
    """
    Adaptive position sizing based on volatility regime.
    
    Low vol → larger positions (capture the trend)
    High vol → smaller positions (protect capital)
    Extreme vol → minimal positions
    
    Also implements inverse-vol targeting: size = target_risk / realized_vol
    """
    
    def __init__(self, config: RiskConfig = None):
        self.config = config or RiskConfig()
    
    def compute_vol_regime(self, closes: np.ndarray) -> Tuple[str, float]:
        """
        Classify current vol regime.
        Returns (regime_name, vol_value)
        """
        if len(closes) < self.config.vol_lookback:
            return "normal", self.config.target_vol
        
        log_rets = np.diff(np.log(closes[-self.config.vol_lookback:]))
        realized_vol = float(np.std(log_rets))
        
        # Classify
        ratio = realized_vol / self.config.target_vol
        
        if ratio < 0.5:
            return "low", realized_vol
        elif ratio < 1.5:
            return "normal", realized_vol
        elif ratio < 3.0:
            return "high", realized_vol
        else:
            return "extreme", realized_vol
    
    def compute_position_size(self, equity: float, closes: np.ndarray,
                               symbol_weight: float = 0.33,
                               dd_multiplier: float = 1.0) -> float:
        """
        Compute position size incorporating vol regime, drawdown state,
        and symbol weight.
        
        Returns USD notional position size.
        """
        regime, realized_vol = self.compute_vol_regime(closes)
        
        # Inverse vol targeting
        if realized_vol > 0:
            vol_scale = self.config.target_vol / realized_vol
        else:
            vol_scale = 1.0
        
        vol_scale = np.clip(vol_scale, self.config.vol_scale_min, self.config.vol_scale_max)
        
        # Base size
        base_size = equity * self.config.max_position_pct * symbol_weight
        
        # Apply all multipliers
        size = base_size * vol_scale * dd_multiplier
        
        # Clamp
        max_size = equity * self.config.max_position_pct
        size = max(self.config.min_position_usd, min(size, max_size))
        
        return size


# ─────────────────────────────────────────────────────────────────
# Kelly Criterion
# ─────────────────────────────────────────────────────────────────

def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float,
                   fraction: float = 0.25) -> float:
    """
    Fractional Kelly criterion for position sizing.
    
    Args:
        win_rate: Historical win rate (0-1)
        avg_win: Average winning trade return
        avg_loss: Average losing trade return (positive number)
        fraction: Kelly fraction (0.25 = quarter-Kelly, conservative)
    
    Returns: Recommended position fraction (0-1)
    """
    if avg_loss <= 0 or avg_win <= 0:
        return 0.0
    
    b = avg_win / avg_loss  # Win/loss ratio
    p = win_rate
    q = 1 - p
    
    # Kelly formula: f = (bp - q) / b
    kelly = (b * p - q) / b
    
    # Apply fraction (quarter-Kelly is standard for trading)
    kelly *= fraction
    
    # Clamp to reasonable range
    return float(np.clip(kelly, 0.0, 0.20))


# ─────────────────────────────────────────────────────────────────
# Portfolio-Level Risk Check
# ─────────────────────────────────────────────────────────────────

class PortfolioRiskManager:
    """
    Portfolio-level risk management overlay.
    Checks and enforces:
    - Total exposure limits
    - Correlated exposure limits
    - Per-asset position limits
    - Overall portfolio heat
    """
    
    def __init__(self, config: RiskConfig = None):
        self.config = config or RiskConfig()
        self.stop_system = HybridTrailingStop(config)
        self.circuit_breaker = DrawdownCircuitBreaker(config)
        self.position_sizer = VolRegimePositionSizer(config)
    
    def check_new_position(self, symbol: str, proposed_size: float,
                            current_positions: Dict[str, float],
                            equity: float,
                            correlations: Dict[Tuple[str, str], float] = None) -> Tuple[float, str]:
        """
        Validate a proposed new position.
        
        Returns (approved_size, reason)
        If approved_size < proposed, position was capped.
        If approved_size == 0, position was rejected.
        """
        if equity <= 0:
            return 0.0, "Zero equity"
        
        # Check total exposure
        total_exposure = sum(abs(v) for v in current_positions.values()) + abs(proposed_size)
        max_exposure = equity * self.config.max_total_exposure_pct
        
        if total_exposure > max_exposure:
            available = max_exposure - sum(abs(v) for v in current_positions.values())
            if available <= self.config.min_position_usd:
                return 0.0, f"Total exposure would exceed {self.config.max_total_exposure_pct*100:.0f}%"
            proposed_size = np.sign(proposed_size) * min(abs(proposed_size), available)
        
        # Check per-asset limit
        max_single = equity * self.config.max_position_pct
        if abs(proposed_size) > max_single:
            proposed_size = np.sign(proposed_size) * max_single
        
        # Check correlated exposure
        if correlations:
            for (sym_a, sym_b), corr in correlations.items():
                if corr > self.config.correlation_threshold:
                    # These assets are correlated
                    other_sym = sym_b if sym_a == symbol else sym_a if sym_b == symbol else None
                    if other_sym and other_sym in current_positions:
                        combined = abs(current_positions[other_sym]) + abs(proposed_size)
                        max_corr = equity * self.config.max_correlated_exposure
                        if combined > max_corr:
                            available = max_corr - abs(current_positions[other_sym])
                            if available <= self.config.min_position_usd:
                                return 0.0, f"Correlated exposure with {other_sym} exceeds limit"
                            proposed_size = np.sign(proposed_size) * min(abs(proposed_size), available)
        
        return proposed_size, "approved"
    
    def get_portfolio_heat(self, positions: Dict[str, float], equity: float) -> dict:
        """
        Compute portfolio heat — how much risk is on.
        
        Returns {
            total_exposure_pct, n_positions, largest_position_pct,
            heat_level: "cool" | "warm" | "hot" | "overheated"
        }
        """
        if equity <= 0:
            return {"total_exposure_pct": 0, "n_positions": 0,
                    "largest_position_pct": 0, "heat_level": "cool"}
        
        total = sum(abs(v) for v in positions.values())
        largest = max(abs(v) for v in positions.values()) if positions else 0
        
        exposure_pct = total / equity * 100
        largest_pct = largest / equity * 100
        
        if exposure_pct < 20:
            level = "cool"
        elif exposure_pct < 40:
            level = "warm"
        elif exposure_pct < 60:
            level = "hot"
        else:
            level = "overheated"
        
        return {
            "total_exposure_pct": exposure_pct,
            "n_positions": len(positions),
            "largest_position_pct": largest_pct,
            "heat_level": level,
        }
