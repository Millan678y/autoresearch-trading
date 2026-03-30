"""
REALISTIC EXECUTION MODEL — Spreads, Slippage, Sessions, Commissions

Replaces the simplistic "price ± slippage" model with:
1. Variable spreads (wider during low liquidity / high vol)
2. Stochastic slippage (size-dependent, volatility-dependent)
3. Market session handling (Gold has hours, BTC is 24/7)
4. Commission models (per-trade, per-lot, tiered)
5. Execution delay simulation
6. Partial fill simulation (optional)
7. Price impact for larger orders

This is what separates toy backtests from realistic ones.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


# ─────────────────────────────────────────────────────────────────
# Asset Profiles
# ─────────────────────────────────────────────────────────────────

class AssetType(Enum):
    CRYPTO = "crypto"
    FOREX = "forex"
    COMMODITY = "commodity"
    INDEX = "index"


@dataclass
class AssetProfile:
    """Realistic execution parameters per asset."""
    name: str
    asset_type: AssetType
    
    # Spread
    base_spread_pct: float          # Base spread as % of price
    spread_vol_multiplier: float    # How much spread widens in high vol
    spread_session_multiplier: dict # Multiplier by session
    
    # Slippage
    base_slippage_pct: float        # Base slippage as % of price
    slippage_size_impact: float     # Additional slippage per $10K notional
    
    # Commission
    commission_pct: float           # Commission as % of notional
    min_commission: float           # Minimum commission per trade
    
    # Sessions
    trading_hours: dict             # {session: (start_hour_utc, end_hour_utc)}
    is_24_7: bool                   # True for crypto
    
    # Tick size
    tick_size: float                # Minimum price increment
    
    # Lot constraints
    min_lot: float
    lot_step: float


# Pre-defined asset profiles
ASSET_PROFILES = {
    "BTCUSDT": AssetProfile(
        name="BTCUSDT",
        asset_type=AssetType.CRYPTO,
        base_spread_pct=0.0001,      # ~$10 on $100K BTC
        spread_vol_multiplier=3.0,
        spread_session_multiplier={"asian": 1.2, "london": 0.9, "ny": 0.8},
        base_slippage_pct=0.0002,
        slippage_size_impact=0.00001,
        commission_pct=0.0004,        # 4 bps (Binance futures)
        min_commission=0.0,
        trading_hours={},
        is_24_7=True,
        tick_size=0.01,
        min_lot=0.001,
        lot_step=0.001,
    ),
    "BTC-USD": AssetProfile(  # Alias
        name="BTC-USD",
        asset_type=AssetType.CRYPTO,
        base_spread_pct=0.0001,
        spread_vol_multiplier=3.0,
        spread_session_multiplier={"asian": 1.2, "london": 0.9, "ny": 0.8},
        base_slippage_pct=0.0002,
        slippage_size_impact=0.00001,
        commission_pct=0.0004,
        min_commission=0.0,
        trading_hours={},
        is_24_7=True,
        tick_size=0.01,
        min_lot=0.001,
        lot_step=0.001,
    ),
    "ETHUSDT": AssetProfile(
        name="ETHUSDT",
        asset_type=AssetType.CRYPTO,
        base_spread_pct=0.00015,
        spread_vol_multiplier=3.0,
        spread_session_multiplier={"asian": 1.3, "london": 0.9, "ny": 0.8},
        base_slippage_pct=0.0003,
        slippage_size_impact=0.00002,
        commission_pct=0.0004,
        min_commission=0.0,
        trading_hours={},
        is_24_7=True,
        tick_size=0.01,
        min_lot=0.01,
        lot_step=0.01,
    ),
    "XAUUSD": AssetProfile(
        name="XAUUSD",
        asset_type=AssetType.COMMODITY,
        base_spread_pct=0.00015,      # ~$0.30 on $2000 gold
        spread_vol_multiplier=4.0,     # Gold spreads blow up in high vol
        spread_session_multiplier={
            "asian": 2.0,              # Widest during Asian session
            "london": 0.8,             # Tightest during London
            "ny": 0.9,
            "closed": 5.0,            # After-hours is terrible
        },
        base_slippage_pct=0.00025,
        slippage_size_impact=0.00003,
        commission_pct=0.0007,         # 7 bps typical gold futures
        min_commission=0.50,
        trading_hours={
            "sunday_open": 22,         # Sunday 22:00 UTC
            "friday_close": 21,        # Friday 21:00 UTC
            "daily_close": 21,         # Daily close 21:00-22:00 UTC
            "daily_open": 22,
        },
        is_24_7=False,
        tick_size=0.01,
        min_lot=0.01,
        lot_step=0.01,
    ),
    "SOLUSDT": AssetProfile(
        name="SOLUSDT",
        asset_type=AssetType.CRYPTO,
        base_spread_pct=0.0003,
        spread_vol_multiplier=4.0,
        spread_session_multiplier={"asian": 1.5, "london": 1.0, "ny": 0.8},
        base_slippage_pct=0.0005,
        slippage_size_impact=0.00005,
        commission_pct=0.0004,
        min_commission=0.0,
        trading_hours={},
        is_24_7=True,
        tick_size=0.001,
        min_lot=0.1,
        lot_step=0.1,
    ),
}


def get_profile(symbol: str) -> AssetProfile:
    """Get asset profile, with fallback to generic crypto."""
    if symbol in ASSET_PROFILES:
        return ASSET_PROFILES[symbol]
    
    # Try partial match
    for key, profile in ASSET_PROFILES.items():
        if symbol.startswith(key[:3]):
            return profile
    
    # Generic fallback
    return AssetProfile(
        name=symbol,
        asset_type=AssetType.CRYPTO,
        base_spread_pct=0.0002,
        spread_vol_multiplier=3.0,
        spread_session_multiplier={},
        base_slippage_pct=0.0003,
        slippage_size_impact=0.00002,
        commission_pct=0.0005,
        min_commission=0.0,
        trading_hours={},
        is_24_7=True,
        tick_size=0.01,
        min_lot=0.001,
        lot_step=0.001,
    )


# ─────────────────────────────────────────────────────────────────
# Session Detection
# ─────────────────────────────────────────────────────────────────

def get_session(timestamp_ms: int) -> str:
    """Determine trading session from timestamp."""
    from datetime import datetime
    dt = datetime.utcfromtimestamp(timestamp_ms / 1000)
    hour = dt.hour
    weekday = dt.weekday()  # 0=Mon, 6=Sun
    
    # Weekend check (for non-crypto)
    if weekday == 5:  # Saturday
        return "closed"
    if weekday == 6 and hour < 22:  # Sunday before open
        return "closed"
    if weekday == 4 and hour >= 21:  # Friday after close
        return "closed"
    
    # Sessions
    if 22 <= hour or hour < 8:
        return "asian"
    elif 8 <= hour < 13:
        return "london"
    elif 13 <= hour < 21:
        return "ny"
    
    return "london"


def is_market_open(symbol: str, timestamp_ms: int) -> bool:
    """Check if market is open for this asset at this time."""
    profile = get_profile(symbol)
    
    if profile.is_24_7:
        return True
    
    session = get_session(timestamp_ms)
    return session != "closed"


# ─────────────────────────────────────────────────────────────────
# Spread Model
# ─────────────────────────────────────────────────────────────────

def compute_spread(symbol: str, price: float, timestamp_ms: int,
                   realized_vol: float = 0.0, vol_target: float = 0.015) -> float:
    """
    Compute realistic spread for a given asset and market conditions.
    
    Returns: spread in price units (half-spread = spread/2 for each side)
    """
    profile = get_profile(symbol)
    
    # Base spread
    spread = price * profile.base_spread_pct
    
    # Vol adjustment (higher vol = wider spread)
    if realized_vol > 0 and vol_target > 0:
        vol_ratio = realized_vol / vol_target
        vol_mult = 1.0 + (vol_ratio - 1.0) * profile.spread_vol_multiplier
        vol_mult = max(0.5, min(vol_mult, 5.0))
        spread *= vol_mult
    
    # Session adjustment
    session = get_session(timestamp_ms)
    session_mult = profile.spread_session_multiplier.get(session, 1.0)
    spread *= session_mult
    
    # Add small random component (real spreads aren't constant)
    jitter = np.random.uniform(0.8, 1.2)
    spread *= jitter
    
    return float(spread)


# ─────────────────────────────────────────────────────────────────
# Slippage Model
# ─────────────────────────────────────────────────────────────────

def compute_slippage(symbol: str, price: float, order_size_usd: float,
                     realized_vol: float = 0.0, is_market_order: bool = True) -> float:
    """
    Compute realistic slippage.
    
    Slippage depends on:
    1. Order size (larger = more slippage)
    2. Market volatility (higher = more slippage)
    3. Order type (market orders have more slippage than limits)
    
    Returns: slippage in price units (always positive — direction applied by caller)
    """
    if not is_market_order:
        return 0.0  # Limit orders get their price (if filled)
    
    profile = get_profile(symbol)
    
    # Base slippage
    slip = price * profile.base_slippage_pct
    
    # Size impact (more slippage for larger orders)
    size_buckets = order_size_usd / 10_000  # Per $10K
    slip += price * profile.slippage_size_impact * size_buckets
    
    # Vol adjustment
    if realized_vol > 0:
        vol_mult = max(1.0, realized_vol / 0.015)
        slip *= vol_mult
    
    # Stochastic component (slippage is random in real life)
    # Use log-normal to model heavy-tailed slippage distribution
    random_mult = np.random.lognormal(0, 0.3)
    slip *= random_mult
    
    return float(max(0, slip))


# ─────────────────────────────────────────────────────────────────
# Commission Model
# ─────────────────────────────────────────────────────────────────

def compute_commission(symbol: str, notional_usd: float) -> float:
    """Compute commission for a trade."""
    profile = get_profile(symbol)
    
    commission = notional_usd * profile.commission_pct
    return float(max(commission, profile.min_commission))


# ─────────────────────────────────────────────────────────────────
# Execution Simulator
# ─────────────────────────────────────────────────────────────────

@dataclass
class ExecutionResult:
    """Result of simulated order execution."""
    executed: bool
    fill_price: float
    spread_cost: float
    slippage_cost: float
    commission: float
    total_cost: float         # Total execution cost
    execution_delay_ms: int   # Simulated delay
    reason: str = ""


class ExecutionSimulator:
    """
    Simulates realistic order execution.
    
    Replaces the naive "price ± fixed_slippage" with proper modeling
    of spreads, slippage, commissions, sessions, and execution delays.
    """
    
    def __init__(self, execution_delay_ms: int = 100,
                 enable_partial_fills: bool = False):
        self.execution_delay_ms = execution_delay_ms
        self.enable_partial_fills = enable_partial_fills
    
    def execute_market_order(
        self,
        symbol: str,
        direction: int,         # 1 = buy, -1 = sell
        price: float,           # Current mid price
        size_usd: float,        # Order size in USD
        timestamp_ms: int,
        realized_vol: float = 0.0,
    ) -> ExecutionResult:
        """
        Simulate a market order execution.
        
        Returns ExecutionResult with realistic fill price and costs.
        """
        profile = get_profile(symbol)
        
        # Check if market is open
        if not is_market_open(symbol, timestamp_ms):
            return ExecutionResult(
                executed=False, fill_price=0, spread_cost=0,
                slippage_cost=0, commission=0, total_cost=0,
                execution_delay_ms=0, reason="market_closed"
            )
        
        # Compute spread
        spread = compute_spread(symbol, price, timestamp_ms, realized_vol)
        half_spread = spread / 2
        
        # Apply spread (buy at ask, sell at bid)
        spread_price = price + half_spread * direction
        
        # Compute slippage
        slippage = compute_slippage(symbol, price, size_usd, realized_vol)
        
        # Apply slippage (always adverse)
        fill_price = spread_price + slippage * direction
        
        # Round to tick size
        if profile.tick_size > 0:
            fill_price = round(fill_price / profile.tick_size) * profile.tick_size
        
        # Commission
        commission = compute_commission(symbol, size_usd)
        
        # Total cost
        spread_cost = abs(fill_price - price) * (size_usd / price) - slippage * (size_usd / price)
        slippage_cost = slippage * (size_usd / price)
        total_cost = spread_cost + slippage_cost + commission
        
        return ExecutionResult(
            executed=True,
            fill_price=float(fill_price),
            spread_cost=float(abs(half_spread * size_usd / price)),
            slippage_cost=float(slippage * size_usd / price),
            commission=float(commission),
            total_cost=float(total_cost),
            execution_delay_ms=self.execution_delay_ms,
        )
    
    def execute_limit_order(
        self,
        symbol: str,
        direction: int,
        limit_price: float,
        current_price: float,
        size_usd: float,
        timestamp_ms: int,
    ) -> ExecutionResult:
        """
        Check if a limit order would fill.
        
        Buy limit fills if price <= limit_price
        Sell limit fills if price >= limit_price
        """
        if not is_market_open(symbol, timestamp_ms):
            return ExecutionResult(
                executed=False, fill_price=0, spread_cost=0,
                slippage_cost=0, commission=0, total_cost=0,
                execution_delay_ms=0, reason="market_closed"
            )
        
        # Check fill condition
        fills = False
        if direction == 1 and current_price <= limit_price:
            fills = True
        elif direction == -1 and current_price >= limit_price:
            fills = True
        
        if not fills:
            return ExecutionResult(
                executed=False, fill_price=0, spread_cost=0,
                slippage_cost=0, commission=0, total_cost=0,
                execution_delay_ms=0, reason="not_triggered"
            )
        
        # Limit orders get better execution (no slippage, just spread + commission)
        commission = compute_commission(symbol, size_usd)
        # Maker gets better spread
        profile = get_profile(symbol)
        spread = current_price * profile.base_spread_pct * 0.3  # Makers get less spread
        
        fill_price = limit_price
        
        return ExecutionResult(
            executed=True,
            fill_price=float(fill_price),
            spread_cost=float(spread * size_usd / current_price),
            slippage_cost=0.0,
            commission=float(commission),
            total_cost=float(spread * size_usd / current_price + commission),
            execution_delay_ms=self.execution_delay_ms,
        )
