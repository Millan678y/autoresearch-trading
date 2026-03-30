"""
Data models for the autonomous trading system.
Shared types across all modules.
"""

import sqlite3
import json
import time
import os
from dataclasses import dataclass, field, asdict
from typing import Optional
from enum import Enum

DB_PATH = os.path.join(os.path.expanduser("~"), ".cache", "autotrader", "strategies.db")


class StrategyStatus(Enum):
    PENDING = "pending"          # generated, not yet tested
    BACKTESTING = "backtesting"  # currently running
    PASSED_IS = "passed_is"      # passed in-sample (2024)
    PASSED_OOS = "passed_oos"    # passed out-of-sample (2025)
    DEPLOYED = "deployed"        # in live forward-test
    KILLED = "killed"            # pruned by Darwin
    ERROR = "error"              # crashed or had bugs


@dataclass
class StrategyRecord:
    id: str                              # unique hash
    name: str                            # human-readable name
    code: str                            # full Python source
    params: dict = field(default_factory=dict)
    signals_used: list = field(default_factory=list)
    
    # In-sample metrics (2024)
    is_sharpe: float = 0.0
    is_return_pct: float = 0.0
    is_max_dd_pct: float = 0.0
    is_win_rate: float = 0.0
    is_num_trades: int = 0
    is_profit_factor: float = 0.0
    is_sortino: float = 0.0
    is_score: float = -999.0
    
    # Out-of-sample metrics (2025)
    oos_sharpe: float = 0.0
    oos_return_pct: float = 0.0
    oos_max_dd_pct: float = 0.0
    oos_win_rate: float = 0.0
    oos_num_trades: int = 0
    oos_score: float = -999.0
    
    # Meta
    parent_id: Optional[str] = None      # strategy it mutated from
    generation: int = 0                  # evolution generation
    status: str = "pending"
    created_at: float = 0.0
    kill_reason: str = ""
    keep_reason: str = ""
    
    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()


def init_db():
    """Initialize the strategy database."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS strategies (
            id TEXT PRIMARY KEY,
            name TEXT,
            code TEXT,
            params TEXT,
            signals_used TEXT,
            is_sharpe REAL DEFAULT 0,
            is_return_pct REAL DEFAULT 0,
            is_max_dd_pct REAL DEFAULT 0,
            is_win_rate REAL DEFAULT 0,
            is_num_trades INTEGER DEFAULT 0,
            is_profit_factor REAL DEFAULT 0,
            is_sortino REAL DEFAULT 0,
            is_score REAL DEFAULT -999,
            oos_sharpe REAL DEFAULT 0,
            oos_return_pct REAL DEFAULT 0,
            oos_max_dd_pct REAL DEFAULT 0,
            oos_win_rate REAL DEFAULT 0,
            oos_num_trades INTEGER DEFAULT 0,
            oos_score REAL DEFAULT -999,
            parent_id TEXT,
            generation INTEGER DEFAULT 0,
            status TEXT DEFAULT 'pending',
            created_at REAL,
            kill_reason TEXT DEFAULT '',
            keep_reason TEXT DEFAULT ''
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS experiment_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_id TEXT,
            timestamp REAL,
            action TEXT,
            details TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_strategy(rec: StrategyRecord):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT OR REPLACE INTO strategies VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
    """, (
        rec.id, rec.name, rec.code, json.dumps(rec.params),
        json.dumps(rec.signals_used),
        rec.is_sharpe, rec.is_return_pct, rec.is_max_dd_pct,
        rec.is_win_rate, rec.is_num_trades, rec.is_profit_factor,
        rec.is_sortino, rec.is_score,
        rec.oos_sharpe, rec.oos_return_pct, rec.oos_max_dd_pct,
        rec.oos_win_rate, rec.oos_num_trades, rec.oos_score,
        rec.parent_id, rec.generation, rec.status,
        rec.created_at, rec.kill_reason, rec.keep_reason
    ))
    conn.commit()
    conn.close()


def load_strategies(status=None, min_score=None, limit=50):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    query = "SELECT * FROM strategies"
    conditions = []
    params = []
    if status:
        conditions.append("status = ?")
        params.append(status)
    if min_score is not None:
        conditions.append("is_score >= ?")
        params.append(min_score)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY is_score DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def log_event(strategy_id: str, action: str, details: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO experiment_log (strategy_id, timestamp, action, details) VALUES (?, ?, ?, ?)",
        (strategy_id, time.time(), action, details)
    )
    conn.commit()
    conn.close()
