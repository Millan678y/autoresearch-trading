"""
Autoresearch Trading Dashboard — Web UI for monitoring the autonomous engine.
Run: python dashboard.py → open http://localhost:5000
"""

from flask import Flask, render_template_string, jsonify
import json
import os
import time
import sqlite3
from pathlib import Path

app = Flask(__name__)

DB_PATH = os.path.join(os.path.expanduser("~"), ".cache", "autotrader", "strategies.db")
INSIGHTS_PATH = os.path.join(os.path.expanduser("~"), ".cache", "autotrader", "insights.json")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Autoresearch Trading Engine</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta http-equiv="refresh" content="30">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'SF Mono', 'Fira Code', monospace; background: #0a0a0a; color: #e0e0e0; padding: 16px; }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { color: #00ff88; margin-bottom: 8px; font-size: 1.4em; }
        h2 { color: #ffcc00; margin: 16px 0 8px; font-size: 1.1em; }
        .subtitle { color: #666; font-size: 0.85em; margin-bottom: 16px; }
        
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; margin-bottom: 16px; }
        .stat-card { background: #1a1a1a; border: 1px solid #333; border-radius: 6px; padding: 12px; }
        .stat-value { font-size: 1.6em; font-weight: bold; }
        .stat-label { color: #888; font-size: 0.8em; margin-top: 2px; }
        .green { color: #00ff88; }
        .red { color: #ff4444; }
        .yellow { color: #ffcc00; }
        .blue { color: #4488ff; }
        
        table { width: 100%; border-collapse: collapse; margin-bottom: 16px; }
        th { color: #ffcc00; text-align: left; padding: 8px; border-bottom: 2px solid #333; font-size: 0.85em; }
        td { padding: 8px; border-bottom: 1px solid #222; font-size: 0.85em; }
        tr:hover { background: #1a1a2a; }
        
        .badge { display: inline-block; padding: 2px 8px; border-radius: 3px; font-size: 0.75em; }
        .badge-green { background: #003300; color: #00ff88; border: 1px solid #006600; }
        .badge-red { background: #330000; color: #ff4444; border: 1px solid #660000; }
        .badge-yellow { background: #332200; color: #ffcc00; border: 1px solid #664400; }
        .badge-blue { background: #001133; color: #4488ff; border: 1px solid #003366; }
        
        .section { background: #111; border: 1px solid #222; border-radius: 6px; padding: 16px; margin-bottom: 16px; }
        .signal-bar { display: inline-block; height: 12px; background: #00ff88; border-radius: 2px; margin-right: 4px; }
        
        .footer { color: #444; font-size: 0.75em; text-align: center; margin-top: 16px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Autoresearch Trading Engine</h1>
        <p class="subtitle">Autonomous strategy generation • BTC/USD • XAU/USD</p>
        
        <!-- Stats Overview -->
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value green">{{ stats.total_strategies }}</div>
                <div class="stat-label">Total Strategies</div>
            </div>
            <div class="stat-card">
                <div class="stat-value blue">{{ stats.survived }}</div>
                <div class="stat-label">Survived OOS</div>
            </div>
            <div class="stat-card">
                <div class="stat-value red">{{ stats.killed }}</div>
                <div class="stat-label">Killed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value yellow">{{ "%.1f"|format(stats.kill_rate) }}%</div>
                <div class="stat-label">Kill Rate</div>
            </div>
            <div class="stat-card">
                <div class="stat-value green">{{ "%.3f"|format(stats.best_score) }}</div>
                <div class="stat-label">Best Score</div>
            </div>
            <div class="stat-card">
                <div class="stat-value blue">{{ stats.errors }}</div>
                <div class="stat-label">Errors / Healed</div>
            </div>
        </div>
        
        <!-- Leaderboard -->
        <div class="section">
            <h2>🏆 Strategy Leaderboard (Top 15)</h2>
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Name</th>
                    <th>OOS Score</th>
                    <th>Sharpe</th>
                    <th>Return</th>
                    <th>Max DD</th>
                    <th>Trades</th>
                    <th>Gen</th>
                    <th>Signals</th>
                </tr>
                {% for s in leaderboard %}
                <tr>
                    <td>{{ loop.index }}</td>
                    <td>{{ s.name[:28] }}</td>
                    <td class="green">{{ "%.3f"|format(s.oos_score) }}</td>
                    <td>{{ "%.2f"|format(s.oos_sharpe) }}</td>
                    <td class="{{ 'green' if s.oos_return_pct > 0 else 'red' }}">{{ "%.1f"|format(s.oos_return_pct) }}%</td>
                    <td class="{{ 'green' if s.oos_max_dd_pct < 5 else 'yellow' if s.oos_max_dd_pct < 15 else 'red' }}">{{ "%.1f"|format(s.oos_max_dd_pct) }}%</td>
                    <td>{{ s.oos_num_trades }}</td>
                    <td>{{ s.generation }}</td>
                    <td><span class="badge badge-blue">{{ s.n_signals }}</span></td>
                </tr>
                {% endfor %}
                {% if not leaderboard %}
                <tr><td colspan="9" style="text-align:center; color:#666;">No survivors yet. Engine still running...</td></tr>
                {% endif %}
            </table>
        </div>
        
        <!-- Signal Performance -->
        {% if signal_stats %}
        <div class="section">
            <h2>📊 Signal Performance</h2>
            <table>
                <tr>
                    <th>Signal</th>
                    <th>Win Rate</th>
                    <th>Visual</th>
                    <th>Avg Sharpe</th>
                    <th>Appearances</th>
                </tr>
                {% for name, stats in signal_stats %}
                <tr>
                    <td>{{ name }}</td>
                    <td class="{{ 'green' if stats.win_rate > 30 else 'red' }}">{{ "%.1f"|format(stats.win_rate) }}%</td>
                    <td><span class="signal-bar" style="width: {{ stats.win_rate * 2 }}px;"></span></td>
                    <td>{{ "%.2f"|format(stats.avg_sharpe_present) }}</td>
                    <td>{{ stats.total_appearances }}</td>
                </tr>
                {% endfor %}
            </table>
        </div>
        {% endif %}
        
        <!-- Failure Patterns -->
        {% if failures %}
        <div class="section">
            <h2>💀 Top Failure Patterns</h2>
            <table>
                <tr>
                    <th>Pattern</th>
                    <th>Count</th>
                    <th>%</th>
                    <th>Recommendation</th>
                </tr>
                {% for f in failures %}
                <tr>
                    <td><span class="badge badge-red">{{ f.pattern }}</span></td>
                    <td>{{ f.count }}</td>
                    <td>{{ "%.1f"|format(f.pct) }}%</td>
                    <td style="color:#888; font-size:0.8em;">{{ f.description[:80] }}</td>
                </tr>
                {% endfor %}
            </table>
        </div>
        {% endif %}
        
        <!-- Recent Activity -->
        <div class="section">
            <h2>⚡ Recent Activity</h2>
            <table>
                <tr>
                    <th>Time</th>
                    <th>Strategy</th>
                    <th>Action</th>
                    <th>Details</th>
                </tr>
                {% for log in recent_logs %}
                <tr>
                    <td style="color:#666;">{{ log.time_ago }}</td>
                    <td>{{ log.strategy_id[:12] }}</td>
                    <td>
                        {% if log.action == 'passed_oos' %}
                        <span class="badge badge-green">✅ survived</span>
                        {% elif log.action == 'killed_is' or log.action == 'killed_oos' %}
                        <span class="badge badge-red">❌ killed</span>
                        {% elif log.action == 'healed' %}
                        <span class="badge badge-yellow">🩹 healed</span>
                        {% elif log.action == 'error' %}
                        <span class="badge badge-red">💥 error</span>
                        {% else %}
                        <span class="badge badge-blue">{{ log.action }}</span>
                        {% endif %}
                    </td>
                    <td style="color:#888; font-size:0.8em;">{{ log.details[:80] }}</td>
                </tr>
                {% endfor %}
                {% if not recent_logs %}
                <tr><td colspan="4" style="text-align:center; color:#666;">No activity yet.</td></tr>
                {% endif %}
            </table>
        </div>
        
        <div class="footer">
            Auto-refreshes every 30s • Autoresearch Trading Engine v2 • BTC/USD + XAU/USD
        </div>
    </div>
</body>
</html>
"""


def _get_db_stats():
    """Get statistics from the strategy database."""
    defaults = {
        "total_strategies": 0, "survived": 0, "killed": 0,
        "errors": 0, "kill_rate": 0.0, "best_score": 0.0,
    }
    
    if not os.path.exists(DB_PATH):
        return defaults
    
    try:
        conn = sqlite3.connect(DB_PATH)
        total = conn.execute("SELECT COUNT(*) FROM strategies").fetchone()[0]
        survived = conn.execute("SELECT COUNT(*) FROM strategies WHERE status='passed_oos'").fetchone()[0]
        killed = conn.execute("SELECT COUNT(*) FROM strategies WHERE status='killed'").fetchone()[0]
        errors = conn.execute("SELECT COUNT(*) FROM strategies WHERE status='error'").fetchone()[0]
        
        best = conn.execute(
            "SELECT oos_score FROM strategies WHERE status='passed_oos' ORDER BY oos_score DESC LIMIT 1"
        ).fetchone()
        
        conn.close()
        
        return {
            "total_strategies": total,
            "survived": survived,
            "killed": killed,
            "errors": errors,
            "kill_rate": killed / max(total, 1) * 100,
            "best_score": best[0] if best else 0.0,
        }
    except:
        return defaults


def _get_leaderboard(limit=15):
    if not os.path.exists(DB_PATH):
        return []
    
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT name, oos_score, oos_sharpe, oos_return_pct, oos_max_dd_pct, "
            "oos_num_trades, generation, signals_used "
            "FROM strategies WHERE status='passed_oos' "
            "ORDER BY oos_score DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        
        results = []
        for r in rows:
            signals = json.loads(r["signals_used"]) if r["signals_used"] else []
            results.append({
                "name": r["name"],
                "oos_score": r["oos_score"] or 0,
                "oos_sharpe": r["oos_sharpe"] or 0,
                "oos_return_pct": r["oos_return_pct"] or 0,
                "oos_max_dd_pct": r["oos_max_dd_pct"] or 0,
                "oos_num_trades": r["oos_num_trades"] or 0,
                "generation": r["generation"] or 0,
                "n_signals": len(signals),
            })
        return results
    except:
        return []


def _get_signal_stats():
    if not os.path.exists(INSIGHTS_PATH):
        return []
    try:
        with open(INSIGHTS_PATH) as f:
            insights = json.load(f)
        perf = insights.get("signal_performance", {})
        # Convert to sorted list of tuples
        items = sorted(perf.items(), key=lambda x: -x[1].get("win_rate", 0))
        # Convert inner dicts to objects for template access
        from types import SimpleNamespace
        return [(name, SimpleNamespace(**stats)) for name, stats in items]
    except:
        return []


def _get_failures():
    if not os.path.exists(INSIGHTS_PATH):
        return []
    try:
        with open(INSIGHTS_PATH) as f:
            insights = json.load(f)
        return insights.get("failure_patterns", [])[:5]
    except:
        return []


def _get_recent_logs(limit=20):
    if not os.path.exists(DB_PATH):
        return []
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT strategy_id, timestamp, action, details "
            "FROM experiment_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        
        now = time.time()
        results = []
        for r in rows:
            elapsed = now - (r["timestamp"] or now)
            if elapsed < 60:
                time_ago = f"{int(elapsed)}s ago"
            elif elapsed < 3600:
                time_ago = f"{int(elapsed/60)}m ago"
            elif elapsed < 86400:
                time_ago = f"{int(elapsed/3600)}h ago"
            else:
                time_ago = f"{int(elapsed/86400)}d ago"
            
            results.append({
                "strategy_id": r["strategy_id"] or "",
                "time_ago": time_ago,
                "action": r["action"] or "",
                "details": r["details"] or "",
            })
        return results
    except:
        return []


@app.route("/")
def index():
    return render_template_string(
        HTML_TEMPLATE,
        stats=_get_db_stats(),
        leaderboard=_get_leaderboard(),
        signal_stats=_get_signal_stats(),
        failures=_get_failures(),
        recent_logs=_get_recent_logs(),
    )


@app.route("/api/stats")
def api_stats():
    return jsonify(_get_db_stats())


@app.route("/api/leaderboard")
def api_leaderboard():
    return jsonify(_get_leaderboard())


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting dashboard on http://localhost:{port}")
    print("Auto-refreshes every 30 seconds")
    app.run(host="0.0.0.0", port=port, debug=False)
