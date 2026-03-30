"""
Autoresearch Trading Dashboard — Glassmorphic UI
Run: python dashboard.py → open http://localhost:5000
"""

from flask import Flask, render_template_string, jsonify
import json
import os
import time
import sqlite3

app = Flask(__name__)

DB_PATH = os.path.join(os.path.expanduser("~"), ".cache", "autotrader", "strategies.db")
INSIGHTS_PATH = os.path.join(os.path.expanduser("~"), ".cache", "autotrader", "insights.json")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <title>Autoresearch Trading</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta http-equiv="refresh" content="30">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0a0e1a;
            --bg-secondary: #0f1629;
            --glass-bg: rgba(15, 22, 41, 0.6);
            --glass-border: rgba(255, 255, 255, 0.08);
            --glass-hover: rgba(255, 255, 255, 0.04);
            --accent-green: #00f5a0;
            --accent-blue: #00d4ff;
            --accent-purple: #a855f7;
            --accent-red: #ff4757;
            --accent-orange: #ff9f43;
            --accent-gold: #ffd700;
            --text-primary: #e8eaf0;
            --text-secondary: #8892a4;
            --text-muted: #4a5568;
            --gradient-1: linear-gradient(135deg, #00f5a0 0%, #00d4ff 100%);
            --gradient-2: linear-gradient(135deg, #a855f7 0%, #6366f1 100%);
            --gradient-3: linear-gradient(135deg, #ff4757 0%, #ff6b81 100%);
        }
        
        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        body {
            font-family: 'Inter', -apple-system, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            overflow-x: hidden;
        }
        
        /* Animated background */
        body::before {
            content: '';
            position: fixed;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle at 20% 80%, rgba(0, 245, 160, 0.03) 0%, transparent 50%),
                        radial-gradient(circle at 80% 20%, rgba(0, 212, 255, 0.03) 0%, transparent 50%),
                        radial-gradient(circle at 50% 50%, rgba(168, 85, 247, 0.02) 0%, transparent 50%);
            z-index: -1;
            animation: bgFloat 20s ease-in-out infinite;
        }
        
        @keyframes bgFloat {
            0%, 100% { transform: translate(0, 0) rotate(0deg); }
            33% { transform: translate(2%, -2%) rotate(1deg); }
            66% { transform: translate(-1%, 1%) rotate(-0.5deg); }
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        /* Header */
        .header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 16px 0;
            margin-bottom: 24px;
            flex-wrap: wrap;
            gap: 12px;
        }
        
        .logo {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .logo-icon {
            width: 40px;
            height: 40px;
            background: var(--gradient-1);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
        }
        
        .logo h1 {
            font-size: 1.3em;
            font-weight: 700;
            background: var(--gradient-1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .logo-sub {
            font-size: 0.75em;
            color: var(--text-secondary);
            font-weight: 400;
        }
        
        .live-badge {
            display: flex;
            align-items: center;
            gap: 6px;
            background: rgba(0, 245, 160, 0.1);
            border: 1px solid rgba(0, 245, 160, 0.2);
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.75em;
            color: var(--accent-green);
            font-weight: 500;
        }
        
        .live-dot {
            width: 6px;
            height: 6px;
            background: var(--accent-green);
            border-radius: 50%;
            animation: pulse 2s ease-in-out infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.5; transform: scale(1.5); }
        }
        
        /* Glass Card */
        .glass {
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--glass-border);
            border-radius: 16px;
            padding: 20px;
            margin-bottom: 16px;
            transition: border-color 0.3s ease;
        }
        
        .glass:hover {
            border-color: rgba(255, 255, 255, 0.12);
        }
        
        /* Stats Grid */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 12px;
            margin-bottom: 20px;
        }
        
        .stat-card {
            background: var(--glass-bg);
            backdrop-filter: blur(15px);
            border: 1px solid var(--glass-border);
            border-radius: 14px;
            padding: 16px;
            position: relative;
            overflow: hidden;
        }
        
        .stat-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
        }
        
        .stat-card:nth-child(1)::before { background: var(--gradient-1); }
        .stat-card:nth-child(2)::before { background: var(--gradient-2); }
        .stat-card:nth-child(3)::before { background: var(--gradient-3); }
        .stat-card:nth-child(4)::before { background: linear-gradient(135deg, var(--accent-orange), var(--accent-gold)); }
        .stat-card:nth-child(5)::before { background: var(--gradient-1); }
        .stat-card:nth-child(6)::before { background: var(--gradient-2); }
        
        .stat-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.8em;
            font-weight: 700;
            line-height: 1;
            margin-bottom: 4px;
        }
        
        .stat-label {
            font-size: 0.7em;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 500;
        }
        
        .green { color: var(--accent-green); }
        .blue { color: var(--accent-blue); }
        .red { color: var(--accent-red); }
        .purple { color: var(--accent-purple); }
        .orange { color: var(--accent-orange); }
        .gold { color: var(--accent-gold); }
        
        /* Section Headers */
        .section-header {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 14px;
            font-size: 0.95em;
            font-weight: 600;
        }
        
        .section-icon {
            font-size: 1.1em;
        }
        
        /* Table */
        table {
            width: 100%;
            border-collapse: collapse;
        }
        
        th {
            text-align: left;
            padding: 10px 12px;
            font-size: 0.7em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-secondary);
            border-bottom: 1px solid var(--glass-border);
            font-weight: 600;
        }
        
        td {
            padding: 10px 12px;
            font-size: 0.82em;
            border-bottom: 1px solid rgba(255, 255, 255, 0.03);
            font-family: 'JetBrains Mono', monospace;
        }
        
        tr:hover td {
            background: var(--glass-hover);
        }
        
        .rank-cell {
            width: 40px;
            text-align: center;
        }
        
        .rank-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 24px;
            height: 24px;
            border-radius: 8px;
            font-size: 0.75em;
            font-weight: 700;
        }
        
        .rank-1 { background: rgba(255, 215, 0, 0.15); color: var(--accent-gold); }
        .rank-2 { background: rgba(192, 192, 192, 0.15); color: #c0c0c0; }
        .rank-3 { background: rgba(205, 127, 50, 0.15); color: #cd7f32; }
        .rank-n { background: rgba(255, 255, 255, 0.05); color: var(--text-secondary); }
        
        /* Badges */
        .badge {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 6px;
            font-size: 0.7em;
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
        }
        
        .badge-survived { background: rgba(0, 245, 160, 0.1); color: var(--accent-green); border: 1px solid rgba(0, 245, 160, 0.2); }
        .badge-killed { background: rgba(255, 71, 87, 0.1); color: var(--accent-red); border: 1px solid rgba(255, 71, 87, 0.2); }
        .badge-healed { background: rgba(255, 159, 67, 0.1); color: var(--accent-orange); border: 1px solid rgba(255, 159, 67, 0.2); }
        .badge-error { background: rgba(168, 85, 247, 0.1); color: var(--accent-purple); border: 1px solid rgba(168, 85, 247, 0.2); }
        .badge-signal { background: rgba(0, 212, 255, 0.1); color: var(--accent-blue); border: 1px solid rgba(0, 212, 255, 0.15); }
        
        /* Signal Bar */
        .signal-bar-container {
            width: 100px;
            height: 6px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 3px;
            overflow: hidden;
            display: inline-block;
            vertical-align: middle;
        }
        
        .signal-bar-fill {
            height: 100%;
            border-radius: 3px;
            background: var(--gradient-1);
            transition: width 0.5s ease;
        }
        
        /* Empty State */
        .empty-state {
            text-align: center;
            padding: 40px 20px;
            color: var(--text-muted);
        }
        
        .empty-icon {
            font-size: 2.5em;
            margin-bottom: 12px;
            opacity: 0.4;
        }
        
        .empty-text {
            font-size: 0.85em;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            padding: 20px 0;
            font-size: 0.7em;
            color: var(--text-muted);
        }
        
        .footer-links {
            display: flex;
            justify-content: center;
            gap: 16px;
            margin-top: 6px;
        }
        
        .footer a {
            color: var(--text-secondary);
            text-decoration: none;
        }
        
        /* Grid Layout */
        .two-col {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }
        
        @media (max-width: 768px) {
            .two-col { grid-template-columns: 1fr; }
            .stats-grid { grid-template-columns: repeat(2, 1fr); }
            .container { padding: 12px; }
            td, th { padding: 8px 6px; font-size: 0.75em; }
            .stat-value { font-size: 1.4em; }
        }
        
        /* Scrollbar */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <div class="logo">
                <div class="logo-icon">⚡</div>
                <div>
                    <h1>Autoresearch Trading</h1>
                    <div class="logo-sub">Autonomous Strategy Research • BTC/USD • XAU/USD</div>
                </div>
            </div>
            <div class="live-badge">
                <div class="live-dot"></div>
                Auto-refresh 30s
            </div>
        </div>
        
        <!-- Stats -->
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
                <div class="stat-value orange">{{ "%.1f"|format(stats.kill_rate) }}%</div>
                <div class="stat-label">Kill Rate</div>
            </div>
            <div class="stat-card">
                <div class="stat-value gold">{{ "%.3f"|format(stats.best_score) }}</div>
                <div class="stat-label">Best Score</div>
            </div>
            <div class="stat-card">
                <div class="stat-value purple">{{ stats.errors }}</div>
                <div class="stat-label">Errors</div>
            </div>
        </div>
        
        <!-- Leaderboard -->
        <div class="glass">
            <div class="section-header">
                <span class="section-icon">🏆</span>
                Strategy Leaderboard
            </div>
            {% if leaderboard %}
            <div style="overflow-x: auto;">
            <table>
                <tr>
                    <th class="rank-cell">#</th>
                    <th>Strategy</th>
                    <th>Score</th>
                    <th>Sharpe</th>
                    <th>Return</th>
                    <th>Max DD</th>
                    <th>Trades</th>
                    <th>Gen</th>
                </tr>
                {% for s in leaderboard %}
                <tr>
                    <td class="rank-cell">
                        <span class="rank-badge {% if loop.index == 1 %}rank-1{% elif loop.index == 2 %}rank-2{% elif loop.index == 3 %}rank-3{% else %}rank-n{% endif %}">{{ loop.index }}</span>
                    </td>
                    <td style="font-family: 'Inter', sans-serif; font-weight: 500;">{{ s.name[:24] }}</td>
                    <td class="green">{{ "%.3f"|format(s.oos_score) }}</td>
                    <td>{{ "%.2f"|format(s.oos_sharpe) }}</td>
                    <td class="{{ 'green' if s.oos_return_pct > 0 else 'red' }}">{{ "%.1f"|format(s.oos_return_pct) }}%</td>
                    <td class="{{ 'green' if s.oos_max_dd_pct < 5 else 'orange' if s.oos_max_dd_pct < 15 else 'red' }}">{{ "%.1f"|format(s.oos_max_dd_pct) }}%</td>
                    <td>{{ s.oos_num_trades }}</td>
                    <td class="purple">{{ s.generation }}</td>
                </tr>
                {% endfor %}
            </table>
            </div>
            {% else %}
            <div class="empty-state">
                <div class="empty-icon">🧬</div>
                <div class="empty-text">No survivors yet. Engine generating strategies...</div>
            </div>
            {% endif %}
        </div>
        
        <div class="two-col">
            <!-- Signal Performance -->
            <div class="glass">
                <div class="section-header">
                    <span class="section-icon">📊</span>
                    Signal Performance
                </div>
                {% if signal_stats %}
                <table>
                    <tr>
                        <th>Signal</th>
                        <th>Win Rate</th>
                        <th style="width:110px;"></th>
                        <th>Sharpe</th>
                    </tr>
                    {% for name, stats in signal_stats %}
                    <tr>
                        <td><span class="badge badge-signal">{{ name }}</span></td>
                        <td class="{{ 'green' if stats.win_rate > 30 else 'red' }}">{{ "%.0f"|format(stats.win_rate) }}%</td>
                        <td>
                            <div class="signal-bar-container">
                                <div class="signal-bar-fill" style="width: {{ stats.win_rate }}%;"></div>
                            </div>
                        </td>
                        <td>{{ "%.2f"|format(stats.avg_sharpe_present) }}</td>
                    </tr>
                    {% endfor %}
                </table>
                {% else %}
                <div class="empty-state">
                    <div class="empty-icon">📈</div>
                    <div class="empty-text">Signal data populates after first generation</div>
                </div>
                {% endif %}
            </div>
            
            <!-- Failure Patterns -->
            <div class="glass">
                <div class="section-header">
                    <span class="section-icon">💀</span>
                    Failure Patterns
                </div>
                {% if failures %}
                <table>
                    <tr>
                        <th>Pattern</th>
                        <th>Count</th>
                        <th>%</th>
                    </tr>
                    {% for f in failures %}
                    <tr>
                        <td><span class="badge badge-killed">{{ f.pattern }}</span></td>
                        <td>{{ f.count }}</td>
                        <td>{{ "%.0f"|format(f.pct) }}%</td>
                    </tr>
                    {% endfor %}
                </table>
                {% else %}
                <div class="empty-state">
                    <div class="empty-icon">🔍</div>
                    <div class="empty-text">No failure data yet</div>
                </div>
                {% endif %}
            </div>
        </div>
        
        <!-- Activity -->
        <div class="glass">
            <div class="section-header">
                <span class="section-icon">⚡</span>
                Recent Activity
            </div>
            {% if recent_logs %}
            <table>
                <tr>
                    <th style="width: 80px;">Time</th>
                    <th>Strategy</th>
                    <th style="width: 100px;">Action</th>
                    <th>Details</th>
                </tr>
                {% for log in recent_logs %}
                <tr>
                    <td style="color: var(--text-muted);">{{ log.time_ago }}</td>
                    <td>{{ log.strategy_id[:12] }}</td>
                    <td>
                        {% if log.action == 'passed_oos' %}<span class="badge badge-survived">survived</span>
                        {% elif 'killed' in log.action %}<span class="badge badge-killed">killed</span>
                        {% elif log.action == 'healed' %}<span class="badge badge-healed">healed</span>
                        {% elif log.action == 'error' %}<span class="badge badge-error">error</span>
                        {% else %}<span class="badge badge-signal">{{ log.action }}</span>{% endif %}
                    </td>
                    <td style="color: var(--text-secondary); font-family: 'Inter', sans-serif; font-size: 0.78em;">{{ log.details[:60] }}</td>
                </tr>
                {% endfor %}
            </table>
            {% else %}
            <div class="empty-state">
                <div class="empty-icon">⏳</div>
                <div class="empty-text">Waiting for engine to start...</div>
            </div>
            {% endif %}
        </div>
        
        <div class="footer">
            Autoresearch Trading Engine v2 • 36 signals • BTC/USD + XAU/USD
            <div class="footer-links">
                <a href="https://github.com/Millan678y/autoresearch-trading">GitHub</a>
            </div>
        </div>
    </div>
</body>
</html>
"""


def _get_db_stats():
    defaults = {"total_strategies": 0, "survived": 0, "killed": 0, "errors": 0, "kill_rate": 0.0, "best_score": 0.0}
    if not os.path.exists(DB_PATH): return defaults
    try:
        conn = sqlite3.connect(DB_PATH)
        total = conn.execute("SELECT COUNT(*) FROM strategies").fetchone()[0]
        survived = conn.execute("SELECT COUNT(*) FROM strategies WHERE status='passed_oos'").fetchone()[0]
        killed = conn.execute("SELECT COUNT(*) FROM strategies WHERE status='killed'").fetchone()[0]
        errors = conn.execute("SELECT COUNT(*) FROM strategies WHERE status='error'").fetchone()[0]
        best = conn.execute("SELECT oos_score FROM strategies WHERE status='passed_oos' ORDER BY oos_score DESC LIMIT 1").fetchone()
        conn.close()
        return {"total_strategies": total, "survived": survived, "killed": killed, "errors": errors,
                "kill_rate": killed / max(total, 1) * 100, "best_score": best[0] if best else 0.0}
    except: return defaults


def _get_leaderboard(limit=15):
    if not os.path.exists(DB_PATH): return []
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT name, oos_score, oos_sharpe, oos_return_pct, oos_max_dd_pct, oos_num_trades, generation, signals_used "
            "FROM strategies WHERE status='passed_oos' ORDER BY oos_score DESC LIMIT ?", (limit,)).fetchall()
        conn.close()
        return [{"name": r["name"], "oos_score": r["oos_score"] or 0, "oos_sharpe": r["oos_sharpe"] or 0,
                 "oos_return_pct": r["oos_return_pct"] or 0, "oos_max_dd_pct": r["oos_max_dd_pct"] or 0,
                 "oos_num_trades": r["oos_num_trades"] or 0, "generation": r["generation"] or 0,
                 "n_signals": len(json.loads(r["signals_used"])) if r["signals_used"] else 0} for r in rows]
    except: return []


def _get_signal_stats():
    if not os.path.exists(INSIGHTS_PATH): return []
    try:
        with open(INSIGHTS_PATH) as f: insights = json.load(f)
        perf = insights.get("signal_performance", {})
        from types import SimpleNamespace
        return [(name, SimpleNamespace(**stats)) for name, stats in sorted(perf.items(), key=lambda x: -x[1].get("win_rate", 0))]
    except: return []


def _get_failures():
    if not os.path.exists(INSIGHTS_PATH): return []
    try:
        with open(INSIGHTS_PATH) as f: return json.load(f).get("failure_patterns", [])[:5]
    except: return []


def _get_recent_logs(limit=15):
    if not os.path.exists(DB_PATH): return []
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT strategy_id, timestamp, action, details FROM experiment_log ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        conn.close()
        now = time.time()
        results = []
        for r in rows:
            elapsed = now - (r["timestamp"] or now)
            if elapsed < 60: time_ago = f"{int(elapsed)}s"
            elif elapsed < 3600: time_ago = f"{int(elapsed/60)}m"
            elif elapsed < 86400: time_ago = f"{int(elapsed/3600)}h"
            else: time_ago = f"{int(elapsed/86400)}d"
            results.append({"strategy_id": r["strategy_id"] or "", "time_ago": time_ago,
                           "action": r["action"] or "", "details": r["details"] or ""})
        return results
    except: return []


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE, stats=_get_db_stats(), leaderboard=_get_leaderboard(),
                                  signal_stats=_get_signal_stats(), failures=_get_failures(), recent_logs=_get_recent_logs())

@app.route("/api/stats")
def api_stats(): return jsonify(_get_db_stats())

@app.route("/api/leaderboard")
def api_leaderboard(): return jsonify(_get_leaderboard())

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting dashboard on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
