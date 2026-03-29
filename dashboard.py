from flask import Flask, render_template_string
import json
import os
from pathlib import Path
import pandas as pd

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Autoresearch Trading</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: monospace; background: #0a0a0a; color: #00ff00; padding: 10px; }
        .container { max-width: 100%; }
        .card { background: #1a1a1a; border: 1px solid #333; padding: 10px; margin: 5px 0; border-radius: 5px; }
        .metric { font-size: 1.2em; font-weight: bold; }
        .positive { color: #00ff00; }
        .negative { color: #ff0000; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 8px; text-align: left; border-bottom: 1px solid #333; }
        th { color: #ffff00; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Autoresearch Trading Dashboard</h1>
        <p>Running on: Poco X6 Pro (Mobile)</p>
        
        <h2>📊 Strategy Rankings</h2>
        <table>
            <tr>
                <th>Rank</th>
                <th>ID</th>
                <th>Sharpe</th>
                <th>Return</th>
                <th>Mode</th>
            </tr>
            {% for exp in experiments %}
            <tr>
                <td>{{ loop.index }}</td>
                <td>{{ exp.id }}</td>
                <td class="{{ 'positive' if exp.sharpe > 1 else 'negative' }}">{{ "%.2f"|format(exp.sharpe) }}</td>
                <td class="{{ 'positive' if exp.return > 0 else 'negative' }}">{{ "%.2f"|format(exp.return*100) }}%</td>
                <td>{{ exp.mode }}</td>
            </tr>
            {% endfor %}
        </table>
        
        <h2>🧠 LLM Progress</h2>
        <div class="card">
            <p>Best Val BPB: <span class="metric">{{ best_llm_loss }}</span></p>
            <p>Experiments: {{ llm_count }}</p>
        </div>
        
        <h2>⚡ Controls</h2>
        <div class="card">
            <p>Current Mode: <strong>{{ current_mode }}</strong></p>
            <p>Next Experiment: {{ next_exp }}</p>
        </div>
    </div>
</body>
</html>
"""

@app.route("/")
def index():
    results_dir = Path("results")
    experiments = []
    
    llm_best = float('inf')
    llm_count = 0
    current_mode = "Unknown"
    
    if results_dir.exists():
        for f in sorted(results_dir.glob("exp_*.json")):
            with open(f) as file:
                data = json.load(file)
                data['id'] = f.stem.replace("exp_", "")
                experiments.append(data)
                
                if data.get('mode') == 'llm':
                    llm_count += 1
                    if data.get('val_bpb', 999) < llm_best:
                        llm_best = data['val_bpb']
                elif data.get('mode') == 'strategy':
                    current_mode = 'strategy'
    
    # Sort by Sharpe (for strategies) or loss (for LLM)
    experiments.sort(key=lambda x: x.get('sharpe', -x.get('val_bpb', 999)), reverse=True)
    
    return render_template_string(
        HTML_TEMPLATE,
        experiments=experiments[:10],  # Top 10
        best_llm_loss=f"{llm_best:.4f}" if llm_best != float('inf') else "N/A",
        llm_count=llm_count,
        current_mode=current_mode,
        next_exp=int(os.time()) if 'os' in dir() else "Soon"
    )

if __name__ == "__main__":
    print("Starting dashboard on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
