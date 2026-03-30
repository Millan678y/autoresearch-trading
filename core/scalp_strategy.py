"""
PARAMETRIC SCALP STRATEGY — No code generation, just config.

Instead of generating Python code as strings (fragile, breaks easily),
this strategy class takes a config dict and evaluates signals internally.
Zero chance of syntax errors.
"""

import numpy as np
from typing import Dict, List, Optional


class ScalpStrategy:
    """
    Configurable scalping strategy. All signal logic is built-in.
    Configure via params dict — no code generation needed.
    """
    
    def __init__(self, params: dict = None):
        self.params = params or {}
        self.signals = self.params.get("signals", ["ema_cross", "rsi_scalp", "micro_momentum"])
        self.min_votes = self.params.get("min_votes", 2)
        self.cooldown = self.params.get("cooldown", 6)
        self.tp_mult = self.params.get("tp_mult", 2.0)
        self.sl_mult = self.params.get("sl_mult", 1.5)
        self.size_pct = self.params.get("size_pct", 0.5)
        self.max_hold = self.params.get("max_hold", 72)
        
        self.bar_count = 0
        self.last_trade_bar = -999
    
    def on_bar(self, bar, position, equity):
        from core.scalp_engine import ScalpSignal
        
        self.bar_count += 1
        h = bar.history
        
        if len(h) < 50:
            return ScalpSignal(symbol=bar.symbol, action="none")
        
        # Cooldown
        if position is None and (self.bar_count - self.last_trade_bar) < self.cooldown:
            return ScalpSignal(symbol=bar.symbol, action="none")
        
        closes = h["close"].values.astype(float)
        highs = h["high"].values.astype(float)
        lows = h["low"].values.astype(float)
        opens = h["open"].values.astype(float)
        volumes = h["volume"].values.astype(float)
        
        # Compute all signals
        bull_votes = 0
        bear_votes = 0
        
        for sig_name in self.signals:
            try:
                b, s = self._eval_signal(sig_name, closes, highs, lows, opens, volumes, bar, h)
                if b:
                    bull_votes += 1
                if s:
                    bear_votes += 1
            except Exception:
                pass  # Skip broken signals
        
        # Position management
        if position is not None:
            bars_held = self.bar_count - position.entry_bar
            if bars_held > self.max_hold:
                return ScalpSignal(symbol=bar.symbol, action="close", reason="max_hold")
            return ScalpSignal(symbol=bar.symbol, action="none")
        
        # Entry
        atr = float(np.mean(highs[-12:] - lows[-12:]))
        if atr <= 0:
            atr = abs(closes[-1]) * 0.001
        
        if bull_votes >= self.min_votes:
            self.last_trade_bar = self.bar_count
            return ScalpSignal(
                symbol=bar.symbol, action="long", size_pct=self.size_pct,
                take_profit=bar.close + atr * self.tp_mult,
                stop_loss=bar.close - atr * self.sl_mult,
                reason="bull"
            )
        elif bear_votes >= self.min_votes:
            self.last_trade_bar = self.bar_count
            return ScalpSignal(
                symbol=bar.symbol, action="short", size_pct=self.size_pct,
                take_profit=bar.close - atr * self.tp_mult,
                stop_loss=bar.close + atr * self.sl_mult,
                reason="bear"
            )
        
        return ScalpSignal(symbol=bar.symbol, action="none")
    
    def _eval_signal(self, name, closes, highs, lows, opens, volumes, bar, h):
        """Evaluate a single signal. Returns (bull, bear)."""
        p = self.params
        
        if name == "ema_cross":
            fast = p.get("ema_cross__fast", 9)
            slow = p.get("ema_cross__slow", 21)
            ema_f = np.mean(closes[-fast:])
            ema_s = np.mean(closes[-slow:])
            return ema_f > ema_s, ema_f < ema_s
        
        elif name == "rsi_scalp":
            period = p.get("rsi_scalp__period", 9)
            bull_t = p.get("rsi_scalp__bull_thresh", 52)
            bear_t = p.get("rsi_scalp__bear_thresh", 48)
            rsi = self._calc_rsi(closes, period)
            return rsi > bull_t and rsi < 70, rsi < bear_t and rsi > 30
        
        elif name == "micro_momentum":
            lb = p.get("micro_momentum__lookback", 6)
            thresh = p.get("micro_momentum__threshold", 0.002)
            if len(closes) < lb + 1:
                return False, False
            ret = (closes[-1] - closes[-lb]) / closes[-lb]
            return ret > thresh, ret < -thresh
        
        elif name == "taker_imbalance":
            bull_t = p.get("taker_imbalance__bull_thresh", 0.58)
            bear_t = p.get("taker_imbalance__bear_thresh", 0.42)
            # Proxy: close position within candle range
            rng = highs[-1] - lows[-1]
            taker = (closes[-1] - lows[-1]) / rng if rng > 0 else 0.5
            return taker > bull_t, taker < bear_t
        
        elif name == "vol_spike":
            thresh = p.get("vol_spike__thresh", 2.0)
            avg_vol = np.mean(volumes[-12:])
            spike = volumes[-1] / max(avg_vol, 1)
            speed = (closes[-1] - closes[-3]) / closes[-3] if len(closes) > 3 else 0
            return spike > thresh and speed > 0, spike > thresh and speed < 0
        
        elif name == "vwap_position":
            bull_t = p.get("vwap_position__bull_thresh", 0.001)
            bear_t = p.get("vwap_position__bear_thresh", -0.001)
            # Rolling VWAP proxy
            tp = (highs[-72:] + lows[-72:] + closes[-72:]) / 3
            vols = volumes[-72:]
            vwap = np.sum(tp * vols) / max(np.sum(vols), 1)
            pvw = (closes[-1] - vwap) / max(vwap, 1)
            return pvw > bull_t, pvw < bear_t
        
        elif name == "candle_body":
            min_body = p.get("candle_body__min_body", 0.6)
            rng = highs[-1] - lows[-1]
            body = abs(closes[-1] - opens[-1])
            ratio = body / rng if rng > 0 else 0
            bullish = closes[-1] > opens[-1]
            return ratio > min_body and bullish, ratio > min_body and not bullish
        
        elif name == "price_position":
            lb = p.get("price_position__lookback", 36)
            oversold = p.get("price_position__oversold", 0.25)
            overbought = p.get("price_position__overbought", 0.75)
            if len(highs) < lb:
                return False, False
            rh = np.max(highs[-lb:])
            rl = np.min(lows[-lb:])
            pos = (closes[-1] - rl) / (rh - rl) if rh > rl else 0.5
            return pos < oversold, pos > overbought
        
        elif name == "speed_acceleration":
            thresh = p.get("speed_acceleration__threshold", 0.0003)
            if len(closes) < 7:
                return False, False
            speed_now = (closes[-1] - closes[-4]) / 3
            speed_prev = (closes[-4] - closes[-7]) / 3
            acc = speed_now - speed_prev
            return acc > thresh, acc < -thresh
        
        elif name == "session_filter":
            good = bar.session in ("london", "ny")
            if not good or len(closes) < 4:
                return False, False
            return closes[-1] > closes[-3], closes[-1] < closes[-3]
        
        elif name == "microvol_regime":
            thresh = p.get("microvol_regime__expand_thresh", 1.3)
            if len(closes) < 14:
                return False, False
            mv6 = np.std(np.diff(closes[-7:]))
            mv12 = np.std(np.diff(closes[-13:]))
            mvr = mv6 / max(mv12, 1e-10)
            expanding = mvr > thresh
            return expanding and closes[-1] > closes[-2], expanding and closes[-1] < closes[-2]
        
        elif name == "macd_fast":
            fast_p = p.get("macd_fast__fast", 8)
            slow_p = p.get("macd_fast__slow", 17)
            sig_p = p.get("macd_fast__signal", 6)
            if len(closes) < slow_p + sig_p + 5:
                return False, False
            ema_f = self._ema(closes, fast_p)
            ema_s = self._ema(closes, slow_p)
            macd_line = ema_f - ema_s
            sig_line = self._ema(macd_line, sig_p)
            hist = macd_line[-1] - sig_line[-1]
            return hist > 0, hist < 0
        
        elif name == "bb_squeeze":
            period = p.get("bb_squeeze__period", 12)
            if len(closes) < period * 2:
                return False, False
            sma = np.mean(closes[-period:])
            std = np.std(closes[-period:])
            width = (2 * std) / sma if sma > 0 else 0
            # Is width below median?
            widths = []
            for i in range(period, min(len(closes), period * 5)):
                w = np.std(closes[-i-period:-i]) * 2 / max(np.mean(closes[-i-period:-i]), 1)
                widths.append(w)
            if widths:
                compressed = width < np.median(widths)
            else:
                compressed = False
            return compressed and closes[-1] > closes[-2], compressed and closes[-1] < closes[-2]
        
        elif name == "obv_trend":
            period = p.get("obv_trend__period", 15)
            if len(closes) < period + 1:
                return False, False
            obv = np.zeros(len(closes))
            for i in range(1, len(closes)):
                if closes[i] > closes[i-1]:
                    obv[i] = obv[i-1] + volumes[i]
                elif closes[i] < closes[i-1]:
                    obv[i] = obv[i-1] - volumes[i]
                else:
                    obv[i] = obv[i-1]
            obv_sma = np.mean(obv[-period:])
            return obv[-1] > obv_sma, obv[-1] < obv_sma
        
        return False, False
    
    def _calc_rsi(self, closes, period):
        if len(closes) < period + 1:
            return 50.0
        deltas = np.diff(closes[-(period+1):])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - 100 / (1 + rs)
    
    def _ema(self, values, span):
        alpha = 2.0 / (span + 1)
        result = np.empty(len(values), dtype=float)
        result[0] = values[0]
        for i in range(1, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
        return result
