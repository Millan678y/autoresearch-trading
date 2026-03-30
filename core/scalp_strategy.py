"""
PARAMETRIC SCALP STRATEGY v2 — Event-based signals, not state-based.

v1 problem: signals like "EMA fast > slow" are TRUE on ~50% of bars,
leading to 1000+ trades and massive drawdown.

v2 fix: signals detect EVENTS (crossovers, reversals, breakouts) that
fire on ~2-5% of bars. Fewer, higher-quality entries.
"""

import numpy as np
from typing import Dict, List, Optional


class ScalpStrategy:
    """
    Configurable scalping strategy with event-based signals.
    """
    
    def __init__(self, params: dict = None):
        self.params = params or {}
        self.signals = self.params.get("signals", ["ema_cross", "rsi_scalp"])
        self.min_votes = self.params.get("min_votes", 2)
        self.cooldown = self.params.get("cooldown", 12)
        self.tp_mult = self.params.get("tp_mult", 2.0)
        self.sl_mult = self.params.get("sl_mult", 1.0)
        self.size_pct = self.params.get("size_pct", 0.15)
        self.max_hold = self.params.get("max_hold", 72)
        
        self.bar_count = 0
        self.last_trade_bar = -999
    
    def on_bar(self, bar, position, equity):
        from core.scalp_engine import ScalpSignal
        
        self.bar_count += 1
        h = bar.history
        
        if len(h) < 60:
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
                pass
        
        # Position management
        if position is not None:
            bars_held = self.bar_count - position.entry_bar
            if bars_held > self.max_hold:
                return ScalpSignal(symbol=bar.symbol, action="close", reason="max_hold")
            return ScalpSignal(symbol=bar.symbol, action="none")
        
        # Entry — need enough votes
        atr = self._atr(highs, lows, closes, 14)
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
        """Evaluate a signal. Returns (bull_event, bear_event).
        
        ALL signals are EVENT-based — they fire on transitions, not states.
        Each should fire on roughly 2-8% of bars.
        """
        p = self.params
        
        if name == "ema_cross":
            # Fires when EMA fast CROSSES above/below slow (not just above/below)
            fast = p.get("ema_cross__fast", 9)
            slow = p.get("ema_cross__slow", 21)
            ema_f = self._ema(closes, fast)
            ema_s = self._ema(closes, slow)
            # Cross: was below, now above (or vice versa)
            bull = ema_f[-1] > ema_s[-1] and ema_f[-2] <= ema_s[-2]
            bear = ema_f[-1] < ema_s[-1] and ema_f[-2] >= ema_s[-2]
            return bull, bear
        
        elif name == "rsi_scalp":
            # Fires when RSI crosses OUT of oversold/overbought
            period = p.get("rsi_scalp__period", 9)
            oversold = p.get("rsi_scalp__oversold", 30)
            overbought = p.get("rsi_scalp__overbought", 70)
            rsi_now = self._calc_rsi(closes, period)
            rsi_prev = self._calc_rsi(closes[:-1], period)
            # Bull: RSI was below oversold, now above (reversal from oversold)
            bull = rsi_prev < oversold and rsi_now >= oversold
            # Bear: RSI was above overbought, now below
            bear = rsi_prev > overbought and rsi_now <= overbought
            return bull, bear
        
        elif name == "macd_cross":
            # Fires when MACD histogram flips sign
            fast_p = p.get("macd_cross__fast", 12)
            slow_p = p.get("macd_cross__slow", 26)
            sig_p = p.get("macd_cross__signal", 9)
            if len(closes) < slow_p + sig_p + 5:
                return False, False
            ema_f = self._ema(closes, fast_p)
            ema_s = self._ema(closes, slow_p)
            macd_line = ema_f - ema_s
            sig_line = self._ema(macd_line[-sig_p*3:], sig_p)
            if len(sig_line) < 2:
                return False, False
            hist_now = macd_line[-1] - sig_line[-1]
            hist_prev = macd_line[-2] - sig_line[-2]
            bull = hist_now > 0 and hist_prev <= 0
            bear = hist_now < 0 and hist_prev >= 0
            return bull, bear
        
        elif name == "bb_breakout":
            # Fires when price breaks out of Bollinger Bands
            period = p.get("bb_breakout__period", 20)
            std_mult = p.get("bb_breakout__std", 2.0)
            if len(closes) < period + 2:
                return False, False
            sma = np.mean(closes[-period:])
            std = np.std(closes[-period:])
            upper = sma + std_mult * std
            lower = sma - std_mult * std
            sma_prev = np.mean(closes[-period-1:-1])
            std_prev = np.std(closes[-period-1:-1])
            upper_prev = sma_prev + std_mult * std_prev
            lower_prev = sma_prev - std_mult * std_prev
            # Breakout: was inside, now outside
            bull = closes[-1] > upper and closes[-2] <= upper_prev
            bear = closes[-1] < lower and closes[-2] >= lower_prev
            return bull, bear
        
        elif name == "vol_breakout":
            # Volume spike + directional move in same bar
            mult = p.get("vol_breakout__mult", 2.5)
            min_move = p.get("vol_breakout__min_move", 0.003)
            avg_vol = np.mean(volumes[-20:])
            vol_spike = volumes[-1] > avg_vol * mult
            bar_return = (closes[-1] - opens[-1]) / opens[-1] if opens[-1] > 0 else 0
            bull = vol_spike and bar_return > min_move
            bear = vol_spike and bar_return < -min_move
            return bull, bear
        
        elif name == "engulfing":
            # Bullish/bearish engulfing candle pattern
            if len(opens) < 2:
                return False, False
            prev_body = closes[-2] - opens[-2]
            curr_body = closes[-1] - opens[-1]
            # Bullish engulfing: prev red, curr green, curr body covers prev
            bull = (prev_body < 0 and curr_body > 0 and 
                    opens[-1] <= closes[-2] and closes[-1] >= opens[-2])
            # Bearish engulfing: prev green, curr red
            bear = (prev_body > 0 and curr_body < 0 and
                    opens[-1] >= closes[-2] and closes[-1] <= opens[-2])
            return bull, bear
        
        elif name == "support_resist":
            # Price bounces off recent support/resistance
            lb = p.get("support_resist__lookback", 48)
            margin = p.get("support_resist__margin", 0.003)
            if len(lows) < lb:
                return False, False
            recent_low = np.min(lows[-lb:-1])
            recent_high = np.max(highs[-lb:-1])
            price = closes[-1]
            # Bull: touched support and bounced
            near_support = abs(lows[-1] - recent_low) / recent_low < margin
            bounced_up = closes[-1] > opens[-1]
            # Bear: touched resistance and rejected
            near_resist = abs(highs[-1] - recent_high) / recent_high < margin
            rejected_down = closes[-1] < opens[-1]
            bull = near_support and bounced_up
            bear = near_resist and rejected_down
            return bull, bear
        
        elif name == "momentum_shift":
            # Short-term momentum reverses direction
            fast_lb = p.get("momentum_shift__fast", 5)
            slow_lb = p.get("momentum_shift__slow", 15)
            if len(closes) < slow_lb + 2:
                return False, False
            mom_fast = closes[-1] - closes[-fast_lb]
            mom_fast_prev = closes[-2] - closes[-fast_lb-1]
            mom_slow = closes[-1] - closes[-slow_lb]
            # Bull: fast momentum flips positive while slow is negative (mean reversion)
            bull = mom_fast > 0 and mom_fast_prev <= 0 and mom_slow < 0
            # Bear: fast momentum flips negative while slow is positive
            bear = mom_fast < 0 and mom_fast_prev >= 0 and mom_slow > 0
            return bull, bear
        
        elif name == "vwap_cross":
            # Price crosses VWAP
            if len(closes) < 72:
                return False, False
            tp = (highs[-72:] + lows[-72:] + closes[-72:]) / 3
            vols = volumes[-72:]
            vwap = np.sum(tp * vols) / max(np.sum(vols), 1)
            tp_prev = (highs[-73:-1] + lows[-73:-1] + closes[-73:-1]) / 3
            vols_prev = volumes[-73:-1]
            vwap_prev = np.sum(tp_prev * vols_prev) / max(np.sum(vols_prev), 1)
            bull = closes[-1] > vwap and closes[-2] <= vwap_prev
            bear = closes[-1] < vwap and closes[-2] >= vwap_prev
            return bull, bear
        
        elif name == "session_open":
            # Strong move in first bars of London/NY session
            min_move = p.get("session_open__min_move", 0.002)
            if bar.session not in ("london", "ny"):
                return False, False
            # Check if this is near session open (first 6 bars = 30 min)
            hour = (bar.timestamp // 3_600_000) % 24
            is_london_open = bar.session == "london" and hour == 8
            is_ny_open = bar.session == "ny" and hour == 13
            if not (is_london_open or is_ny_open):
                return False, False
            bar_return = (closes[-1] - opens[-1]) / opens[-1] if opens[-1] > 0 else 0
            bull = bar_return > min_move
            bear = bar_return < -min_move
            return bull, bear
        
        elif name == "range_break":
            # Price breaks out of N-bar consolidation range
            n = p.get("range_break__period", 24)
            if len(highs) < n + 1:
                return False, False
            range_high = np.max(highs[-n-1:-1])
            range_low = np.min(lows[-n-1:-1])
            bull = closes[-1] > range_high and closes[-2] <= range_high
            bear = closes[-1] < range_low and closes[-2] >= range_low
            return bull, bear
        
        elif name == "obv_divergence":
            # Price makes new low/high but OBV doesn't (divergence)
            lb = p.get("obv_divergence__lookback", 24)
            if len(closes) < lb + 1:
                return False, False
            obv = np.zeros(len(closes))
            for i in range(1, len(closes)):
                if closes[i] > closes[i-1]:
                    obv[i] = obv[i-1] + volumes[i]
                elif closes[i] < closes[i-1]:
                    obv[i] = obv[i-1] - volumes[i]
                else:
                    obv[i] = obv[i-1]
            # Bull div: price lower low, OBV higher low
            price_ll = closes[-1] < np.min(closes[-lb:-1])
            obv_hl = obv[-1] > np.min(obv[-lb:-1])
            # Bear div: price higher high, OBV lower high
            price_hh = closes[-1] > np.max(closes[-lb:-1])
            obv_lh = obv[-1] < np.max(obv[-lb:-1])
            return price_ll and obv_hl, price_hh and obv_lh
        
        elif name == "wick_rejection":
            # Long wick rejection (pin bar)
            min_wick = p.get("wick_rejection__min_ratio", 2.0)
            rng = highs[-1] - lows[-1]
            if rng <= 0:
                return False, False
            body = abs(closes[-1] - opens[-1])
            body = max(body, rng * 0.01)
            upper_wick = highs[-1] - max(closes[-1], opens[-1])
            lower_wick = min(closes[-1], opens[-1]) - lows[-1]
            # Bull: long lower wick (hammer)
            bull = lower_wick / body > min_wick and upper_wick < body
            # Bear: long upper wick (shooting star)
            bear = upper_wick / body > min_wick and lower_wick < body
            return bull, bear
        
        # Legacy signal names — map to new ones
        elif name == "micro_momentum":
            return self._eval_signal("momentum_shift", closes, highs, lows, opens, volumes, bar, h)
        elif name == "macd_fast":
            return self._eval_signal("macd_cross", closes, highs, lows, opens, volumes, bar, h)
        elif name == "bb_squeeze":
            return self._eval_signal("bb_breakout", closes, highs, lows, opens, volumes, bar, h)
        elif name in ("taker_imbalance", "price_position", "speed_acceleration", 
                       "microvol_regime", "candle_body", "vol_spike", "session_filter",
                       "vwap_position", "obv_trend"):
            # Old state-based signals — redirect to closest event-based version
            remap = {
                "taker_imbalance": "engulfing",
                "price_position": "support_resist",
                "speed_acceleration": "momentum_shift",
                "microvol_regime": "vol_breakout",
                "candle_body": "engulfing",
                "vol_spike": "vol_breakout",
                "session_filter": "session_open",
                "vwap_position": "vwap_cross",
                "obv_trend": "obv_divergence",
            }
            return self._eval_signal(remap[name], closes, highs, lows, opens, volumes, bar, h)
        
        return False, False
    
    def _atr(self, highs, lows, closes, period=14):
        """Average True Range."""
        if len(closes) < period + 1:
            return np.mean(highs[-period:] - lows[-period:])
        tr = np.maximum(
            highs[-period:] - lows[-period:],
            np.maximum(
                np.abs(highs[-period:] - closes[-period-1:-1]),
                np.abs(lows[-period:] - closes[-period-1:-1])
            )
        )
        return float(np.mean(tr))
    
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
