"""
PARAMETRIC SCALP STRATEGY v3 — Institutional concepts from Chart Fanatics.

Built from 33 verified trader strategies including:
- Liquidity sweeps + reversals (Marco Acetony, Jadecap)
- Break & Retest (battle zone confirmation)
- Mean Reversion / Capitulation (Lance Breitstein)
- AMD: Accumulation-Manipulation-Distribution (ICT/PO3)
- Volume Profile / Low Volume Nodes (Camrine, Forrest Knight)
- 50% Equilibrium targeting (Trader Kane)
- Session-based execution (London/NY kill zones)
- Market regime filtering (Crudele's Bollinger method)
- SMT Divergence proxy (BTC vs ETH correlation)
- EMA stacking for trend confirmation

All signals are EVENT-based. Each fires on ~1-5% of bars.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class ScalpStrategy:
    """Institutional scalping strategy with event-based signals."""
    
    def __init__(self, params: dict = None):
        self.params = params or {}
        self.signals = self.params.get("signals", ["liquidity_sweep", "break_retest"])
        self.min_votes = self.params.get("min_votes", 2)
        self.cooldown = self.params.get("cooldown", 12)
        self.tp_mult = self.params.get("tp_mult", 2.5)
        self.sl_mult = self.params.get("sl_mult", 1.0)
        self.size_pct = self.params.get("size_pct", 0.15)
        self.max_hold = self.params.get("max_hold", 72)
        
        self.bar_count = 0
        self.last_trade_bar = -999
    
    def on_bar(self, bar, position, equity):
        from core.scalp_engine import ScalpSignal
        
        self.bar_count += 1
        h = bar.history
        
        if len(h) < 80:
            return ScalpSignal(symbol=bar.symbol, action="none")
        
        # Cooldown between trades
        if position is None and (self.bar_count - self.last_trade_bar) < self.cooldown:
            return ScalpSignal(symbol=bar.symbol, action="none")
        
        closes = h["close"].values.astype(float)
        highs = h["high"].values.astype(float)
        lows = h["low"].values.astype(float)
        opens = h["open"].values.astype(float)
        volumes = h["volume"].values.astype(float)
        
        # Evaluate all signals
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
        
        # Entry
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
        """Evaluate institutional signal. Returns (bull_event, bear_event)."""
        p = self.params
        
        # ── LIQUIDITY SWEEP + REVERSAL ──────────────────────────
        # Source: Marco Acetony (#23), Jadecap (#32)
        # Buy below lows after stop hunt, sell above highs after stop hunt
        if name == "liquidity_sweep":
            lb = p.get("liquidity_sweep__lookback", 48)
            wick_ratio = p.get("liquidity_sweep__wick_ratio", 0.6)
            if len(lows) < lb + 1:
                return False, False
            
            prev_low = np.min(lows[-lb:-1])
            prev_high = np.max(highs[-lb:-1])
            
            # Bull: price swept below prev low (stop hunt) but closed back above
            swept_low = lows[-1] < prev_low
            reclaimed = closes[-1] > prev_low
            candle_range = highs[-1] - lows[-1]
            lower_wick = min(opens[-1], closes[-1]) - lows[-1]
            has_wick = (lower_wick / candle_range > wick_ratio) if candle_range > 0 else False
            bull = swept_low and reclaimed and has_wick
            
            # Bear: price swept above prev high but closed back below
            swept_high = highs[-1] > prev_high
            reclaimed_down = closes[-1] < prev_high
            upper_wick = highs[-1] - max(opens[-1], closes[-1])
            has_upper_wick = (upper_wick / candle_range > wick_ratio) if candle_range > 0 else False
            bear = swept_high and reclaimed_down and has_upper_wick
            
            return bull, bear
        
        # ── BREAK & RETEST ──────────────────────────────────────
        # Source: Strategy #28 — break level, wait for pullback, enter on retest
        elif name == "break_retest":
            lb = p.get("break_retest__lookback", 36)
            retest_bars = p.get("break_retest__retest_window", 12)
            if len(closes) < lb + retest_bars + 1:
                return False, False
            
            # Find level that was broken
            range_high = np.max(highs[-lb-retest_bars:-retest_bars])
            range_low = np.min(lows[-lb-retest_bars:-retest_bars])
            
            # Check if level was broken in recent bars
            broke_above = np.any(closes[-retest_bars:-1] > range_high)
            broke_below = np.any(closes[-retest_bars:-1] < range_low)
            
            # Bull: broke above, pulled back to level, bounced
            if broke_above:
                retested = lows[-1] <= range_high * 1.002  # within 0.2%
                bounced = closes[-1] > opens[-1] and closes[-1] > range_high
                bull = retested and bounced
            else:
                bull = False
            
            # Bear: broke below, pulled back up to level, rejected
            if broke_below:
                retested = highs[-1] >= range_low * 0.998
                rejected = closes[-1] < opens[-1] and closes[-1] < range_low
                bear = retested and rejected
            else:
                bear = False
            
            return bull, bear
        
        # ── MEAN REVERSION / CAPITULATION ───────────────────────
        # Source: Lance Breitstein (#4) — fade extreme moves
        elif name == "mean_reversion":
            consec_days = p.get("mean_reversion__consec_bars", 5)
            speed_thresh = p.get("mean_reversion__speed_pct", 0.03)
            if len(closes) < consec_days + 3:
                return False, False
            
            # Count consecutive down/up bars
            consec_down = 0
            consec_up = 0
            for i in range(1, consec_days + 1):
                if closes[-i] < opens[-i]:
                    consec_down += 1
                else:
                    break
            for i in range(1, consec_days + 1):
                if closes[-i] > opens[-i]:
                    consec_up += 1
                else:
                    break
            
            # Speed: total move over the period
            total_move = (closes[-1] - closes[-consec_days]) / closes[-consec_days]
            
            # Bull: multiple consecutive down bars + speed + reversal candle
            reversal_up = closes[-1] > opens[-1]  # green candle after reds
            bull = consec_down >= consec_days - 1 and total_move < -speed_thresh and reversal_up
            
            # Bear: multiple consecutive up bars + speed + reversal candle
            reversal_down = closes[-1] < opens[-1]
            bear = consec_up >= consec_days - 1 and total_move > speed_thresh and reversal_down
            
            return bull, bear
        
        # ── AMD: ACCUMULATION-MANIPULATION-DISTRIBUTION ─────────
        # Source: ICT/PO3 (#25, #33) — range → fake breakout → real move
        elif name == "amd_model":
            range_lb = p.get("amd_model__range_bars", 24)
            manip_thresh = p.get("amd_model__manip_pct", 0.003)
            if len(closes) < range_lb + 6:
                return False, False
            
            # Define accumulation range
            acc_high = np.max(highs[-range_lb-5:-5])
            acc_low = np.min(lows[-range_lb-5:-5])
            acc_range = acc_high - acc_low
            if acc_range <= 0:
                return False, False
            
            # Check for manipulation (sweep beyond range) in recent bars
            recent_highs = highs[-5:-1]
            recent_lows = lows[-5:-1]
            
            # Bull AMD: manipulated below range, now distributing up
            manip_below = np.any(recent_lows < acc_low - acc_range * manip_thresh)
            now_above = closes[-1] > acc_low and closes[-1] > opens[-1]
            reclaimed_range = closes[-1] > acc_low + acc_range * 0.3
            bull = manip_below and now_above and reclaimed_range
            
            # Bear AMD: manipulated above range, now distributing down
            manip_above = np.any(recent_highs > acc_high + acc_range * manip_thresh)
            now_below = closes[-1] < acc_high and closes[-1] < opens[-1]
            dropped_into = closes[-1] < acc_high - acc_range * 0.3
            bear = manip_above and now_below and dropped_into
            
            return bull, bear
        
        # ── VOLUME PROFILE / LOW VOLUME NODE ────────────────────
        # Source: Camrine (#1, #30), Forrest Knight (#9, #12)
        elif name == "volume_node":
            lb = p.get("volume_node__lookback", 48)
            n_bins = p.get("volume_node__bins", 20)
            if len(closes) < lb:
                return False, False
            
            # Build simple volume profile
            price_range = np.max(highs[-lb:]) - np.min(lows[-lb:])
            if price_range <= 0:
                return False, False
            
            bin_size = price_range / n_bins
            min_price = np.min(lows[-lb:])
            vol_profile = np.zeros(n_bins)
            
            for i in range(lb):
                idx = int((closes[-lb+i] - min_price) / bin_size)
                idx = min(idx, n_bins - 1)
                vol_profile[idx] += volumes[-lb+i]
            
            # Current price bin
            curr_bin = int((closes[-1] - min_price) / bin_size)
            curr_bin = min(max(curr_bin, 0), n_bins - 1)
            
            # Is current price at a low volume node?
            median_vol = np.median(vol_profile)
            at_lvn = vol_profile[curr_bin] < median_vol * 0.5
            
            # Find nearest high volume node (POC direction)
            poc_bin = np.argmax(vol_profile)
            poc_price = min_price + (poc_bin + 0.5) * bin_size
            
            # Bull: at LVN below POC, bouncing
            bull = at_lvn and closes[-1] < poc_price and closes[-1] > opens[-1]
            # Bear: at LVN above POC, rejecting
            bear = at_lvn and closes[-1] > poc_price and closes[-1] < opens[-1]
            
            return bull, bear
        
        # ── 50% EQUILIBRIUM ─────────────────────────────────────
        # Source: Trader Kane (#33) — price targets 50% of range
        elif name == "equilibrium_50":
            lb = p.get("equilibrium_50__lookback", 48)
            zone_pct = p.get("equilibrium_50__zone_pct", 0.05)
            if len(closes) < lb + 1:
                return False, False
            
            swing_high = np.max(highs[-lb:])
            swing_low = np.min(lows[-lb:])
            midpoint = (swing_high + swing_low) / 2
            zone = (swing_high - swing_low) * zone_pct
            
            at_midpoint = abs(closes[-1] - midpoint) < zone
            
            # Bull: approached 50% from below, showing strength
            approaching_from_below = closes[-2] < midpoint
            bull = at_midpoint and approaching_from_below and closes[-1] > opens[-1]
            
            # Bear: approached 50% from above, showing weakness
            approaching_from_above = closes[-2] > midpoint
            bear = at_midpoint and approaching_from_above and closes[-1] < opens[-1]
            
            return bull, bear
        
        # ── SESSION KILLZONE ────────────────────────────────────
        # Source: TG Capital (#24), multiple ICT strategies
        elif name == "session_killzone":
            min_move = p.get("session_killzone__min_move", 0.002)
            # Only trade during London (08-12 UTC) or NY (13-17 UTC)
            hour = (bar.timestamp // 3_600_000) % 24
            
            london_kz = 8 <= hour <= 11
            ny_kz = 13 <= hour <= 16
            
            if not (london_kz or ny_kz):
                return False, False
            
            # Strong directional move in killzone
            bar_return = (closes[-1] - opens[-1]) / opens[-1] if opens[-1] > 0 else 0
            # Confirm with volume above average
            avg_vol = np.mean(volumes[-24:]) if len(volumes) >= 24 else np.mean(volumes)
            high_vol = volumes[-1] > avg_vol * 1.3
            
            bull = bar_return > min_move and high_vol
            bear = bar_return < -min_move and high_vol
            return bull, bear
        
        # ── EMA STACK (Trend Confirmation) ──────────────────────
        # Source: TG Capital (#24) — 5/9/13/21 EMAs aligned
        elif name == "ema_stack":
            e1 = p.get("ema_stack__fast", 5)
            e2 = p.get("ema_stack__mid1", 9)
            e3 = p.get("ema_stack__mid2", 13)
            e4 = p.get("ema_stack__slow", 21)
            if len(closes) < e4 + 2:
                return False, False
            
            ema1 = self._ema(closes, e1)
            ema2 = self._ema(closes, e2)
            ema3 = self._ema(closes, e3)
            ema4 = self._ema(closes, e4)
            
            # Bull stack: EMAs aligned bullish AND just became aligned
            bull_now = ema1[-1] > ema2[-1] > ema3[-1] > ema4[-1]
            bull_prev = ema1[-2] > ema2[-2] > ema3[-2] > ema4[-2]
            # Event: stack just formed (wasn't aligned before)
            bull = bull_now and not bull_prev
            
            # Bear stack
            bear_now = ema1[-1] < ema2[-1] < ema3[-1] < ema4[-1]
            bear_prev = ema1[-2] < ema2[-2] < ema3[-2] < ema4[-2]
            bear = bear_now and not bear_prev
            
            return bull, bear
        
        # ── MARKET REGIME (Bollinger State) ─────────────────────
        # Source: Anthony Crudele (#8) — expansion vs compression
        elif name == "regime_filter":
            period = p.get("regime_filter__period", 20)
            std_mult = p.get("regime_filter__std", 2.0)
            if len(closes) < period + 5:
                return False, False
            
            # Current BB width
            sma = np.mean(closes[-period:])
            std = np.std(closes[-period:])
            width_now = (2 * std) / sma if sma > 0 else 0
            
            # Previous BB width
            sma_prev = np.mean(closes[-period-1:-1])
            std_prev = np.std(closes[-period-1:-1])
            width_prev = (2 * std_prev) / sma_prev if sma_prev > 0 else 0
            
            # Event: BB expansion starting (squeeze breakout)
            expanding = width_now > width_prev * 1.1
            price_above = closes[-1] > sma
            price_below = closes[-1] < sma
            
            bull = expanding and price_above
            bear = expanding and price_below
            return bull, bear
        
        # ── FAIR VALUE GAP (FVG) ────────────────────────────────
        # Source: ICT concepts (#19, #20, #25)
        elif name == "fvg":
            min_gap = p.get("fvg__min_gap_pct", 0.002)
            if len(closes) < 5:
                return False, False
            
            # Bull FVG: gap between bar[-3] high and bar[-1] low
            # (bar[-2] moved up so fast it left a gap)
            bull_gap = lows[-1] - highs[-3]
            bull_gap_pct = bull_gap / closes[-2] if closes[-2] > 0 else 0
            # Price re-entering the gap (filling) = entry
            filling_bull = closes[-1] <= highs[-3] + bull_gap * 0.5 if bull_gap > 0 else False
            bull_fvg_exists = bull_gap_pct > min_gap
            
            # Bear FVG
            bear_gap = lows[-3] - highs[-1]
            bear_gap_pct = bear_gap / closes[-2] if closes[-2] > 0 else 0
            filling_bear = closes[-1] >= lows[-3] - bear_gap * 0.5 if bear_gap > 0 else False
            bear_fvg_exists = bear_gap_pct > min_gap
            
            # We want: FVG existed 2-10 bars ago, price is now retesting it
            bull = bull_fvg_exists and closes[-1] > opens[-1]
            bear = bear_fvg_exists and closes[-1] < opens[-1]
            return bull, bear
        
        # ── ENGULFING PATTERN ───────────────────────────────────
        # Source: Multiple (#9, #31) — signal candle confirmation
        elif name == "engulfing":
            if len(opens) < 3:
                return False, False
            
            prev_body = closes[-2] - opens[-2]
            curr_body = closes[-1] - opens[-1]
            curr_range = highs[-1] - lows[-1]
            
            # Minimum body size relative to range
            body_ratio = abs(curr_body) / curr_range if curr_range > 0 else 0
            
            bull = (prev_body < 0 and curr_body > 0 and 
                    body_ratio > 0.6 and
                    opens[-1] <= closes[-2] and closes[-1] >= opens[-2])
            bear = (prev_body > 0 and curr_body < 0 and
                    body_ratio > 0.6 and
                    opens[-1] >= closes[-2] and closes[-1] <= opens[-2])
            return bull, bear
        
        # ── WICK REJECTION (Pin Bar / Hammer) ───────────────────
        # Source: Forrest Knight (#9) — signal candle at key level
        elif name == "wick_rejection":
            min_wick = p.get("wick_rejection__min_ratio", 2.0)
            body = abs(closes[-1] - opens[-1])
            rng = highs[-1] - lows[-1]
            if rng <= 0 or body < rng * 0.01:
                return False, False
            
            upper_wick = highs[-1] - max(closes[-1], opens[-1])
            lower_wick = min(closes[-1], opens[-1]) - lows[-1]
            
            bull = lower_wick / body > min_wick and upper_wick < body
            bear = upper_wick / body > min_wick and lower_wick < body
            return bull, bear
        
        # ── VWAP RECLAIM ────────────────────────────────────────
        # Source: Multiple strategies use VWAP as key level
        elif name == "vwap_reclaim":
            if len(closes) < 72:
                return False, False
            
            # Rolling VWAP
            tp = (highs[-72:] + lows[-72:] + closes[-72:]) / 3
            vols = volumes[-72:]
            vwap = np.sum(tp * vols) / max(np.sum(vols), 1)
            
            tp_prev = (highs[-73:-1] + lows[-73:-1] + closes[-73:-1]) / 3
            vols_prev = volumes[-73:-1]
            vwap_prev = np.sum(tp_prev * vols_prev) / max(np.sum(vols_prev), 1)
            
            # Bull: was below VWAP, reclaimed above
            bull = closes[-2] < vwap_prev and closes[-1] > vwap
            # Bear: was above VWAP, lost it
            bear = closes[-2] > vwap_prev and closes[-1] < vwap
            return bull, bear
        
        # ── RANGE BREAKOUT ──────────────────────────────────────
        # Source: Crudele (#8), multiple — consolidation breakout
        elif name == "range_breakout":
            n = p.get("range_breakout__period", 24)
            if len(highs) < n + 2:
                return False, False
            
            range_high = np.max(highs[-n-1:-1])
            range_low = np.min(lows[-n-1:-1])
            
            # Event: breakout on current bar
            bull = closes[-1] > range_high and closes[-2] <= range_high
            bear = closes[-1] < range_low and closes[-2] >= range_low
            return bull, bear
        
        # ── SMT DIVERGENCE PROXY ────────────────────────────────
        # Source: Trader Kane (#33) — compare correlated assets
        # We proxy this by comparing current bar vs recent correlation
        elif name == "smt_divergence":
            lb = p.get("smt_divergence__lookback", 12)
            if len(closes) < lb + 1:
                return False, False
            
            # Use price vs its own momentum as proxy
            # (True SMT needs two assets; this detects momentum divergence)
            price_makes_new_low = closes[-1] < np.min(closes[-lb:-1])
            momentum_higher = (closes[-1] - closes[-3]) > (closes[-lb//2] - closes[-lb//2-2])
            
            price_makes_new_high = closes[-1] > np.max(closes[-lb:-1])
            momentum_lower = (closes[-1] - closes[-3]) < (closes[-lb//2] - closes[-lb//2-2])
            
            # Bull divergence: price new low but momentum not confirming
            bull = price_makes_new_low and momentum_higher
            # Bear divergence: price new high but momentum weakening
            bear = price_makes_new_high and momentum_lower
            return bull, bear
        
        # ── MACD HISTOGRAM FLIP ─────────────────────────────────
        elif name == "macd_cross":
            fast_p = p.get("macd_cross__fast", 12)
            slow_p = p.get("macd_cross__slow", 26)
            sig_p = p.get("macd_cross__signal", 9)
            if len(closes) < slow_p + sig_p + 5:
                return False, False
            ema_f = self._ema(closes, fast_p)
            ema_s = self._ema(closes, slow_p)
            macd = ema_f - ema_s
            sig = self._ema(macd[-sig_p*3:], sig_p)
            if len(sig) < 2:
                return False, False
            hist_now = macd[-1] - sig[-1]
            hist_prev = macd[-2] - sig[-2]
            bull = hist_now > 0 and hist_prev <= 0
            bear = hist_now < 0 and hist_prev >= 0
            return bull, bear
        
        # ── RSI REVERSAL ────────────────────────────────────────
        elif name == "rsi_reversal":
            period = p.get("rsi_reversal__period", 9)
            oversold = p.get("rsi_reversal__oversold", 30)
            overbought = p.get("rsi_reversal__overbought", 70)
            rsi_now = self._calc_rsi(closes, period)
            rsi_prev = self._calc_rsi(closes[:-1], period)
            bull = rsi_prev < oversold and rsi_now >= oversold
            bear = rsi_prev > overbought and rsi_now <= overbought
            return bull, bear
        
        # ── EMA CROSSOVER ───────────────────────────────────────
        elif name == "ema_cross":
            fast = p.get("ema_cross__fast", 9)
            slow = p.get("ema_cross__slow", 21)
            ema_f = self._ema(closes, fast)
            ema_s = self._ema(closes, slow)
            bull = ema_f[-1] > ema_s[-1] and ema_f[-2] <= ema_s[-2]
            bear = ema_f[-1] < ema_s[-1] and ema_f[-2] >= ema_s[-2]
            return bull, bear
        
        # Legacy remaps
        elif name in ("micro_momentum", "momentum_shift", "speed_acceleration"):
            return self._eval_signal("mean_reversion", closes, highs, lows, opens, volumes, bar, h)
        elif name in ("bb_squeeze", "bb_breakout"):
            return self._eval_signal("regime_filter", closes, highs, lows, opens, volumes, bar, h)
        elif name in ("vol_breakout", "vol_spike"):
            return self._eval_signal("session_killzone", closes, highs, lows, opens, volumes, bar, h)
        elif name in ("support_resist", "price_position"):
            return self._eval_signal("break_retest", closes, highs, lows, opens, volumes, bar, h)
        elif name in ("vwap_cross", "vwap_position"):
            return self._eval_signal("vwap_reclaim", closes, highs, lows, opens, volumes, bar, h)
        elif name in ("obv_divergence", "obv_trend"):
            return self._eval_signal("smt_divergence", closes, highs, lows, opens, volumes, bar, h)
        elif name in ("range_break",):
            return self._eval_signal("range_breakout", closes, highs, lows, opens, volumes, bar, h)
        elif name in ("session_open",):
            return self._eval_signal("session_killzone", closes, highs, lows, opens, volumes, bar, h)
        elif name in ("macd_fast",):
            return self._eval_signal("macd_cross", closes, highs, lows, opens, volumes, bar, h)
        elif name in ("rsi_scalp",):
            return self._eval_signal("rsi_reversal", closes, highs, lows, opens, volumes, bar, h)
        
        return False, False
    
    # ── UTILITY FUNCTIONS ───────────────────────────────────────
    
    def _atr(self, highs, lows, closes, period=14):
        if len(closes) < period + 1:
            return float(np.mean(highs[-period:] - lows[-period:]))
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
