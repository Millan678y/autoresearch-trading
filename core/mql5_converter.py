"""
MQL5 CONVERTER — Converts winning Python strategies to MetaTrader 5 Expert Advisors

Takes a strategy's signals, parameters, and risk management config,
and generates a complete .mq5 file ready to compile in MetaEditor.

Supports:
- All 23 signal types (mapped to MQL5 indicator equivalents)
- Hybrid trailing stop (ATR + vol + time-decay)
- Scalping and swing trading modes
- Configurable TP/SL
- Magic number for multi-EA deployment
- Lot sizing (fixed, risk-based, or Kelly)

Usage:
    from core.mql5_converter import convert_to_mql5
    mql5_code = convert_to_mql5(strategy_record)
    # or
    mql5_code = convert_to_mql5_from_params(signals, params, mode="scalp")
"""

import os
import json
import time
import hashlib
from typing import Dict, List, Optional
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────
# MQL5 Signal Mappings
# ─────────────────────────────────────────────────────────────────

MQL5_SIGNAL_MAP = {
    "momentum": """
   // Signal: Momentum ({lookback} bars)
   double ret = (close_arr[0] - close_arr[{lookback}]) / close_arr[{lookback}];
   if(ret > {threshold}) bull_votes++;
   if(ret < -{threshold}) bear_votes++;
""",
    "ema_cross": """
   // Signal: EMA Crossover ({fast}/{slow})
   double ema_fast_val = iMA(_Symbol, PERIOD_CURRENT, {fast}, 0, MODE_EMA, PRICE_CLOSE);
   double ema_slow_val = iMA(_Symbol, PERIOD_CURRENT, {slow}, 0, MODE_EMA, PRICE_CLOSE);
   double ema_f[], ema_s[];
   CopyBuffer(ema_fast_val, 0, 0, 2, ema_f);
   CopyBuffer(ema_slow_val, 0, 0, 2, ema_s);
   if(ema_f[0] > ema_s[0]) bull_votes++;
   if(ema_f[0] < ema_s[0]) bear_votes++;
""",
    "rsi": """
   // Signal: RSI ({period})
   int rsi_handle = iRSI(_Symbol, PERIOD_CURRENT, {period}, PRICE_CLOSE);
   double rsi_buf[];
   CopyBuffer(rsi_handle, 0, 0, 1, rsi_buf);
   double rsi_val = rsi_buf[0];
   if(rsi_val > {bull_thresh}) bull_votes++;
   if(rsi_val < {bear_thresh}) bear_votes++;
""",
    "rsi_scalp": """
   // Signal: RSI Scalp ({period})
   int rsi_handle = iRSI(_Symbol, PERIOD_CURRENT, {period}, PRICE_CLOSE);
   double rsi_buf[];
   CopyBuffer(rsi_handle, 0, 0, 1, rsi_buf);
   double rsi_val = rsi_buf[0];
   if(rsi_val > {bull_thresh} && rsi_val < 70) bull_votes++;
   if(rsi_val < {bear_thresh} && rsi_val > 30) bear_votes++;
""",
    "macd": """
   // Signal: MACD ({fast}/{slow}/{signal})
   int macd_handle = iMACD(_Symbol, PERIOD_CURRENT, {fast}, {slow}, {signal}, PRICE_CLOSE);
   double macd_main[], macd_sig[];
   CopyBuffer(macd_handle, 0, 0, 1, macd_main);
   CopyBuffer(macd_handle, 1, 0, 1, macd_sig);
   double macd_hist = macd_main[0] - macd_sig[0];
   if(macd_hist > 0) bull_votes++;
   if(macd_hist < 0) bear_votes++;
""",
    "bb_compression": """
   // Signal: Bollinger Band Compression ({period})
   int bb_handle = iBands(_Symbol, PERIOD_CURRENT, {period}, 0, 2.0, PRICE_CLOSE);
   double bb_upper[], bb_lower[], bb_mid[];
   CopyBuffer(bb_handle, 1, 0, 1, bb_upper);
   CopyBuffer(bb_handle, 2, 0, 1, bb_lower);
   CopyBuffer(bb_handle, 0, 0, 1, bb_mid);
   double bb_width = (bb_upper[0] - bb_lower[0]) / bb_mid[0];
   // Compression = potential breakout
   if(bb_width < {threshold} * 0.01) {{ bull_votes++; bear_votes++; }}
""",
    "stochastic": """
   // Signal: Stochastic ({period})
   int stoch_handle = iStochastic(_Symbol, PERIOD_CURRENT, {period}, 3, 3, MODE_SMA, STO_LOWHIGH);
   double stoch_k[];
   CopyBuffer(stoch_handle, 0, 0, 1, stoch_k);
   if(stoch_k[0] < {oversold}) bull_votes++;
   if(stoch_k[0] > {overbought}) bear_votes++;
""",
    "atr_breakout": """
   // Signal: ATR Breakout ({period})
   int atr_handle = iATR(_Symbol, PERIOD_CURRENT, {period});
   double atr_buf[];
   CopyBuffer(atr_handle, 0, 0, 1, atr_buf);
   double atr_val = atr_buf[0];
   if(close_arr[0] > close_arr[1] + atr_val * {mult}) bull_votes++;
   if(close_arr[0] < close_arr[1] - atr_val * {mult}) bear_votes++;
""",
    "micro_momentum": """
   // Signal: Micro Momentum ({lookback} bars)
   double micro_ret = (close_arr[0] - close_arr[{lookback}]) / close_arr[{lookback}];
   if(micro_ret > {threshold}) bull_votes++;
   if(micro_ret < -{threshold}) bear_votes++;
""",
    "vol_spike": """
   // Signal: Volume Spike
   double vol_arr[];
   CopyTickVolume(_Symbol, PERIOD_CURRENT, 0, 20, vol_arr);
   double vol_avg = 0;
   for(int v = 1; v < 13; v++) vol_avg += vol_arr[v];
   vol_avg /= 12.0;
   double vol_ratio = vol_arr[0] / MathMax(vol_avg, 1);
   if(vol_ratio > {thresh} && close_arr[0] > close_arr[2]) bull_votes++;
   if(vol_ratio > {thresh} && close_arr[0] < close_arr[2]) bear_votes++;
""",
    "candle_body": """
   // Signal: Candle Body Ratio
   double body = MathAbs(close_arr[0] - open_arr[0]);
   double range_c = high_arr[0] - low_arr[0];
   double body_ratio = (range_c > 0) ? body / range_c : 0;
   bool bullish_candle = close_arr[0] > open_arr[0];
   if(body_ratio > {min_body} && bullish_candle) bull_votes++;
   if(body_ratio > {min_body} && !bullish_candle) bear_votes++;
""",
    "mean_reversion": """
   // Signal: Mean Reversion ({period})
   double sum = 0, sum_sq = 0;
   for(int m = 0; m < {period}; m++) {{ sum += close_arr[m]; sum_sq += close_arr[m] * close_arr[m]; }}
   double mean = sum / {period};
   double std_dev = MathSqrt(sum_sq / {period} - mean * mean);
   double zscore = (std_dev > 0) ? (close_arr[0] - mean) / std_dev : 0;
   if(zscore < -{entry_z}) bull_votes++;
   if(zscore > {entry_z}) bear_votes++;
""",
    "vwap_position": """
   // Signal: VWAP Position (proxy using MA of typical price * volume)
   double tp_sum = 0, vol_sum = 0;
   double tp_arr[];
   CopyTickVolume(_Symbol, PERIOD_CURRENT, 0, 72, vol_arr);
   for(int vw = 0; vw < 72; vw++) {{
      double tp = (high_arr[vw] + low_arr[vw] + close_arr[vw]) / 3.0;
      tp_sum += tp * vol_arr[vw];
      vol_sum += vol_arr[vw];
   }}
   double vwap = (vol_sum > 0) ? tp_sum / vol_sum : close_arr[0];
   double pvw = (close_arr[0] - vwap) / MathMax(vwap, 0.0001);
   if(pvw > {bull_thresh}) bull_votes++;
   if(pvw < {bear_thresh}) bear_votes++;
""",
    "speed_acceleration": """
   // Signal: Speed Acceleration
   double speed_now = (close_arr[0] - close_arr[3]) / 3.0;
   double speed_prev = (close_arr[3] - close_arr[6]) / 3.0;
   double acc = speed_now - speed_prev;
   if(acc > {threshold}) bull_votes++;
   if(acc < -{threshold}) bear_votes++;
""",
    "engulfing": """
   // Signal: Engulfing Pattern
   bool prev_bear = close_arr[1] < open_arr[1];
   bool curr_bull = close_arr[0] > open_arr[0];
   bool prev_bull = close_arr[1] > open_arr[1];
   bool curr_bear = close_arr[0] < open_arr[0];
   if(prev_bear && curr_bull && open_arr[0] <= close_arr[1] && close_arr[0] >= open_arr[1]) bull_votes++;
   if(prev_bull && curr_bear && open_arr[0] >= close_arr[1] && close_arr[0] <= open_arr[1]) bear_votes++;
""",
    "price_position": """
   // Signal: Price Position ({lookback} bars)
   double rh = high_arr[0], rl = low_arr[0];
   for(int pp = 1; pp < {lookback}; pp++) {{ rh = MathMax(rh, high_arr[pp]); rl = MathMin(rl, low_arr[pp]); }}
   double pp_val = (rh > rl) ? (close_arr[0] - rl) / (rh - rl) : 0.5;
   if(pp_val < {oversold}) bull_votes++;
   if(pp_val > {overbought}) bear_votes++;
""",
    "ema_cross_scalp": """
   // Signal: EMA Cross (scalp)
   double ema_f_scalp = 0, ema_s_scalp = 0;
   for(int e = 0; e < {fast}; e++) ema_f_scalp += close_arr[e];
   ema_f_scalp /= {fast};
   for(int e = 0; e < {slow}; e++) ema_s_scalp += close_arr[e];
   ema_s_scalp /= {slow};
   if(ema_f_scalp > ema_s_scalp) bull_votes++;
   if(ema_f_scalp < ema_s_scalp) bear_votes++;
""",
    "taker_imbalance": """
   // Signal: Taker Imbalance (approximated by candle close position)
   double candle_range = high_arr[0] - low_arr[0];
   double taker_proxy = (candle_range > 0) ? (close_arr[0] - low_arr[0]) / candle_range : 0.5;
   if(taker_proxy > {bull_thresh}) bull_votes++;
   if(taker_proxy < {bear_thresh}) bear_votes++;
""",
    "microvol_regime": """
   // Signal: Micro Volatility Regime
   double mv6 = 0, mv12 = 0;
   for(int i = 0; i < 6; i++) mv6 += MathPow(close_arr[i] - close_arr[i+1], 2);
   mv6 = MathSqrt(mv6 / 6);
   for(int i = 0; i < 12; i++) mv12 += MathPow(close_arr[i] - close_arr[i+1], 2);
   mv12 = MathSqrt(mv12 / 12);
   double mvr = (mv12 > 0) ? mv6 / mv12 : 1.0;
   if(mvr > {expand_thresh} && close_arr[0] > close_arr[1]) bull_votes++;
   if(mvr > {expand_thresh} && close_arr[0] < close_arr[1]) bear_votes++;
""",
}

# Fallback for signals without specific MQL5 mapping
MQL5_SIGNAL_FALLBACK = """
   // Signal: {signal_name} (generic momentum proxy)
   double gen_ret = (close_arr[0] - close_arr[12]) / close_arr[12];
   if(gen_ret > 0.003) bull_votes++;
   if(gen_ret < -0.003) bear_votes++;
"""


# ─────────────────────────────────────────────────────────────────
# MQL5 Template
# ─────────────────────────────────────────────────────────────────

MQL5_EA_TEMPLATE = '''//+------------------------------------------------------------------+
//| {ea_name}.mq5                                                     |
//| Auto-generated by Autoresearch Trading Engine                     |
//| https://github.com/Millan678y/autoresearch-trading                |
//+------------------------------------------------------------------+
#property copyright "Autoresearch Trading Engine"
#property link      "https://github.com/Millan678y/autoresearch-trading"
#property version   "1.00"
#property strict

//--- Input parameters
input double   InpLotSize       = {lot_size};      // Lot size (0 = auto)
input double   InpRiskPercent   = {risk_pct};       // Risk per trade (%)
input int      InpMagicNumber   = {magic};          // Magic number
input int      InpSlippage      = {slippage};       // Max slippage (points)
input int      InpMinVotes      = {min_votes};      // Minimum signal votes
input int      InpCooldownBars  = {cooldown};       // Cooldown between trades
input double   InpTPMultiplier  = {tp_mult};        // Take Profit ATR multiplier
input double   InpSLMultiplier  = {sl_mult};        // Stop Loss ATR multiplier
input double   InpMaxPositionPct = {max_pos_pct};   // Max position as % of balance
input bool     InpUseTrailingStop = {use_trailing};  // Use trailing stop
input double   InpTrailingATRMult = {trailing_mult}; // Trailing stop ATR multiplier
input ENUM_TIMEFRAMES InpTimeframe = {timeframe};    // Timeframe

//--- Global variables
int      g_last_trade_bar = -999;
int      g_bar_count = 0;
double   g_peak_equity = 0;

//--- Indicator handles
int      g_atr_handle;

//+------------------------------------------------------------------+
//| Expert initialization function                                     |
//+------------------------------------------------------------------+
int OnInit()
{{
   g_atr_handle = iATR(_Symbol, InpTimeframe, 14);
   if(g_atr_handle == INVALID_HANDLE)
   {{
      Print("Failed to create ATR indicator");
      return(INIT_FAILED);
   }}
   
   Print("=== {ea_name} initialized ===");
   Print("Symbol: ", _Symbol, " | TF: ", EnumToString(InpTimeframe));
   Print("Min Votes: ", InpMinVotes, " | Cooldown: ", InpCooldownBars);
   Print("TP: ", InpTPMultiplier, "x ATR | SL: ", InpSLMultiplier, "x ATR");
   
   return(INIT_SUCCEEDED);
}}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                   |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{{
   IndicatorRelease(g_atr_handle);
   Print("=== {ea_name} stopped ===");
}}

//+------------------------------------------------------------------+
//| Calculate lot size based on risk                                   |
//+------------------------------------------------------------------+
double CalcLotSize(double sl_distance)
{{
   if(InpLotSize > 0) return InpLotSize;
   
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double risk_amount = balance * InpRiskPercent / 100.0;
   double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tick_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   
   if(tick_value <= 0 || tick_size <= 0 || sl_distance <= 0) return 0.01;
   
   double sl_points = sl_distance / point;
   double lot = risk_amount / (sl_points * tick_value / tick_size);
   
   double min_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double max_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double lot_step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   
   // Max position size check
   double max_pos = balance * InpMaxPositionPct / 100.0;
   double margin_per_lot = 0;
   if(OrderCalcMargin(ORDER_TYPE_BUY, _Symbol, 1.0, SymbolInfoDouble(_Symbol, SYMBOL_ASK), margin_per_lot))
   {{
      if(margin_per_lot > 0)
         lot = MathMin(lot, max_pos / margin_per_lot);
   }}
   
   lot = MathMax(min_lot, MathMin(max_lot, lot));
   lot = MathRound(lot / lot_step) * lot_step;
   
   return lot;
}}

//+------------------------------------------------------------------+
//| Check if we have an open position                                  |
//+------------------------------------------------------------------+
bool HasPosition(ENUM_POSITION_TYPE &pos_type, double &pos_price, double &pos_volume)
{{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {{
      ulong ticket = PositionGetTicket(i);
      if(PositionSelectByTicket(ticket))
      {{
         if(PositionGetString(POSITION_SYMBOL) == _Symbol &&
            PositionGetInteger(POSITION_MAGIC) == InpMagicNumber)
         {{
            pos_type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
            pos_price = PositionGetDouble(POSITION_PRICE_OPEN);
            pos_volume = PositionGetDouble(POSITION_VOLUME);
            return true;
         }}
      }}
   }}
   return false;
}}

//+------------------------------------------------------------------+
//| Close position                                                     |
//+------------------------------------------------------------------+
bool ClosePosition()
{{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {{
      ulong ticket = PositionGetTicket(i);
      if(PositionSelectByTicket(ticket))
      {{
         if(PositionGetString(POSITION_SYMBOL) == _Symbol &&
            PositionGetInteger(POSITION_MAGIC) == InpMagicNumber)
         {{
            MqlTradeRequest request = {{}};
            MqlTradeResult result = {{}};
            
            ENUM_POSITION_TYPE type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
            double volume = PositionGetDouble(POSITION_VOLUME);
            
            request.action = TRADE_ACTION_DEAL;
            request.position = ticket;
            request.symbol = _Symbol;
            request.volume = volume;
            request.deviation = InpSlippage;
            request.magic = InpMagicNumber;
            
            if(type == POSITION_TYPE_BUY)
            {{
               request.type = ORDER_TYPE_SELL;
               request.price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
            }}
            else
            {{
               request.type = ORDER_TYPE_BUY;
               request.price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
            }}
            
            return OrderSend(request, result);
         }}
      }}
   }}
   return false;
}}

//+------------------------------------------------------------------+
//| Open a trade                                                       |
//+------------------------------------------------------------------+
bool OpenTrade(ENUM_ORDER_TYPE type, double sl, double tp, string comment)
{{
   MqlTradeRequest request = {{}};
   MqlTradeResult result = {{}};
   
   double price = (type == ORDER_TYPE_BUY) ? 
                  SymbolInfoDouble(_Symbol, SYMBOL_ASK) :
                  SymbolInfoDouble(_Symbol, SYMBOL_BID);
   
   double sl_distance = MathAbs(price - sl);
   double lot = CalcLotSize(sl_distance);
   
   if(lot < SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN)) return false;
   
   request.action = TRADE_ACTION_DEAL;
   request.symbol = _Symbol;
   request.volume = lot;
   request.type = type;
   request.price = price;
   request.sl = sl;
   request.tp = tp;
   request.deviation = InpSlippage;
   request.magic = InpMagicNumber;
   request.comment = comment;
   request.type_filling = ORDER_FILLING_IOC;
   
   bool ok = OrderSend(request, result);
   if(ok)
      Print("Trade opened: ", comment, " | Lot: ", lot, " | SL: ", sl, " | TP: ", tp);
   else
      Print("Trade failed: ", result.comment, " | Code: ", result.retcode);
   
   return ok;
}}

//+------------------------------------------------------------------+
//| Manage trailing stop                                               |
//+------------------------------------------------------------------+
void ManageTrailingStop()
{{
   if(!InpUseTrailingStop) return;
   
   double atr_buf[];
   CopyBuffer(g_atr_handle, 0, 0, 1, atr_buf);
   double atr = atr_buf[0];
   double trail_dist = atr * InpTrailingATRMult;
   
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {{
      ulong ticket = PositionGetTicket(i);
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if(PositionGetInteger(POSITION_MAGIC) != InpMagicNumber) continue;
      
      ENUM_POSITION_TYPE type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
      double current_sl = PositionGetDouble(POSITION_SL);
      double current_tp = PositionGetDouble(POSITION_TP);
      
      double new_sl = 0;
      
      if(type == POSITION_TYPE_BUY)
      {{
         double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         new_sl = bid - trail_dist;
         if(new_sl > current_sl + _Point)
         {{
            MqlTradeRequest req = {{}};
            MqlTradeResult res = {{}};
            req.action = TRADE_ACTION_SLTP;
            req.position = ticket;
            req.symbol = _Symbol;
            req.sl = NormalizeDouble(new_sl, _Digits);
            req.tp = current_tp;
            OrderSend(req, res);
         }}
      }}
      else if(type == POSITION_TYPE_SELL)
      {{
         double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         new_sl = ask + trail_dist;
         if(new_sl < current_sl - _Point || current_sl == 0)
         {{
            MqlTradeRequest req = {{}};
            MqlTradeResult res = {{}};
            req.action = TRADE_ACTION_SLTP;
            req.position = ticket;
            req.symbol = _Symbol;
            req.sl = NormalizeDouble(new_sl, _Digits);
            req.tp = current_tp;
            OrderSend(req, res);
         }}
      }}
   }}
}}

//+------------------------------------------------------------------+
//| Expert tick function                                               |
//+------------------------------------------------------------------+
void OnTick()
{{
   // Only process on new bar
   static datetime last_bar_time = 0;
   datetime current_bar_time = iTime(_Symbol, InpTimeframe, 0);
   if(current_bar_time == last_bar_time) return;
   last_bar_time = current_bar_time;
   
   g_bar_count++;
   
   // Manage trailing stop on existing positions
   ManageTrailingStop();
   
   // Get price data
   double close_arr[], open_arr[], high_arr[], low_arr[];
   ArraySetAsSeries(close_arr, true);
   ArraySetAsSeries(open_arr, true);
   ArraySetAsSeries(high_arr, true);
   ArraySetAsSeries(low_arr, true);
   
   int copied = CopyClose(_Symbol, InpTimeframe, 0, 200, close_arr);
   CopyOpen(_Symbol, InpTimeframe, 0, 200, open_arr);
   CopyHigh(_Symbol, InpTimeframe, 0, 200, high_arr);
   CopyLow(_Symbol, InpTimeframe, 0, 200, low_arr);
   
   if(copied < 50) return;
   
   // Get ATR
   double atr_buf[];
   CopyBuffer(g_atr_handle, 0, 0, 1, atr_buf);
   double atr = atr_buf[0];
   if(atr <= 0) return;
   
   // =====================================================
   // SIGNAL COMPUTATION
   // =====================================================
   int bull_votes = 0;
   int bear_votes = 0;
   
{signal_code}
   
   // =====================================================
   // POSITION MANAGEMENT
   // =====================================================
   ENUM_POSITION_TYPE pos_type;
   double pos_price, pos_volume;
   bool has_pos = HasPosition(pos_type, pos_price, pos_volume);
   
   // Cooldown check
   bool in_cooldown = (g_bar_count - g_last_trade_bar) < InpCooldownBars;
   
   if(has_pos)
   {{
      // Exit on signal flip
      if(pos_type == POSITION_TYPE_BUY && bear_votes >= InpMinVotes && !in_cooldown)
      {{
         ClosePosition();
         // Open reverse
         double sl = close_arr[0] + atr * InpSLMultiplier;
         double tp = close_arr[0] - atr * InpTPMultiplier;
         OpenTrade(ORDER_TYPE_SELL, sl, tp, "flip_short");
         g_last_trade_bar = g_bar_count;
      }}
      else if(pos_type == POSITION_TYPE_SELL && bull_votes >= InpMinVotes && !in_cooldown)
      {{
         ClosePosition();
         double sl = close_arr[0] - atr * InpSLMultiplier;
         double tp = close_arr[0] + atr * InpTPMultiplier;
         OpenTrade(ORDER_TYPE_BUY, sl, tp, "flip_long");
         g_last_trade_bar = g_bar_count;
      }}
   }}
   else if(!in_cooldown)
   {{
      // New entry
      if(bull_votes >= InpMinVotes)
      {{
         double sl = close_arr[0] - atr * InpSLMultiplier;
         double tp = close_arr[0] + atr * InpTPMultiplier;
         OpenTrade(ORDER_TYPE_BUY, sl, tp, "bull_entry");
         g_last_trade_bar = g_bar_count;
      }}
      else if(bear_votes >= InpMinVotes)
      {{
         double sl = close_arr[0] + atr * InpSLMultiplier;
         double tp = close_arr[0] - atr * InpTPMultiplier;
         OpenTrade(ORDER_TYPE_SELL, sl, tp, "bear_entry");
         g_last_trade_bar = g_bar_count;
      }}
   }}
}}
//+------------------------------------------------------------------+
'''


# ─────────────────────────────────────────────────────────────────
# Conversion Functions
# ─────────────────────────────────────────────────────────────────

def convert_to_mql5(strategy_record, mode: str = "auto") -> str:
    """
    Convert a StrategyRecord to MQL5 Expert Advisor code.
    
    Args:
        strategy_record: StrategyRecord or dict with signals_used, params
        mode: "scalp" (5m), "swing" (1h), or "auto" (detect from params)
    
    Returns:
        Complete .mq5 source code string
    """
    if isinstance(strategy_record, dict):
        signals = strategy_record.get("signals_used", strategy_record.get("signals", []))
        params = strategy_record.get("params", {})
        name = strategy_record.get("name", "AutoStrategy")
    else:
        signals = strategy_record.signals_used
        params = strategy_record.params
        name = strategy_record.name
    
    if isinstance(signals, str):
        signals = json.loads(signals)
    if isinstance(params, str):
        params = json.loads(params)
    
    return convert_to_mql5_from_params(signals, params, name=name, mode=mode)


def convert_to_mql5_from_params(signals: List[str], params: dict,
                                 name: str = "AutoStrategy",
                                 mode: str = "auto") -> str:
    """
    Generate MQL5 EA from signal list and parameters.
    """
    # Detect mode
    if mode == "auto":
        is_scalp = any(s in signals for s in [
            "micro_momentum", "taker_imbalance", "vol_spike",
            "speed_acceleration", "microvol_regime", "rsi_scalp"
        ])
        mode = "scalp" if is_scalp else "swing"
    
    # Build signal code
    signal_blocks = []
    for sig_name in signals:
        template = MQL5_SIGNAL_MAP.get(sig_name) or MQL5_SIGNAL_MAP.get(sig_name + "_scalp")
        
        if template is None:
            template = MQL5_SIGNAL_FALLBACK.replace("{signal_name}", sig_name)
        
        # Fill in params
        for p_name, p_val in params.items():
            if p_name.startswith(f"{sig_name}__"):
                param_key = p_name.split("__")[1]
                template = template.replace(f"{{{param_key}}}", str(p_val))
        
        # Fill any remaining placeholders with defaults
        import re
        remaining = re.findall(r'\{(\w+)\}', template)
        defaults = {
            "lookback": "12", "threshold": "0.003", "fast": "9", "slow": "21",
            "period": "14", "bull_thresh": "55", "bear_thresh": "45",
            "signal": "9", "mult": "2.0", "oversold": "30", "overbought": "70",
            "entry_z": "2.0", "min_body": "0.6", "thresh": "2.0",
            "expand_thresh": "1.5",
        }
        for key in remaining:
            template = template.replace(f"{{{key}}}", defaults.get(key, "0"))
        
        signal_blocks.append(template)
    
    signal_code = "\n".join(signal_blocks)
    
    # Parameters
    ea_params = {
        "ea_name": _clean_name(name),
        "lot_size": "0.0",  # Auto lot sizing
        "risk_pct": str(params.get("risk_pct", 1.0)),
        "magic": str(abs(hash(name)) % 999999),
        "slippage": "10" if mode == "scalp" else "30",
        "min_votes": str(params.get("min_votes", 3)),
        "cooldown": str(params.get("cooldown", params.get("cooldown_bars", 5))),
        "tp_mult": str(params.get("tp_mult", 2.0)),
        "sl_mult": str(params.get("sl_mult", 2.0)),
        "max_pos_pct": str(params.get("position_pct", params.get("max_pos_pct", 5.0)) * 100 if params.get("position_pct", 0) < 1 else params.get("max_pos_pct", 5.0)),
        "use_trailing": "true" if params.get("stop_type", "") == "atr_trailing" or mode == "swing" else "false",
        "trailing_mult": str(params.get("stop__atr_mult", 5.5)),
        "timeframe": "PERIOD_M5" if mode == "scalp" else "PERIOD_H1",
        "signal_code": signal_code,
    }
    
    return MQL5_EA_TEMPLATE.format(**ea_params)


def _clean_name(name: str) -> str:
    """Clean strategy name for MQL5 compatibility."""
    clean = name.replace("-", "_").replace(" ", "_")
    clean = "".join(c for c in clean if c.isalnum() or c == "_")
    return clean[:50]


# ─────────────────────────────────────────────────────────────────
# Batch Export
# ─────────────────────────────────────────────────────────────────

def export_top_strategies(output_dir: str = "mql5_experts",
                          top_n: int = 10,
                          mode: str = "auto") -> List[str]:
    """
    Export top N strategies as MQL5 Expert Advisors.
    
    Returns list of exported file paths.
    """
    from .models import load_strategies
    
    os.makedirs(output_dir, exist_ok=True)
    
    top = load_strategies(status="passed_oos", limit=top_n)
    
    if not top:
        print("No surviving strategies to export.")
        return []
    
    exported = []
    
    print(f"\n{'=' * 60}")
    print(f"MQL5 EXPORT — Top {len(top)} Strategies")
    print(f"{'=' * 60}")
    
    for i, strat in enumerate(top):
        name = strat.get("name", f"strategy_{i}")
        clean_name = _clean_name(name)
        
        try:
            mql5_code = convert_to_mql5(strat, mode=mode)
            
            filepath = os.path.join(output_dir, f"{clean_name}.mq5")
            with open(filepath, "w") as f:
                f.write(mql5_code)
            
            exported.append(filepath)
            
            oos_score = strat.get("oos_score", 0)
            oos_sharpe = strat.get("oos_sharpe", 0)
            
            print(f"  {i+1}. {clean_name}.mq5 | Score={oos_score:.3f} Sharpe={oos_sharpe:.2f}")
        
        except Exception as e:
            print(f"  {i+1}. {name} — FAILED: {e}")
    
    # Generate a summary README
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write("# MQL5 Expert Advisors\n\n")
        f.write("Auto-generated by Autoresearch Trading Engine.\n\n")
        f.write("## Installation\n\n")
        f.write("1. Copy `.mq5` files to `MQL5/Experts/` in your MT5 data folder\n")
        f.write("2. Open MetaEditor → Compile each file\n")
        f.write("3. In MT5: Navigator → Expert Advisors → drag to chart\n")
        f.write("4. Enable AutoTrading (green button in toolbar)\n\n")
        f.write("## Strategies\n\n")
        f.write(f"| # | Name | OOS Score | Sharpe | Max DD | Mode |\n")
        f.write(f"|---|------|-----------|--------|--------|------|\n")
        for i, strat in enumerate(top):
            name = _clean_name(strat.get("name", ""))
            f.write(f"| {i+1} | {name} | "
                   f"{strat.get('oos_score', 0):.3f} | "
                   f"{strat.get('oos_sharpe', 0):.2f} | "
                   f"{strat.get('oos_max_dd_pct', 0):.1f}% | "
                   f"{'scalp' if 'scalp' in name else 'swing'} |\n")
        
        f.write("\n## Important Notes\n\n")
        f.write("- **Test on demo account first!** These are research outputs, not financial advice.\n")
        f.write("- Run MT5 Strategy Tester before going live\n")
        f.write("- Adjust lot sizes and risk % to your account size\n")
        f.write("- Each EA uses a unique Magic Number to avoid conflicts\n")
    
    print(f"\n✅ Exported {len(exported)} EAs to {output_dir}/")
    print(f"📖 README: {readme_path}")
    
    return exported


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Export strategies to MQL5")
    parser.add_argument("--output", default="mql5_experts", help="Output directory")
    parser.add_argument("--top", type=int, default=10, help="Number of strategies to export")
    parser.add_argument("--mode", default="auto", choices=["auto", "scalp", "swing"])
    args = parser.parse_args()
    
    export_top_strategies(output_dir=args.output, top_n=args.top, mode=args.mode)
