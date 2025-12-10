"""
Enhanced data sources beyond basic yfinance - social sentiment, options data, etc.
Adds guardrails for missing/NaN frames and confidence flags for thin coverage.
"""
import json
import time
from datetime import datetime, timedelta

import pandas as pd
import ta
import yfinance as yf
from langchain.tools import tool
from typing import Optional
from src.utils.logging_config import ToolLogger
from src.validation.data_validators import MarketDataValidator

logger = ToolLogger("enhanced_data")
validator = MarketDataValidator()


def _volume_percentile(volumes: pd.Series) -> float:
    """Return percentile rank of the latest volume within the window."""
    if volumes is None or volumes.empty:
        return 0.0
    try:
        return float(volumes.rank(pct=True).iloc[-1])
    except Exception:
        return 0.0


def _up_down_volume_ratio(df: pd.DataFrame, lookback: int = 20) -> float:
    """Ratio of up-day volume to down-day volume to gauge participation quality."""
    recent = df.tail(lookback)
    if recent.empty:
        return 0.0
    up_vol = recent.loc[recent["Close"] >= recent["Open"], "Volume"].sum()
    down_vol = recent.loc[recent["Close"] < recent["Open"], "Volume"].sum()
    return float(up_vol / max(down_vol, 1e-9))

@tool
def get_options_sentiment(ticker: str) -> str:
    """
    Analyzes options data for sentiment signals (put/call ratio, implied volatility).
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Options sentiment analysis with put/call ratio and IV trends
    """
    try:
        stock = yf.Ticker(ticker)
        
        # Get options dates
        dates = stock.options
        if not dates or len(dates) == 0:
            return f"No options data available for {ticker}"
        
        # Get nearest expiry
        nearest_expiry = dates[0]
        opt = stock.option_chain(nearest_expiry)

        calls = opt.calls if hasattr(opt, "calls") else pd.DataFrame()
        puts = opt.puts if hasattr(opt, "puts") else pd.DataFrame()

        # Calculate put/call ratio by volume with NaN safety
        call_volume_series = pd.to_numeric(
            calls.get("volume", pd.Series(dtype=float)), errors="coerce"
        ).fillna(0)
        put_volume_series = pd.to_numeric(
            puts.get("volume", pd.Series(dtype=float)), errors="coerce"
        ).fillna(0)
        call_volume = call_volume_series.sum()
        put_volume = put_volume_series.sum()
        pc_ratio = put_volume / call_volume if call_volume > 0 else 0

        # Get at-the-money implied volatility
        hist_df = stock.history(period="1d")
        if hist_df.empty:
            return f"Options data available but price unavailable for {ticker}"
        current_price = hist_df["Close"].iloc[-1]

        # Find ATM options
        calls_atm = calls[
            (calls.get("strike", pd.Series(dtype=float)) - current_price).abs() < current_price * 0.05
        ] if not calls.empty else pd.DataFrame()
        atm_iv = (
            calls_atm.get("impliedVolatility", pd.Series(dtype=float)).dropna().mean() * 100
            if not calls_atm.empty
            else 0
        )

        # Interpret signals
        if pc_ratio > 1.2:
            sentiment = "Bearish (High put buying)"
        elif pc_ratio < 0.7:
            sentiment = "Bullish (High call buying)"
        else:
            sentiment = "Neutral"

        if atm_iv > 40:
            iv_signal = "High IV - Expect volatility"
        elif atm_iv > 25:
            iv_signal = "Moderate IV"
        else:
            iv_signal = "Low IV - Calm market"

        data_points = len(calls) + len(puts)
        confidence = "low" if data_points < 10 or (call_volume + put_volume) == 0 else "medium"

        report = f"""
OPTIONS SENTIMENT: {ticker} (Expiry: {nearest_expiry})
{'='*50}
Put/Call Ratio: {pc_ratio:.2f} - {sentiment}
Implied Volatility (ATM): {atm_iv:.1f}% - {iv_signal}

Call Volume: {call_volume:,.0f}
Put Volume: {put_volume:,.0f}

Trading Signal:
- P/C > 1.2 = Bearish positioning
- P/C < 0.7 = Bullish positioning
- High IV = Uncertainty, potential big move expected
{'='*50}
"""
        payload = {
            "ticker": ticker,
            "expiry": nearest_expiry,
            "put_call_ratio": pc_ratio,
            "call_volume": call_volume,
            "put_volume": put_volume,
            "atm_iv_pct": atm_iv,
            "sentiment": sentiment,
            "iv_signal": iv_signal,
            "confidence": confidence,
            "data_points": data_points,
        }

        return report + f"\nJSON:{json.dumps(payload, default=str)}"

    except Exception as e:
        return f"Options analysis error for {ticker}: {str(e)}"


@tool
def get_insider_trading_summary(ticker: str) -> str:
    """
    Checks for recent insider trading activity.
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Summary of insider buying/selling activity
    """
    try:
        stock = yf.Ticker(ticker)
        
        # Get insider transactions
        insiders = stock.insider_transactions
        
        if insiders is None or insiders.empty:
            return f"No recent insider trading data for {ticker}"
        
        # Analyze last 6 months
        six_months_ago = datetime.now() - timedelta(days=180)
        recent = insiders[insiders.index > six_months_ago] if hasattr(insiders.index, 'date') else insiders.head(20)
        
        if recent.empty:
            return f"No recent insider transactions for {ticker}"
        
        # Count buys vs sells
        buys = recent[recent['Shares'] > 0] if 'Shares' in recent.columns else recent[recent['Transaction'] == 'Buy']
        sells = recent[recent['Shares'] < 0] if 'Shares' in recent.columns else recent[recent['Transaction'] == 'Sale']
        
        buy_count = len(buys)
        sell_count = len(sells)

        if buy_count > sell_count * 2:
            signal = "Bullish - Heavy insider buying"
        elif sell_count > buy_count * 2:
            signal = "Bearish - Heavy insider selling"
        else:
            signal = "Neutral - Mixed activity"

        confidence = "low" if len(recent) < 3 else "medium"
        recent_preview = recent.head(5).to_dict(orient="list")

        report = f"""
INSIDER TRADING: {ticker} (Last 6 months)
{'='*50}
Insider Buys: {buy_count}
Insider Sells: {sell_count}

Signal: {signal}

Note: Insider buying often signals confidence
Insider selling can be for various reasons (diversification, taxes)
{'='*50}
"""
        payload = {
            "ticker": ticker,
            "buy_count": buy_count,
            "sell_count": sell_count,
            "signal": signal,
            "confidence": confidence,
            "recent_sample": recent_preview,
        }

        return report + f"\nJSON:{json.dumps(payload, default=str)}"

    except Exception as e:
        return f"Insider trading analysis error for {ticker}: {str(e)}"


@tool  
def get_institutional_holdings_change(ticker: str) -> str:
    """
    Analyzes changes in institutional holdings (mutual funds, FIIs).
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Institutional ownership trends
    """
    try:
        stock = yf.Ticker(ticker)
        
        # Get institutional holders
        inst_holders = stock.institutional_holders
        
        if inst_holders is None or inst_holders.empty:
            return f"No institutional holdings data for {ticker}"
        
        # Get major holders summary
        major_holders = stock.major_holders
        
        if major_holders is not None and not major_holders.empty:
            inst_ownership = major_holders.iloc[0, 0] if len(major_holders) > 0 else "N/A"
        else:
            inst_ownership = "N/A"
        
        # Count top holders
        top_count = len(inst_holders)

        top_sample = inst_holders.head(5)
        confidence = "low" if top_count < 3 else "medium"

        report = f"""
INSTITUTIONAL HOLDINGS: {ticker}
{'='*50}
Institutional Ownership: {inst_ownership}
Number of Major Institutional Holders: {top_count}

Top 5 Holders:
"""
        for idx, row in top_sample.iterrows():
            holder = row.get('Holder', 'Unknown')
            shares = row.get('Shares', 0)
            report += f"  - {holder}: {shares:,.0f} shares\n"
        
        report += f"\n{'='*50}"
        payload = {
            "ticker": ticker,
            "institutional_ownership": inst_ownership,
            "holder_count": top_count,
            "confidence": confidence,
            "top_holders": top_sample.to_dict(orient="records"),
        }

        return report + f"\nJSON:{json.dumps(payload, default=str)}"

    except Exception as e:
        return f"Institutional holdings error for {ticker}: {str(e)}"


@tool
def get_swing_feature_snapshot(
    ticker: str,
    period: str = "1y",
    benchmark: str = "^NSEI",
    bank_benchmark: str = "^NSEBANK",
) -> str:
    """
    Generate swing-horizon features (returns, trend, volatility, volume/breadth) for a ticker.

    Emphasizes days-to-weeks regime detection: multi-lookback returns, ATR/realized vol,
    EMA stack distance, MACD histogram slope, volume percentile, up/down volume balance,
    breakout strength, and relative performance vs NIFTY/BANKNIFTY.
    """
    start_time = time.time()
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        df = validator.sanitize_dataframe(ticker, df)

        if df is None or df.empty:
            return f"No data available for {ticker}"

        # Validate data quality (reject hard failures, allow corporate-action warnings)
        validations = validator.validate_all(ticker, df)
        for check, (passed, reason) in validations.items():
            if not passed and check != "corporate_actions":
                return f"Data quality issue for {ticker}: {reason}"

        df = df.dropna(subset=["Close"])
        if df.empty:
            return f"No clean price data for {ticker}"

        def _safe_float(val: float) -> Optional[float]:
            return float(val) if pd.notna(val) else None

        # Core price/volatility features
        df["ret_5d"] = df["Close"].pct_change(5)
        df["ret_20d"] = df["Close"].pct_change(20)
        df["ret_60d"] = df["Close"].pct_change(60)

        atr = ta.volatility.AverageTrueRange(
            high=df["High"], low=df["Low"], close=df["Close"], window=14
        ).average_true_range()
        df["atr_pct"] = atr / df["Close"]
        df["realized_vol_20"] = df["Close"].pct_change().rolling(20).std() * (252 ** 0.5)

        ema_50 = ta.trend.EMAIndicator(df["Close"], window=50).ema_indicator()
        ema_200 = ta.trend.EMAIndicator(df["Close"], window=200).ema_indicator()
        df["ema_50"] = ema_50
        df["ema_200"] = ema_200
        df["ema_distance_pct"] = (df["Close"] - ema_50) / ema_50

        macd = ta.trend.MACD(df["Close"])
        macd_hist = macd.macd_diff()
        df["macd_hist_slope"] = macd_hist.diff()

        # Volume and breadth proxies
        vol_window = df["Volume"].tail(60)
        vol_pct_60 = _volume_percentile(vol_window)
        up_down_vol_20 = _up_down_volume_ratio(df, 20)

        # Breakout strength and gap behaviour
        roll_max_20 = df["Close"].rolling(20).max()
        roll_min_20 = df["Close"].rolling(20).min()
        df["breakout_strength_20d"] = (df["Close"] - roll_min_20) / (roll_max_20 - roll_min_20 + 1e-9)

        gap_pct = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)
        gap_tail = gap_pct.tail(20)

        # Benchmark relative returns (best effort, non-fatal)
        rel_ret_20d = None
        rel_ret_bank_20d = None
        try:
            bench = yf.Ticker(benchmark).history(period="6mo")
            bench = validator.sanitize_dataframe(benchmark, bench)
            if not bench.empty:
                rel_ret_20d = (
                    df["Close"].pct_change(20).iloc[-1] - bench["Close"].pct_change(20).iloc[-1]
                )
        except Exception:
            rel_ret_20d = None

        try:
            bank = yf.Ticker(bank_benchmark).history(period="6mo")
            bank = validator.sanitize_dataframe(bank_benchmark, bank)
            if not bank.empty:
                rel_ret_bank_20d = (
                    df["Close"].pct_change(20).iloc[-1] - bank["Close"].pct_change(20).iloc[-1]
                )
        except Exception:
            rel_ret_bank_20d = None

        latest = df.iloc[-1]
        payload = {
            "ticker": ticker,
            "as_of": str(latest.name),
            "ret_5d": _safe_float(latest["ret_5d"]),
            "ret_20d": _safe_float(latest["ret_20d"]),
            "ret_60d": _safe_float(latest["ret_60d"]),
            "atr_pct": _safe_float(latest["atr_pct"]),
            "realized_vol_20": _safe_float(latest["realized_vol_20"]),
            "ema_distance_pct": _safe_float(latest["ema_distance_pct"]),
            "ema_50": _safe_float(latest["ema_50"]),
            "ema_200": _safe_float(latest["ema_200"]),
            "macd_hist_slope": _safe_float(latest["macd_hist_slope"]),
            "volume_pct_60": vol_pct_60,
            "up_down_volume_ratio_20": up_down_vol_20,
            "breakout_strength_20d": _safe_float(latest["breakout_strength_20d"]),
            "gap_mean_20": _safe_float(gap_tail.mean()),
            "gap_std_20": _safe_float(gap_tail.std()),
            "rel_ret_vs_nifty_20d": _safe_float(rel_ret_20d),
            "rel_ret_vs_banknifty_20d": _safe_float(rel_ret_bank_20d),
        }

        bias_notes = []
        if payload["ema_distance_pct"] is not None:
            if payload["ema_distance_pct"] > 0 and (payload["ema_50"] or 0) > (payload["ema_200"] or 0):
                bias_notes.append("Price above EMA50/200 (uptrend)")
            elif payload["ema_distance_pct"] < 0 and (payload["ema_50"] or 0) < (payload["ema_200"] or 0):
                bias_notes.append("Price below EMA50/200 (downtrend)")
        if payload["macd_hist_slope"] is not None:
            bias_notes.append(
                "MACD hist rising" if payload["macd_hist_slope"] > 0 else "MACD hist falling"
            )
        if payload["breakout_strength_20d"] is not None:
            if payload["breakout_strength_20d"] > 0.8:
                bias_notes.append("Near 20d highs (breakout)")
            elif payload["breakout_strength_20d"] < 0.2:
                bias_notes.append("Near 20d lows (breakdown)")

        latency_ms = (time.time() - start_time) * 1000
        logger.log_fetch(
            ticker=ticker,
            data_type="swing_features",
            success=True,
            latency_ms=latency_ms,
            records_fetched=len(df),
        )

        fmt_pct = lambda v: f"{v:+.2%}" if v is not None else "n/a"  # noqa: E731
        fmt_pct_plain = lambda v: f"{v:.2%}" if v is not None else "n/a"  # noqa: E731
        fmt_float = lambda v, pattern="{:+.4f}": pattern.format(v) if v is not None else "n/a"  # noqa: E731

        summary = f"""Swing feature snapshot for {ticker} (as of {latest.name.date()}):
- 5d/20d/60d returns: {fmt_pct(payload['ret_5d'])} / {fmt_pct(payload['ret_20d'])} / {fmt_pct(payload['ret_60d'])}
- ATR14 as % of price: {fmt_pct_plain(payload['atr_pct'])} | Realized vol20 (ann): {fmt_pct_plain(payload['realized_vol_20'])}
- EMA distance: {fmt_pct(payload['ema_distance_pct'])} | MACD hist slope: {fmt_float(payload['macd_hist_slope'])}
- Volume pct (60d): {vol_pct_60:.1%} | Up/Down vol (20d): {up_down_vol_20:.2f}x
- Breakout strength (20d): {fmt_float(payload['breakout_strength_20d'], '{:.2f}')} | Gap μ/σ (20d): {fmt_pct(payload['gap_mean_20'])}/{fmt_pct_plain(payload['gap_std_20'])}
- Relative 20d vs NIFTY/BANKNIFTY: {fmt_pct(payload['rel_ret_vs_nifty_20d'])}/{fmt_pct(payload['rel_ret_vs_banknifty_20d'])}
Signals: {', '.join(bias_notes) if bias_notes else 'None'}
"""
        return summary + f"\nJSON:{json.dumps(payload, default=str)}"

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        logger.log_fetch(
            ticker=ticker,
            data_type="swing_features",
            success=False,
            latency_ms=latency_ms,
            records_fetched=0,
            error=str(e),
        )
        return f"Swing feature extraction error for {ticker}: {str(e)}"
