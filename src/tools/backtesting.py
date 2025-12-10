"""
Backtesting tools for validating trade calls against historical data.
Adds input validation, retries, and structured outputs to help downstream agents.
"""
import json
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple

import pandas as pd
import yfinance as yf
from langchain.tools import tool

from src.middleware.retry_handler import retry_yfinance
from src.utils.logging_config import ToolLogger
from src.validation.data_validators import MarketDataValidator

bt_logger = ToolLogger("backtesting")
bt_validator = MarketDataValidator()
_HISTORY_CACHE_TTL_SECONDS = 90
_history_cache: Dict[Tuple[str, str, str], Dict[str, pd.DataFrame]] = {}


def _compute_dynamic_cost(df: pd.DataFrame, slippage_bps: int, fee_bps: int) -> float:
    """
    Estimate round-trip cost (%) using liquidity and intraday volatility.

    Returns percentage (not decimal) to align with downstream calculations.
    """
    base_per_leg = (slippage_bps + fee_bps) / 10000  # decimal
    if df is None or df.empty:
        return base_per_leg * 2 * 100

    range_pct = ((df["High"] - df["Low"]) / df["Close"]).tail(20).mean()
    vol_factor = 1 + min(range_pct * 5, 1.0) if pd.notna(range_pct) else 1.0

    volume_mean = df["Volume"].tail(20).mean()
    liquidity_factor = 1.0
    if volume_mean and not pd.isna(volume_mean):
        if volume_mean < 500_000:
            liquidity_factor = 1.2
        elif volume_mean > 2_000_000:
            liquidity_factor = 0.9

    return base_per_leg * vol_factor * liquidity_factor * 2 * 100


@retry_yfinance
def _fetch_history(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    cache_key = (ticker, start.date().isoformat(), end.date().isoformat())
    now = time.time()
    cached = _history_cache.get(cache_key)
    if cached and (now - cached["fetched_at"]) < _HISTORY_CACHE_TTL_SECONDS:
        return cached["data"].copy()

    stock = yf.Ticker(ticker)
    df = stock.history(start=start, end=end)
    df = bt_validator.sanitize_dataframe(ticker, df)
    _history_cache[cache_key] = {"fetched_at": now, "data": df.copy()}
    return df


def _first_hit_index(df: pd.DataFrame, target: float, column: str, comparison) -> Optional[pd.Timestamp]:
    mask = comparison(df[column], target)
    if not mask.any():
        return None
    return mask.idxmax()


def _validate_inputs(entry_price: float, target_price: float, stop_loss: float) -> Optional[str]:
    if entry_price <= 0 or target_price <= 0 or stop_loss <= 0:
        return "Prices must be positive numbers."
    if stop_loss >= entry_price:
        return "Stop loss should be below entry price for long setups."
    if target_price <= entry_price:
        return "Target price should exceed entry price for long setups."
    return None

@tool
def backtest_trade_call(
    ticker: str,
    entry_price: float,
    target_price: float,
    stop_loss: float,
    days_back: int = 30,
    slippage_bps: int = 10,
    fee_bps: int = 5,
) -> str:
    """
    Backtests a trade call against the last N days of historical data.

    Adds validation, ordered hit detection, and structured payloads.
    """
    start_time = time.time()
    validation_error = _validate_inputs(entry_price, target_price, stop_loss)
    if validation_error:
        return f"Invalid inputs: {validation_error}"

    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back * 2)  # Get extra data for context
        df = _fetch_history(ticker, start=start_date, end=end_date)

        if df.empty:
            return f"No historical data available for {ticker}"

        df = df.tail(days_back)
        if df.empty:
            return f"No data in requested window for {ticker}"

        # Basic data quality checks
        validations = bt_validator.validate_all(ticker, df)
        failures = [r for r in validations.values() if not r[0] and r[1]]
        if failures:
            reason = failures[0][1]
            return f"Data quality issue for {ticker}: {reason}"

        current_price = float(df["Close"].iloc[-1])
        high_in_period = float(df["High"].max())
        low_in_period = float(df["Low"].min())

        # Ordered hit detection (target vs stop)
        target_hit_at = _first_hit_index(df, target_price, "High", lambda series, t: series >= t)
        stop_hit_at = _first_hit_index(df, stop_loss, "Low", lambda series, t: series <= t)

        would_hit_target = target_hit_at is not None
        would_hit_stop = stop_hit_at is not None

        if would_hit_target and would_hit_stop:
            if target_hit_at < stop_hit_at:
                outcome = "WIN - Target reached before stop"
                confidence = 75
                first_event = target_hit_at
            elif stop_hit_at < target_hit_at:
                outcome = "LOSS - Stop reached before target"
                confidence = 35
                first_event = stop_hit_at
            else:
                outcome = "MIXED - Target and stop hit same session"
                confidence = 50
                first_event = target_hit_at
        elif would_hit_target:
            outcome = "WIN - Target would have been hit"
            confidence = 80
            first_event = target_hit_at
        elif would_hit_stop:
            outcome = "LOSS - Stop loss would have been hit"
            confidence = 30
            first_event = stop_hit_at
        else:
            outcome = "PENDING - Neither target nor stop hit"
            confidence = 60
            first_event = None

        potential_gain = ((target_price - entry_price) / entry_price) * 100
        potential_loss = ((entry_price - stop_loss) / entry_price) * 100
        risk_reward = potential_gain / potential_loss if potential_loss > 0 else 0
        price_change = ((current_price - entry_price) / entry_price) * 100
        round_trip_cost_pct = _compute_dynamic_cost(df, slippage_bps, fee_bps)
        potential_gain_net = max(potential_gain - round_trip_cost_pct, 0)
        potential_loss_net = potential_loss + round_trip_cost_pct
        risk_reward_net = potential_gain_net / potential_loss_net if potential_loss_net > 0 else 0

        payload = {
            "ticker": ticker,
            "window_days": days_back,
            "entry_price": entry_price,
            "target_price": target_price,
            "stop_loss": stop_loss,
            "current_price": current_price,
            "high": high_in_period,
            "low": low_in_period,
            "target_hit_at": str(target_hit_at) if target_hit_at is not None else None,
            "stop_hit_at": str(stop_hit_at) if stop_hit_at is not None else None,
            "outcome": outcome,
            "confidence": confidence,
            "risk_reward": risk_reward,
            "risk_reward_net": risk_reward_net,
            "price_change_pct": price_change,
            "round_trip_cost_pct": round_trip_cost_pct,
            "slippage_bps": slippage_bps,
            "fee_bps": fee_bps,
            "first_event": str(first_event) if first_event is not None else None,
        }

        report = f"""
BACKTEST RESULTS ({days_back} days):
{'='*50}
Ticker: {ticker}
Entry Price: Rs{entry_price:.2f}
Target: Rs{target_price:.2f} (+{potential_gain:.2f}%)
Stop Loss: Rs{stop_loss:.2f} (-{potential_loss:.2f}%)

Historical Range:
- High: Rs{high_in_period:.2f}
- Low: Rs{low_in_period:.2f}
- Current: Rs{current_price:.2f}

Outcome: {outcome}
Risk-Reward Ratio: {risk_reward:.2f}:1
Risk-Reward (after costs): {risk_reward_net:.2f}:1 | Costs: {round_trip_cost_pct:.2f}% round-trip (slip {slippage_bps}bps, fees {fee_bps}bps)
Price Change from Entry: {price_change:+.2f}%
Confidence Score: {confidence}% 
{'='*50}
"""
        latency_ms = (time.time() - start_time) * 1000
        bt_logger.log_fetch(
            ticker=ticker,
            data_type="backtest_trade_call",
            success=True,
            latency_ms=latency_ms,
            records_fetched=len(df),
        )
        return report + f"\nJSON:{json.dumps(payload, default=str)}"

    except Exception as e:  # noqa: PERF203 - user-facing error path
        latency_ms = (time.time() - start_time) * 1000
        bt_logger.log_fetch(
            ticker=ticker,
            data_type="backtest_trade_call",
            success=False,
            latency_ms=latency_ms,
            records_fetched=0,
            error=str(e),
        )
        return f"Backtest error for {ticker}: {str(e)}"
        
@tool
def perform_walk_forward_analysis(ticker: str, strategy_type: str = "Trend Following", windows: int = 4) -> str:
    """
    Performs Walk-Forward Analysis to validate strategy robustness.
    
    Splits history into sliding Train/Test windows to check if the strategy
    performs well on unseen data.
    
    Args:
        ticker: Stock ticker
        strategy_type: Type of strategy to test (Trend Following, Mean Reversion)
        windows: Number of walk-forward windows to test
        
    Returns:
        Comprehensive backtest report with Out-of-Sample performance.
    """
    try:
        stock = yf.Ticker(ticker)
        # Get 2 years of data
        df = stock.history(period="2y")

        if df.empty or len(df) < 200:
            return f"Insufficient data for walk-forward analysis for {ticker}"

        df = df.dropna(subset=["Close"])
        if df.empty:
            return f"Insufficient clean price data for {ticker}"

        # Define window size (e.g., 6 months train, 2 months test)
        total_days = len(df)
        window_size = total_days // (windows + 1)
        train_size = int(window_size * 0.75)
        test_size = window_size - train_size
        
        results = []
        
        for i in range(windows):
            start_idx = i * test_size
            train_end = start_idx + train_size
            test_end = train_end + test_size
            
            if test_end > total_days:
                break
                
            train_data = df.iloc[start_idx:train_end]
            test_data = df.iloc[train_end:test_end]
            
            # Simulate Strategy (Simplified for this tool)
            # In a real system, this would call the actual strategy logic
            # Here we simulate a simple Moving Average Crossover for "Trend Following"
            # and RSI for "Mean Reversion"
            
            if strategy_type == "Trend Following":
                # Optimize on Train (Find best MA combo)
                best_fast, best_slow, best_return = _optimize_ma_strategy(train_data)
                # Test on Test
                test_return, trades = _run_ma_strategy(test_data, best_fast, best_slow)
                
                results.append({
                    "window": i+1,
                    "train_return": best_return,
                    "test_return": test_return,
                    "params": f"MA({best_fast}/{best_slow})",
                    "trades": trades
                })
                
            else: # Mean Reversion (RSI)
                # Optimize RSI thresholds
                best_low, best_high, best_return = _optimize_rsi_strategy(train_data)
                # Test
                test_return, trades = _run_rsi_strategy(test_data, best_low, best_high)
                
                results.append({
                    "window": i+1,
                    "train_return": best_return,
                    "test_return": test_return,
                    "params": f"RSI({best_low}/{best_high})",
                    "trades": trades
                })
        
        # Aggregate Results
        avg_test_return = sum(r['test_return'] for r in results) / len(results)
        robustness = sum(1 for r in results if r['test_return'] > 0) / len(results) * 100

        report = f"""
WALK-FORWARD ANALYSIS: {ticker} ({strategy_type})
{'='*50}
Robustness Score: {robustness:.0f}% (Windows Profitable)
Avg Out-of-Sample Return: {avg_test_return:.2f}% per window

DETAILS:
"""
        for r in results:
            report += f"Window {r['window']}: Train {r['train_return']:.1f}% -> Test {r['test_return']:.1f}% | Params: {r['params']} | Trades: {r['trades']}\n"

        report += "="*50

        payload = {
            "ticker": ticker,
            "strategy_type": strategy_type,
            "windows": results,
            "avg_test_return": avg_test_return,
            "robustness": robustness,
        }

        return report + f"\nJSON:{json.dumps(payload, default=str)}"

    except Exception as e:  # noqa: PERF203 - user-facing error path
        return f"Walk-forward analysis error: {str(e)}"

def _optimize_ma_strategy(df):
    # Simplified optimization
    best_ret = -999
    best_f, best_s = 10, 50
    
    # Grid search small range
    for f in [10, 20]:
        for s in [50, 100]:
            ret, _ = _run_ma_strategy(df, f, s)
            if ret > best_ret:
                best_ret = ret
                best_f, best_s = f, s
    return best_f, best_s, best_ret

def _run_ma_strategy(df, fast, slow):
    # Vectorized backtest would be better, but keeping it simple
    df = df.copy()
    df['Fast'] = df['Close'].rolling(fast).mean()
    df['Slow'] = df['Close'].rolling(slow).mean()
    
    position = 0
    trades = 0
    returns = 0.0
    
    for i in range(1, len(df)):
        if df['Fast'].iloc[i] > df['Slow'].iloc[i] and df['Fast'].iloc[i-1] <= df['Slow'].iloc[i-1]:
            position = 1 # Buy
            entry = df['Close'].iloc[i]
            trades += 1
        elif df['Fast'].iloc[i] < df['Slow'].iloc[i] and df['Fast'].iloc[i-1] >= df['Slow'].iloc[i-1]:
            if position == 1:
                position = 0 # Sell
                exit_price = df['Close'].iloc[i]
                returns += (exit_price - entry) / entry * 100
    
    # Close open position
    if position == 1:
        returns += (df['Close'].iloc[-1] - entry) / entry * 100
        
    return returns, trades

def _optimize_rsi_strategy(df):
    import ta
    best_ret = -999
    best_l, best_h = 30, 70
    
    # Simple check
    ret, _ = _run_rsi_strategy(df, 30, 70)
    return 30, 70, ret

def _run_rsi_strategy(df, low, high):
    import ta
    df = df.copy()
    rsi = ta.momentum.RSIIndicator(df['Close']).rsi()
    
    position = 0
    trades = 0
    returns = 0.0
    entry = 0
    
    for i in range(14, len(df)):
        if rsi.iloc[i] < low and position == 0:
            position = 1
            entry = df['Close'].iloc[i]
            trades += 1
        elif rsi.iloc[i] > high and position == 1:
            position = 0
            returns += (df['Close'].iloc[i] - entry) / entry * 100
            
    if position == 1:
        returns += (df['Close'].iloc[-1] - entry) / entry * 100
        
    return returns, trades


@tool  
def calculate_confidence_score(ticker: str, technical_bias: str, fundamental_rating: str, sentiment: str) -> str:
    """
    Calculates a confidence score based on alignment of analysis components.
    
    Args:
        ticker: Stock ticker
        technical_bias: "Bullish", "Bearish", or "Neutral"
        fundamental_rating: "Positive", "Negative", or "Neutral"  
        sentiment: "Positive", "Negative", or "Neutral"
    
    Returns:
        Confidence score (0-100%) with reasoning
    """
    try:
        # Normalize inputs
        tech = technical_bias.lower()
        fund = fundamental_rating.lower()
        sent = sentiment.lower()
        
        # Map to numeric scores
        score_map = {
            "bullish": 1, "positive": 1,
            "neutral": 0,
            "bearish": -1, "negative": -1
        }
        
        tech_score = score_map.get(tech, 0)
        fund_score = score_map.get(fund, 0)
        sent_score = score_map.get(sent, 0)
        
        # Calculate alignment
        total_score = tech_score + fund_score + sent_score
        
        # Calculate confidence based on alignment
        if abs(total_score) == 3:
            # Perfect alignment (all bullish or all bearish)
            confidence = 90
            reasoning = "Strong alignment across all analyses"
        elif abs(total_score) == 2:
            # Two align, one neutral or opposite
            confidence = 75
            reasoning = "Good alignment with minor divergence"
        elif abs(total_score) == 1:
            # Mixed signals
            confidence = 55
            reasoning = "Mixed signals detected - proceed with caution"
        else:
            # All neutral or conflicting
            confidence = 40
            reasoning = "Low confidence - conflicting or neutral signals"
        
        # Boost/reduce based on specific patterns
        if tech_score == 1 and fund_score == 1 and sent_score == -1:
            confidence -= 10
            reasoning = "Sentiment diverges from fundamentals - market may be wrong"
        
        report = f"""
CONFIDENCE ANALYSIS:
{'='*50}
Technical: {technical_bias} ({tech_score:+d})
Fundamental: {fundamental_rating} ({fund_score:+d})
Sentiment: {sentiment} ({sent_score:+d})

Confidence Score: {confidence}%
Reasoning: {reasoning}

Recommendation: {'PROCEED' if confidence >= 60 else 'CAUTION' if confidence >= 50 else 'AVOID'}
{'='*50}
"""
        return report
        
    except Exception as e:
        return f"Confidence calculation error: {str(e)}"
