import yfinance as yf
import pandas as pd
import ta
import time
from langchain.tools import tool
from src.utils.logging_config import ToolLogger
from src.validation.data_validators import MarketDataValidator, ValidationError
from src.middleware.retry_handler import retry_yfinance

# Initialize logger and validator
logger = ToolLogger("market_data")
validator = MarketDataValidator()

@tool
def get_stock_history(ticker: str, period: str = "1y") -> str:
    """
    Fetches historical stock data for a given ticker with validation and retry.
    Args:
        ticker: The stock ticker symbol (e.g., "RELIANCE.NS").
        period: The data period to fetch (default: "1y").
    Returns:
        A string representation of the last 30 days and validation status,
        or an error message.
    """
    start_time = time.time()

    try:
        # Fetch data with retry logic
        df = _fetch_stock_data_with_retry(ticker, period)

        if df.empty:
            logger.log_fetch(ticker, "history", False, 0, 0, "Empty DataFrame")
            return f"No data found for ticker {ticker}."

        # Validate data quality
        validation_results = validator.validate_all(ticker, df)
        validation_warnings = []

        # Check for failures
        for check, (passed, reason) in validation_results.items():
            if not passed and check != 'corporate_actions':
                error_msg = f"Data validation failed - {check}: {reason}"
                logger.log_fetch(ticker, "history", False, 0, len(df), error_msg)
                return f"Data quality issue for {ticker}: {reason}"
            elif check == 'corporate_actions' and passed:
                validation_warnings.append(f"WARNING: {reason}")

        # Log successful fetch
        latency_ms = (time.time() - start_time) * 1000
        logger.log_fetch(ticker, "history", True, latency_ms, len(df))

        # Return summary with validation status
        last_30 = df.tail(30).to_csv()
        validation_status = "Data validated" if not validation_warnings else "\n".join(validation_warnings)

        return f"""Data fetched successfully. Shape: {df.shape}
{validation_status}
Last 30 days:
{last_30}"""

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        logger.log_fetch(ticker, "history", False, latency_ms, 0, str(e))
        return f"Error fetching data for {ticker}: {str(e)}"


@retry_yfinance
def _fetch_stock_data_with_retry(ticker: str, period: str) -> pd.DataFrame:
    """Helper function to fetch stock data with automatic retry."""
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    return df

@tool
def calculate_indicators(ticker: str, period: str = "1y") -> str:
    """
    Calculates technical indicators (RSI, MACD, EMAs, Bollinger Bands) for a stock with validation.
    Args:
        ticker: The stock ticker symbol.
        period: The data period (default: "1y").
    Returns:
        A summary of the latest indicator values with data quality status.
    """
    start_time = time.time()

    try:
        # Fetch data with retry
        df = _fetch_stock_data_with_retry(ticker, period)

        if df.empty:
            logger.log_fetch(ticker, "indicators", False, 0, 0, "Empty DataFrame")
            return f"No data found for ticker {ticker}."

        # Validate data first
        validation_results = validator.validate_all(ticker, df)
        for check, (passed, reason) in validation_results.items():
            if not passed and check != 'corporate_actions':
                logger.log_fetch(ticker, "indicators", False, 0, len(df), f"Validation failed: {reason}")
                return f"Data quality issue for {ticker}: {reason}. Cannot calculate reliable indicators."

        # Calculate RSI
        df['rsi'] = ta.momentum.rsi(df['Close'], window=14)

        # Calculate MACD
        macd = ta.trend.MACD(df['Close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()

        # Calculate EMAs
        df['ema_20'] = ta.trend.ema_indicator(df['Close'], window=20)
        df['ema_50'] = ta.trend.ema_indicator(df['Close'], window=50)
        df['ema_200'] = ta.trend.ema_indicator(df['Close'], window=200)

        # Calculate Bollinger Bands
        bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()

        # Get latest values
        latest = df.iloc[-1]

        # Log successful calculation
        latency_ms = (time.time() - start_time) * 1000
        logger.log_fetch(ticker, "indicators", True, latency_ms, len(df))

        report = f"""
Technical Indicators for {ticker} (Date: {latest.name.date()}):
Data validated and fresh

Price: {latest['Close']:.2f}
RSI (14): {latest['rsi']:.2f}
MACD: {latest['macd']:.2f} (Signal: {latest['macd_signal']:.2f}, Diff: {latest['macd_diff']:.2f})
EMA 20: {latest['ema_20']:.2f}
EMA 50: {latest['ema_50']:.2f}
EMA 200: {latest['ema_200']:.2f}
Bollinger Bands: High {latest['bb_high']:.2f}, Low {latest['bb_low']:.2f}
"""
        return report

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        logger.log_fetch(ticker, "indicators", False, latency_ms, 0, str(e))
        return f"Error calculating indicators for {ticker}: {str(e)}"


@tool
def multi_timeframe_analysis(ticker: str) -> str:
    """
    Analyze stock across multiple timeframes (Weekly, Daily, 4-Hour) to identify high-probability setups.

    Checks for timeframe alignment:
    - All aligned = High confidence setup
    - Daily/Weekly aligned = Good setup
    - Conflicting = Wait for clarity

    Args:
        ticker: Stock ticker symbol

    Returns:
        Multi-timeframe analysis with alignment assessment and confidence adjustment
    """
    start_time = time.time()

    try:
        # Fetch data for different timeframes
        weekly_df = _fetch_stock_data_with_retry(ticker, "1y")  # 1 year for weekly context
        daily_df = _fetch_stock_data_with_retry(ticker, "6mo")  # 6 months for daily
        hourly_df = _fetch_hourly_data(ticker)  # 1 month of hourly data

        if weekly_df.empty or daily_df.empty:
            logger.log_fetch(ticker, "multi_timeframe", False, 0, 0, "Missing data")
            return f"Insufficient data for multi-timeframe analysis for {ticker}"

        # Analyze each timeframe
        weekly_analysis = _analyze_timeframe(weekly_df, "Weekly")
        daily_analysis = _analyze_timeframe(daily_df, "Daily")
        hourly_analysis = _analyze_timeframe(hourly_df, "4-Hour") if not hourly_df.empty else None

        # Determine alignment
        alignment_result = _check_timeframe_alignment(weekly_analysis, daily_analysis, hourly_analysis)

        # Log successful analysis
        latency_ms = (time.time() - start_time) * 1000
        logger.log_fetch(ticker, "multi_timeframe", True, latency_ms, len(daily_df))

        # Build report
        report = _build_multi_timeframe_report(ticker, weekly_analysis, daily_analysis, hourly_analysis, alignment_result)

        return report

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        logger.log_fetch(ticker, "multi_timeframe", False, latency_ms, 0, str(e))
        return f"Error in multi-timeframe analysis for {ticker}: {str(e)}"


def _fetch_hourly_data(ticker: str) -> pd.DataFrame:
    """Fetch hourly data for intraday analysis."""
    try:
        stock = yf.Ticker(ticker)
        # Get 1 month of hourly data
        df = stock.history(period="1mo", interval="1h")
        return df
    except:
        return pd.DataFrame()  # Return empty if hourly data not available


def _analyze_timeframe(df: pd.DataFrame, timeframe_name: str) -> dict:
    """Analyze a single timeframe and return trend assessment."""

    if df.empty or len(df) < 50:
        return {
            'timeframe': timeframe_name,
            'bias': 'INSUFFICIENT_DATA',
            'trend': 'UNKNOWN',
            'key_level': None
        }

    # Calculate indicators
    df['ema_20'] = ta.trend.ema_indicator(df['Close'], window=20)
    df['ema_50'] = ta.trend.ema_indicator(df['Close'], window=50)
    df['rsi'] = ta.momentum.rsi(df['Close'], window=14)

    latest = df.iloc[-1]
    current_price = latest['Close']
    ema_20 = latest['ema_20']
    ema_50 = latest['ema_50']
    rsi = latest['rsi']

    # Determine trend and bias
    if current_price > ema_20 > ema_50:
        trend = "UPTREND"
        if rsi > 50:
            bias = "Bullish"
        else:
            bias = "Bullish (weakening)"
        key_level = f"Support at EMA50: Rs{ema_50:.2f}"

    elif current_price < ema_20 < ema_50:
        trend = "DOWNTREND"
        if rsi < 50:
            bias = "Bearish"
        else:
            bias = "Bearish (weakening)"
        key_level = f"Resistance at EMA50: Rs{ema_50:.2f}"

    else:
        trend = "SIDEWAYS"
        if rsi > 55:
            bias = "Neutral (slight bullish)"
        elif rsi < 45:
            bias = "Neutral (slight bearish)"
        else:
            bias = "Neutral"
        key_level = f"Range: Rs{ema_50:.2f} - Rs{ema_20:.2f}"

    return {
        'timeframe': timeframe_name,
        'bias': bias,
        'trend': trend,
        'key_level': key_level,
        'price': current_price,
        'ema_20': ema_20,
        'ema_50': ema_50,
        'rsi': rsi
    }


def _check_timeframe_alignment(weekly: dict, daily: dict, hourly: dict = None) -> dict:
    """
    Check alignment across timeframes and calculate confidence adjustment.

    Returns dict with alignment status and confidence modifier.
    """

    # Extract trend directions (simplified to UP/DOWN/SIDEWAYS)
    weekly_dir = _simplify_trend(weekly['trend'])
    daily_dir = _simplify_trend(daily['trend'])
    hourly_dir = _simplify_trend(hourly['trend']) if hourly else None

    # Check alignment
    if hourly_dir:
        # All three timeframes available
        if weekly_dir == daily_dir == hourly_dir and weekly_dir != 'SIDEWAYS':
            alignment = "PERFECT"
            confidence_adj = +10
            message = "All timeframes aligned - STRONG SETUP"

        elif weekly_dir == daily_dir and weekly_dir != 'SIDEWAYS':
            alignment = "GOOD"
            confidence_adj = +5
            message = "Weekly/Daily aligned (hourly diverges) - Good setup"

        elif weekly_dir != daily_dir:
            alignment = "CONFLICTING"
            confidence_adj = -15
            message = "Timeframes conflicting - WAIT FOR CLARITY"

        else:
            alignment = "NEUTRAL"
            confidence_adj = 0
            message = "Sideways/mixed - No strong directional bias"

    else:
        # Only weekly and daily
        if weekly_dir == daily_dir and weekly_dir != 'SIDEWAYS':
            alignment = "GOOD"
            confidence_adj = +5
            message = "Weekly/Daily aligned - Good setup"

        elif weekly_dir != daily_dir:
            alignment = "CONFLICTING"
            confidence_adj = -10
            message = "Weekly/Daily conflicting - Caution advised"

        else:
            alignment = "NEUTRAL"
            confidence_adj = 0
            message = "Sideways/mixed"

    return {
        'alignment': alignment,
        'confidence_adjustment': confidence_adj,
        'message': message
    }


def _simplify_trend(trend: str) -> str:
    """Simplify trend to UP/DOWN/SIDEWAYS."""
    if 'UP' in trend:
        return 'UP'
    elif 'DOWN' in trend:
        return 'DOWN'
    else:
        return 'SIDEWAYS'


def _build_multi_timeframe_report(ticker: str, weekly: dict, daily: dict, hourly: dict, alignment: dict) -> str:
    """Build comprehensive multi-timeframe report."""

    report = f"""
MULTI-TIMEFRAME ANALYSIS: {ticker}

WEEKLY (Trend Context):
Bias: {weekly['bias']}
Trend: {weekly['trend']}
{weekly['key_level']}
RSI: {weekly['rsi']:.1f}

DAILY (Swing Trade Setup):
Bias: {daily['bias']}
Trend: {daily['trend']}
{daily['key_level']}
RSI: {daily['rsi']:.1f}
"""

    if hourly and hourly['bias'] != 'INSUFFICIENT_DATA':
        report += f"""
4-HOUR (Entry Timing):
Bias: {hourly['bias']}
Trend: {hourly['trend']}
{hourly['key_level']}
RSI: {hourly['rsi']:.1f}
"""

    report += f"""
{'='*50}
ALIGNMENT ASSESSMENT:
{alignment['message']}

Confidence Adjustment: {alignment['confidence_adjustment']:+d}%

TRADING IMPLICATIONS:
"""

    # Add trading implications based on alignment
    if alignment['alignment'] == "PERFECT":
        report += """- High probability directional trade
- Can use tighter stops
- Expect strong follow-through
- Ideal for swing trades"""

    elif alignment['alignment'] == "GOOD":
        report += """- Good directional setup
- Use normal stop distances
- Monitor hourly for entry timing
- Favorable risk/reward"""

    elif alignment['alignment'] == "CONFLICTING":
        report += """- Mixed signals across timeframes
- Higher risk of false breakouts
- Consider waiting for alignment
- If trading, use wider stops"""

    else:  # NEUTRAL
        report += """- No clear directional bias
- Better to wait for setup
- Range-bound trading possible
- Avoid momentum strategies"""

    return report
