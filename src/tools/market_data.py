import yfinance as yf
import pandas as pd
import ta
import time
import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple, List
from langchain.tools import tool
from src.utils.logging_config import ToolLogger
from src.validation.data_validators import MarketDataValidator, ValidationError
from src.middleware.retry_handler import retry_yfinance

# Initialize logger and validator
logger = ToolLogger("market_data")
validator = MarketDataValidator()

# Cache configuration
_quote_cache: Dict[str, Dict[str, Any]] = {}
_QUOTE_CACHE_TTL_SECONDS = 45
_INTRADAY_PERIOD = "1d"
_INTRADAY_INTERVAL = "1m"
_DAILY_PERIOD = "5d"
_MAX_INTRADAY_AGE_DAYS = 2
_MAX_DAILY_AGE_DAYS = 5
_VOLUME_DAILY_THRESHOLD = 100_000  # baseline minimum average daily volume
_INTRADAY_BARS_PER_DAY = 390       # approx. 6.5 trading hours at 1m bars
_MIN_PER_BAR_VOLUME = 200          # floor to avoid zero/near-zero threshold
_HISTORY_CACHE_TTL_SECONDS = 90
_history_cache: Dict[Tuple[str, str, Optional[str]], Dict[str, Any]] = {}


def _get_attr_or_item(obj: Any, key: str) -> Optional[Any]:
    """Safely extract value from mapping or object attribute."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _extract_fast_info(stock: Any) -> Dict[str, Any]:
    """Extract price/currency from yfinance fast_info if available."""
    fast_info = _get_attr_or_item(stock, "fast_info")
    price_keys = ["last_price", "lastPrice", "regularMarketPrice"]
    currency_keys = ["currency", "currencyCode"]

    price = None
    for key in price_keys:
        val = _get_attr_or_item(fast_info, key)
        if val is not None:
            price = float(val)
            break

    currency = None
    for key in currency_keys:
        val = _get_attr_or_item(fast_info, key)
        if val:
            currency = val
            break
    if isinstance(currency, str):
        currency = currency.upper()

    prev_close = _get_attr_or_item(fast_info, "previous_close") or _get_attr_or_item(fast_info, "previousClose")

    return {
        "price": price,
        "currency": currency,
        "previous_close": prev_close,
    }


@retry_yfinance
def _fetch_history(ticker: str, period: str, interval: Optional[str] = None) -> pd.DataFrame:
    """Centralized history fetch with retry/backoff."""
    cache_key = (ticker, period, interval)
    now = time.time()
    cached = _history_cache.get(cache_key)
    if cached and (now - cached["fetched_at"]) < _HISTORY_CACHE_TTL_SECONDS:
        return cached["data"].copy()

    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    df = validator.sanitize_dataframe(ticker, df)
    _history_cache[cache_key] = {"fetched_at": now, "data": df.copy()}
    return df


def _collect_validation_failures(results: Dict[str, Tuple[bool, Optional[str]]]) -> List[str]:
    """Return a list of validation failures excluding non-critical warnings."""
    failures: List[str] = []
    for check, (passed, reason) in results.items():
        if passed:
            continue
        if check == "corporate_actions":
            # Corporate actions are warnings, not hard failures
            continue
        if reason:
            failures.append(reason)
    return failures


def _per_bar_volume_threshold(df: pd.DataFrame, interval: Optional[str]) -> int:
    """
    Compute an interval-aware volume threshold to avoid flagging intraday bars as illiquid.

    For intraday 1m data, scale the daily threshold by expected bars-per-day.
    """
    if interval == "1m":
        bars = min(max(len(df), 1), _INTRADAY_BARS_PER_DAY)
        per_bar = max(_VOLUME_DAILY_THRESHOLD // max(bars, 1), _MIN_PER_BAR_VOLUME)
        return int(per_bar)
    return _VOLUME_DAILY_THRESHOLD


def _assess_coverage(df: pd.DataFrame, expected_bars: int, label: str) -> Tuple[bool, Optional[str]]:
    """
    Flag datasets that are too sparse to be relied on.

    Keeps the threshold lenient to avoid false alarms during live sessions or holidays.
    """
    if df is None or df.empty or expected_bars <= 0:
        return False, f"{label} data missing"

    coverage = len(df) / expected_bars
    if coverage < 0.05:
        return False, f"{label} data too sparse ({coverage:.1%} of expected bars)"
    return True, None


def _isoformat(ts: Optional[datetime]) -> str:
    if ts is None:
        return datetime.now(timezone.utc).isoformat()
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.isoformat()


def fetch_realtime_quote_structured(ticker: str) -> Dict[str, Any]:
    """
    Fetch a near real-time quote for a ticker with freshness validation and caching.

    Returns structured data:
    {
        "ticker": str,
        "price": float | None,
        "currency": str | None,
        "previous_close": float | None,
        "as_of": iso timestamp str,
        "stale": bool,
        "stale_reason": str | None,
        "source": "yfinance"
    }
    """
    now = time.time()
    cached = _quote_cache.get(ticker)
    if cached and (now - cached["fetched_at"]) < _QUOTE_CACHE_TTL_SECONDS:
        return cached["data"]

    start_time = time.time()
    error_type = None
    validation_failures: List[str] = []
    try:
        stock = yf.Ticker(ticker)
        fast_bits = _extract_fast_info(stock)

        # Try to get 1-minute data for freshness
        intraday_error = None
        try:
            intraday_df = _fetch_history(ticker, _INTRADAY_PERIOD, _INTRADAY_INTERVAL)
        except Exception as e:  # noqa: PERF203 - needs broad catch to classify errors
            intraday_df = pd.DataFrame()
            intraday_error = str(e)

        last_ts = None
        hist_price = None
        if intraday_df is not None and not intraday_df.empty:
            last_row = intraday_df.tail(1)
            hist_price = float(last_row["Close"].iloc[0])
            last_ts = pd.to_datetime(last_row.index[-1]).tz_localize(None)

        price = fast_bits["price"] or hist_price
        currency = fast_bits["currency"]
        previous_close = fast_bits["previous_close"]

        stale = False
        stale_reason = None
        data_age_seconds = None

        if last_ts:
            now_ts = datetime.now(timezone.utc).replace(tzinfo=None)
            data_age_seconds = (now_ts - last_ts).total_seconds()

        if intraday_df is not None and not intraday_df.empty:
            volume_threshold = _per_bar_volume_threshold(intraday_df, _INTRADAY_INTERVAL)
            coverage_ok, coverage_reason = _assess_coverage(
                intraday_df, _INTRADAY_BARS_PER_DAY, "Intraday"
            )
            # Enforce freshness and overall data quality
            valid_ts, ts_reason = validator.validate_timestamp_freshness(
                ticker, intraday_df, max_age_days=_MAX_INTRADAY_AGE_DAYS
            )
            validations = validator.validate_all(
                ticker,
                intraday_df,
                current_price=price,
                min_avg_volume=volume_threshold,
                volume_window=min(len(intraday_df), 200),
            )
            validation_failures = _collect_validation_failures(validations)

            if not coverage_ok:
                stale = True
                stale_reason = coverage_reason
                error_type = "partial_intraday"
            if not valid_ts:
                stale = True
                stale_reason = stale_reason or ts_reason
                error_type = error_type or "stale_data"
            if validation_failures:
                stale = True
                joined_failures = "; ".join(validation_failures)
                stale_reason = stale_reason or joined_failures
                error_type = error_type or "validation_failed"
        else:
            stale = True
            stale_reason = intraday_error or "No intraday data returned"
            error_type = "no_intraday_data"

        # If we still have no price, try daily history as last resort
        if price is None:
            daily_error = None
            try:
                volume_threshold = _VOLUME_DAILY_THRESHOLD
                daily_df = _fetch_history(ticker, _DAILY_PERIOD)
                if daily_df is not None and not daily_df.empty:
                    price = float(daily_df["Close"].iloc[-1])
                    last_ts = pd.to_datetime(daily_df.index[-1]).tz_localize(None)
                    now_ts = datetime.now(timezone.utc).replace(tzinfo=None)
                    data_age_seconds = (now_ts - last_ts).total_seconds()

                    coverage_ok, coverage_reason = _assess_coverage(daily_df, 5, "Daily")
                    valid_ts, ts_reason = validator.validate_timestamp_freshness(
                        ticker, daily_df, max_age_days=_MAX_DAILY_AGE_DAYS
                    )
                    validations = validator.validate_all(
                        ticker,
                        daily_df,
                        current_price=price,
                        min_avg_volume=volume_threshold,
                        volume_window=min(len(daily_df), 60),
                    )
                    validation_failures = _collect_validation_failures(validations)

                    if not coverage_ok:
                        stale = True
                        stale_reason = stale_reason or coverage_reason
                        error_type = error_type or "partial_daily"
                    if not valid_ts:
                        stale = True
                        stale_reason = ts_reason
                        error_type = "stale_data"
                    if validation_failures:
                        stale = True
                        joined_failures = "; ".join(validation_failures)
                        stale_reason = stale_reason or joined_failures
                        error_type = error_type or "validation_failed"
                else:
                    stale = True
                    stale_reason = daily_error or stale_reason or "Daily history unavailable"
            except Exception as e:  # noqa: PERF203 - classification
                stale = True
                daily_error = str(e)
                stale_reason = stale_reason or f"Daily history unavailable: {daily_error}"
                error_type = error_type or "daily_history_error"

        if price is None:
            stale = True
            stale_reason = stale_reason or "Price unavailable from data source"
            error_type = error_type or "no_price"
        else:
            price = float(price)

        as_of = _isoformat(last_ts)

        currency = (currency or "N/A")
        if isinstance(currency, str):
            currency = currency.upper()

        quote = {
            "ticker": ticker,
            "price": float(price) if price is not None else None,
            "currency": currency or "N/A",
            "previous_close": float(previous_close) if previous_close is not None else None,
            "as_of": as_of,
            "data_age_seconds": data_age_seconds,
            "stale": bool(stale),
            "stale_reason": stale_reason,
            "error_type": error_type,
            "source": "yfinance",
            "validation_failures": validation_failures,
        }

        latency_ms = (time.time() - start_time) * 1000
        logger.log_fetch(
            ticker=ticker,
            data_type="realtime_quote",
            success=price is not None and not validation_failures,
            latency_ms=latency_ms,
            records_fetched=len(intraday_df) if intraday_df is not None else 0,
            error=stale_reason if price is None else None,
            data_age_seconds=data_age_seconds,
            source="yfinance",
        )

        _quote_cache[ticker] = {"fetched_at": now, "data": quote}
        return quote
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        logger.log_fetch(
            ticker=ticker,
            data_type="realtime_quote",
            success=False,
            latency_ms=latency_ms,
            records_fetched=0,
            error=str(e),
            source="yfinance",
        )
        return {
            "ticker": ticker,
            "price": None,
            "currency": "N/A",
            "previous_close": None,
            "as_of": datetime.now(timezone.utc).isoformat(),
            "stale": True,
            "stale_reason": str(e),
            "source": "yfinance",
            "error": str(e),
            "error_type": "exception",
        }


@tool
def get_realtime_quote(ticker: str) -> str:
    """
    Fetches a near real-time quote with validation and caching.

    Returns a human summary plus JSON payload for downstream agents.
    """
    data = fetch_realtime_quote_structured(ticker)

    if data.get("error"):
        return f"Live quote unavailable for {ticker}: {data['error']}"

    price = data.get("price")
    currency = data.get("currency") or "N/A"
    stale = data.get("stale")
    stale_reason = data.get("stale_reason")
    as_of = data.get("as_of")

    freshness = "STALE" if stale else "FRESH"
    freshness_detail = f"Reason: {stale_reason}" if stale_reason else ""

    if price is None:
        summary = (
            f"Live quote for {ticker}: Price unavailable—do not fabricate price or levels. "
            f"{freshness_detail}".strip()
        )
    else:
        summary = (
            f"Live quote for {ticker}: {price} {currency} as of {as_of} "
            f"({freshness}). {freshness_detail}".strip()
        )

    # Make the JSON payload easy to parse downstream
    return summary + f"\nJSON:{json.dumps(data, default=str)}"

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
    cache_key = (ticker, period, None)
    now = time.time()
    cached = _history_cache.get(cache_key)
    if cached and (now - cached["fetched_at"]) < _HISTORY_CACHE_TTL_SECONDS:
        return cached["data"].copy()

    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    df = validator.sanitize_dataframe(ticker, df)
    _history_cache[cache_key] = {"fetched_at": now, "data": df.copy()}
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

        # Average True Range (volatility) and volume trend
        atr_indicator = ta.volatility.AverageTrueRange(
            high=df["High"], low=df["Low"], close=df["Close"], window=14
        )
        df["atr_14"] = atr_indicator.average_true_range()
        df["vol_avg_20"] = df["Volume"].rolling(20).mean()

        # Get latest values
        latest = df.iloc[-1]
        atr_14 = latest["atr_14"]
        vol_avg_20 = latest["vol_avg_20"]
        vol_ratio = (latest["Volume"] / vol_avg_20) if vol_avg_20 and vol_avg_20 > 0 else None

        # Log successful calculation
        latency_ms = (time.time() - start_time) * 1000
        logger.log_fetch(ticker, "indicators", True, latency_ms, len(df))

        if vol_ratio is not None:
            volume_line = f"Volume vs 20d avg: {vol_ratio:.2f}x (Current: {latest['Volume']:.0f}, 20d avg: {vol_avg_20:.0f})"
        else:
            volume_line = f"Volume: {latest['Volume']:.0f} (20d avg unavailable)"

        # Derive directional bias and confidence from multiple indicators
        bull_points = 0
        bear_points = 0
        signal_notes: List[str] = []

        if latest['Close'] > latest['ema_50'] and latest['ema_50'] > latest['ema_200']:
            bull_points += 1
            signal_notes.append("Price above EMA50/200 (uptrend)")
        elif latest['Close'] < latest['ema_50'] and latest['ema_50'] < latest['ema_200']:
            bear_points += 1
            signal_notes.append("Price below EMA50/200 (downtrend)")
        else:
            signal_notes.append("Price near EMAs (range)")

        if latest['macd'] > latest['macd_signal']:
            bull_points += 1
            signal_notes.append("MACD > signal (bullish momentum)")
        else:
            bear_points += 1
            signal_notes.append("MACD < signal (bearish momentum)")

        if latest['rsi'] >= 60:
            bull_points += 1
            signal_notes.append("RSI > 60 (buying pressure)")
        elif latest['rsi'] <= 40:
            bear_points += 1
            signal_notes.append("RSI < 40 (selling pressure)")
        else:
            signal_notes.append("RSI in neutral band (40-60)")

        if vol_ratio is not None:
            if vol_ratio >= 1.2:
                bull_points += 1
                signal_notes.append(f"Volume {vol_ratio:.2f}x avg (participation strong)")
            elif vol_ratio <= 0.8:
                bear_points += 1
                signal_notes.append(f"Volume {vol_ratio:.2f}x avg (participation weak)")
            else:
                signal_notes.append(f"Volume {vol_ratio:.2f}x avg (neutral)")

        score = bull_points - bear_points
        conflict = bull_points > 0 and bear_points > 0

        if score >= 2:
            bias = "Bullish"
            confidence = 65 + min(score, 3) * 5
        elif score <= -2:
            bias = "Bearish"
            confidence = 65 + min(abs(score), 3) * 5
        else:
            bias = "Neutral"
            confidence = 55 if score == 1 or score == -1 else 50

        if conflict:
            confidence -= 10

        confidence = max(20, min(90, confidence))
        signal_summary = "; ".join(signal_notes[:4])

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
ATR (14): {atr_14:.2f}
{volume_line}

Bias: {bias} | Confidence: {confidence:.0f}%
Signals: {signal_summary}
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
Overall Confidence: {max(20, min(90, 50 + alignment['confidence_adjustment'])):.0f}% (base 50% + alignment adj)

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
