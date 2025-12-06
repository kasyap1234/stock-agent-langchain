"""
Market regime detection tool for identifying current market conditions.

Detects:
- Trending vs Ranging markets
- Volatility levels
- Market direction

Used by ensemble agent to weight strategies appropriately.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import ta
import time
from langchain.tools import tool
from src.utils.logging_config import ToolLogger
from src.middleware.retry_handler import retry_yfinance

# Initialize logger
logger = ToolLogger("regime_detection")


@tool
def detect_market_regime(ticker: str, period: str = "6mo") -> str:
    """
    Detect the current market regime for a stock to inform strategy selection.

    Analyzes:
    - Trend strength (ADX)
    - Volatility (Bollinger Band width, ATR)
    - Trend direction (EMA alignment)

    Args:
        ticker: Stock ticker symbol
        period: Data period for analysis (default: 6mo)

    Returns:
        String describing the current regime and strategic recommendations
    """
    start_time = time.time()

    try:
        # Fetch data with retry
        df = _fetch_regime_data(ticker, period)

        if df.empty or len(df) < 50:
            logger.log_fetch(ticker, "regime", False, 0, 0, "Insufficient data")
            return f"Insufficient data for regime detection for {ticker}"

        # Calculate regime indicators
        regime_data = _calculate_regime_indicators(df)

        # Classify regime
        regime_classification = _classify_regime(regime_data)

        # Log successful detection
        latency_ms = (time.time() - start_time) * 1000
        logger.log_fetch(ticker, "regime", True, latency_ms, len(df))

        # Build report
        report = _build_regime_report(ticker, regime_data, regime_classification)

        return report

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        logger.log_fetch(ticker, "regime", False, latency_ms, 0, str(e))
        return f"Error detecting regime for {ticker}: {str(e)}"


@retry_yfinance
def _fetch_regime_data(ticker: str, period: str) -> pd.DataFrame:
    """Fetch stock data for regime analysis with retry."""
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    return df


def _calculate_regime_indicators(df: pd.DataFrame) -> dict:
    """Calculate all indicators needed for regime detection."""

    # ADX (Average Directional Index) - Trend Strength
    adx_indicator = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14)
    adx = adx_indicator.adx().iloc[-1]

    # Bollinger Bands - Volatility
    bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    bb_width = (bb.bollinger_hband() - bb.bollinger_lband()) / df['Close']
    current_bb_width = bb_width.iloc[-1]
    avg_bb_width = bb_width.mean()

    # ATR (Average True Range) - Volatility
    atr = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14)
    current_atr = atr.average_true_range().iloc[-1]
    atr_pct = (current_atr / df['Close'].iloc[-1]) * 100

    # EMAs - Trend Direction
    ema_20 = ta.trend.ema_indicator(df['Close'], window=20).iloc[-1]
    ema_50 = ta.trend.ema_indicator(df['Close'], window=50).iloc[-1]
    ema_200 = ta.trend.ema_indicator(df['Close'], window=200).iloc[-1]
    current_price = df['Close'].iloc[-1]

    # Calculate historical volatility
    returns = df['Close'].pct_change()
    historical_vol = returns.std() * np.sqrt(252) * 100  # Annualized %

    # Recent price action
    price_change_1m = ((df['Close'].iloc[-1] / df['Close'].iloc[-20]) - 1) * 100
    price_change_3m = ((df['Close'].iloc[-1] / df['Close'].iloc[-60]) - 1) * 100

    return {
        'adx': adx,
        'bb_width': current_bb_width,
        'avg_bb_width': avg_bb_width,
        'bb_width_ratio': current_bb_width / avg_bb_width,
        'atr_pct': atr_pct,
        'historical_vol': historical_vol,
        'ema_20': ema_20,
        'ema_50': ema_50,
        'ema_200': ema_200,
        'current_price': current_price,
        'price_change_1m': price_change_1m,
        'price_change_3m': price_change_3m,
    }


def _classify_regime(data: dict) -> dict:
    """
    Classify market regime based on indicators.

    Returns dict with:
    - trend_strength: STRONG/MODERATE/WEAK
    - trend_direction: UPTREND/DOWNTREND/SIDEWAYS
    - volatility: HIGH/NORMAL/LOW
    - regime: Overall regime classification
    - strategy_weights: Recommended strategy weights
    """

    # 1. Trend Strength (ADX)
    if data['adx'] > 40:
        trend_strength = "STRONG"
    elif data['adx'] > 25:
        trend_strength = "MODERATE"
    else:
        trend_strength = "WEAK"

    # 2. Trend Direction (EMA alignment)
    price = data['current_price']
    ema_20 = data['ema_20']
    ema_50 = data['ema_50']
    ema_200 = data['ema_200']

    if price > ema_20 > ema_50 > ema_200:
        trend_direction = "STRONG_UPTREND"
    elif price > ema_20 > ema_50:
        trend_direction = "UPTREND"
    elif price < ema_20 < ema_50 < ema_200:
        trend_direction = "STRONG_DOWNTREND"
    elif price < ema_20 < ema_50:
        trend_direction = "DOWNTREND"
    else:
        trend_direction = "SIDEWAYS"

    # 3. Volatility
    if data['historical_vol'] > 40:
        volatility = "HIGH"
    elif data['historical_vol'] > 25:
        volatility = "ELEVATED"
    elif data['historical_vol'] > 15:
        volatility = "NORMAL"
    else:
        volatility = "LOW"

    # Also check if volatility is expanding or contracting
    if data['bb_width_ratio'] > 1.3:
        vol_state = "EXPANDING"
    elif data['bb_width_ratio'] < 0.7:
        vol_state = "CONTRACTING"
    else:
        vol_state = "STABLE"

    # 4. Overall Regime Classification
    regime = _determine_overall_regime(trend_strength, trend_direction, volatility, vol_state)

    # 5. Strategy Weights
    strategy_weights = _calculate_strategy_weights(regime, trend_strength, trend_direction, volatility)

    return {
        'trend_strength': trend_strength,
        'trend_direction': trend_direction,
        'volatility': volatility,
        'vol_state': vol_state,
        'regime': regime,
        'strategy_weights': strategy_weights,
        'confidence_modifier': _calculate_confidence_modifier(regime, volatility)
    }


def _determine_overall_regime(trend_strength: str, trend_direction: str, volatility: str, vol_state: str) -> str:
    """Determine overall market regime."""

    if trend_strength == "STRONG" and trend_direction in ["STRONG_UPTREND", "STRONG_DOWNTREND"]:
        if volatility in ["HIGH", "ELEVATED"]:
            return "VOLATILE_TRENDING"
        else:
            return "STRONG_TRENDING"

    elif trend_strength in ["MODERATE", "STRONG"] and trend_direction in ["UPTREND", "DOWNTREND"]:
        return "TRENDING"

    elif trend_strength == "WEAK" and volatility in ["HIGH", "ELEVATED"]:
        return "VOLATILE_RANGING"

    elif trend_strength == "WEAK":
        return "RANGING"

    elif vol_state == "CONTRACTING" and trend_strength == "WEAK":
        return "CONSOLIDATION"

    else:
        return "MIXED"


def _calculate_strategy_weights(regime: str, trend_strength: str, trend_direction: str, volatility: str) -> dict:
    """
    Calculate recommended weights for each strategy based on regime.

    Returns:
        Dict with weights for momentum, mean_reversion, trend_following
    """

    # Default equal weights
    weights = {'momentum': 0.33, 'mean_reversion': 0.33, 'trend_following': 0.34}

    if regime == "STRONG_TRENDING":
        # Strong trends: favor momentum and trend following
        weights = {'momentum': 0.45, 'trend_following': 0.45, 'mean_reversion': 0.10}

    elif regime == "TRENDING":
        # Moderate trends: favor trend following with some momentum
        weights = {'momentum': 0.35, 'trend_following': 0.45, 'mean_reversion': 0.20}

    elif regime == "RANGING":
        # Ranging market: heavily favor mean reversion
        weights = {'momentum': 0.10, 'trend_following': 0.10, 'mean_reversion': 0.80}

    elif regime == "CONSOLIDATION":
        # Consolidation: mean reversion with preparation for breakout
        weights = {'momentum': 0.20, 'trend_following': 0.20, 'mean_reversion': 0.60}

    elif regime == "VOLATILE_TRENDING":
        # Volatile but trending: balanced with slight trend bias
        weights = {'momentum': 0.30, 'trend_following': 0.40, 'mean_reversion': 0.30}

    elif regime == "VOLATILE_RANGING":
        # High volatility, no clear trend: reduce all confidence
        weights = {'momentum': 0.33, 'trend_following': 0.33, 'mean_reversion': 0.34}

    else:  # MIXED
        # Mixed signals: equal weights
        weights = {'momentum': 0.33, 'trend_following': 0.33, 'mean_reversion': 0.34}

    return weights


def _calculate_confidence_modifier(regime: str, volatility: str) -> int:
    """
    Calculate confidence modifier based on regime.

    Returns:
        Integer modifier to add/subtract from base confidence (-20 to +10)
    """

    if regime == "STRONG_TRENDING":
        return +10  # High confidence in strong trends

    elif regime == "TRENDING":
        return +5  # Moderate confidence boost

    elif regime == "RANGING":
        return 0  # Neutral

    elif regime == "CONSOLIDATION":
        return -5  # Slightly reduce confidence (breakout pending)

    elif regime == "VOLATILE_TRENDING":
        return -5  # Volatility reduces confidence

    elif regime == "VOLATILE_RANGING":
        return -15  # High volatility + no trend = low confidence

    else:  # MIXED
        return -10  # Mixed signals reduce confidence

    # Additional volatility penalty
    if volatility == "HIGH":
        return -20


def _build_regime_report(ticker: str, data: dict, classification: dict) -> str:
    """Build human-readable regime report."""

    regime = classification['regime']
    trend_str = classification['trend_strength']
    direction = classification['trend_direction']
    vol = classification['volatility']
    vol_state = classification['vol_state']
    weights = classification['strategy_weights']
    conf_mod = classification['confidence_modifier']

    report = f"""
MARKET REGIME ANALYSIS: {ticker}

REGIME: {regime}
Trend Strength: {trend_str} (ADX: {data['adx']:.1f})
Trend Direction: {direction}
Volatility: {vol} ({vol_state}) - Annual: {data['historical_vol']:.1f}%
ATR: {data['atr_pct']:.2f}% of price

PRICE ACTION:
Current: Rs{data['current_price']:.2f}
1-Month Change: {data['price_change_1m']:+.2f}%
3-Month Change: {data['price_change_3m']:+.2f}%

EMA Alignment:
- EMA 20: Rs{data['ema_20']:.2f}
- EMA 50: Rs{data['ema_50']:.2f}
- EMA 200: Rs{data['ema_200']:.2f}

RECOMMENDED STRATEGY WEIGHTS:
- Momentum: {weights['momentum']*100:.0f}%
- Trend Following: {weights['trend_following']*100:.0f}%
- Mean Reversion: {weights['mean_reversion']*100:.0f}%

CONFIDENCE MODIFIER: {conf_mod:+d}%

STRATEGIC IMPLICATIONS:
{_get_strategic_implications(regime, direction, vol)}
"""

    return report


def _get_strategic_implications(regime: str, direction: str, volatility: str) -> str:
    """Get strategic implications for the current regime."""

    implications = []

    if regime == "STRONG_TRENDING":
        implications.append("Excellent conditions for trend-following strategies")
        implications.append("High probability of continuation")
        if direction in ["STRONG_UPTREND", "UPTREND"]:
            implications.append("Favor long positions with momentum")
        else:
            implications.append("Consider short positions or wait for reversal")

    elif regime == "TRENDING":
        implications.append("Good conditions for directional trades")
        implications.append("Monitor for potential reversal signals")

    elif regime == "RANGING":
        implications.append("Ideal for mean reversion strategies")
        implications.append("Avoid momentum breakout trades")
        implications.append("Buy near support, sell near resistance")

    elif regime == "CONSOLIDATION":
        implications.append("Market preparing for next move")
        implications.append("Low conviction - wait for breakout confirmation")
        implications.append("Watch for volume expansion and volatility spike")

    elif regime == "VOLATILE_RANGING":
        implications.append("High risk environment")
        implications.append("Reduce position sizes significantly")
        implications.append("Use wider stops or avoid trading")

    elif regime == "VOLATILE_TRENDING":
        implications.append("Trending but with high volatility")
        implications.append("Reduce position size, use wider stops")
        implications.append("Can still trade with trend, but cautiously")

    else:  # MIXED
        implications.append("Mixed signals - no clear regime")
        implications.append("Wait for clarity before taking positions")
        implications.append("Low probability environment")

    if volatility == "HIGH":
        implications.append("HIGH VOLATILITY: Expect large swings")

    return "\n".join(implications)
