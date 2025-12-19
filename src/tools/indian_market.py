"""
Indian Market Context Module

Provides India-specific market data and validations:
- NSE/BSE trading hours and holidays
- Circuit breaker detection
- Nifty 50 beta/correlation
- FII/DII flow tracking
- Indian corporate actions
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from typing import Dict, Tuple, Optional, List
from langchain.tools import tool
import pytz
from src.utils.logging_config import ToolLogger
from src.middleware.retry_handler import retry_yfinance

logger = ToolLogger("indian_market")

# NSE Trading Hours (IST)
NSE_OPEN = time(9, 15)
NSE_CLOSE = time(15, 30)
NSE_PRE_OPEN_START = time(9, 0)
NSE_PRE_OPEN_END = time(9, 8)

# NSE Holidays 2024-2025 (update annually)
NSE_HOLIDAYS_2024 = [
    "2024-01-26",  # Republic Day
    "2024-03-08",  # Maha Shivaratri
    "2024-03-25",  # Holi
    "2024-03-29",  # Good Friday
    "2024-04-11",  # Id-Ul-Fitr
    "2024-04-14",  # Dr. Ambedkar Jayanti
    "2024-04-17",  # Ram Navami
    "2024-04-21",  # Mahavir Jayanti
    "2024-05-01",  # Maharashtra Day
    "2024-05-23",  # Buddha Purnima
    "2024-06-17",  # Eid ul-Adha
    "2024-07-17",  # Muharram
    "2024-08-15",  # Independence Day
    "2024-10-02",  # Gandhi Jayanti
    "2024-11-01",  # Diwali Laxmi Pujan
    "2024-11-15",  # Guru Nanak Jayanti
    "2024-12-25",  # Christmas
]

NSE_HOLIDAYS_2025 = [
    "2025-01-26",  # Republic Day
    "2025-02-26",  # Maha Shivaratri
    "2025-03-14",  # Holi
    "2025-03-31",  # Id-Ul-Fitr
    "2025-04-10",  # Mahavir Jayanti
    "2025-04-14",  # Dr. Ambedkar Jayanti
    "2025-04-18",  # Good Friday
    "2025-05-01",  # Maharashtra Day
    "2025-05-12",  # Buddha Purnima
    "2025-06-07",  # Eid ul-Adha
    "2025-07-06",  # Muharram
    "2025-08-15",  # Independence Day
    "2025-08-16",  # Janmashtami
    "2025-10-02",  # Gandhi Jayanti/Dussehra
    "2025-10-21",  # Diwali Laxmi Pujan
    "2025-10-22",  # Diwali Balipratipada
    "2025-11-05",  # Guru Nanak Jayanti
    "2025-12-25",  # Christmas
]

NSE_HOLIDAYS = set(NSE_HOLIDAYS_2024 + NSE_HOLIDAYS_2025)

# Circuit Breaker Limits (NSE)
CIRCUIT_LIMITS = {
    'stage_1': 0.10,   # 10% - First halt
    'stage_2': 0.15,   # 15% - Second halt
    'stage_3': 0.20,   # 20% - Market closed for day
}

# Stock-specific circuit limits based on price band
PRICE_BAND_CIRCUITS = {
    'no_band': None,           # F&O stocks - no circuit
    '2_percent': 0.02,
    '5_percent': 0.05,
    '10_percent': 0.10,
    '20_percent': 0.20,
}

IST = pytz.timezone('Asia/Kolkata')


def is_nse_trading_day(date: datetime = None) -> Tuple[bool, str]:
    """
    Check if given date is an NSE trading day.

    Args:
        date: Date to check (default: today)

    Returns:
        (is_trading_day, reason)
    """
    if date is None:
        date = datetime.now(IST)

    date_str = date.strftime('%Y-%m-%d')
    weekday = date.weekday()

    # Weekend check
    if weekday >= 5:
        return False, f"Weekend ({['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][weekday]})"

    # Holiday check
    if date_str in NSE_HOLIDAYS:
        return False, f"NSE Holiday ({date_str})"

    return True, "Trading day"


def is_nse_market_open(dt: datetime = None) -> Tuple[bool, str]:
    """
    Check if NSE market is currently open.

    Args:
        dt: Datetime to check (default: now IST)

    Returns:
        (is_open, status_message)
    """
    if dt is None:
        dt = datetime.now(IST)
    elif dt.tzinfo is None:
        dt = IST.localize(dt)

    # Check if trading day
    is_trading, reason = is_nse_trading_day(dt)
    if not is_trading:
        return False, reason

    current_time = dt.time()

    if current_time < NSE_PRE_OPEN_START:
        return False, f"Pre-market (opens at 9:00 AM IST)"
    elif NSE_PRE_OPEN_START <= current_time < NSE_OPEN:
        return False, f"Pre-open session (trading starts at 9:15 AM IST)"
    elif NSE_OPEN <= current_time <= NSE_CLOSE:
        return True, "Market open"
    else:
        return False, f"Market closed (closed at 3:30 PM IST)"


def get_next_trading_day(from_date: datetime = None) -> datetime:
    """Get the next NSE trading day."""
    if from_date is None:
        from_date = datetime.now(IST)

    next_day = from_date + timedelta(days=1)

    # Skip weekends and holidays
    while True:
        is_trading, _ = is_nse_trading_day(next_day)
        if is_trading:
            return next_day
        next_day += timedelta(days=1)

        # Safety: don't loop forever
        if (next_day - from_date).days > 30:
            return next_day


def get_trading_days_between(start_date: datetime, end_date: datetime) -> int:
    """Count NSE trading days between two dates."""
    count = 0
    current = start_date

    while current <= end_date:
        is_trading, _ = is_nse_trading_day(current)
        if is_trading:
            count += 1
        current += timedelta(days=1)

    return count


@tool
def get_nse_market_status() -> str:
    """
    Get current NSE market status including trading hours and next session info.

    Returns:
        Market status report with timing information
    """
    now = datetime.now(IST)
    is_open, status = is_nse_market_open(now)

    report = f"""
NSE MARKET STATUS
{'='*40}
Current Time (IST): {now.strftime('%Y-%m-%d %H:%M:%S')}
Status: {'OPEN' if is_open else 'CLOSED'}
Reason: {status}

Trading Hours:
- Pre-open: 9:00 AM - 9:08 AM IST
- Normal Session: 9:15 AM - 3:30 PM IST

"""

    if not is_open:
        next_trading = get_next_trading_day(now)
        report += f"Next Trading Day: {next_trading.strftime('%Y-%m-%d (%A)')}\n"

    # Check upcoming holidays
    upcoming = []
    for holiday in sorted(NSE_HOLIDAYS):
        h_date = datetime.strptime(holiday, '%Y-%m-%d')
        if h_date.date() > now.date() and len(upcoming) < 3:
            upcoming.append(holiday)

    if upcoming:
        report += f"\nUpcoming Holidays: {', '.join(upcoming)}\n"

    return report


@tool
def detect_circuit_breaker(ticker: str, lookback_days: int = 5) -> str:
    """
    Detect if stock hit circuit breaker limits in recent sessions.

    NSE Circuit Limits:
    - Index: 10%, 15%, 20% from previous close
    - Stocks: Price band dependent (2%, 5%, 10%, 20% or no limit for F&O)

    Args:
        ticker: Stock ticker (e.g., "RELIANCE.NS")
        lookback_days: Days to check

    Returns:
        Circuit breaker analysis report
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="1mo")

        if df.empty:
            return f"No data available for {ticker}"

        # Check if F&O stock (no circuit for F&O stocks in India)
        info = stock.info
        is_fno = info.get('quoteType') == 'EQUITY'  # Simplified check

        # Calculate daily moves
        df['daily_change'] = (df['Close'] - df['Open']) / df['Open']
        df['gap_change'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        df['total_change'] = (df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)

        recent = df.tail(lookback_days)

        # Detect circuit hits
        circuits_hit = []
        for idx, row in recent.iterrows():
            daily_move = abs(row['total_change']) * 100

            if daily_move >= 20:
                circuits_hit.append(f"{idx.date()}: {row['total_change']*100:+.1f}% (UPPER/LOWER CIRCUIT - 20%)")
            elif daily_move >= 15:
                circuits_hit.append(f"{idx.date()}: {row['total_change']*100:+.1f}% (Near 15% circuit)")
            elif daily_move >= 10:
                circuits_hit.append(f"{idx.date()}: {row['total_change']*100:+.1f}% (Near 10% circuit)")

        # Calculate average daily move
        avg_move = recent['total_change'].abs().mean() * 100
        max_move = recent['total_change'].abs().max() * 100

        report = f"""
CIRCUIT BREAKER ANALYSIS: {ticker}
{'='*40}
Period: Last {lookback_days} trading days
Average Daily Move: {avg_move:.2f}%
Max Single-Day Move: {max_move:.2f}%

"""

        if circuits_hit:
            report += "CIRCUIT EVENTS DETECTED:\n"
            for event in circuits_hit:
                report += f"  - {event}\n"
            report += "\nWARNING: Stock showing high volatility near circuit limits.\n"
            report += "Consider: Wider stops, smaller position size, limit orders only.\n"
        else:
            report += "No circuit breaker events in recent sessions.\n"
            report += "Stock trading within normal range.\n"

        # Add intraday volatility warning
        if avg_move > 3:
            report += f"\nHIGH VOLATILITY WARNING: Avg daily move {avg_move:.1f}% exceeds 3%\n"

        logger.log_fetch(ticker, "circuit_breaker", True, 0, len(df))
        return report

    except Exception as e:
        logger.log_fetch(ticker, "circuit_breaker", False, 0, 0, str(e))
        return f"Error detecting circuit breakers for {ticker}: {str(e)}"


@tool
@retry_yfinance
def get_nifty_correlation(ticker: str, period: str = "1y") -> str:
    """
    Calculate stock's correlation and beta with Nifty 50 index.

    Beta > 1: Stock moves more than market
    Beta < 1: Stock moves less than market
    Beta < 0: Stock moves opposite to market

    Args:
        ticker: Stock ticker (e.g., "TCS.NS")
        period: Analysis period (default: 1y)

    Returns:
        Correlation analysis with Nifty 50
    """
    try:
        # Fetch stock data
        stock_df = yf.download(ticker, period=period, progress=False)
        nifty_df = yf.download("^NSEI", period=period, progress=False)

        if stock_df.empty or nifty_df.empty:
            return f"Unable to fetch data for correlation analysis"

        # Calculate returns
        stock_returns = stock_df['Close'].pct_change().dropna()
        nifty_returns = nifty_df['Close'].pct_change().dropna()

        # Align dates
        common_dates = stock_returns.index.intersection(nifty_returns.index)
        stock_returns = stock_returns.loc[common_dates]
        nifty_returns = nifty_returns.loc[common_dates]

        if len(common_dates) < 30:
            return "Insufficient data for correlation analysis (need 30+ days)"

        # Calculate correlation
        correlation = stock_returns.corr(nifty_returns)

        # Calculate beta (covariance / variance)
        covariance = stock_returns.cov(nifty_returns)
        variance = nifty_returns.var()
        beta = covariance / variance

        # Calculate alpha (Jensen's alpha - simplified)
        stock_avg_return = stock_returns.mean() * 252  # Annualized
        nifty_avg_return = nifty_returns.mean() * 252
        risk_free_rate = 0.065  # ~6.5% India 10Y bond yield
        alpha = stock_avg_return - (risk_free_rate + beta * (nifty_avg_return - risk_free_rate))

        # Rolling correlation (30-day)
        rolling_corr = stock_returns.rolling(30).corr(nifty_returns)
        recent_corr = rolling_corr.iloc[-1] if not rolling_corr.empty else correlation

        # Interpret beta
        if beta > 1.5:
            beta_interpretation = "HIGH BETA - Amplifies market moves significantly"
            risk_note = "High risk in market downturns, high reward in uptrends"
        elif beta > 1.0:
            beta_interpretation = "ABOVE MARKET - Moves slightly more than Nifty"
            risk_note = "Moderate amplification of market moves"
        elif beta > 0.7:
            beta_interpretation = "MARKET ALIGNED - Moves with Nifty"
            risk_note = "Good market proxy, diversification limited"
        elif beta > 0:
            beta_interpretation = "DEFENSIVE - Moves less than market"
            risk_note = "Lower volatility, potential hedge"
        else:
            beta_interpretation = "NEGATIVE BETA - Counter-market moves"
            risk_note = "Natural hedge against market declines"

        # Correlation interpretation
        if abs(correlation) > 0.8:
            corr_note = "Strong correlation - moves closely with Nifty"
        elif abs(correlation) > 0.5:
            corr_note = "Moderate correlation - partially tracks Nifty"
        else:
            corr_note = "Weak correlation - stock-specific factors dominate"

        report = f"""
NIFTY 50 CORRELATION ANALYSIS: {ticker}
{'='*50}

KEY METRICS:
- Correlation (1Y): {correlation:.3f}
- Correlation (30D): {recent_corr:.3f}
- Beta: {beta:.2f}
- Alpha (Annualized): {alpha*100:.1f}%

BETA INTERPRETATION:
{beta_interpretation}
{risk_note}

CORRELATION NOTE:
{corr_note}

TRADING IMPLICATIONS:
"""

        if beta > 1.2 and correlation > 0.7:
            report += "- Monitor Nifty 50 closely - stock amplifies index moves\n"
            report += "- In Nifty uptrend: Strong buy candidate\n"
            report += "- In Nifty downtrend: Avoid or hedge position\n"
        elif beta < 0.8 and correlation > 0.5:
            report += "- Defensive play in uncertain markets\n"
            report += "- Lower upside but better downside protection\n"
        else:
            report += "- Stock has independent movement patterns\n"
            report += "- Focus on stock-specific catalysts over market direction\n"

        # Add current Nifty context
        nifty_current = nifty_df['Close'].iloc[-1]
        nifty_50d_ma = nifty_df['Close'].rolling(50).mean().iloc[-1]
        nifty_trend = "BULLISH" if nifty_current > nifty_50d_ma else "BEARISH"

        report += f"""
CURRENT NIFTY CONTEXT:
- Nifty 50: {nifty_current:.0f}
- 50-Day MA: {nifty_50d_ma:.0f}
- Trend: {nifty_trend}

RECOMMENDATION:
"""
        if nifty_trend == "BULLISH" and beta > 1:
            report += "Nifty bullish + High beta = Favorable for long positions"
        elif nifty_trend == "BEARISH" and beta > 1:
            report += "Nifty bearish + High beta = CAUTION - Consider hedging/avoiding"
        elif nifty_trend == "BEARISH" and beta < 0.8:
            report += "Nifty bearish + Low beta = Relative safety, can hold"
        else:
            report += "Monitor both stock-specific and market factors"

        logger.log_fetch(ticker, "nifty_correlation", True, 0, len(common_dates))
        return report

    except Exception as e:
        logger.log_fetch(ticker, "nifty_correlation", False, 0, 0, str(e))
        return f"Error calculating Nifty correlation for {ticker}: {str(e)}"


@tool
def get_fii_dii_sentiment() -> str:
    """
    Get FII/DII (Foreign & Domestic Institutional Investor) flow sentiment.

    FII buying = Bullish signal (foreign money inflow)
    DII buying during FII selling = Support signal (domestic absorption)
    Both selling = Bearish signal

    Note: This provides general guidance. For real-time data,
    check NSE website or financial portals.

    Returns:
        FII/DII sentiment analysis and implications
    """
    try:
        # Fetch Nifty 50 as proxy for institutional activity
        nifty = yf.download("^NSEI", period="3mo", progress=False)

        if nifty.empty:
            return "Unable to fetch market data for FII/DII analysis"

        # Calculate volume trends as proxy for institutional activity
        nifty['Volume_MA20'] = nifty['Volume'].rolling(20).mean()
        nifty['Volume_Ratio'] = nifty['Volume'] / nifty['Volume_MA20']

        # Price momentum
        nifty['Returns_5D'] = nifty['Close'].pct_change(5)
        nifty['Returns_20D'] = nifty['Close'].pct_change(20)

        recent = nifty.tail(5)
        last = nifty.iloc[-1]

        # Infer institutional sentiment from price + volume
        price_trend = "UP" if last['Returns_20D'] > 0.02 else ("DOWN" if last['Returns_20D'] < -0.02 else "FLAT")
        volume_trend = "HIGH" if last['Volume_Ratio'] > 1.2 else ("LOW" if last['Volume_Ratio'] < 0.8 else "NORMAL")

        # Sentiment inference
        if price_trend == "UP" and volume_trend == "HIGH":
            fii_sentiment = "LIKELY BUYING"
            dii_sentiment = "MIXED"
            overall = "BULLISH - Strong institutional support"
        elif price_trend == "DOWN" and volume_trend == "HIGH":
            fii_sentiment = "LIKELY SELLING"
            dii_sentiment = "LIKELY BUYING (absorbing)"
            overall = "CAUTIOUS - FII outflow, DII support"
        elif price_trend == "DOWN" and volume_trend == "LOW":
            fii_sentiment = "INACTIVE"
            dii_sentiment = "INACTIVE"
            overall = "NEUTRAL - Low conviction either way"
        else:
            fii_sentiment = "MIXED"
            dii_sentiment = "MIXED"
            overall = "NEUTRAL - No clear institutional bias"

        report = f"""
FII/DII SENTIMENT ANALYSIS
{'='*50}

IMPORTANT: This is an inference based on market data.
For actual FII/DII figures, check:
- NSE: https://www.nseindia.com/reports/fii-dii
- MoneyControl: https://www.moneycontrol.com/stocks/marketstats/fii_dii_activity/

INFERRED SENTIMENT (Based on Nifty Price + Volume):
- FII (Foreign Institutional): {fii_sentiment}
- DII (Domestic Institutional): {dii_sentiment}
- Overall Institutional Mood: {overall}

MARKET CONTEXT:
- Nifty 5-Day Return: {last['Returns_5D']*100:.1f}%
- Nifty 20-Day Return: {last['Returns_20D']*100:.1f}%
- Volume vs 20D Avg: {last['Volume_Ratio']:.2f}x

INTERPRETATION GUIDE:
- FII Buying + DII Buying = Strong Rally likely
- FII Selling + DII Buying = Support, but upside limited
- FII Selling + DII Selling = Correction/Bear phase
- FII Buying + DII Selling = Early stage rally

TRADING IMPLICATIONS:
"""

        if "BUYING" in fii_sentiment:
            report += "- Favor momentum strategies on large-caps\n"
            report += "- IT, Banking stocks typically benefit from FII flows\n"
        elif "SELLING" in fii_sentiment:
            report += "- Be cautious on high-beta, FII-heavy stocks\n"
            report += "- Domestic-focused sectors may outperform\n"
            report += "- Watch for rupee depreciation impact\n"
        else:
            report += "- Wait for clearer institutional direction\n"
            report += "- Focus on stock-specific opportunities\n"

        return report

    except Exception as e:
        return f"Error analyzing FII/DII sentiment: {str(e)}"


@tool
def get_india_vix_context() -> str:
    """
    Get India VIX (Volatility Index) context for risk assessment.

    India VIX < 15: Low fear, complacency (potential reversal)
    India VIX 15-20: Normal volatility
    India VIX 20-25: Elevated fear
    India VIX > 25: High fear, potential capitulation

    Returns:
        India VIX analysis and trading implications
    """
    try:
        # India VIX ticker on Yahoo Finance
        vix = yf.download("^INDIAVIX", period="3mo", progress=False)

        if vix.empty:
            return "Unable to fetch India VIX data"

        current_vix = vix['Close'].iloc[-1]
        vix_20d_ma = vix['Close'].rolling(20).mean().iloc[-1]
        vix_high_52w = vix['Close'].max()
        vix_low_52w = vix['Close'].min()

        # VIX trend
        vix_5d_change = (current_vix - vix['Close'].iloc[-5]) / vix['Close'].iloc[-5] * 100

        # Interpret VIX level
        if current_vix < 13:
            vix_interpretation = "VERY LOW - Market complacent"
            implication = "Potential for sudden spike, consider hedging"
        elif current_vix < 17:
            vix_interpretation = "LOW - Calm markets"
            implication = "Favorable for swing trades, normal stops"
        elif current_vix < 22:
            vix_interpretation = "MODERATE - Normal volatility"
            implication = "Standard trading conditions"
        elif current_vix < 28:
            vix_interpretation = "ELEVATED - Increased fear"
            implication = "Widen stops, reduce position size"
        else:
            vix_interpretation = "HIGH - Fear/Panic"
            implication = "Potential capitulation, contrarian opportunity OR stay out"

        report = f"""
INDIA VIX ANALYSIS
{'='*50}

CURRENT STATUS:
- India VIX: {current_vix:.2f}
- 20-Day MA: {vix_20d_ma:.2f}
- 5-Day Change: {vix_5d_change:+.1f}%
- 52-Week Range: {vix_low_52w:.1f} - {vix_high_52w:.1f}

INTERPRETATION:
{vix_interpretation}

MARKET IMPLICATION:
{implication}

VIX LEVELS GUIDE:
- Below 13: Extreme calm (contrarian bearish)
- 13-17: Low volatility (bullish, but watch for spikes)
- 17-22: Normal range (standard trading)
- 22-28: Elevated (reduce risk exposure)
- Above 28: High fear (potential buying opportunity if brave)

POSITION SIZING ADJUSTMENT:
"""

        if current_vix < 15:
            report += "- Standard position size OK\n"
            report += "- Consider buying puts for insurance\n"
        elif current_vix < 20:
            report += "- Standard position size OK\n"
            report += "- Normal stop-loss distances\n"
        elif current_vix < 25:
            report += "- REDUCE position size by 25-50%\n"
            report += "- WIDEN stops by 1.5x\n"
        else:
            report += "- REDUCE position size by 50-75%\n"
            report += "- WIDEN stops by 2x OR stay in cash\n"
            report += "- Only trade high-conviction setups\n"

        logger.log_fetch("INDIAVIX", "vix_analysis", True, 0, len(vix))
        return report

    except Exception as e:
        return f"Error fetching India VIX: {str(e)}"


@tool
def get_earnings_calendar(ticker: str) -> str:
    """
    Get upcoming earnings date and recent results context for Indian stock.

    Args:
        ticker: Stock ticker (e.g., "TCS.NS")

    Returns:
        Earnings calendar and historical performance around results
    """
    try:
        stock = yf.Ticker(ticker)

        # Get earnings dates
        try:
            earnings_dates = stock.earnings_dates
            if earnings_dates is not None and not earnings_dates.empty:
                upcoming = earnings_dates[earnings_dates.index > datetime.now()]
                if not upcoming.empty:
                    next_earnings = upcoming.index[0]
                    days_until = (next_earnings - datetime.now()).days
                else:
                    next_earnings = None
                    days_until = None
            else:
                next_earnings = None
                days_until = None
        except:
            next_earnings = None
            days_until = None

        # Get historical data for earnings reaction analysis
        hist = stock.history(period="2y")

        report = f"""
EARNINGS CALENDAR: {ticker}
{'='*50}

"""

        if next_earnings and days_until:
            report += f"NEXT EARNINGS: {next_earnings.strftime('%Y-%m-%d')}\n"
            report += f"Days Until: {days_until}\n\n"

            if days_until <= 7:
                report += "WARNING: Earnings within 1 week!\n"
                report += "- Expect increased volatility\n"
                report += "- Consider reducing position or hedging\n"
                report += "- Gap risk is HIGH\n\n"
            elif days_until <= 21:
                report += "CAUTION: Earnings within 3 weeks\n"
                report += "- Plan exit before results if swing trading\n"
                report += "- IV likely elevated for options\n\n"
        else:
            report += "Earnings date: Not available (check company announcements)\n\n"

        # Indian financial calendar context
        now = datetime.now()
        current_month = now.month

        # Q1: Apr-Jun, Q2: Jul-Sep, Q3: Oct-Dec, Q4: Jan-Mar
        if current_month in [4, 5]:
            report += "SEASON: Q4 results (Jan-Mar) being announced\n"
        elif current_month in [7, 8]:
            report += "SEASON: Q1 results (Apr-Jun) being announced\n"
        elif current_month in [10, 11]:
            report += "SEASON: Q2 results (Jul-Sep) being announced\n"
        elif current_month in [1, 2]:
            report += "SEASON: Q3 results (Oct-Dec) being announced\n"

        report += """
INDIAN RESULTS PATTERN:
- IT Sector: Usually first to report (TCS kicks off)
- Banks: Mid-month of earnings season
- PSUs: Often delayed announcements

TRADING AROUND EARNINGS:
- Avoid new positions 3 days before results
- If holding, decide: Exit before OR hold through
- Post-results gaps can be 5-10% for IT/Banks
"""

        logger.log_fetch(ticker, "earnings_calendar", True, 0, 1)
        return report

    except Exception as e:
        return f"Error fetching earnings calendar for {ticker}: {str(e)}"


def get_indian_market_context(ticker: str) -> Dict:
    """
    Get comprehensive Indian market context for a stock.

    Returns dict with all relevant Indian market factors.
    Used internally by agents.
    """
    context = {
        'market_open': is_nse_market_open(),
        'ticker_exchange': 'NSE' if '.NS' in ticker else 'BSE' if '.BO' in ticker else 'Unknown',
    }

    try:
        # Get beta with Nifty
        stock_df = yf.download(ticker, period="1y", progress=False)
        nifty_df = yf.download("^NSEI", period="1y", progress=False)

        if not stock_df.empty and not nifty_df.empty:
            stock_returns = stock_df['Close'].pct_change().dropna()
            nifty_returns = nifty_df['Close'].pct_change().dropna()

            common = stock_returns.index.intersection(nifty_returns.index)
            if len(common) > 30:
                correlation = stock_returns.loc[common].corr(nifty_returns.loc[common])
                beta = stock_returns.loc[common].cov(nifty_returns.loc[common]) / nifty_returns.loc[common].var()

                context['nifty_correlation'] = float(correlation)
                context['nifty_beta'] = float(beta)

        # Get VIX
        vix = yf.download("^INDIAVIX", period="5d", progress=False)
        if not vix.empty:
            context['india_vix'] = float(vix['Close'].iloc[-1])

    except Exception as e:
        context['error'] = str(e)

    return context


# Export all tools
INDIAN_MARKET_TOOLS = [
    get_nse_market_status,
    detect_circuit_breaker,
    get_nifty_correlation,
    get_fii_dii_sentiment,
    get_india_vix_context,
    get_earnings_calendar,
]
