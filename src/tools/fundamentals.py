import yfinance as yf
from langchain.tools import tool
from typing import Dict, Union
from datetime import datetime
from src.utils.logging_config import ToolLogger
from src.middleware.retry_handler import retry_yfinance

logger = ToolLogger("fundamentals")

@tool
@retry_yfinance
def get_fundamental_metrics(ticker: str) -> str:
    """
    Fetches key fundamental metrics for a stock.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        String report containing valuation, profitability, and financial health metrics.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        if not info:
            logger.log_fetch(ticker, "fundamentals", False, 0, 0, "Empty fundamentals from provider")
            return f"Fundamental data unavailable for {ticker} (provider returned empty info)."
        
        # Valuation
        pe = info.get('trailingPE', 'N/A')
        fwd_pe = info.get('forwardPE', 'N/A')
        pb = info.get('priceToBook', 'N/A')
        peg = info.get('pegRatio', 'N/A')
        
        # Profitability
        roe = info.get('returnOnEquity', 'N/A')
        roa = info.get('returnOnAssets', 'N/A')
        profit_margin = info.get('profitMargins', 'N/A')
        op_margin = info.get('operatingMargins', 'N/A')
        
        # Financial Health
        debt_to_equity = info.get('debtToEquity', 'N/A')
        current_ratio = info.get('currentRatio', 'N/A')
        free_cash_flow = info.get('freeCashflow', 'N/A')
        
        # Formatting
        if isinstance(roe, (int, float)): roe = f"{roe*100:.2f}%"
        if isinstance(profit_margin, (int, float)): profit_margin = f"{profit_margin*100:.2f}%"
        if isinstance(free_cash_flow, (int, float)): free_cash_flow = f"Rs{free_cash_flow/10_000_000:.2f} Cr"
        
        report = f"""
FUNDAMENTAL ANALYSIS: {ticker}
{'='*40}
VALUATION:
- P/E (Trailing): {pe}
- P/E (Forward): {fwd_pe}
- P/B Ratio: {pb}
- PEG Ratio: {peg}

PROFITABILITY:
- ROE: {roe}
- Profit Margin: {profit_margin}
- Operating Margin: {op_margin}

FINANCIAL HEALTH:
- Debt/Equity: {debt_to_equity}
- Current Ratio: {current_ratio}
- Free Cash Flow: {free_cash_flow}
{'='*40}
"""
        logger.log_fetch(ticker, "fundamentals", True, 0, len(report))
        return report
        
    except Exception as e:
        logger.log_fetch(ticker, "fundamentals", False, 0, 0, str(e))
        return f"Error fetching fundamentals for {ticker}: {str(e)}"

@tool
@retry_yfinance
def get_growth_metrics(ticker: str) -> str:
    """
    Fetches growth metrics (Revenue, Earnings) for a stock.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        if not info:
            return f"Growth data unavailable for {ticker} (provider returned empty info)."
        
        rev_growth = info.get('revenueGrowth', 'N/A')
        earnings_growth = info.get('earningsGrowth', 'N/A')
        
        # Formatting
        if isinstance(rev_growth, (int, float)): rev_growth = f"{rev_growth*100:.2f}%"
        if isinstance(earnings_growth, (int, float)): earnings_growth = f"{earnings_growth*100:.2f}%"
        
        return f"""
GROWTH METRICS: {ticker}
- Revenue Growth (YoY): {rev_growth}
- Earnings Growth (YoY): {earnings_growth}
"""
    except Exception as e:
        return f"Error fetching growth metrics: {str(e)}"
