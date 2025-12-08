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
        ev = info.get('enterpriseValue')
        ebitda = info.get('ebitda')
        ev_ebitda = ev / ebitda if ev and ebitda else "N/A"
        ps = info.get('priceToSalesTrailing12Months', 'N/A')
        
        # Profitability
        roe = info.get('returnOnEquity', 'N/A')
        roa = info.get('returnOnAssets', 'N/A')
        profit_margin = info.get('profitMargins', 'N/A')
        op_margin = info.get('operatingMargins', 'N/A')
        
        # Cash flow & earnings quality
        net_income = info.get('netIncome')
        operating_cf = info.get('operatingCashflow')
        free_cash_flow = info.get('freeCashflow')
        cash_conversion = (operating_cf / net_income) if (operating_cf and net_income) else "N/A"
        accrual_ratio = ((net_income - operating_cf) / info.get('totalRevenue')) if (operating_cf and net_income and info.get('totalRevenue')) else "N/A"
        
        # Financial Health / Leverage
        debt_to_equity = info.get('debtToEquity', 'N/A')
        current_ratio = info.get('currentRatio', 'N/A')
        total_debt = info.get('totalDebt')
        total_cash = info.get('totalCash')
        ocf_to_debt = (operating_cf / total_debt) if (operating_cf and total_debt) else "N/A"
        fcf_to_debt = (free_cash_flow / total_debt) if (free_cash_flow and total_debt) else "N/A"
        debt_to_ebitda = (total_debt / ebitda) if (total_debt and ebitda) else "N/A"
        
        # Formatting
        if isinstance(roe, (int, float)): roe = f"{roe*100:.2f}%"
        if isinstance(profit_margin, (int, float)): profit_margin = f"{profit_margin*100:.2f}%"
        if isinstance(free_cash_flow, (int, float)): free_cash_flow = f"Rs{free_cash_flow/10_000_000:.2f} Cr"
        if isinstance(cash_conversion, (int, float)): cash_conversion = f"{cash_conversion:.2f}x"
        if isinstance(accrual_ratio, (int, float)): accrual_ratio = f"{accrual_ratio*100:.1f}%"
        if isinstance(ocf_to_debt, (int, float)): ocf_to_debt = f"{ocf_to_debt:.2f}x"
        if isinstance(fcf_to_debt, (int, float)): fcf_to_debt = f"{fcf_to_debt:.2f}x"
        if isinstance(debt_to_ebitda, (int, float)): debt_to_ebitda = f"{debt_to_ebitda:.2f}x"
        if isinstance(ev_ebitda, (int, float)): ev_ebitda = f"{ev_ebitda:.2f}x"
        
        report = f"""
FUNDAMENTAL ANALYSIS: {ticker}
{'='*40}
VALUATION:
- P/E (Trailing): {pe}
- P/E (Forward): {fwd_pe}
- P/B Ratio: {pb}
- PEG Ratio: {peg}
- EV/EBITDA: {ev_ebitda}
- P/S (TTM): {ps}

PROFITABILITY:
- ROE: {roe}
- Profit Margin: {profit_margin}
- Operating Margin: {op_margin}

EARNINGS QUALITY:
- Cash Conversion (OCF/Net Income): {cash_conversion}
- Accrual Ratio (NI - OCF as % Revenue): {accrual_ratio}

CASH FLOW & COVERAGE:
- Free Cash Flow: {free_cash_flow}
- OCF / Total Debt: {ocf_to_debt}
- FCF / Total Debt: {fcf_to_debt}

LEVERAGE & LIQUIDITY:
- Debt/Equity: {debt_to_equity}
- Debt/EBITDA: {debt_to_ebitda}
- Current Ratio: {current_ratio}
- Cash: {total_cash}
- Total Debt: {total_debt}
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
