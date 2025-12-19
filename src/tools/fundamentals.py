import yfinance as yf
from langchain.tools import tool
from typing import Dict, Union
from datetime import datetime
from src.utils.logging_config import ToolLogger
from src.middleware.retry_handler import retry_yfinance

logger = ToolLogger("fundamentals")

# Indian sector-specific metric thresholds and benchmarks
INDIAN_SECTOR_BENCHMARKS = {
    'Technology': {
        'pe_range': (20, 35),
        'roe_min': 20,
        'key_metrics': ['Revenue Growth', 'Deal Pipeline', 'Attrition Rate'],
        'watch': 'USD-INR impact, US IT spending, Digital revenue %'
    },
    'Information Technology': {
        'pe_range': (20, 35),
        'roe_min': 20,
        'key_metrics': ['Revenue Growth', 'Deal Pipeline', 'Attrition Rate'],
        'watch': 'USD-INR impact, US IT spending, Digital revenue %'
    },
    'Financial Services': {
        'pe_range': (10, 20),
        'pb_range': (1.5, 4.0),
        'roe_min': 12,
        'key_metrics': ['NIM', 'GNPA', 'NNPA', 'CASA Ratio', 'PCR'],
        'watch': 'RBI policy, Credit growth, Slippages'
    },
    'Financials': {
        'pe_range': (10, 20),
        'pb_range': (1.5, 4.0),
        'roe_min': 12,
        'key_metrics': ['NIM', 'GNPA', 'NNPA', 'CASA Ratio', 'PCR'],
        'watch': 'RBI policy, Credit growth, Slippages'
    },
    'Healthcare': {
        'pe_range': (20, 40),
        'roe_min': 12,
        'key_metrics': ['R&D Spend', 'ANDA Pipeline', 'US Revenue %'],
        'watch': 'USFDA observations, Price erosion in US generics'
    },
    'Pharmaceuticals': {
        'pe_range': (20, 40),
        'roe_min': 12,
        'key_metrics': ['R&D Spend', 'ANDA Pipeline', 'US Revenue %'],
        'watch': 'USFDA observations, Price erosion in US generics'
    },
    'Consumer Defensive': {
        'pe_range': (40, 70),
        'roe_min': 20,
        'key_metrics': ['Volume Growth', 'Rural Mix', 'Distribution Reach'],
        'watch': 'Rural demand, Monsoon, Input costs, GST'
    },
    'Consumer Cyclical': {
        'pe_range': (15, 35),
        'roe_min': 12,
        'key_metrics': ['Same Store Sales', 'Footfall', 'Inventory Days'],
        'watch': 'Discretionary spending, Festival season'
    },
    'Basic Materials': {
        'pe_range': (8, 15),
        'roe_min': 10,
        'key_metrics': ['EBITDA/Ton', 'Capacity Utilization', 'Debt/EBITDA'],
        'watch': 'China demand, LME prices, Import duties'
    },
    'Energy': {
        'pe_range': (8, 15),
        'roe_min': 10,
        'key_metrics': ['GRM', 'Marketing Margin', 'Upstream Realization'],
        'watch': 'Crude prices, Govt subsidy policy, Gas pricing'
    },
    'Industrials': {
        'pe_range': (20, 40),
        'roe_min': 12,
        'key_metrics': ['Order Book', 'Book-to-Bill', 'Execution Rate'],
        'watch': 'Govt capex, PLI schemes, Infrastructure spending'
    },
    'Real Estate': {
        'pe_range': (15, 30),
        'pb_range': (1.0, 3.0),
        'key_metrics': ['Pre-sales', 'Collections', 'Unsold Inventory'],
        'watch': 'Home loan rates, RERA, Launches pipeline'
    },
}


def _get_sector_context(sector: str, info: dict) -> str:
    """Get Indian sector-specific context and analysis."""
    benchmarks = INDIAN_SECTOR_BENCHMARKS.get(sector, {})
    if not benchmarks:
        return ""

    context = f"\nINDIAN SECTOR CONTEXT ({sector}):\n"

    # P/E assessment
    pe = info.get('trailingPE')
    if pe and 'pe_range' in benchmarks:
        low, high = benchmarks['pe_range']
        if pe < low:
            context += f"- P/E ({pe:.1f}) BELOW sector range ({low}-{high}) - Potentially undervalued\n"
        elif pe > high:
            context += f"- P/E ({pe:.1f}) ABOVE sector range ({low}-{high}) - Premium valuation\n"
        else:
            context += f"- P/E ({pe:.1f}) within sector range ({low}-{high}) - Fair valuation\n"

    # P/B assessment (mainly for banks/financials)
    pb = info.get('priceToBook')
    if pb and 'pb_range' in benchmarks:
        low, high = benchmarks['pb_range']
        if pb < low:
            context += f"- P/B ({pb:.2f}) BELOW sector range - Check for asset quality issues\n"
        elif pb > high:
            context += f"- P/B ({pb:.2f}) ABOVE sector range - Premium for quality\n"

    # ROE assessment
    roe = info.get('returnOnEquity')
    if roe and 'roe_min' in benchmarks:
        roe_pct = roe * 100 if roe < 1 else roe
        if roe_pct < benchmarks['roe_min']:
            context += f"- ROE ({roe_pct:.1f}%) BELOW sector benchmark ({benchmarks['roe_min']}%) - Efficiency concern\n"
        else:
            context += f"- ROE ({roe_pct:.1f}%) meets sector benchmark - Good capital efficiency\n"

    # Key metrics to watch
    if 'key_metrics' in benchmarks:
        context += f"- Key Metrics to Track: {', '.join(benchmarks['key_metrics'])}\n"

    if 'watch' in benchmarks:
        context += f"- Watch For: {benchmarks['watch']}\n"

    return context


@tool
@retry_yfinance
def get_fundamental_metrics(ticker: str) -> str:
    """
    Fetches key fundamental metrics for a stock with Indian sector context.

    Args:
        ticker: Stock ticker symbol

    Returns:
        String report containing valuation, profitability, financial health,
        and Indian sector-specific analysis.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Check if Indian stock
        is_indian = ticker.endswith('.NS') or ticker.endswith('.BO')
        sector = info.get('sector', 'Unknown')

        # Valuation
        pe = info.get('trailingPE', 'N/A')
        fwd_pe = info.get('forwardPE', 'N/A')
        pb = info.get('priceToBook', 'N/A')
        peg = info.get('pegRatio', 'N/A')
        ev_ebitda = info.get('enterpriseToEbitda', 'N/A')

        # Profitability
        roe = info.get('returnOnEquity', 'N/A')
        roa = info.get('returnOnAssets', 'N/A')
        profit_margin = info.get('profitMargins', 'N/A')
        op_margin = info.get('operatingMargins', 'N/A')
        gross_margin = info.get('grossMargins', 'N/A')

        # Financial Health
        debt_to_equity = info.get('debtToEquity', 'N/A')
        current_ratio = info.get('currentRatio', 'N/A')
        quick_ratio = info.get('quickRatio', 'N/A')
        free_cash_flow = info.get('freeCashflow', 'N/A')
        total_debt = info.get('totalDebt', 'N/A')

        # Dividends (important for Indian investors)
        div_yield = info.get('dividendYield', 'N/A')
        payout_ratio = info.get('payoutRatio', 'N/A')

        # Formatting
        if isinstance(roe, (int, float)): roe = f"{roe*100:.2f}%"
        if isinstance(roa, (int, float)): roa = f"{roa*100:.2f}%"
        if isinstance(profit_margin, (int, float)): profit_margin = f"{profit_margin*100:.2f}%"
        if isinstance(op_margin, (int, float)): op_margin = f"{op_margin*100:.2f}%"
        if isinstance(gross_margin, (int, float)): gross_margin = f"{gross_margin*100:.2f}%"
        if isinstance(free_cash_flow, (int, float)): free_cash_flow = f"Rs{free_cash_flow/10_000_000:.2f} Cr"
        if isinstance(total_debt, (int, float)): total_debt = f"Rs{total_debt/10_000_000:.2f} Cr"
        if isinstance(div_yield, (int, float)): div_yield = f"{div_yield*100:.2f}%"
        if isinstance(payout_ratio, (int, float)): payout_ratio = f"{payout_ratio*100:.1f}%"

        report = f"""
FUNDAMENTAL ANALYSIS: {ticker}
{'='*50}
Sector: {sector}

VALUATION:
- P/E (Trailing): {pe}
- P/E (Forward): {fwd_pe}
- P/B Ratio: {pb}
- PEG Ratio: {peg}
- EV/EBITDA: {ev_ebitda}

PROFITABILITY:
- ROE: {roe}
- ROA: {roa}
- Gross Margin: {gross_margin}
- Operating Margin: {op_margin}
- Net Profit Margin: {profit_margin}

FINANCIAL HEALTH:
- Debt/Equity: {debt_to_equity}
- Current Ratio: {current_ratio}
- Quick Ratio: {quick_ratio}
- Total Debt: {total_debt}
- Free Cash Flow: {free_cash_flow}

SHAREHOLDER RETURNS:
- Dividend Yield: {div_yield}
- Payout Ratio: {payout_ratio}
"""

        # Add Indian sector-specific context
        if is_indian and sector:
            sector_context = _get_sector_context(sector, info)
            if sector_context:
                report += sector_context

        report += f"{'='*50}\n"

        logger.log_fetch(ticker, "fundamentals", True, 0, len(info))
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


@tool
def get_usdinr_context() -> str:
    """
    Get USD-INR exchange rate context - critical for IT, Pharma exporters.

    Rupee depreciation = Positive for IT/Pharma exporters
    Rupee appreciation = Negative for exporters, positive for importers

    Returns:
        USD-INR analysis with sector implications
    """
    try:
        # Fetch USD-INR data
        usdinr = yf.download("USDINR=X", period="6mo", progress=False)

        if usdinr.empty:
            return "Unable to fetch USD-INR data"

        current_rate = usdinr['Close'].iloc[-1]
        rate_30d_ago = usdinr['Close'].iloc[-22] if len(usdinr) > 22 else usdinr['Close'].iloc[0]
        rate_90d_ago = usdinr['Close'].iloc[-66] if len(usdinr) > 66 else usdinr['Close'].iloc[0]

        change_30d = ((current_rate - rate_30d_ago) / rate_30d_ago) * 100
        change_90d = ((current_rate - rate_90d_ago) / rate_90d_ago) * 100

        # 52-week range
        high_52w = usdinr['Close'].max()
        low_52w = usdinr['Close'].min()

        # Volatility
        daily_returns = usdinr['Close'].pct_change()
        volatility = daily_returns.std() * (252 ** 0.5) * 100

        # Trend assessment
        if change_30d > 1:
            trend = "DEPRECIATING (INR weakening)"
            exporter_impact = "POSITIVE for IT, Pharma exporters"
            importer_impact = "NEGATIVE for Oil, Capital goods importers"
        elif change_30d < -1:
            trend = "APPRECIATING (INR strengthening)"
            exporter_impact = "NEGATIVE for IT, Pharma exporters"
            importer_impact = "POSITIVE for Oil, Capital goods importers"
        else:
            trend = "STABLE"
            exporter_impact = "NEUTRAL"
            importer_impact = "NEUTRAL"

        report = f"""
USD-INR EXCHANGE RATE ANALYSIS
{'='*50}

CURRENT STATUS:
- USD-INR Rate: {current_rate:.2f}
- 30-Day Change: {change_30d:+.2f}%
- 90-Day Change: {change_90d:+.2f}%
- 52-Week Range: {low_52w:.2f} - {high_52w:.2f}
- Volatility (Annualized): {volatility:.1f}%

TREND: {trend}

SECTOR IMPACT:
- IT Services (TCS, Infosys, Wipro): {exporter_impact}
  Every 1% INR depreciation = ~0.3-0.4% EBITDA margin boost
- Pharma Exporters (Sun, Dr Reddy's): {exporter_impact}
- Oil & Gas Importers (IOC, BPCL): {importer_impact}
- Capital Goods (imports): {importer_impact}

TRADING IMPLICATIONS:
"""

        if change_30d > 2:
            report += "- Strong tailwind for IT stocks - consider overweighting\n"
            report += "- Headwind for OMCs - be cautious on oil marketing companies\n"
        elif change_30d < -2:
            report += "- Headwind for IT stocks - margins under pressure\n"
            report += "- Positive for oil importers - input costs lower\n"
        else:
            report += "- Currency neutral - focus on company-specific factors\n"

        report += f"""
RBI CONTEXT:
- RBI typically intervenes at extremes (>84 or <82 levels)
- Watch RBI forex reserves and intervention patterns
- Global dollar strength (DXY) drives INR moves
"""

        return report

    except Exception as e:
        return f"Error fetching USD-INR data: {str(e)}"
