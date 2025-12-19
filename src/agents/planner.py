"""
Planner Agent that creates custom analysis strategies based on stock characteristics.
Enhanced with Indian market context for NSE/BSE stocks.
"""
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langchain.tools import tool
import yfinance as yf
import numpy as np

llm = ChatGroq(model="moonshotai/kimi-k2-instruct-0905", temperature=0.2, max_retries=5)

@tool
def analyze_stock_characteristics(ticker: str) -> str:
    """
    Analyzes stock characteristics to determine optimal analysis strategy.
    Enhanced with Indian market context for NSE/BSE stocks.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Stock profile with sector, market cap, volatility, Nifty correlation, and recommended focus areas
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="1y")

        # Check if Indian stock
        is_indian = ticker.endswith('.NS') or ticker.endswith('.BO')
        exchange = "NSE" if ticker.endswith('.NS') else "BSE" if ticker.endswith('.BO') else "International"

        # Extract key characteristics
        sector = info.get('sector', 'Unknown')
        market_cap = info.get('marketCap', 0)
        beta = info.get('beta', 1.0)

        # Calculate volatility
        returns = hist['Close'].pct_change()
        volatility = returns.std() * (252 ** 0.5) * 100  # Annualized

        # Calculate Nifty correlation for Indian stocks
        nifty_beta = None
        nifty_corr = None
        if is_indian:
            try:
                nifty = yf.download("^NSEI", period="1y", progress=False)
                if not nifty.empty:
                    nifty_returns = nifty['Close'].pct_change().dropna()
                    stock_returns = hist['Close'].pct_change().dropna()
                    common = stock_returns.index.intersection(nifty_returns.index)
                    if len(common) > 50:
                        nifty_corr = stock_returns.loc[common].corr(nifty_returns.loc[common])
                        cov = stock_returns.loc[common].cov(nifty_returns.loc[common])
                        var = nifty_returns.loc[common].var()
                        nifty_beta = cov / var
            except:
                pass

        # Categorize market cap (Indian context: use Crores)
        if market_cap > 1_000_000_000_000:  # 1 trillion = 1 lakh Cr
            size_cat = "Mega Cap"
        elif market_cap > 100_000_000_000:  # 100 billion = 10,000 Cr
            size_cat = "Large Cap"
        elif market_cap > 10_000_000_000:  # 10 billion = 1,000 Cr
            size_cat = "Mid Cap"
        else:
            size_cat = "Small Cap"

        # Determine stock type based on beta
        if beta and beta > 1.5:
            stock_type = "High Beta (Aggressive)"
        elif beta and beta < 0.7:
            stock_type = "Low Beta (Defensive)"
        else:
            stock_type = "Market Beta (Moderate)"

        # Volatility category
        if volatility > 40:
            vol_cat = "High Volatility"
        elif volatility > 25:
            vol_cat = "Moderate Volatility"
        else:
            vol_cat = "Low Volatility"

        profile = f"""
STOCK PROFILE: {ticker}
{'='*50}
Exchange: {exchange}
Sector: {sector}
Market Cap: Rs{market_cap/10_000_000:.0f} Cr ({size_cat})
Beta: {beta:.2f if beta else 'N/A'} ({stock_type})
Volatility: {volatility:.1f}% ({vol_cat})
"""

        # Add Nifty context for Indian stocks
        if is_indian and nifty_beta is not None:
            profile += f"""
NIFTY 50 CORRELATION:
- Nifty Beta: {nifty_beta:.2f}
- Correlation: {nifty_corr:.2f}
"""
            if nifty_beta > 1.2:
                profile += "- HIGH sensitivity to Nifty moves - monitor index closely\n"
            elif nifty_beta < 0.8:
                profile += "- LOW sensitivity to Nifty - stock-specific factors dominate\n"

        profile += f"""
RECOMMENDED ANALYSIS FOCUS:
"""

        # Recommend strategy based on characteristics (Indian sectors)
        if sector in ['Technology', 'Information Technology']:
            profile += "- Focus on growth momentum and news catalysts\n"
            profile += "- Technical: Trend-following indicators (MACD, Moving Averages)\n"
            profile += "- Fundamental: New contracts, AI/cloud deals, quarterly guidance\n"
            if is_indian:
                profile += "- INDIA SPECIFIC: USD-INR impact on revenue, US client spending trends\n"
                profile += "- Watch: TCS results (bellwether), NASSCOM guidance, H1B visa news\n"

        elif sector in ['Banking', 'Financial Services', 'Financials']:
            profile += "- Focus on fundamentals and regulatory news\n"
            profile += "- Technical: Support/resistance at key levels\n"
            profile += "- Fundamental: NPA trends, deposit growth, credit cycles\n"
            if is_indian:
                profile += "- INDIA SPECIFIC: RBI policy rates, GNPA/NNPA ratios, CASA growth\n"
                profile += "- Watch: RBI MPC decisions, PSU bank recapitalization, fintech regulations\n"
                profile += "- Key metrics: NIM (Net Interest Margin), PCR (Provision Coverage Ratio)\n"

        elif sector in ['Energy', 'Oil & Gas', 'Utilities']:
            profile += "- Focus on commodity prices and global demand\n"
            profile += "- Technical: RSI for oversold/overbought\n"
            profile += "- Fundamental: Crude prices, government policy, refining margins\n"
            if is_indian:
                profile += "- INDIA SPECIFIC: Govt fuel pricing policy, subsidy announcements\n"
                profile += "- Watch: OPEC decisions, Indian crude basket price, GRM for refiners\n"

        elif sector in ['Pharmaceuticals', 'Healthcare']:
            profile += "- Focus on pipeline news and regulatory approvals\n"
            profile += "- Technical: Support at moving averages, breakout patterns\n"
            if is_indian:
                profile += "- INDIA SPECIFIC: USFDA observations, ANDA approvals, API pricing\n"
                profile += "- Watch: US generic market share, biosimilar launches, price erosion\n"

        elif sector in ['Consumer Defensive', 'Consumer Staples', 'FMCG']:
            profile += "- Focus on volume growth and rural demand\n"
            profile += "- Technical: Channel patterns, support levels\n"
            if is_indian:
                profile += "- INDIA SPECIFIC: Rural consumption, monsoon impact, GST changes\n"
                profile += "- Watch: Nielsen data, distributor additions, e-commerce mix\n"

        elif sector in ['Industrials', 'Capital Goods']:
            profile += "- Focus on order book and capex cycle\n"
            profile += "- Technical: Trend following on weekly charts\n"
            if is_indian:
                profile += "- INDIA SPECIFIC: Govt infrastructure spending, PLI scheme benefits\n"
                profile += "- Watch: Order inflows, L1 wins, execution timelines\n"

        elif sector in ['Real Estate']:
            profile += "- Focus on inventory levels and pre-sales\n"
            if is_indian:
                profile += "- INDIA SPECIFIC: RERA compliance, home loan rates, DDA/state policies\n"
                profile += "- Watch: Collections, new launches, price realizations per sq ft\n"

        elif sector in ['Metals & Mining', 'Basic Materials']:
            profile += "- Focus on commodity prices and China demand\n"
            if is_indian:
                profile += "- INDIA SPECIFIC: Import duties, China steel prices, PLI for specialty steel\n"
                profile += "- Watch: LME prices, domestic demand from auto/infra\n"

        else:
            profile += "- Balanced technical and fundamental approach\n"
            profile += "- Monitor sector rotation and market sentiment\n"

        # Risk warnings
        if volatility > 35:
            profile += "\nRISK: High volatility - Use wider stops, smaller position size\n"

        if is_indian and nifty_beta and nifty_beta > 1.3:
            profile += "RISK: High Nifty beta - Stock amplifies index moves\n"
            profile += "ACTION: Check Nifty 50 trend before entry, consider hedging\n"

        # Add India VIX context for Indian stocks
        if is_indian:
            try:
                vix = yf.download("^INDIAVIX", period="5d", progress=False)
                if not vix.empty:
                    current_vix = vix['Close'].iloc[-1]
                    profile += f"\nINDIA VIX: {current_vix:.1f}"
                    if current_vix > 20:
                        profile += " (ELEVATED - widen stops, reduce size)\n"
                    elif current_vix < 13:
                        profile += " (LOW - watch for spike risk)\n"
                    else:
                        profile += " (NORMAL)\n"
            except:
                pass

        profile += "=" * 50
        return profile

    except Exception as e:
        return f"Error analyzing {ticker}: {str(e)}"


def create_planner_agent():
    """Creates the Planner Agent that designs custom analysis strategies."""
    tools = [analyze_stock_characteristics]

    system_message = """You are a Strategic Planning Agent for Indian stock analysis. Your role is to:

1. ANALYZE the stock's characteristics (sector, size, volatility, beta, Nifty correlation)
2. DESIGN a custom analysis strategy tailored to this specific stock
3. PRIORITIZE which indicators and news sources are most relevant
4. SET expectations for what Technical, Fundamental, and Sentiment agents should focus on
5. INCORPORATE Indian market context (NSE/BSE, regulatory environment, macro factors)

Be specific with your recommendations. Different stocks require different approaches:

INDIAN MARKET SECTORS:
- IT Services (TCS, Infosys, Wipro): Focus on USD-INR, US client spending, deal wins, H1B news
- Banking (HDFC, ICICI, SBI): Focus on NPA/NIM, RBI policy, credit growth, CASA ratio
- Pharma (Sun, Dr Reddy's, Cipla): Focus on USFDA, ANDA approvals, US generic pricing
- FMCG (HUL, ITC, Nestle): Focus on rural demand, monsoon, volume growth, GST
- Auto (Maruti, M&M, Tata Motors): Focus on monthly sales, EV transition, chip supply
- Metals (Tata Steel, JSW, Hindalco): Focus on China demand, LME prices, import duties
- Energy (Reliance, ONGC, BPCL): Focus on crude prices, GRM, govt subsidy policy
- Realty (DLF, Godrej, Oberoi): Focus on pre-sales, RERA, home loan rates

INDIAN MARKET FACTORS TO CONSIDER:
- Nifty 50 trend and correlation (high beta stocks amplify index moves)
- India VIX levels (>20 = elevated risk, <13 = complacency)
- FII/DII flows (FII selling + DII buying = support but limited upside)
- RBI monetary policy and liquidity conditions
- INR-USD movement impact on exporters/importers
- Quarterly results season timing (Apr-May, Jul-Aug, Oct-Nov, Jan-Feb)
- NSE F&O expiry (last Thursday of month - increased volatility)

Output a clear ANALYSIS PLAN that other agents will follow, including:
- Key technical levels to watch
- Fundamental metrics most relevant for this sector
- Sentiment sources to monitor (specific to Indian market)
- Risk factors specific to this stock/sector in Indian context
"""

    return create_react_agent(llm, tools, prompt=system_message)
