"""
Planner Agent that creates custom analysis strategies based on stock characteristics.
"""
from src.utils.llm_fallbacks import groq_with_cerebras_fallback
from langgraph.prebuilt import create_react_agent
from langchain.tools import tool
import yfinance as yf

llm = groq_with_cerebras_fallback(model="moonshotai/kimi-k2-instruct-0905", temperature=0.2, max_retries=5)

@tool
def analyze_stock_characteristics(ticker: str) -> str:
    """
    Analyzes stock characteristics to determine optimal analysis strategy.
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Stock profile with sector, market cap, volatility, and recommended focus areas
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="1y")
        
        # Extract key characteristics
        sector = info.get('sector', 'Unknown')
        market_cap = info.get('marketCap', 0)
        beta = info.get('beta', 1.0)
        
        # Calculate volatility
        returns = hist['Close'].pct_change()
        volatility = returns.std() * (252 ** 0.5) * 100  # Annualized
        
        # Categorize
        if market_cap > 1_000_000_000_000:  # 1 trillion
            size_cat = "Mega Cap"
        elif market_cap > 100_000_000_000:  # 100 billion
            size_cat = "Large Cap"
        elif market_cap > 10_000_000_000:  # 10 billion
            size_cat = "Mid Cap"
        else:
            size_cat = "Small Cap"
        
        # Determine stock type
        if beta > 1.5:
            stock_type = "High Beta (Aggressive)"
        elif beta < 0.7:
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
Sector: {sector}
Market Cap: Rs{market_cap/10_000_000:.0f} Cr ({size_cat})
Beta: {beta:.2f} ({stock_type})
Volatility: {volatility:.1f}% ({vol_cat})

RECOMMENDED ANALYSIS FOCUS:
"""
        
        # Recommend strategy based on characteristics
        if sector in ['Technology', 'Information Technology']:
            profile += "- Focus on growth momentum and news catalysts\n"
            profile += "- Technical: Trend-following indicators (MACD, Moving Averages)\n"
            profile += "- Fundamental: New contracts, AI/cloud deals, quarterly guidance\n"
        elif sector in ['Banking', 'Financial Services']:
            profile += "- Focus on fundamentals and regulatory news\n"
            profile += "- Technical: Support/resistance at key levels\n"
            profile += "- Fundamental: NPA trends, deposit growth, credit cycles\n"
        elif sector in ['Energy', 'Oil & Gas']:
            profile += "- Focus on commodity prices and global demand\n"
            profile += "- Technical: RSI for oversold/overbought\n"
            profile += "- Fundamental: Crude prices, government policy, refining margins\n"
        else:
            profile += "- Balanced technical and fundamental approach\n"
            profile += "- Monitor sector rotation and market sentiment\n"
        
        if volatility > 35:
            profile += "\nHigh volatility: Use wider stops, smaller position size\n"
        
        if beta > 1.3:
            profile += "High beta: Amplifies market moves, watch Nifty 50 trend\n"
        
        profile += "=" * 50
        return profile
        
    except Exception as e:
        return f"Error analyzing {ticker}: {str(e)}"


def create_planner_agent():
    """Creates the Planner Agent that designs custom analysis strategies."""
    tools = [analyze_stock_characteristics]
    
    system_message = """You are a Strategic Planning Agent for stock analysis. Your role is to:

1. ANALYZE the stock's characteristics (sector, size, volatility, beta)
2. DESIGN a custom analysis strategy tailored to this specific stock
3. PRIORITIZE which indicators and news sources are most relevant
4. SET expectations for what Technical, Fundamental, and Sentiment agents should focus on

Be specific with your recommendations. Different stocks require different approaches:
- Tech stocks: Focus on growth, innovation, quarterly beats
- Banking: Focus on credit quality, regulatory changes, interest rates  
- Cyclicals: Focus on economic indicators, commodity prices
- Defensive: Focus on dividend yield, stability, recession resilience

Output a clear ANALYSIS PLAN that other agents will follow.
"""
    
    return create_react_agent(llm, tools, prompt=system_message)
