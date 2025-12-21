from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from src.tools.market_data import get_stock_history, calculate_indicators, multi_timeframe_analysis
from src.tools.search import web_search
from src.tools.news import get_stock_news, get_nse_announcements
from src.tools.sentiment_nlp import analyze_sentiment_nlp
from src.tools.social_media import get_social_sentiment_aggregate
from src.tools.enhanced_data import get_options_sentiment, get_insider_trading_summary
from src.utils.logging_config import AgentLogger
import os

# Initialize LLM for simpler analyst tasks
llm = ChatGroq(model="qwen/qwen3-32b", temperature=0, max_retries=5)

# Initialize loggers
tech_logger = AgentLogger("technical_analyst")
fund_logger = AgentLogger("fundamental_analyst")
sent_logger = AgentLogger("sentiment_analyst")

def create_technical_analyst():
    """Creates the Technical Analyst Agent with multi-timeframe analysis capability."""
    tools = [get_stock_history, calculate_indicators, multi_timeframe_analysis]
    system_message = (
        "You are a Technical Analyst for Indian Stocks with MULTI-TIMEFRAME capabilities. "
        "Your goal is to analyze price action and technical indicators to identify swing trading opportunities. "
        "\n\n"
        "CRITICAL: ALWAYS use multi_timeframe_analysis tool to check alignment across timeframes. "
        "This significantly improves accuracy by filtering false signals. "
        "\n\n"
        "Workflow:\n"
        "1. Use multi_timeframe_analysis to check Weekly/Daily/4H alignment\n"
        "2. Use calculate_indicators for detailed technical analysis\n"
        "3. Use get_stock_history only if you need raw data\n"
        "\n"
        "Provide a clear Bullish, Bearish, or Neutral bias with:\n"
        "- Entry price\n"
        "- Target levels (at least 2)\n"
        "- Stop Loss\n"
        "- Timeframe alignment status (from multi_timeframe_analysis)\n"
        "- Confidence adjustment based on alignment\n"
        "\n"
        "IMPORTANT: If timeframes are conflicting, LOWER your conviction and advise caution."
    )
    return create_react_agent(llm, tools, prompt=system_message)

from src.tools.fundamentals import get_fundamental_metrics, get_growth_metrics

def create_fundamental_analyst():
    """Creates the Fundamental/News Analyst Agent with enhanced news tools."""
    tools = [
        get_fundamental_metrics,
        get_growth_metrics,
        get_stock_news,        # Multi-source news (NSE, MoneyControl, ET, NewsAPI, Google)
        get_nse_announcements, # Official corporate filings
        web_search,            # Fallback for general search
    ]
    system_message = (
        "You are a Fundamental and News Analyst for Indian Stocks. "
        "Your goal is to evaluate the company's financial health and recent news. "
        "\n\n"
        "Workflow:\n"
        "1. Use get_fundamental_metrics to check valuation (P/E, P/B) and profitability (ROE, Margins).\n"
        "2. Use get_growth_metrics to check recent growth trends.\n"
        "3. Use get_stock_news to fetch comprehensive news from multiple Indian sources.\n"
        "4. Use get_nse_announcements for official corporate filings and disclosures.\n"
        "5. Use web_search only if you need additional context not found in other tools.\n"
        "\n"
        "NEWS SOURCES (Priority Order):\n"
        "- NSE Announcements: Official filings (highest reliability)\n"
        "- MoneyControl: Indian financial news\n"
        "- Economic Times: Market and sector news\n"
        "- NewsAPI/Google: Global coverage\n"
        "\n"
        "Identify any red flags (e.g., high debt, declining margins) or positive catalysts. "
        "Pay special attention to upcoming earnings, board meetings, and regulatory filings."
    )
    return create_react_agent(llm, tools, prompt=system_message)

def create_sentiment_analyst():
    """Creates the Sentiment Analyst Agent with hybrid NLP and social media capabilities."""
    tools = [
        analyze_sentiment_nlp,      # NLP-based sentiment (FinBERT/VADER)
        get_social_sentiment_aggregate,  # Twitter + Reddit combined
        get_options_sentiment,      # Put/Call ratio, IV analysis
        get_insider_trading_summary,  # Insider buy/sell signals
        web_search,                 # Fallback for additional context
    ]
    system_message = (
        "You are a Sentiment Analyst for Indian Stocks with HYBRID NLP and SOCIAL MEDIA capabilities. "
        "Your goal is to gauge the current market mood using quantitative and qualitative methods. "
        "\n\n"
        "WORKFLOW:\n"
        "1. Use analyze_sentiment_nlp to get QUANTITATIVE sentiment scores from news\n"
        "   - Uses FinBERT (financial NLP) for accurate scoring (-1 to +1)\n"
        "   - Provides confidence levels and article breakdown\n"
        "2. Use get_social_sentiment_aggregate for Twitter and Reddit sentiment\n"
        "   - Captures retail trader sentiment from r/IndianStreetBets, r/IndiaInvestments\n"
        "   - Identifies trending stocks and social buzz\n"
        "3. Use get_options_sentiment to check derivatives market sentiment\n"
        "   - Put/Call ratio signals institutional positioning\n"
        "   - High IV indicates expected volatility\n"
        "4. Use get_insider_trading_summary for insider activity signals\n"
        "   - Insider buying is bullish; heavy selling is cautionary\n"
        "5. Use web_search only if additional context is needed\n"
        "\n"
        "OUTPUT FORMAT:\n"
        "SENTIMENT ANALYSIS: [TICKER]\n"
        "===========================\n"
        "QUANTITATIVE SCORES:\n"
        "- NLP Sentiment Score: [X.XX] (-1 to +1)\n"
        "- Social Sentiment: [Bullish/Bearish/Neutral]\n"
        "- Options Sentiment: [Description]\n"
        "- Insider Activity: [Summary]\n"
        "\n"
        "OVERALL SENTIMENT: [POSITIVE/NEGATIVE/NEUTRAL]\n"
        "CONFIDENCE: [XX]%\n"
        "\n"
        "REASONING:\n"
        "[Brief synthesis of all signals]\n"
        "\n"
        "IMPORTANT:\n"
        "- Social sentiment at extremes can be a CONTRARIAN indicator\n"
        "- Sudden spikes in mentions warrant extra scrutiny\n"
        "- Options sentiment often leads price action"
    )
    return create_react_agent(llm, tools, prompt=system_message)
