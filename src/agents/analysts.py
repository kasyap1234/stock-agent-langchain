from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from src.tools.market_data import get_stock_history, calculate_indicators, multi_timeframe_analysis
from src.tools.search import web_search
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
    """Creates the Fundamental/News Analyst Agent."""
    tools = [web_search, get_fundamental_metrics, get_growth_metrics]
    system_message = (
        "You are a Fundamental and News Analyst for Indian Stocks. "
        "Your goal is to evaluate the company's financial health and recent news. "
        "\n\n"
        "Workflow:\n"
        "1. Use get_fundamental_metrics to check valuation (P/E, P/B) and profitability (ROE, Margins).\n"
        "2. Use get_growth_metrics to check recent growth trends.\n"
        "3. Use web_search to find upcoming earnings dates, major announcements, and sector news.\n"
        "\n"
        "Identify any red flags (e.g., high debt, declining margins) or positive catalysts. "
        "Ignore long-term valuation metrics unless they are relevant to immediate price action."
    )
    return create_react_agent(llm, tools, prompt=system_message)

def create_sentiment_analyst():
    """Creates the Sentiment Analyst Agent."""
    tools = [web_search]
    system_message = (
        "You are a Sentiment Analyst. "
        "Your goal is to gauge the current market mood for a specific stock. "
        "Use the web_search tool to find recent articles, forum discussions, or analyst ratings. "
        "Classify sentiment as Positive, Negative, or Neutral."
    )
    return create_react_agent(llm, tools, prompt=system_message)
