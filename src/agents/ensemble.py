"""
Multi-strategy ensemble that runs multiple analysis approaches and votes.
Enhanced with regime-aware dynamic weighting.
"""
from src.utils.llm_fallbacks import groq_with_cerebras_fallback
from langgraph.prebuilt import create_react_agent
from langchain.tools import tool
import yfinance as yf
import pandas as pd
import time
from src.utils.logging_config import AgentLogger
from src.middleware.retry_handler import retry_yfinance

llm = groq_with_cerebras_fallback(model="qwen/qwen3-32b", temperature=0, max_retries=5)
logger = AgentLogger("ensemble")

@tool
def momentum_strategy_signal(ticker: str) -> str:
    """
    Momentum breakout strategy - looks for strong trends and breakouts.
    
    Returns: BUY/SELL/HOLD with reasoning
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="6mo")
        
        if df.empty:
            return f"No data for {ticker}"
        
        # Calculate momentum indicators
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['RSI'] = 100 - (100 / (1 + df['Close'].diff().apply(lambda x: x if x > 0 else 0).rolling(14).mean() / 
                                          df['Close'].diff().apply(lambda x: -x if x < 0 else 0).rolling(14).mean()))
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Momentum signals
        golden_cross = latest['SMA_20'] > latest['SMA_50'] and prev['SMA_20'] <= prev['SMA_50']
        price_above_sma = latest['Close'] > latest['SMA_20'] > latest['SMA_50']
        rsi_bullish = 50 < latest['RSI'] < 70
        
        if golden_cross or (price_above_sma and rsi_bullish):
            signal = "BUY"
            reason = "Momentum: Golden cross or strong uptrend"
        elif latest['Close'] < latest['SMA_50']:
            signal = "SELL"
            reason = "Momentum: Below 50-SMA, downtrend"
        else:
            signal = "HOLD"
            reason = "Momentum: Neutral, waiting for clear trend"
        
        return f"{signal} - {reason} (RSI: {latest['RSI']:.1f})"
        
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def mean_reversion_strategy_signal(ticker: str) -> str:
    """
    Mean reversion strategy - looks for oversold/overbought for reversal.
    
    Returns: BUY/SELL/HOLD with reasoning
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="3mo")
        
        if df.empty:
            return f"No data for {ticker}"
        
        # Bollinger Bands
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['BB_upper'] = df['SMA_20'] + 2 * df['Close'].rolling(20).std()
        df['BB_lower'] = df['SMA_20'] - 2 * df['Close'].rolling(20).std()
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        latest = df.iloc[-1]
        
        # Mean reversion signals
        oversold = latest['Close'] < latest['BB_lower'] and latest['RSI'] < 30
        overbought = latest['Close'] > latest['BB_upper'] and latest['RSI'] > 70
        
        if oversold:
            signal = "BUY"
            reason = "Mean Reversion: Oversold, expect bounce"
        elif overbought:
            signal = "SELL"
            reason = "Mean Reversion: Overbought, expect pullback"
        else:
            signal = "HOLD"
            reason = "Mean Reversion: Price near mean, no clear signal"
        
        return f"{signal} - {reason} (RSI: {latest['RSI']:.1f}, Price vs BB: {((latest['Close'] - latest['SMA_20']) / latest['SMA_20'] * 100):.1f}%)"
        
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def trend_following_strategy_signal(ticker: str) -> str:
    """
    Trend following strategy - rides established trends.
    
    Returns: BUY/SELL/HOLD with reasoning
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="1y")
        
        if df.empty:
            return f"No data for {ticker}"
        
        # Multiple timeframe EMAs
        df['EMA_20'] = df['Close'].ewm(span=20).mean()
        df['EMA_50'] = df['Close'].ewm(span=50).mean()
        df['EMA_200'] = df['Close'].ewm(span=200).mean()
        
        latest = df.iloc[-1]
        
        # Trend alignment
        strong_uptrend = latest['EMA_20'] > latest['EMA_50'] > latest['EMA_200']
        strong_downtrend = latest['EMA_20'] < latest['EMA_50'] < latest['EMA_200']
        
        if strong_uptrend and latest['Close'] > latest['EMA_20']:
            signal = "BUY"
            reason = "Trend Following: Strong uptrend, all EMAs aligned"
        elif strong_downtrend:
            signal = "SELL"
            reason = "Trend Following: Strong downtrend"
        else:
            signal = "HOLD"
            reason = "Trend Following: Mixed trend, wait for clarity"
        
        return f"{signal} - {reason}"
        
    except Exception as e:
        return f"Error: {str(e)}"


def create_ensemble_agent():
    """Creates an Ensemble Strategy Agent with regime-aware weighting."""
    # Import regime detection tool
    from src.tools.regime_detection import detect_market_regime

    tools = [
        momentum_strategy_signal,
        mean_reversion_strategy_signal,
        trend_following_strategy_signal,
        detect_market_regime  # Add regime detection
    ]

    system_message = """You are a Multi-Strategy Ensemble Agent with REGIME-AWARE VOTING. Your role is to:

1. FIRST, detect the current market regime using detect_market_regime tool
   - This tells you which strategies work best in current conditions

2. RUN all three trading strategies (Momentum, Mean Reversion, Trend Following)

3. APPLY WEIGHTED VOTING based on regime:
   - Use the strategy_weights provided by regime detection
   - Weight each strategy's signal by its recommended weight
   - Example: In STRONG_TRENDING regime:
     * Momentum vote weight: 45%
     * Trend Following vote weight: 45%
     * Mean Reversion vote weight: 10%

4. CALCULATE final recommendation:
   - Convert BUY/SELL/HOLD to scores: BUY=+1, HOLD=0, SELL=-1
   - Multiply each strategy's score by its weight
   - Sum weighted scores to get final signal
   - Final > 0.3 = BUY, Final < -0.3 = SELL, otherwise HOLD

5. REPORT the weighted consensus:
   - Show all three strategy signals
   - Show the regime and recommended weights
   - Show the weighted calculation
   - Explain the final recommendation

CRITICAL: Market regime determines which strategies to trust more.
- TRENDING markets: Trust momentum and trend-following
- RANGING markets: Trust mean reversion heavily
- VOLATILE markets: Reduce overall confidence, use balanced weights

This regime-aware approach reduces false signals by adapting to market conditions.
"""

    return create_react_agent(llm, tools, prompt=system_message)
