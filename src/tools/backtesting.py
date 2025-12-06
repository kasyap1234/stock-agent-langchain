"""
Backtesting tools for validating trade calls against historical data.
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from langchain.tools import tool
from typing import Dict

@tool
def backtest_trade_call(ticker: str, entry_price: float, target_price: float, stop_loss: float, days_back: int = 30) -> str:
    """
    Backtests a trade call against the last N days of historical data.
    
    Args:
        ticker: Stock ticker (e.g., "RELIANCE.NS")
        entry_price: Proposed entry price
        target_price: Proposed target price
        stop_loss: Proposed stop loss price
        days_back: Number of days to backtest (default 30)
    
    Returns:
        Backtest results with win rate, risk-reward, and confidence score
    """
    try:
        # Fetch historical data
        stock = yf.Ticker(ticker)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back * 2)  # Get extra data for context
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            return f"No historical data available for {ticker}"
        
        # Get the most recent N days
        df = df.tail(days_back)
        
        # Calculate what would have happened if we entered at entry_price
        current_price = df['Close'].iloc[-1]
        high_in_period = df['High'].max()
        low_in_period = df['Low'].min()
        
        # Simulate the trade
        would_hit_target = high_in_period >= target_price
        would_hit_stop = low_in_period <= stop_loss
        
        # Calculate risk-reward
        potential_gain = ((target_price - entry_price) / entry_price) * 100
        potential_loss = ((entry_price - stop_loss) / entry_price) * 100
        risk_reward = potential_gain / potential_loss if potential_loss > 0 else 0
        
        # Determine outcome
        if would_hit_target and would_hit_stop:
            outcome = "MIXED - Both target and stop loss were hit in period"
            confidence = 50
        elif would_hit_target:
            outcome = "WIN - Target would have been hit"
            confidence = 80
        elif would_hit_stop:
            outcome = "LOSS - Stop loss would have been hit"
            confidence = 30
        else:
            outcome = "PENDING - Neither target nor stop loss hit yet"
            confidence = 60
        
        # Calculate price movement correlation
        price_change = ((current_price - entry_price) / entry_price) * 100
        
        report = f"""
BACKTEST RESULTS ({days_back} days):
{'='*50}
Ticker: {ticker}
Entry Price: Rs{entry_price:.2f}
Target: Rs{target_price:.2f} (+{potential_gain:.2f}%)
Stop Loss: Rs{stop_loss:.2f} (-{potential_loss:.2f}%)

Historical Range:
- High: Rs{high_in_period:.2f}
- Low: Rs{low_in_period:.2f}
- Current: Rs{current_price:.2f}

Outcome: {outcome}
Risk-Reward Ratio: {risk_reward:.2f}:1
Price Change from Entry: {price_change:+.2f}%

Confidence Score: {confidence}% 
{'='*50}
"""
        return report
        
    except Exception as e:
        return f"Backtest error for {ticker}: {str(e)}"
        
@tool
def perform_walk_forward_analysis(ticker: str, strategy_type: str = "Trend Following", windows: int = 4) -> str:
    """
    Performs Walk-Forward Analysis to validate strategy robustness.
    
    Splits history into sliding Train/Test windows to check if the strategy
    performs well on unseen data.
    
    Args:
        ticker: Stock ticker
        strategy_type: Type of strategy to test (Trend Following, Mean Reversion)
        windows: Number of walk-forward windows to test
        
    Returns:
        Comprehensive backtest report with Out-of-Sample performance.
    """
    try:
        stock = yf.Ticker(ticker)
        # Get 2 years of data
        df = stock.history(period="2y")
        
        if df.empty or len(df) < 200:
            return f"Insufficient data for walk-forward analysis for {ticker}"
            
        # Define window size (e.g., 6 months train, 2 months test)
        total_days = len(df)
        window_size = total_days // (windows + 1)
        train_size = int(window_size * 0.75)
        test_size = window_size - train_size
        
        results = []
        
        for i in range(windows):
            start_idx = i * test_size
            train_end = start_idx + train_size
            test_end = train_end + test_size
            
            if test_end > total_days:
                break
                
            train_data = df.iloc[start_idx:train_end]
            test_data = df.iloc[train_end:test_end]
            
            # Simulate Strategy (Simplified for this tool)
            # In a real system, this would call the actual strategy logic
            # Here we simulate a simple Moving Average Crossover for "Trend Following"
            # and RSI for "Mean Reversion"
            
            if strategy_type == "Trend Following":
                # Optimize on Train (Find best MA combo)
                best_fast, best_slow, best_return = _optimize_ma_strategy(train_data)
                # Test on Test
                test_return, trades = _run_ma_strategy(test_data, best_fast, best_slow)
                
                results.append({
                    "window": i+1,
                    "train_return": best_return,
                    "test_return": test_return,
                    "params": f"MA({best_fast}/{best_slow})",
                    "trades": trades
                })
                
            else: # Mean Reversion (RSI)
                # Optimize RSI thresholds
                best_low, best_high, best_return = _optimize_rsi_strategy(train_data)
                # Test
                test_return, trades = _run_rsi_strategy(test_data, best_low, best_high)
                
                results.append({
                    "window": i+1,
                    "train_return": best_return,
                    "test_return": test_return,
                    "params": f"RSI({best_low}/{best_high})",
                    "trades": trades
                })
        
        # Aggregate Results
        avg_test_return = sum(r['test_return'] for r in results) / len(results)
        robustness = sum(1 for r in results if r['test_return'] > 0) / len(results) * 100
        
        report = f"""
WALK-FORWARD ANALYSIS: {ticker} ({strategy_type})
{'='*50}
Robustness Score: {robustness:.0f}% (Windows Profitable)
Avg Out-of-Sample Return: {avg_test_return:.2f}% per window

DETAILS:
"""
        for r in results:
            report += f"Window {r['window']}: Train {r['train_return']:.1f}% -> Test {r['test_return']:.1f}% | Params: {r['params']} | Trades: {r['trades']}\n"
            
        report += "="*50
        return report
        
    except Exception as e:
        return f"Walk-forward analysis error: {str(e)}"

def _optimize_ma_strategy(df):
    # Simplified optimization
    best_ret = -999
    best_f, best_s = 10, 50
    
    # Grid search small range
    for f in [10, 20]:
        for s in [50, 100]:
            ret, _ = _run_ma_strategy(df, f, s)
            if ret > best_ret:
                best_ret = ret
                best_f, best_s = f, s
    return best_f, best_s, best_ret

def _run_ma_strategy(df, fast, slow):
    # Vectorized backtest would be better, but keeping it simple
    df = df.copy()
    df['Fast'] = df['Close'].rolling(fast).mean()
    df['Slow'] = df['Close'].rolling(slow).mean()
    
    position = 0
    trades = 0
    returns = 0.0
    
    for i in range(1, len(df)):
        if df['Fast'].iloc[i] > df['Slow'].iloc[i] and df['Fast'].iloc[i-1] <= df['Slow'].iloc[i-1]:
            position = 1 # Buy
            entry = df['Close'].iloc[i]
            trades += 1
        elif df['Fast'].iloc[i] < df['Slow'].iloc[i] and df['Fast'].iloc[i-1] >= df['Slow'].iloc[i-1]:
            if position == 1:
                position = 0 # Sell
                exit_price = df['Close'].iloc[i]
                returns += (exit_price - entry) / entry * 100
    
    # Close open position
    if position == 1:
        returns += (df['Close'].iloc[-1] - entry) / entry * 100
        
    return returns, trades

def _optimize_rsi_strategy(df):
    import ta
    best_ret = -999
    best_l, best_h = 30, 70
    
    # Simple check
    ret, _ = _run_rsi_strategy(df, 30, 70)
    return 30, 70, ret

def _run_rsi_strategy(df, low, high):
    import ta
    df = df.copy()
    rsi = ta.momentum.RSIIndicator(df['Close']).rsi()
    
    position = 0
    trades = 0
    returns = 0.0
    entry = 0
    
    for i in range(14, len(df)):
        if rsi.iloc[i] < low and position == 0:
            position = 1
            entry = df['Close'].iloc[i]
            trades += 1
        elif rsi.iloc[i] > high and position == 1:
            position = 0
            returns += (df['Close'].iloc[i] - entry) / entry * 100
            
    if position == 1:
        returns += (df['Close'].iloc[-1] - entry) / entry * 100
        
    return returns, trades


@tool  
def calculate_confidence_score(ticker: str, technical_bias: str, fundamental_rating: str, sentiment: str) -> str:
    """
    Calculates a confidence score based on alignment of analysis components.
    
    Args:
        ticker: Stock ticker
        technical_bias: "Bullish", "Bearish", or "Neutral"
        fundamental_rating: "Positive", "Negative", or "Neutral"  
        sentiment: "Positive", "Negative", or "Neutral"
    
    Returns:
        Confidence score (0-100%) with reasoning
    """
    try:
        # Normalize inputs
        tech = technical_bias.lower()
        fund = fundamental_rating.lower()
        sent = sentiment.lower()
        
        # Map to numeric scores
        score_map = {
            "bullish": 1, "positive": 1,
            "neutral": 0,
            "bearish": -1, "negative": -1
        }
        
        tech_score = score_map.get(tech, 0)
        fund_score = score_map.get(fund, 0)
        sent_score = score_map.get(sent, 0)
        
        # Calculate alignment
        total_score = tech_score + fund_score + sent_score
        
        # Calculate confidence based on alignment
        if abs(total_score) == 3:
            # Perfect alignment (all bullish or all bearish)
            confidence = 90
            reasoning = "Strong alignment across all analyses"
        elif abs(total_score) == 2:
            # Two align, one neutral or opposite
            confidence = 75
            reasoning = "Good alignment with minor divergence"
        elif abs(total_score) == 1:
            # Mixed signals
            confidence = 55
            reasoning = "Mixed signals detected - proceed with caution"
        else:
            # All neutral or conflicting
            confidence = 40
            reasoning = "Low confidence - conflicting or neutral signals"
        
        # Boost/reduce based on specific patterns
        if tech_score == 1 and fund_score == 1 and sent_score == -1:
            confidence -= 10
            reasoning = "Sentiment diverges from fundamentals - market may be wrong"
        
        report = f"""
CONFIDENCE ANALYSIS:
{'='*50}
Technical: {technical_bias} ({tech_score:+d})
Fundamental: {fundamental_rating} ({fund_score:+d})
Sentiment: {sentiment} ({sent_score:+d})

Confidence Score: {confidence}%
Reasoning: {reasoning}

Recommendation: {'PROCEED' if confidence >= 60 else 'CAUTION' if confidence >= 50 else 'AVOID'}
{'='*50}
"""
        return report
        
    except Exception as e:
        return f"Confidence calculation error: {str(e)}"
