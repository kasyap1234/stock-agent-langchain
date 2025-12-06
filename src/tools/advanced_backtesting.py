"""
Advanced backtesting framework with detailed metrics and visualization.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from langchain.tools import tool
from typing import Dict, List

@tool
def advanced_backtest(ticker: str, strategy: str = "swing", days_back: int = 90) -> str:
    """
    Runs comprehensive backtest with detailed performance metrics.
    
    Args:
        ticker: Stock ticker
        strategy: "swing", "momentum", "reversion"
        days_back: Historical period to test
    
    Returns:
        Detailed backtest report with Sharpe ratio, max drawdown, win rate, etc.
    """
    try:
        stock = yf.Ticker(ticker)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back * 2)
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            return f"No data for {ticker}"
        
        # Calculate indicators
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['RSI'] = calculate_rsi(df['Close'], 14)
        
        # Simulate trades based on strategy
        trades = []
        position = None
        
        for i in range(50, len(df)):
            if strategy == "momentum":
                # Golden cross strategy
                if df['SMA_20'].iloc[i] > df['SMA_50'].iloc[i] and df['SMA_20'].iloc[i-1] <= df['SMA_50'].iloc[i-1]:
                    if position is None:
                        position = {
                            'entry_price': df['Close'].iloc[i],
                            'entry_date': df.index[i],
                            'type': 'LONG'
                        }
                elif position and df['SMA_20'].iloc[i] < df['SMA_50'].iloc[i]:
                    # Exit
                    exit_price = df['Close'].iloc[i]
                    pnl = ((exit_price - position['entry_price']) / position['entry_price']) * 100
                    trades.append({
                        'entry': position['entry_price'],
                        'exit': exit_price,
                        'pnl': pnl,
                        'days': (df.index[i] - position['entry_date']).days
                    })
                    position = None
        
        # Close open position
        if position:
            exit_price = df['Close'].iloc[-1]
            pnl = ((exit_price - position['entry_price']) / position['entry_price']) * 100
            trades.append({
                'entry': position['entry_price'],
                'exit': exit_price,
                'pnl': pnl,
                'days': (df.index[-1] - position['entry_date']).days
            })
        
        if not trades:
            return "No trades generated in backtest period"
        
        # Calculate metrics
        total_trades = len(trades)
        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] <= 0]
        
        win_rate = (len(wins) / total_trades) * 100
        avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
        avg_loss = np.mean([t['pnl'] for t in losses]) if losses else 0
        
        total_return = sum([t['pnl'] for t in trades])
        avg_holding_days = np.mean([t['days'] for t in trades])
        
        # Sharpe ratio (simplified)
        returns = [t['pnl'] for t in trades]
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252 / avg_holding_days) if np.std(returns) > 0 else 0
        
        # Max drawdown
        cumulative = np.cumsum([t['pnl'] for t in trades])
        max_dd = np.min(cumulative - np.maximum.accumulate(cumulative))
        
        report = f"""
ADVANCED BACKTEST: {ticker} ({strategy.upper()} Strategy)
{'='*60}
Period: {days_back} days | Total Trades: {total_trades}

PERFORMANCE METRICS:
Win Rate: {win_rate:.1f}% ({len(wins)}W / {len(losses)}L)
Avg Win: +{avg_win:.2f}%
Avg Loss: {avg_loss:.2f}%
Profit Factor: {abs(avg_win / avg_loss) if avg_loss != 0 else 0:.2f}

Total Return: {total_return:.2f}%
Sharpe Ratio: {sharpe:.2f}
Max Drawdown: {max_dd:.2f}%
Avg Holding Period: {avg_holding_days:.1f} days

RISK ASSESSMENT:
"""
        if sharpe > 1.5:
            report += "Excellent risk-adjusted returns\n"
        elif sharpe > 1.0:
            report += "Good risk-adjusted returns\n"
        elif sharpe > 0.5:
            report += "Moderate risk-adjusted returns\n"
        else:
            report += "Poor risk-adjusted returns\n"
        
        if max_dd < -15:
            report += "High drawdown risk - use small position size\n"
        elif max_dd < -10:
            report += "Moderate drawdown - manage risk carefully\n"
        else:
            report += "Acceptable drawdown levels\n"
        
        report += f"\n{'='*60}"
        return report
        
    except Exception as e:
        return f"Backtest error: {str(e)}"


def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


@tool
def monte_carlo_simulation(ticker: str, days_ahead: int = 30, simulations: int = 1000, scenario: str = "normal") -> str:
    """
    Runs Monte Carlo simulation to estimate potential outcomes.
    
    Args:
        ticker: Stock ticker
        days_ahead: Days to simulate
        simulations: Number of simulations to run
        scenario: "normal", "bull", "bear", or "fat_tail"
    
    Returns:
        Probability distribution of returns
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="1y")
        
        if df.empty:
            return f"No data for {ticker}"
        
        # Calculate daily returns statistics
        returns = df['Close'].pct_change().dropna()
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Adjust parameters based on scenario
        if scenario == "bull":
            mean_return += (std_return * 0.1) # Slight drift up
        elif scenario == "bear":
            mean_return -= (std_return * 0.1) # Slight drift down
        
        current_price = df['Close'].iloc[-1]
        
        # Run simulations
        final_prices = []
        for _ in range(simulations):
            price = current_price
            for _ in range(days_ahead):
                if scenario == "fat_tail":
                    # Use Student's t-distribution (df=3) for fatter tails
                    daily_return = mean_return + np.random.standard_t(3) * std_return
                else:
                    daily_return = np.random.normal(mean_return, std_return)
                price *= (1 + daily_return)
            final_prices.append(price)
        
        final_prices = np.array(final_prices)
        
        # Calculate percentiles
        p10 = np.percentile(final_prices, 10)
        p50 = np.percentile(final_prices, 50)
        p90 = np.percentile(final_prices, 90)
        
        # Expected returns
        ret_10 = ((p10 - current_price) / current_price) * 100
        ret_50 = ((p50 - current_price) / current_price) * 100
        ret_90 = ((p90 - current_price) / current_price) * 100
        
        report = f"""
MONTE CARLO SIMULATION: {ticker} ({scenario.upper()} Scenario)
{'='*60}
Simulations: {simulations} | Forecast: {days_ahead} days
Current Price: Rs{current_price:.2f}

EXPECTED OUTCOMES:
90th Percentile (Optimistic): Rs{p90:.2f} ({ret_90:+.1f}%)
50th Percentile (Median): Rs{p50:.2f} ({ret_50:+.1f}%)
10th Percentile (Pessimistic): Rs{p10:.2f} ({ret_10:+.1f}%)

PROBABILITY ANALYSIS:
Chance of >5% gain: {(final_prices > current_price * 1.05).mean() * 100:.1f}%
Chance of >10% gain: {(final_prices > current_price * 1.10).mean() * 100:.1f}%
Chance of <5% loss: {(final_prices < current_price * 0.95).mean() * 100:.1f}%

TRADING IMPLICATIONS:
"""
        if ret_50 > 5:
            report += "Positive expected value - favorable setup\n"
        elif ret_50 > 0:
            report += "Slightly positive - marginal trade\n"
        else:
            report += "Negative expected value - avoid\n"
        
        risk_reward = abs(ret_90 - ret_50) / abs(ret_50 - ret_10) if ret_10 != ret_50 else 0
        report += f"Risk-Reward Asymmetry: {risk_reward:.2f}\n"
        
        report += f"{'='*60}"
        return report
        
    except Exception as e:
        return f"Monte Carlo error: {str(e)}"
