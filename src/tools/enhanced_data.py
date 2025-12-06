"""
Enhanced data sources beyond basic yfinance - social sentiment, options data, etc.
"""
from langchain.tools import tool
import yfinance as yf
from datetime import datetime, timedelta
import json

@tool
def get_options_sentiment(ticker: str) -> str:
    """
    Analyzes options data for sentiment signals (put/call ratio, implied volatility).
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Options sentiment analysis with put/call ratio and IV trends
    """
    try:
        stock = yf.Ticker(ticker)
        
        # Get options dates
        dates = stock.options
        if not dates or len(dates) == 0:
            return f"No options data available for {ticker}"
        
        # Get nearest expiry
        nearest_expiry = dates[0]
        opt = stock.option_chain(nearest_expiry)
        
        calls = opt.calls
        puts = opt.puts
        
        # Calculate put/call ratio by volume
        call_volume = calls['volume'].sum()
        put_volume = puts['volume'].sum()
        pc_ratio = put_volume / call_volume if call_volume > 0 else 0
        
        # Get at-the-money implied volatility
        current_price = stock.history(period="1d")['Close'].iloc[-1]
        
        # Find ATM options
        calls_atm = calls[abs(calls['strike'] - current_price) < current_price * 0.05]
        if not calls_atm.empty:
            atm_iv = calls_atm['impliedVolatility'].mean() * 100
        else:
            atm_iv = 0
        
        # Interpret signals
        if pc_ratio > 1.2:
            sentiment = "Bearish (High put buying)"
        elif pc_ratio < 0.7:
            sentiment = "Bullish (High call buying)"
        else:
            sentiment = "Neutral"
        
        if atm_iv > 40:
            iv_signal = "High IV - Expect volatility"
        elif atm_iv > 25:
            iv_signal = "Moderate IV"
        else:
            iv_signal = "Low IV - Calm market"
        
        report = f"""
OPTIONS SENTIMENT: {ticker} (Expiry: {nearest_expiry})
{'='*50}
Put/Call Ratio: {pc_ratio:.2f} - {sentiment}
Implied Volatility (ATM): {atm_iv:.1f}% - {iv_signal}

Call Volume: {call_volume:,.0f}
Put Volume: {put_volume:,.0f}

Trading Signal:
- P/C > 1.2 = Bearish positioning
- P/C < 0.7 = Bullish positioning
- High IV = Uncertainty, potential big move expected
{'='*50}
"""
        return report
        
    except Exception as e:
        return f"Options analysis error for {ticker}: {str(e)}"


@tool
def get_insider_trading_summary(ticker: str) -> str:
    """
    Checks for recent insider trading activity.
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Summary of insider buying/selling activity
    """
    try:
        stock = yf.Ticker(ticker)
        
        # Get insider transactions
        insiders = stock.insider_transactions
        
        if insiders is None or insiders.empty:
            return f"No recent insider trading data for {ticker}"
        
        # Analyze last 6 months
        six_months_ago = datetime.now() - timedelta(days=180)
        recent = insiders[insiders.index > six_months_ago] if hasattr(insiders.index, 'date') else insiders.head(20)
        
        if recent.empty:
            return f"No recent insider transactions for {ticker}"
        
        # Count buys vs sells
        buys = recent[recent['Shares'] > 0] if 'Shares' in recent.columns else recent[recent['Transaction'] == 'Buy']
        sells = recent[recent['Shares'] < 0] if 'Shares' in recent.columns else recent[recent['Transaction'] == 'Sale']
        
        buy_count = len(buys)
        sell_count = len(sells)
        
        if buy_count > sell_count * 2:
            signal = "Bullish - Heavy insider buying"
        elif sell_count > buy_count * 2:
            signal = "Bearish - Heavy insider selling"
        else:
            signal = "Neutral - Mixed activity"
        
        report = f"""
INSIDER TRADING: {ticker} (Last 6 months)
{'='*50}
Insider Buys: {buy_count}
Insider Sells: {sell_count}

Signal: {signal}

Note: Insider buying often signals confidence
Insider selling can be for various reasons (diversification, taxes)
{'='*50}
"""
        return report
        
    except Exception as e:
        return f"Insider trading analysis error for {ticker}: {str(e)}"


@tool  
def get_institutional_holdings_change(ticker: str) -> str:
    """
    Analyzes changes in institutional holdings (mutual funds, FIIs).
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Institutional ownership trends
    """
    try:
        stock = yf.Ticker(ticker)
        
        # Get institutional holders
        inst_holders = stock.institutional_holders
        
        if inst_holders is None or inst_holders.empty:
            return f"No institutional holdings data for {ticker}"
        
        # Get major holders summary
        major_holders = stock.major_holders
        
        if major_holders is not None and not major_holders.empty:
            inst_ownership = major_holders.iloc[0, 0] if len(major_holders) > 0 else "N/A"
        else:
            inst_ownership = "N/A"
        
        # Count top holders
        top_count = len(inst_holders)
        
        report = f"""
INSTITUTIONAL HOLDINGS: {ticker}
{'='*50}
Institutional Ownership: {inst_ownership}
Number of Major Institutional Holders: {top_count}

Top 5 Holders:
"""
        for idx, row in inst_holders.head(5).iterrows():
            holder = row.get('Holder', 'Unknown')
            shares = row.get('Shares', 0)
            report += f"  - {holder}: {shares:,.0f} shares\n"
        
        report += f"\n{'='*50}"
        return report
        
    except Exception as e:
        return f"Institutional holdings error for {ticker}: {str(e)}"
