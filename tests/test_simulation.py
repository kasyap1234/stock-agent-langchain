import sys
import os
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock ChatGroq globally before importing workflow
with patch('langchain_groq.ChatGroq') as MockChatGroq:
    # Configure the mock to return a callable that returns an AIMessage
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="Simulated analysis result.")
    MockChatGroq.return_value = mock_llm
    
    from src.graph.workflow import app

def test_simulation():
    tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "ICICIBANK.NS"]
    
    print("Starting Simulation for tickers:", tickers)
    
    # Mock yfinance
    with patch('yfinance.Ticker') as mock_ticker:
        # Setup mock data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        prices = np.linspace(100, 200, 100)
        df = pd.DataFrame({'Close': prices, 'High': prices+5, 'Low': prices-5, 'Open': prices, 'Volume': 1000000}, index=dates)
        
        mock_instance = MagicMock()
        mock_instance.history.return_value = df
        mock_instance.info = {
            'sector': 'Technology', 
            'trailingPE': 20, 
            'forwardPE': 18,
            'returnOnEquity': 0.25,
            'debtToEquity': 0.1,
            'profitMargins': 0.15,
            'operatingMargins': 0.20,
            'revenueGrowth': 0.10,
            'earningsGrowth': 0.12,
            'freeCashflow': 1000000000,
            'currentRatio': 1.5,
            'priceToBook': 5.0,
            'pegRatio': 1.2,
            'beta': 1.0,
            'marketCap': 5000000000000
        }
        mock_ticker.return_value = mock_instance
        
        for ticker in tickers:
            print(f"\nTesting {ticker}...")
            try:
                initial_state = {
                    "messages": [HumanMessage(content=f"Analyze {ticker} for swing trade.")],
                    "ticker": ticker,
                    "next": "ContextSetup"
                }
                
                # Run the graph
                result = app.invoke(initial_state)
                
                final_msg = result["messages"][-1].content
                print(f"{ticker} Analysis Complete.")
                print(f"Final Output: {final_msg[:100]}...") # Print first 100 chars
                
            except Exception as e:
                print(f"Error analyzing {ticker}: {str(e)}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    test_simulation()
