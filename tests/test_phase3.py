import sys
import os
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.tools.backtesting import perform_walk_forward_analysis
from src.tools.advanced_backtesting import monte_carlo_simulation
from src.tools.fundamentals import get_fundamental_metrics

def test_fundamentals():
    print("Testing Fundamental Metrics...")
    with patch('yfinance.Ticker') as mock_ticker:
        mock_instance = MagicMock()
        mock_instance.info = {
            'trailingPE': 25.5,
            'forwardPE': 22.0,
            'returnOnEquity': 0.18,
            'debtToEquity': 0.5
        }
        mock_ticker.return_value = mock_instance
        
        report = get_fundamental_metrics.invoke({"ticker": "RELIANCE.NS"})
        print(report)
        
        if "P/E (Trailing): 25.5" in report and "ROE: 18.00%" in report:
            print("Fundamentals test passed.")
        else:
            print("Fundamentals test failed.")

def test_walk_forward():
    print("\nTesting Walk-Forward Analysis...")
    with patch('yfinance.Ticker') as mock_ticker:
        # Create dummy historical data
        dates = pd.date_range(start='2023-01-01', periods=300, freq='D')
        prices = np.linspace(100, 200, 300) + np.random.normal(0, 5, 300)
        df = pd.DataFrame({'Close': prices, 'High': prices+2, 'Low': prices-2}, index=dates)
        
        mock_instance = MagicMock()
        mock_instance.history.return_value = df
        mock_ticker.return_value = mock_instance
        
        report = perform_walk_forward_analysis.invoke({"ticker": "RELIANCE.NS", "windows": 2})
        print(report)
        
        if "WALK-FORWARD ANALYSIS" in report and "Robustness Score" in report:
            print("Walk-Forward test passed.")
        else:
            print("Walk-Forward test failed.")

def test_monte_carlo():
    print("\nTesting Monte Carlo (Fat Tail)...")
    with patch('yfinance.Ticker') as mock_ticker:
        # Create dummy historical data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        prices = np.linspace(100, 110, 100)
        df = pd.DataFrame({'Close': prices}, index=dates)
        
        mock_instance = MagicMock()
        mock_instance.history.return_value = df
        mock_ticker.return_value = mock_instance
        
        report = monte_carlo_simulation.invoke({"ticker": "RELIANCE.NS", "scenario": "fat_tail"})
        print(report)
        
        if "FAT_TAIL Scenario" in report and "90th Percentile" in report:
            print("Monte Carlo test passed.")
        else:
            print("Monte Carlo test failed.")

if __name__ == "__main__":
    test_fundamentals()
    test_walk_forward()
    test_monte_carlo()
