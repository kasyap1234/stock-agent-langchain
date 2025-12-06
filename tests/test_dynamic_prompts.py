import sys
import os
import json
from unittest.mock import patch, MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agents.dynamic_prompts import get_dynamic_prompt
from src.agents.context_setup import setup_analysis_context

def test_dynamic_prompt_generation():
    print("Testing Dynamic Prompt Generation...")
    
    # Mock find_similar_examples
    with patch('src.agents.dynamic_prompts.find_similar_examples') as mock_find:
        mock_find.return_value = [
            {
                "ticker": "TCS.NS",
                "outcome": "WIN",
                "sector": "Technology",
                "regime": "TRENDING",
                "recommendation": "Buy on dip",
                "entry": 3500,
                "target": 3800
            }
        ]
        
        prompt = get_dynamic_prompt("Technical_Analyst", "Technology", "TRENDING")
        
        print(f"Generated Prompt:\n{prompt}")
        
        if "TCS.NS" in prompt and "Buy on dip" in prompt:
            print("Prompt generation successful.")
        else:
            print("Prompt generation failed.")

def test_context_setup():
    print("\nTesting Context Setup...")
    
    # Mock yfinance and regime detection
    with patch('yfinance.Ticker') as mock_ticker, \
         patch('src.agents.context_setup._fetch_regime_data') as mock_fetch, \
         patch('src.agents.context_setup._calculate_regime_indicators') as mock_calc, \
         patch('src.agents.context_setup._classify_regime') as mock_classify:
             
        # Mock Ticker info
        mock_instance = MagicMock()
        mock_instance.info = {"sector": "Technology"}
        mock_ticker.return_value = mock_instance
        
        # Mock Regime
        mock_fetch.return_value = MagicMock(empty=False, __len__=lambda x: 100)
        mock_classify.return_value = {"regime": "STRONG_TRENDING"}
        
        state = {"ticker": "INFY.NS"}
        context = setup_analysis_context(state)
        
        print(f"Context: {context}")
        
        if context["sector"] == "Technology" and context["regime"] == "STRONG_TRENDING":
            print("Context setup successful.")
        else:
            print("Context setup failed.")

if __name__ == "__main__":
    test_dynamic_prompt_generation()
    test_context_setup()
