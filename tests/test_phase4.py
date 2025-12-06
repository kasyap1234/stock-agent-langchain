import sys
import os
import json
from unittest.mock import patch, MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock ChatGroq before importing critic
with patch('langchain_groq.ChatGroq'):
    from src.agents.critic import calculate_advanced_confidence

def test_confidence_scoring():
    print("Testing Advanced Confidence Scoring...")
    
    # Mock data representing a strong setup
    strong_data = {
        "technical_analysis": "Bullish trend confirmed",
        "fundamental_analysis": "Positive earnings growth",
        "sentiment_analysis": "Positive market sentiment",
        "sector": "Technology",
        "regime": "STRONG_TRENDING",
        "strategy": "Trend Following",
        "backtest_report": "Robustness Score: 100%",
        "monte_carlo_report": "Positive expected value"
    }
    
    json_input = json.dumps(strong_data)
    report = calculate_advanced_confidence.invoke({"analysis_json": json_input})
    print(f"\nStrong Setup Report:\n{report}")
    
    if "Final Score" in report and "SCORE BREAKDOWN" in report:
        print("Strong setup test passed.")
    else:
        print("Strong setup test failed.")

    # Mock data representing a weak setup
    weak_data = {
        "technical_analysis": "Bearish divergence",
        "fundamental_analysis": "Negative outlook",
        "sentiment_analysis": "Neutral",
        "sector": "Unknown",
        "regime": "VOLATILE",
        "strategy": "Trend Following", # Mismatch with volatile
        "backtest_report": "Robustness Score: 0%",
        "monte_carlo_report": "Negative expected value"
    }
    
    json_input_weak = json.dumps(weak_data)
    report_weak = calculate_advanced_confidence.invoke({"analysis_json": json_input_weak})
    print(f"\nWeak Setup Report:\n{report_weak}")
    
    # Extract scores to compare
    import re
    score_match_strong = re.search(r"Final Score: (\d+\.\d+)", report)
    score_match_weak = re.search(r"Final Score: (\d+\.\d+)", report_weak)
    
    if score_match_strong and score_match_weak:
        strong_score = float(score_match_strong.group(1))
        weak_score = float(score_match_weak.group(1))
        
        if strong_score > weak_score:
            print(f"Logic check passed: Strong ({strong_score}) > Weak ({weak_score})")
        else:
            print(f"Logic check failed: Strong ({strong_score}) <= Weak ({weak_score})")

if __name__ == "__main__":
    test_confidence_scoring()
