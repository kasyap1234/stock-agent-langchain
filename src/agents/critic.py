"""
Reflection/Critic Agent that reviews and validates other agents' analyses.
"""
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from src.tools.backtesting import backtest_trade_call
from src.confidence.advanced_scoring import ConfidenceScorer
from src.confidence.calibration import Calibrator
from langchain.tools import tool
import json
import os

# Initialize LLM for critic - test Gemini availability and fall back to Groq if needed
llm = None
try:
    # Try Gemini first as the primary LLM
    gemini_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.3)
    # Test if Gemini actually works by making a quick API call
    test_response = gemini_llm.invoke("Hello")
    print(f"Critic initialized with Gemini: {type(gemini_llm)}")
    llm = gemini_llm
except Exception as e:
    print(f"Warning: Gemini API failed test call, falling back to Groq/OpenAI: {e}")
    # Fallback to Groq with OpenAI model
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.3, max_retries=5)

@tool
def calculate_advanced_confidence(analysis_json: str) -> str:
    """
    Calculates advanced multi-dimensional confidence score.
    
    Args:
        analysis_json: JSON string containing analysis results (technical, fundamental, sentiment, etc.)
        
    Returns:
        Detailed confidence report with breakdown and calibrated score.
    """
    try:
        data = json.loads(analysis_json)
        scorer = ConfidenceScorer()
        calibrator = Calibrator()
        
        # Calculate raw score
        result = scorer.calculate_score(data)
        raw_score = result["score"]
        breakdown = result["breakdown"]
        
        # Calibrate
        sector = data.get("sector", "Unknown")
        final_score = calibrator.calibrate_score(raw_score, sector)
        
        report = f"""
ADVANCED CONFIDENCE ANALYSIS
{'='*40}
Final Score: {final_score:.1f}/100 (Calibrated)
Raw Score: {raw_score:.1f}/100

SCORE BREAKDOWN:
{breakdown}

CALIBRATION:
Sector: {sector}
Adjustment Factor: {calibrator.sector_adjustments.get(sector, 0.90):.2f}
{'='*40}
"""
        return report
    except Exception as e:
        return f"Error calculating confidence: {str(e)}"

def create_critic_agent():
    """Creates the Critic/Reflection Agent."""
    tools = [backtest_trade_call, calculate_advanced_confidence]
    
    system_message = """You are a Critical Analyst and Risk Manager. Your role is to:

1. REVIEW all analyses from Technical, Fundamental, and Sentiment analysts
2. IDENTIFY potential flaws, biases, or missing information
3. CHALLENGE assumptions and ask tough questions
4. VALIDATE trade levels using backtesting tools
5. CALCULATE advanced confidence scores using the 'calculate_advanced_confidence' tool
   - You MUST pass a JSON string with all available analysis data to this tool.
   - Include 'technical_analysis', 'fundamental_analysis', 'sentiment_analysis', 'sector', 'regime', etc.
6. PROVIDE counter-arguments and alternative perspectives

Be skeptical but constructive. Your goal is to improve accuracy, not to be negative.

Key areas to critique:
- Are technical levels realistic given recent price action?
- Do fundamentals truly support the bias?
- Is sentiment analysis based on reliable sources?
- What could go wrong with this trade?
- Are there conflicting indicators being ignored?
- Is the risk-reward truly favorable?

Use the backtesting and confidence scoring tools to validate claims with data.
"""
    
    return create_react_agent(llm, tools, prompt=system_message)
