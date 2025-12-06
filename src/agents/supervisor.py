"""
Optimized supervisor with enhanced synthesis.
"""
from typing import TypedDict, Annotated, List, Dict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
import operator

# Define State
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next: str
    ticker: str
    sector: str
    regime: str
    analysis_results: Dict[str, str]

# Initialize LLM
llm = ChatGroq(model="moonshotai/kimi-k2-instruct-0905", temperature=0, max_retries=5)

# Final synthesis
def synthesize_final_call(state: AgentState) -> Dict:
    """Synthesize all reports into final trade call."""
    messages = state["messages"]
    ticker = state["ticker"]
    
    # Extract all reports
    reports = {}
    for msg in messages:
        if hasattr(msg, 'name') and msg.name:
            reports[msg.name] = msg.content
    
    synthesis_prompt = f"""Senior Trading Strategist - Final Call for {ticker}

PLANNER'S STRATEGY:
{reports.get('Planner', 'N/A')}

PARALLEL ANALYSTS:
Technical: {reports.get('Technical_Analyst', 'N/A')}
Fundamental: {reports.get('Fundamental_Analyst', 'N/A')}
Sentiment: {reports.get('Sentiment_Analyst', 'N/A')}

ENSEMBLE STRATEGIES:
{reports.get('Ensemble', 'N/A')}

CRITIC'S VALIDATION:
{reports.get('Critic', 'N/A')}

================================================
FINAL SWING TRADE CALL: {ticker}
================================================

RECOMMENDATION: [BUY/SELL/HOLD]
CONFIDENCE: [XX]%
ENSEMBLE: [vote result]

CURRENT PRICE: Rs[price]

TRADE SETUP:
   Entry: Rs[price]
   Target 1: Rs[price] ([X]%)
   Target 2: Rs[price] ([X]%)
   Stop Loss: Rs[price] ([X]%)

BIAS: [Bullish/Bearish/Neutral]
RISK-REWARD: [X:X]

BACKTEST: [results]
MONTE CARLO: [probability analysis]

KEY RISKS:
   - [from Critic]

CATALYSTS:
   - [from analysis]

SUMMARY:
[Final recommendation]

================================================
"""
    
    response = llm.invoke(synthesis_prompt)
    return {"messages": [AIMessage(content=response.content, name="Supervisor")]}
