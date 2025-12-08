"""
Optimized supervisor with enhanced synthesis.
"""
from typing import TypedDict, Annotated, List, Dict, Any, Optional
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
    quote: Optional[Dict[str, Any]]

# Initialize LLM
llm = ChatGroq(model="moonshotai/kimi-k2-instruct-0905", temperature=0, max_retries=5)

# Final synthesis
def synthesize_final_call(state: AgentState) -> Dict:
    """Synthesize all reports into final trade call."""
    messages = state["messages"]
    ticker = state["ticker"]
    quote = state.get("quote") or {}
    
    # Extract all reports
    reports = {}
    for msg in messages:
        if hasattr(msg, 'name') and msg.name:
            reports[msg.name] = msg.content

    # Build explicit price line to avoid hallucinated placeholders
    price = quote.get("price")
    currency = quote.get("currency", "N/A")
    as_of = quote.get("as_of")
    stale = quote.get("stale")
    stale_reason = quote.get("stale_reason")

    if price is not None:
        price_line = f"CURRENT PRICE: {currency} {price:.2f} (as of {as_of})"
        if stale:
            price_line += f" [STALE: {stale_reason}]"
    else:
        price_line = "CURRENT PRICE: Unavailable (live quote missing; do not invent a price)"
    
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

{price_line}
RECOMMENDATION: Base entries/targets on the price above; if unavailable, explicitly say price unavailable.
CONFIDENCE: [0-100]%
ENSEMBLE: [vote result]

TRADE SETUP:
   Entry: Use live price as anchor (or state cannot due to missing price)
   Target 1: Quote level plus % move
   Target 2: Quote level plus % move
   Stop Loss: Protective level with % move from entry

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
