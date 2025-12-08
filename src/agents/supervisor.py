"""
Optimized supervisor with enhanced synthesis.
"""
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from src.utils.llm_fallbacks import groq_with_cerebras_fallback
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

# Initialize LLM with Cerebras fallback on Groq 429
llm = groq_with_cerebras_fallback(model="moonshotai/kimi-k2-instruct-0905", temperature=0, max_retries=5)

# Final synthesis
def synthesize_final_call(state: AgentState) -> Dict:
    """Synthesize all reports into final trade call with guardrails against hallucinated pricing."""
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
    currency = quote.get("currency") or "N/A"
    as_of = quote.get("as_of") or "N/A"
    stale = bool(quote.get("stale"))
    stale_reason = quote.get("stale_reason")

    if price is not None:
        price_line = f"CURRENT PRICE: {currency} {price:.2f} (as of {as_of})"
        if stale:
            price_line += f" [STALE: {stale_reason}]"
    else:
        price_line = (
            "CURRENT PRICE: Unavailable (live quote missing; do not invent a price)"
        )

    # If price is unavailable, short-circuit to a deterministic, non-numeric message
    if price is None:
        safe_summary = f"""FINAL SWING TRADE CALL: {ticker}
{price_line}

Price unavailable → No numeric entries, targets, or stops are provided. Do not trade without a validated live quote.

Planner: {reports.get('Planner', 'N/A')}
Technical: {reports.get('Technical_Analyst', 'N/A')}
Fundamental: {reports.get('Fundamental_Analyst', 'N/A')}
Sentiment: {reports.get('Sentiment_Analyst', 'N/A')}
Ensemble: {reports.get('Ensemble', 'N/A')}
Critic: {reports.get('Critic', 'N/A')}

Action: Wait for a fresh quote, then re-run analysis. If price remains unavailable, skip the trade."""
        return {"messages": [AIMessage(content=safe_summary, name="Supervisor")]}

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
GUARDRAILS:
- Never invent a price. Use the exact price/currency/as-of above.
- If price is marked STALE, clearly label all levels as stale and lower conviction.
- Base entries/targets/stops on the price above. If price becomes unavailable, explicitly say 'Price unavailable—no levels.'

CHECKLIST (down-rank confidence if missing):
- Technical: multi-timeframe alignment stated, volume vs 20d avg cited, ATR(14) cited; no breakout claim without volume/ATR.
- Fundamental: valuation sanity (PE/FwdPE/EV-EBITDA/PB/PS), earnings quality (OCF/NI, accruals), leverage/coverage (Debt/EBITDA, OCF/FCF vs debt), liquidity noted.
- Sentiment: at least two distinct recent sources with recency; otherwise mark 'insufficient sentiment evidence.'
- Data freshness: flag stale price/data and reduce conviction.

CONFIDENCE: [0-100]%
ENSEMBLE: [vote result]

TRADE SETUP:
   Entry: Anchor to live price (or say 'cannot compute' if stale/invalid)
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
