"""
Lightweight smoke checks to ensure guardrails work without hitting live LLMs.
"""
from typing import Dict, Any
from langchain_core.messages import HumanMessage
from src.agents.supervisor import synthesize_final_call


def price_unavailable_smoke() -> str:
    """
    Ensures the supervisor responds with a price-unavailable guardrail instead of hallucinating levels.
    This uses a synthetic state and does not call external services.
    """
    state: Dict[str, Any] = {
        "messages": [
            HumanMessage(content="Planner: synthetic plan", name="Planner"),
            HumanMessage(content="Technical: synthetic tech view", name="Technical_Analyst"),
            HumanMessage(content="Fundamental: synthetic fundamental view", name="Fundamental_Analyst"),
            HumanMessage(content="Sentiment: synthetic sentiment view", name="Sentiment_Analyst"),
            HumanMessage(content="Ensemble: synthetic ensemble view", name="Ensemble"),
            HumanMessage(content="Critic: synthetic critic view", name="Critic"),
        ],
        "next": "",
        "ticker": "TEST",
        "sector": "Unknown",
        "regime": "Unknown",
        "analysis_results": {},
        "quote": {
            "ticker": "TEST",
            "price": None,
            "currency": "N/A",
            "as_of": None,
            "stale": True,
            "stale_reason": "synthetic smoke test",
            "source": "yfinance",
        },
    }

    result = synthesize_final_call(state)["messages"][-1].content
    if "price unavailable" in result.lower():
        return "PASS: supervisor emits price-unavailable guardrail."
    return "FAIL: supervisor did not emit price-unavailable guardrail."

