import sys
import os
from unittest.mock import MagicMock

from langchain_core.messages import HumanMessage, AIMessage

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents import supervisor  # noqa: E402


def test_synthesize_final_call_includes_live_price(monkeypatch):
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content="Final content")
    monkeypatch.setattr(supervisor, "llm", fake_llm)

    state = {
        "messages": [
            HumanMessage(content="Plan details", name="Planner"),
            AIMessage(content="Tech analysis", name="Technical_Analyst"),
            AIMessage(content="Fundamental analysis", name="Fundamental_Analyst"),
            AIMessage(content="Sentiment analysis", name="Sentiment_Analyst"),
            AIMessage(content="Ensemble view", name="Ensemble"),
            AIMessage(content="Critic notes", name="Critic"),
        ],
        "ticker": "TEST.NS",
        "next": "FinalSynthesis",
        "sector": "Technology",
        "regime": "TRENDING",
        "analysis_results": {},
        "quote": {
            "price": 123.45,
            "currency": "INR",
            "previous_close": 122.5,
            "as_of": "2024-01-02T10:00:00",
            "stale": False,
            "stale_reason": None,
            "source": "test",
        },
    }

    supervisor.synthesize_final_call(state)

    prompt = fake_llm.invoke.call_args[0][0]
    assert "CURRENT PRICE: INR 123.45" in prompt
    assert "price unavailable" not in prompt.lower()


