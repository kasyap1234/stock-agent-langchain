import sys
import os
import json
from types import SimpleNamespace
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tools import backtesting, advanced_backtesting  # noqa: E402


def test_backtest_trade_call_prefers_earliest_hit(monkeypatch):
    def fake_history(ticker, start, end):
        idx = pd.date_range(end=pd.Timestamp.utcnow().normalize(), periods=3, freq="D")
        return pd.DataFrame(
            {
                "Open": [100, 100, 100],
                "High": [111, 108, 107],  # Target hit day 1
                "Low": [95, 85, 80],      # Stop breached day 2
                "Close": [105, 90, 82],
                "Volume": [1_000_000, 900_000, 800_000],
            },
            index=idx,
        )

    monkeypatch.setattr(backtesting, "_fetch_history", fake_history)

    result = backtesting.backtest_trade_call.invoke(
        {
            "ticker": "TEST",
            "entry_price": 100,
            "target_price": 110,
            "stop_loss": 90,
            "days_back": 3,
        }
    )

    assert "Target reached before stop" in result
    assert "JSON:" in result


def test_backtest_trade_call_includes_costs(monkeypatch):
    def fake_history(ticker, start, end):
        idx = pd.date_range(end=pd.Timestamp.utcnow().normalize(), periods=3, freq="D")
        return pd.DataFrame(
            {
                "Open": [100, 100, 100],
                "High": [111, 108, 107],
                "Low": [95, 85, 80],
                "Close": [105, 90, 82],
                "Volume": [1_000_000, 900_000, 800_000],
            },
            index=idx,
        )

    monkeypatch.setattr(backtesting, "_fetch_history", fake_history)

    result = backtesting.backtest_trade_call.invoke(
        {
            "ticker": "TEST",
            "entry_price": 100,
            "target_price": 110,
            "stop_loss": 90,
            "days_back": 3,
            "slippage_bps": 20,
            "fee_bps": 10,
        }
    )

    assert "Risk-Reward (after costs)" in result
    payload = json.loads(result.split("JSON:")[1])
    assert payload["risk_reward_net"] >= 0
    assert payload["round_trip_cost_pct"] > 0


def test_model_signal_backtest_outputs_json(monkeypatch):
    end = pd.Timestamp.utcnow().normalize()
    idx = pd.date_range(end=end, periods=400, freq="B")
    close = pd.Series(np.linspace(100, 125, len(idx)), index=idx)
    df = pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": [1_000_000] * len(idx),
        },
        index=idx,
    )

    class FakeTicker:
        def __init__(self, *_args, **_kwargs):
            pass

        def history(self, period="1y"):
            return df.copy()

    monkeypatch.setattr(advanced_backtesting, "yf", SimpleNamespace(Ticker=FakeTicker))

    result = advanced_backtesting.model_signal_backtest.invoke(
        {"ticker": "TEST", "period": "1y", "horizon_days": 5, "threshold_pct": -0.05}
    )

    assert "MODEL SIGNAL BACKTEST" in result
    assert "JSON:" in result
