import sys
import os
from types import SimpleNamespace
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tools import enhanced_data  # noqa: E402


class _FakeTickerBase:
    def __init__(self, *args, **kwargs):
        self.options = ["2024-01-19"]

    def history(self, period="1d"):
        idx = pd.date_range(start="2024-01-01", periods=1, freq="D")
        return pd.DataFrame({"Close": [100]}, index=idx)


class _FakeTickerOptions(_FakeTickerBase):
    def option_chain(self, expiry):
        calls = pd.DataFrame(
            {
                "strike": [100.0],
                "volume": [None],
                "impliedVolatility": [0.2],
            }
        )
        puts = pd.DataFrame(
            {
                "strike": [100.0],
                "volume": [None],
                "impliedVolatility": [0.25],
            }
        )
        return SimpleNamespace(calls=calls, puts=puts)

    @property
    def insider_transactions(self):
        return pd.DataFrame()

    @property
    def institutional_holders(self):
        return pd.DataFrame()


class _FakeTickerNoInsiders(_FakeTickerBase):
    def option_chain(self, expiry):
        return SimpleNamespace(calls=pd.DataFrame(), puts=pd.DataFrame())

    @property
    def insider_transactions(self):
        return None


def test_options_sentiment_returns_json(monkeypatch):
    monkeypatch.setattr(enhanced_data, "yf", SimpleNamespace(Ticker=_FakeTickerOptions))

    result = enhanced_data.get_options_sentiment.invoke({"ticker": "TEST"})

    assert "OPTIONS SENTIMENT" in result
    assert "JSON:" in result


def test_insider_summary_handles_missing(monkeypatch):
    monkeypatch.setattr(enhanced_data, "yf", SimpleNamespace(Ticker=_FakeTickerNoInsiders))

    result = enhanced_data.get_insider_trading_summary.invoke({"ticker": "TEST"})

    assert "No recent insider trading data" in result or "No insider" in result


def test_swing_feature_snapshot_returns_json(monkeypatch):
    end = pd.Timestamp.utcnow().normalize()
    idx = pd.date_range(end=end, periods=260, freq="B")
    close = pd.Series(np.linspace(100, 130, len(idx)), index=idx)
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

    class _FakeTickerSwing(_FakeTickerBase):
        def history(self, period="1y"):
            return df.copy()

    monkeypatch.setattr(enhanced_data, "yf", SimpleNamespace(Ticker=_FakeTickerSwing))

    result = enhanced_data.get_swing_feature_snapshot.invoke({"ticker": "TEST"})

    assert "Swing feature snapshot" in result
    assert "JSON:" in result
