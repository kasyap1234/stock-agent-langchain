import sys
import os
from types import SimpleNamespace
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tools import market_data  # noqa: E402


def test_realtime_quote_marks_stale_when_intraday_missing(monkeypatch):
    market_data._quote_cache.clear()

    def fake_fetch_history(ticker, period, interval=None):
        if interval:
            return pd.DataFrame()  # No intraday data
        idx = pd.date_range(start="2024-01-01", periods=2, freq="D", tz="UTC")
        return pd.DataFrame(
            {
                "Open": [200.0, 201.0],
                "High": [202.0, 203.0],
                "Low": [198.0, 199.0],
                "Close": [201.5, 202.5],
                "Volume": [1_000_000, 1_200_000],
            },
            index=idx,
        )

    class FakeTicker:
        def __init__(self, *_args, **_kwargs):
            self.fast_info = {"last_price": None, "currency": "USD", "previous_close": 199.0}

    monkeypatch.setattr(market_data, "_fetch_history", fake_fetch_history)
    monkeypatch.setattr(market_data, "yf", SimpleNamespace(Ticker=FakeTicker))

    data = market_data.fetch_realtime_quote_structured("TST")

    assert data["price"] == 202.5
    assert data["stale"] is True
    assert data["error_type"] in {"no_intraday_data", "stale_data", "validation_failed"}


def test_intraday_volume_scaling_reduces_false_stale(monkeypatch):
    market_data._quote_cache.clear()

    # 1-minute bars with modest per-bar volume that should pass after scaling
    idx = pd.date_range(start="2024-01-02 09:30", periods=120, freq="min", tz="UTC")
    intraday_df = pd.DataFrame(
        {
            "Open": [100.0] * len(idx),
            "High": [101.0] * len(idx),
            "Low": [99.0] * len(idx),
            "Close": [100.5] * len(idx),
            "Volume": [500.0] * len(idx),
        },
        index=idx,
    )

    class FakeTicker:
        def __init__(self, *_args, **_kwargs):
            self.fast_info = {"last_price": 100.5, "currency": "USD", "previous_close": 99.5}

    def fake_fetch_history(ticker, period, interval=None):
        return intraday_df.copy()

    monkeypatch.setattr(market_data, "_fetch_history", fake_fetch_history)
    monkeypatch.setattr(market_data, "yf", SimpleNamespace(Ticker=FakeTicker))

    data = market_data.fetch_realtime_quote_structured("SCALING")

    assert data["stale"] is False
    assert data["stale_reason"] is None


def test_calculate_indicators_includes_bias_and_confidence(monkeypatch):
    # Construct a simple uptrend dataset
    idx = pd.date_range(start="2024-01-01", periods=260, freq="D")
    prices = pd.Series(range(260, 520), index=idx) / 10  # monotonically rising
    df = pd.DataFrame(
        {
            "Open": prices.values,
            "High": prices.values + 0.5,
            "Low": prices.values - 0.5,
            "Close": prices.values,
            "Volume": [1_000_000] * len(idx),
        },
        index=idx,
    )

    monkeypatch.setattr(market_data, "_fetch_stock_data_with_retry", lambda *_args, **_kwargs: df)
    monkeypatch.setattr(
        market_data,
        "validator",
        SimpleNamespace(
            validate_all=lambda *args, **kwargs: {
                "completeness": (True, None),
                "timestamp_freshness": (True, None),
                "volume_adequacy": (True, None),
                "corporate_actions": (False, None),
            }
        ),
    )

    result = market_data.calculate_indicators.invoke({"ticker": "TST"})

    assert "Bias:" in result
    assert "Confidence:" in result
