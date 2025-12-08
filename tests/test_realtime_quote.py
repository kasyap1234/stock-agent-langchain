import sys
import os
from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tools import market_data  # noqa: E402


def test_get_realtime_quote_caches_and_formats(monkeypatch):
    market_data._quote_cache.clear()

    call_count = {"calls": 0}
    now = pd.date_range(start="2024-01-02 10:00", periods=1, freq="T", tz="UTC")
    intraday_df = pd.DataFrame(
        {
            "Open": [123.0],
            "High": [124.0],
            "Low": [122.0],
            "Close": [123.45],
            "Volume": [1_000_000],
        },
        index=now,
    )

    class FakeTicker:
        def __init__(self, ticker):
            call_count["calls"] += 1
            self.fast_info = {"last_price": 123.45, "currency": "INR", "previous_close": 122.5}

        def history(self, period=None, interval=None):
            return intraday_df.copy()

    monkeypatch.setattr(market_data, "yf", MagicMock(Ticker=FakeTicker))

    result_1 = market_data.get_realtime_quote("RELIANCE.NS")
    result_2 = market_data.get_realtime_quote("RELIANCE.NS")

    assert "123.45" in result_1
    assert "as of" in result_1
    assert "JSON:" in result_1
    # Cache should avoid a second Ticker instantiation
    assert call_count["calls"] == 1

    market_data._quote_cache.clear()

