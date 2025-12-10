import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.tools import forecast_model


def _make_trending_df(rows: int = 400) -> pd.DataFrame:
    end = pd.Timestamp.utcnow().normalize()
    idx = pd.date_range(end=end, periods=rows, freq="B")
    close = pd.Series(np.linspace(100, 130, len(idx)), index=idx)
    return pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": np.linspace(800_000, 1_200_000, len(idx)),
        },
        index=idx,
    )


def test_run_return_forecast_produces_metrics(monkeypatch):
    df = _make_trending_df()

    class FakeTicker:
        def __init__(self, *_args, **_kwargs):
            pass

        def history(self, period="1y"):
            return df.copy()

    monkeypatch.setattr(forecast_model, "yf", SimpleNamespace(Ticker=FakeTicker))

    result = forecast_model.run_return_forecast.invoke(
        {"ticker": "TEST", "horizon_days": 5, "period": "1y", "n_splits": 3, "purge": 2}
    )

    assert "GBDT forecast" in result
    payload = json.loads(result.split("JSON:")[1])
    assert payload["latest_prediction"] is not None
    assert payload["metrics"]["rmse_cv_mean"] is not None
