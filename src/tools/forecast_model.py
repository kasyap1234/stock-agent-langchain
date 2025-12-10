"""
Gradient-boosting return forecaster for swing horizons (5-20 trading days).

Provides:
- Feature engineering on OHLCV (returns, trend, vol, volume regime, breakouts, gaps)
- Purged time-series cross-validation to reduce lookahead bias
- GradientBoostingRegressor pipeline with scaling
- Isotonic calibration on CV predictions
- Diagnostics: RMSE/MAPE, feature importances, latest forecast
"""
import json
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import ta
import yfinance as yf
from langchain.tools import tool
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.logging_config import ToolLogger
from src.validation.data_validators import MarketDataValidator

logger = ToolLogger("forecast_model")
validator = MarketDataValidator()


def _build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer swing-horizon features from OHLCV.

    Note: Assumes sanitized, chronologically ordered data.
    """
    feats = pd.DataFrame(index=df.index)
    close = df["Close"]
    volume = df["Volume"]

    feats["ret_1d"] = close.pct_change(1)
    feats["ret_5d"] = close.pct_change(5)
    feats["ret_20d"] = close.pct_change(20)
    feats["ret_60d"] = close.pct_change(60)

    feats["range_pct"] = (df["High"] - df["Low"]) / close.replace(0, np.nan)
    feats["gap_pct"] = (df["Open"] - close.shift(1)) / close.shift(1)

    feats["vol_10d"] = close.pct_change().rolling(10).std()
    feats["vol_20d"] = close.pct_change().rolling(20).std()
    feats["vol_60d"] = close.pct_change().rolling(60).std()

    atr = ta.volatility.AverageTrueRange(
        high=df["High"], low=df["Low"], close=close, window=14
    ).average_true_range()
    feats["atr_pct"] = atr / close

    ema_20 = ta.trend.EMAIndicator(close, window=20).ema_indicator()
    ema_50 = ta.trend.EMAIndicator(close, window=50).ema_indicator()
    ema_200 = ta.trend.EMAIndicator(close, window=200).ema_indicator()
    feats["ema_20"] = ema_20
    feats["ema_50"] = ema_50
    feats["ema_200"] = ema_200
    feats["ema_stack_distance"] = (ema_50 - ema_200) / ema_200
    feats["ema_distance"] = (close - ema_50) / ema_50
    feats["ema_slope_20"] = ema_20.diff()

    macd = ta.trend.MACD(close)
    feats["macd"] = macd.macd()
    feats["macd_signal"] = macd.macd_signal()
    feats["macd_hist"] = macd.macd_diff()
    feats["macd_hist_slope"] = feats["macd_hist"].diff()

    vol_mean_20 = volume.rolling(20).mean()
    vol_std_20 = volume.rolling(20).std().replace(0, np.nan)
    feats["volume_z_20"] = ((volume - vol_mean_20) / vol_std_20).fillna(0)

    vol_mean_60 = volume.rolling(60).mean()
    vol_std_60 = volume.rolling(60).std().replace(0, np.nan)
    feats["volume_z_60"] = ((volume - vol_mean_60) / vol_std_60).fillna(0)

    roll_max_20 = close.rolling(20).max()
    roll_min_20 = close.rolling(20).min()
    feats["breakout_20d"] = (close - roll_min_20) / (roll_max_20 - roll_min_20 + 1e-9)

    feats = feats.replace([np.inf, -np.inf], np.nan).dropna()
    return feats


def _make_dataset(df: pd.DataFrame, horizon: int) -> Tuple[pd.DataFrame, pd.Series]:
    """Align features and forward-return targets."""
    features = _build_feature_frame(df)
    target = df["Close"].pct_change(periods=horizon).shift(-horizon)
    aligned = features.join(target.rename("target")).dropna()
    return aligned.drop(columns=["target"]), aligned["target"]


def _purged_split(n_samples: int, n_splits: int, purge: int):
    """Yield purged time-series splits to reduce leakage."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, test_idx in tscv.split(np.arange(n_samples)):
        if purge > 0:
            cutoff = max(test_idx[0] - purge, 0)
            train_idx = train_idx[train_idx < cutoff]
        if len(train_idx) == 0:
            continue
        yield train_idx, test_idx


@dataclass
class ForecastResult:
    model: Pipeline
    calibrator: Optional[IsotonicRegression]
    metrics: Dict[str, float]
    cv_metrics: List[Dict[str, float]]
    cv_predictions: List[Tuple[float, float]]
    feature_importances: Dict[str, float]
    latest_prediction: float
    latest_prediction_calibrated: Optional[float]


def train_forecast_model(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 4,
    purge: int = 5,
    random_state: int = 42,
) -> ForecastResult:
    """Train gradient boosting regressor with purged CV and optional calibration."""
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                GradientBoostingRegressor(
                    n_estimators=400,
                    learning_rate=0.05,
                    max_depth=3,
                    subsample=0.9,
                    random_state=random_state,
                ),
            ),
        ]
    )

    cv_metrics: List[Dict[str, float]] = []
    cv_preds: List[float] = []
    cv_true: List[float] = []

    for train_idx, test_idx in _purged_split(len(X), n_splits=n_splits, purge=purge):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        rmse = mean_squared_error(y_test, preds) ** 0.5
        mape = mean_absolute_percentage_error(y_test, preds)
        cv_metrics.append({"rmse": rmse, "mape": mape})

        cv_preds.extend(preds.tolist())
        cv_true.extend(y_test.tolist())

    # Fit final model on full dataset
    pipeline.fit(X, y)

    calibrator: Optional[IsotonicRegression] = None
    if len(cv_preds) >= 10:
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(cv_preds, cv_true)

    feature_importances: Dict[str, float] = {}
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        for name, importance in zip(X.columns, model.feature_importances_):
            feature_importances[name] = float(importance)

    latest_pred = float(pipeline.predict(X.tail(1))[0])
    latest_calibrated = (
        float(calibrator.predict([latest_pred])[0]) if calibrator is not None else None
    )

    metrics = {
        "rmse_cv_mean": float(np.mean([m["rmse"] for m in cv_metrics])) if cv_metrics else None,
        "mape_cv_mean": float(np.mean([m["mape"] for m in cv_metrics])) if cv_metrics else None,
        "rmse_cv_last": float(cv_metrics[-1]["rmse"]) if cv_metrics else None,
        "mape_cv_last": float(cv_metrics[-1]["mape"]) if cv_metrics else None,
    }

    return ForecastResult(
        model=pipeline,
        calibrator=calibrator,
        metrics=metrics,
        cv_metrics=cv_metrics,
        cv_predictions=list(zip(cv_preds, cv_true)),
        feature_importances=feature_importances,
        latest_prediction=latest_pred,
        latest_prediction_calibrated=latest_calibrated,
    )


@tool
def run_return_forecast(
    ticker: str,
    horizon_days: int = 10,
    period: str = "1y",
    n_splits: int = 4,
    purge: int = 5,
) -> str:
    """
    Train and evaluate a swing-horizon return forecaster, returning diagnostics plus latest forecast.

    Args:
        ticker: Equity ticker (e.g., "RELIANCE.NS")
        horizon_days: Forecast horizon in trading days (5-20 recommended)
        period: History window for training (default 1y)
        n_splits: Number of CV folds (time-series split)
        purge: Bars to purge before validation fold to avoid leakage
    """
    start_time = time.time()
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        df = validator.sanitize_dataframe(ticker, df)

        if df is None or df.empty:
            return f"No data available for {ticker}"

        validations = validator.validate_all(ticker, df)
        for check, (passed, reason) in validations.items():
            if not passed and check != "corporate_actions":
                return f"Data quality issue for {ticker}: {reason}"

        X, y = _make_dataset(df, horizon=horizon_days)
        if X.empty or len(y) < 80:
            return f"Insufficient training data for {ticker} at horizon {horizon_days}d"

        result = train_forecast_model(X, y, n_splits=n_splits, purge=purge)

        payload = {
            "ticker": ticker,
            "horizon_days": horizon_days,
            "cv_metrics": result.cv_metrics,
            "metrics": result.metrics,
            "latest_prediction": result.latest_prediction,
            "latest_prediction_calibrated": result.latest_prediction_calibrated,
            "feature_importances": result.feature_importances,
            "samples": len(y),
        }

        latency_ms = (time.time() - start_time) * 1000
        logger.log_fetch(
            ticker=ticker,
            data_type="forecast_model",
            success=True,
            latency_ms=latency_ms,
            records_fetched=len(df),
        )

        rmse_mean = payload["metrics"]["rmse_cv_mean"]
        mape_mean = payload["metrics"]["mape_cv_mean"]
        rmse_str = f"{rmse_mean:.4f}" if rmse_mean is not None else "n/a"
        mape_str = f"{mape_mean:.4f}" if mape_mean is not None else "n/a"
        latest_raw_str = f"{payload['latest_prediction']:+.4%}"
        latest_cal = payload["latest_prediction_calibrated"]
        latest_cal_str = f"{latest_cal:+.4%}" if latest_cal is not None else "n/a"
        top_factors = ", ".join(
            sorted(result.feature_importances, key=result.feature_importances.get, reverse=True)[:5]
        ) or "n/a"

        summary = f"""
GBDT forecast for {ticker} ({horizon_days}d fwd return)
Samples: {len(y)} | CV folds: {n_splits} (purge={purge})
CV RMSE/ MAPE (mean): {rmse_str} / {mape_str}
Latest forecast (raw / calibrated): {latest_raw_str} / {latest_cal_str}
Top factors: {top_factors}
"""
        return summary + f"\nJSON:{json.dumps(payload, default=str)}"

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        logger.log_fetch(
            ticker=ticker,
            data_type="forecast_model",
            success=False,
            latency_ms=latency_ms,
            records_fetched=0,
            error=str(e),
        )
        return f"Forecasting error for {ticker}: {str(e)}"
