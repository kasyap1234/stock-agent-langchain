"""
Data validation layer for ensuring quality of market data before analysis.

Prevents "garbage in, garbage out" by validating:
- Price ranges (detect anomalies)
- Data freshness (no stale data)
- Completeness (no missing values)
- Corporate actions (splits, dividends)
- Volume adequacy (detect illiquid stocks)
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional
from src.utils.logging_config import ToolLogger


class ValidationError(Exception):
    """Raised when data validation fails."""
    pass


class MarketDataValidator:
    """Validates market data quality before agent analysis."""

    def __init__(self):
        self.logger = ToolLogger("MarketDataValidator")

    def validate_price_range(
        self,
        ticker: str,
        current_price: float,
        df: pd.DataFrame,
        tolerance: float = 0.5
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate that current price is within reasonable range of 52-week high/low.

        Args:
            ticker: Stock ticker symbol
            current_price: Current stock price
            df: Historical price DataFrame
            tolerance: Tolerance multiplier (0.5 = +/-50% of 52-week range)

        Returns:
            (is_valid, reason) tuple
        """
        try:
            if df.empty or current_price <= 0:
                reason = f"Invalid data: Empty DataFrame or price <= 0"
                self.logger.log_validation(ticker, "price_range", False, reason)
                return False, reason

            # Calculate 52-week high and low
            high_52w = df['High'].rolling(window=252, min_periods=1).max().iloc[-1]
            low_52w = df['Low'].rolling(window=252, min_periods=1).min().iloc[-1]

            # Calculate acceptable range with tolerance
            range_52w = high_52w - low_52w
            upper_bound = high_52w + (range_52w * tolerance)
            lower_bound = low_52w - (range_52w * tolerance)

            if not (lower_bound <= current_price <= upper_bound):
                reason = (
                    f"Price {current_price:.2f} outside acceptable range "
                    f"[{lower_bound:.2f}, {upper_bound:.2f}] "
                    f"(52W: {low_52w:.2f}-{high_52w:.2f})"
                )
                self.logger.log_validation(ticker, "price_range", False, reason)
                return False, reason

            self.logger.log_validation(ticker, "price_range", True)
            return True, None

        except Exception as e:
            reason = f"Price validation error: {str(e)}"
            self.logger.log_validation(ticker, "price_range", False, reason)
            return False, reason

    def validate_timestamp_freshness(
        self,
        ticker: str,
        df: pd.DataFrame,
        max_age_days: int = 7
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate that data is recent (not stale).

        Args:
            ticker: Stock ticker symbol
            df: Historical price DataFrame with DatetimeIndex
            max_age_days: Maximum acceptable age in days

        Returns:
            (is_valid, reason) tuple
        """
        try:
            if df.empty:
                reason = "Empty DataFrame"
                self.logger.log_validation(ticker, "timestamp_freshness", False, reason)
                return False, reason

            last_date = pd.to_datetime(df.index[-1])
            # Normalize timezone: remove timezone info if present
            if last_date.tzinfo is not None:
                last_date = last_date.tz_localize(None)
            current_date = pd.Timestamp.now()

            # Account for weekends and market holidays
            age_days = (current_date - last_date).days

            # Allow extra days for weekends (up to 3 days for weekend + holiday)
            adjusted_max_age = max_age_days + 3

            if age_days > adjusted_max_age:
                reason = (
                    f"Data is stale: Last date {last_date.date()}, "
                    f"age {age_days} days (max {adjusted_max_age})"
                )
                self.logger.log_validation(ticker, "timestamp_freshness", False, reason)
                return False, reason

            self.logger.log_validation(ticker, "timestamp_freshness", True)
            return True, None

        except Exception as e:
            reason = f"Timestamp validation error: {str(e)}"
            self.logger.log_validation(ticker, "timestamp_freshness", False, reason)
            return False, reason

    def validate_completeness(
        self,
        ticker: str,
        df: pd.DataFrame,
        required_columns: list = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate that required columns exist and have no missing values.

        Args:
            ticker: Stock ticker symbol
            df: Historical price DataFrame
            required_columns: List of required columns (default: OHLCV)

        Returns:
            (is_valid, reason) tuple
        """
        if required_columns is None:
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']

        try:
            if df.empty:
                reason = "Empty DataFrame"
                self.logger.log_validation(ticker, "completeness", False, reason)
                return False, reason

            # Check for missing columns
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                reason = f"Missing required columns: {missing_columns}"
                self.logger.log_validation(ticker, "completeness", False, reason)
                return False, reason

            # Check for NaN values in required columns
            null_counts = df[required_columns].isnull().sum()
            if null_counts.any():
                null_cols = null_counts[null_counts > 0].to_dict()
                reason = f"Missing values detected: {null_cols}"
                self.logger.log_validation(ticker, "completeness", False, reason)
                return False, reason

            # Check for zero or negative prices
            for col in ['Open', 'High', 'Low', 'Close']:
                if (df[col] <= 0).any():
                    reason = f"Invalid {col} prices (<=0) detected"
                    self.logger.log_validation(ticker, "completeness", False, reason)
                    return False, reason

            self.logger.log_validation(ticker, "completeness", True)
            return True, None

        except Exception as e:
            reason = f"Completeness validation error: {str(e)}"
            self.logger.log_validation(ticker, "completeness", False, reason)
            return False, reason

    def detect_corporate_actions(
        self,
        ticker: str,
        df: pd.DataFrame,
        lookback_days: int = 90
    ) -> Tuple[bool, Optional[str]]:
        """
        Detect potential stock splits or significant corporate actions.

        Large price gaps can indicate splits that affect historical patterns.

        Args:
            ticker: Stock ticker symbol
            df: Historical price DataFrame
            lookback_days: Days to look back for corporate actions

        Returns:
            (has_action, description) tuple
        """
        try:
            if df.empty or len(df) < 2:
                self.logger.log_validation(ticker, "corporate_actions", True, "Insufficient data")
                return False, None

            # Look at recent data
            recent_df = df.tail(lookback_days)

            # Calculate daily price changes
            daily_change = recent_df['Close'].pct_change().abs()

            # Detect splits (>40% single-day move with corresponding volume spike)
            split_threshold = 0.40
            volume_spike_threshold = 2.0

            volume_ratio = recent_df['Volume'] / recent_df['Volume'].rolling(window=20).mean()

            potential_splits = (daily_change > split_threshold) & (volume_ratio > volume_spike_threshold)

            if potential_splits.any():
                split_dates = recent_df[potential_splits].index
                reason = f"Potential corporate action detected on {split_dates.tolist()}"
                self.logger.log_validation(
                    ticker,
                    "corporate_actions",
                    False,
                    reason,
                    warning=True
                )
                return True, reason

            self.logger.log_validation(ticker, "corporate_actions", True)
            return False, None

        except Exception as e:
            reason = f"Corporate action detection error: {str(e)}"
            self.logger.log_validation(ticker, "corporate_actions", True, reason)
            return False, None

    def validate_volume_adequacy(
        self,
        ticker: str,
        df: pd.DataFrame,
        min_avg_volume: int = 100000
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate that stock has adequate trading volume (not illiquid).

        Args:
            ticker: Stock ticker symbol
            df: Historical price DataFrame
            min_avg_volume: Minimum acceptable average daily volume

        Returns:
            (is_valid, reason) tuple
        """
        try:
            if df.empty:
                reason = "Empty DataFrame"
                self.logger.log_validation(ticker, "volume_adequacy", False, reason)
                return False, reason

            # Calculate average volume over recent period
            avg_volume = df['Volume'].tail(30).mean()

            if avg_volume < min_avg_volume:
                reason = (
                    f"Low liquidity: Avg volume {avg_volume:,.0f} "
                    f"below threshold {min_avg_volume:,.0f}"
                )
                self.logger.log_validation(ticker, "volume_adequacy", False, reason)
                return False, reason

            self.logger.log_validation(ticker, "volume_adequacy", True)
            return True, None

        except Exception as e:
            reason = f"Volume validation error: {str(e)}"
            self.logger.log_validation(ticker, "volume_adequacy", False, reason)
            return False, reason

    def validate_all(
        self,
        ticker: str,
        df: pd.DataFrame,
        current_price: Optional[float] = None
    ) -> Dict[str, Tuple[bool, Optional[str]]]:
        """
        Run all validation checks and return results.

        Args:
            ticker: Stock ticker symbol
            df: Historical price DataFrame
            current_price: Current stock price (if available)

        Returns:
            Dictionary of validation results
        """
        results = {
            'completeness': self.validate_completeness(ticker, df),
            'timestamp_freshness': self.validate_timestamp_freshness(ticker, df),
            'volume_adequacy': self.validate_volume_adequacy(ticker, df),
            'corporate_actions': self.detect_corporate_actions(ticker, df),
        }

        if current_price is not None:
            results['price_range'] = self.validate_price_range(ticker, current_price, df)

        # Log overall validation result
        all_passed = all(result[0] for key, result in results.items() if key != 'corporate_actions')
        self.logger.logger.info(
            "full_validation",
            ticker=ticker,
            all_passed=all_passed,
            results={k: v[0] for k, v in results.items()}
        )

        return results


def validate_market_data(ticker: str, df: pd.DataFrame) -> None:
    """
    Convenience function to validate market data and raise exception if invalid.

    Args:
        ticker: Stock ticker symbol
        df: Historical price DataFrame

    Raises:
        ValidationError: If validation fails
    """
    validator = MarketDataValidator()
    results = validator.validate_all(ticker, df)

    # Check for critical failures (exclude corporate_actions warning)
    failures = [
        (key, reason)
        for key, (passed, reason) in results.items()
        if not passed and key != 'corporate_actions'
    ]

    if failures:
        error_msg = "; ".join([f"{key}: {reason}" for key, reason in failures])
        raise ValidationError(f"Data validation failed for {ticker}: {error_msg}")

    # Warn about corporate actions
    if results.get('corporate_actions', (False, None))[0]:
        _, reason = results['corporate_actions']
        validator.logger.logger.warning("corporate_action_detected", ticker=ticker, reason=reason)
