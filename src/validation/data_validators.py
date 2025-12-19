"""
Data validation layer for ensuring quality of market data before analysis.

Prevents "garbage in, garbage out" by validating:
- Price ranges (detect anomalies)
- Data freshness (no stale data)
- Completeness (no missing values)
- Corporate actions (splits, dividends)
- Volume adequacy (detect illiquid stocks)
- Indian market specifics (NSE hours, holidays, circuit breakers)
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, time
from typing import Dict, Tuple, Optional, Set
from src.utils.logging_config import ToolLogger


class ValidationError(Exception):
    """Raised when data validation fails."""
    pass


# NSE Holidays 2024-2025
NSE_HOLIDAYS: Set[str] = {
    # 2024
    "2024-01-26", "2024-03-08", "2024-03-25", "2024-03-29", "2024-04-11",
    "2024-04-14", "2024-04-17", "2024-04-21", "2024-05-01", "2024-05-23",
    "2024-06-17", "2024-07-17", "2024-08-15", "2024-10-02", "2024-11-01",
    "2024-11-15", "2024-12-25",
    # 2025
    "2025-01-26", "2025-02-26", "2025-03-14", "2025-03-31", "2025-04-10",
    "2025-04-14", "2025-04-18", "2025-05-01", "2025-05-12", "2025-06-07",
    "2025-07-06", "2025-08-15", "2025-08-16", "2025-10-02", "2025-10-21",
    "2025-10-22", "2025-11-05", "2025-12-25",
}

# NSE Trading Hours
NSE_OPEN = time(9, 15)
NSE_CLOSE = time(15, 30)


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
        Uses NSE holiday calendar for Indian stocks (.NS/.BO suffix).

        Args:
            ticker: Stock ticker symbol
            df: Historical price DataFrame with DatetimeIndex
            max_age_days: Maximum acceptable age in business days

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

            # Check if Indian stock
            is_indian = ticker.endswith('.NS') or ticker.endswith('.BO')

            if is_indian:
                # Calculate business days excluding NSE holidays
                business_days = self._count_nse_trading_days(last_date, current_date)
                adjusted_max_age = max_age_days

                if business_days > adjusted_max_age:
                    reason = (
                        f"Data is stale: Last date {last_date.date()}, "
                        f"{business_days} NSE trading days old (max {adjusted_max_age})"
                    )
                    self.logger.log_validation(ticker, "timestamp_freshness", False, reason)
                    return False, reason
            else:
                # Non-Indian stocks: use calendar days with weekend buffer
                age_days = (current_date - last_date).days
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

    def _count_nse_trading_days(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp
    ) -> int:
        """Count NSE trading days between two dates (excluding weekends and NSE holidays)."""
        count = 0
        current = start_date + timedelta(days=1)

        while current <= end_date:
            date_str = current.strftime('%Y-%m-%d')
            # Skip weekends (5=Sat, 6=Sun)
            if current.weekday() < 5 and date_str not in NSE_HOLIDAYS:
                count += 1
            current += timedelta(days=1)

        return count

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

    def detect_circuit_breaker_risk(
        self,
        ticker: str,
        df: pd.DataFrame,
        lookback_days: int = 10
    ) -> Tuple[bool, Optional[str]]:
        """
        Detect if stock has hit or is near circuit breaker limits.
        Indian stocks have circuit limits at 2%, 5%, 10%, or 20% depending on category.
        F&O stocks have no circuit limits.

        Args:
            ticker: Stock ticker symbol
            df: Historical price DataFrame
            lookback_days: Days to analyze

        Returns:
            (has_risk, description) tuple - True means circuit risk detected
        """
        try:
            if df.empty or len(df) < 2:
                return False, None

            # Check if Indian stock
            is_indian = ticker.endswith('.NS') or ticker.endswith('.BO')
            if not is_indian:
                return False, None

            recent_df = df.tail(lookback_days)

            # Calculate daily moves
            daily_change = recent_df['Close'].pct_change().abs()

            # Check for circuit-like moves (>=10% single day)
            circuit_threshold = 0.10
            circuit_hits = daily_change[daily_change >= circuit_threshold]

            if len(circuit_hits) > 0:
                max_move = daily_change.max() * 100
                circuit_dates = circuit_hits.index.tolist()
                reason = (
                    f"Circuit breaker risk: {len(circuit_hits)} days with >=10% moves "
                    f"(max: {max_move:.1f}%) on {[d.strftime('%Y-%m-%d') for d in circuit_dates[-3:]]}"
                )
                self.logger.log_validation(ticker, "circuit_breaker", True, reason, warning=True)
                return True, reason

            # Check for elevated volatility (avg daily move > 5%)
            avg_move = daily_change.mean() * 100
            if avg_move > 5:
                reason = f"High volatility: Avg daily move {avg_move:.1f}% - near circuit risk"
                self.logger.log_validation(ticker, "circuit_breaker", True, reason, warning=True)
                return True, reason

            self.logger.log_validation(ticker, "circuit_breaker", False)
            return False, None

        except Exception as e:
            reason = f"Circuit breaker detection error: {str(e)}"
            self.logger.log_validation(ticker, "circuit_breaker", False, reason)
            return False, None

    def validate_all(
        self,
        ticker: str,
        df: pd.DataFrame,
        current_price: Optional[float] = None
    ) -> Dict[str, Tuple[bool, Optional[str]]]:
        """
        Run all validation checks and return results.
        Includes Indian market specific validations for .NS/.BO tickers.

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

        # Indian market specific validations
        is_indian = ticker.endswith('.NS') or ticker.endswith('.BO')
        if is_indian:
            results['circuit_breaker'] = self.detect_circuit_breaker_risk(ticker, df)

        # Log overall validation result
        # Exclude warning-type checks from pass/fail (corporate_actions, circuit_breaker)
        warning_checks = {'corporate_actions', 'circuit_breaker'}
        all_passed = all(
            result[0] for key, result in results.items()
            if key not in warning_checks
        )
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
