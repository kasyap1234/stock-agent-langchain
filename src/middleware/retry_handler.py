"""
Retry middleware for handling transient API failures with exponential backoff.

Provides:
- Automatic retry logic for failed API calls
- Exponential backoff to avoid hammering failing services
- Configurable retry policies per tool
- Logging of all retry attempts
"""

import time
import functools
from typing import Callable, Any, Type, Tuple
from src.utils.logging_config import ToolLogger


logger = ToolLogger("RetryHandler")


class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_attempts: int = 3,
        backoff_factor: float = 2.0,
        initial_delay: float = 1.0,
        max_delay: float = 30.0,
        retry_on: Tuple[Type[Exception], ...] = (Exception,),
    ):
        """
        Initialize retry configuration.

        Args:
            max_attempts: Maximum number of total attempts (including first)
            backoff_factor: Multiplier for exponential backoff
            initial_delay: Initial delay in seconds
            max_delay: Maximum delay between retries in seconds
            retry_on: Tuple of exception types to retry on
        """
        self.max_attempts = max_attempts
        self.backoff_factor = backoff_factor
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.retry_on = retry_on


def with_retry(
    config: RetryConfig = None,
    fallback_value: Any = None
) -> Callable:
    """
    Decorator to add retry logic to a function.

    Args:
        config: RetryConfig instance (uses defaults if None)
        fallback_value: Value to return if all retries fail (raises if None)

    Returns:
        Decorated function with retry logic

    Example:
        @with_retry(RetryConfig(max_attempts=3))
        def get_stock_data(ticker):
            return yf.download(ticker)
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            delay = config.initial_delay

            # Extract ticker for logging (if available in args/kwargs)
            ticker = None
            if args and isinstance(args[0], str):
                ticker = args[0]
            elif 'ticker' in kwargs:
                ticker = kwargs['ticker']

            for attempt in range(config.max_attempts):
                try:
                    # Attempt the function call
                    result = func(*args, **kwargs)

                    # Success - log if this wasn't the first attempt
                    if attempt > 0:
                        logger.logger.info(
                            "retry_success",
                            function=func.__name__,
                            ticker=ticker,
                            attempt=attempt + 1,
                            total_attempts=config.max_attempts
                        )

                    return result

                except config.retry_on as e:
                    last_exception = e

                    # Log retry attempt
                    if ticker:
                        logger.log_retry(
                            ticker=ticker,
                            attempt=attempt + 1,
                            max_attempts=config.max_attempts,
                            error=str(e),
                            retryable=True,
                        )
                    else:
                        logger.logger.warning(
                            "retry_attempt",
                            function=func.__name__,
                            attempt=attempt + 1,
                            max_attempts=config.max_attempts,
                            error=str(e)
                        )

                    # Don't sleep after the last attempt
                    if attempt < config.max_attempts - 1:
                        time.sleep(min(delay, config.max_delay))
                        delay *= config.backoff_factor

            # All retries exhausted
            logger.logger.error(
                "retry_exhausted",
                function=func.__name__,
                ticker=ticker,
                attempts=config.max_attempts,
                error=str(last_exception),
                retryable=False,
            )

            if fallback_value is not None:
                return fallback_value
            else:
                raise last_exception

        return wrapper
    return decorator


# Pre-configured retry decorators for common use cases

def retry_api_call(func: Callable) -> Callable:
    """Retry for general API calls (3 attempts, 1s -> 2s -> 4s)."""
    return with_retry(
        RetryConfig(
            max_attempts=3,
            backoff_factor=2.0,
            initial_delay=1.0,
            retry_on=(ConnectionError, TimeoutError, Exception)
        )
    )(func)


def retry_web_search(func: Callable) -> Callable:
    """Retry for web searches (2 attempts, faster backoff)."""
    return with_retry(
        RetryConfig(
            max_attempts=2,
            backoff_factor=1.5,
            initial_delay=0.5,
            retry_on=(ConnectionError, TimeoutError)
        )
    )(func)


def retry_yfinance(func: Callable) -> Callable:
    """Retry for yfinance calls (4 attempts, aggressive backoff for rate limits)."""
    return with_retry(
        RetryConfig(
            max_attempts=4,
            backoff_factor=3.0,
            initial_delay=2.0,
            max_delay=30.0,
            retry_on=(Exception,)  # yfinance can raise various exceptions
        )
    )(func)
