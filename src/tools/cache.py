"""
Caching layer for API calls with TTL and rate limiting.

Provides:
- Thread-safe TTL-based caching for API responses
- Rate limiting to prevent API abuse
- Decorator for easy caching integration
"""

import time
import threading
from typing import Any, Optional, Dict, Callable, List
from functools import wraps
from datetime import datetime, timedelta


class SentimentCache:
    """Thread-safe cache with TTL for sentiment and news data."""

    def __init__(self, default_ttl_seconds: int = 900):
        """
        Initialize cache.

        Args:
            default_ttl_seconds: Default time-to-live in seconds (default: 15 min)
        """
        self._cache: Dict[str, tuple] = {}  # key -> (value, expiry_time)
        self._lock = threading.RLock()
        self._default_ttl = default_ttl_seconds

    def get(self, key: str) -> Optional[Any]:
        """
        Get cached value if not expired.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            if key in self._cache:
                value, expiry = self._cache[key]
                if datetime.now() < expiry:
                    return value
                else:
                    del self._cache[key]
        return None

    def set(self, key: str, value: Any, ttl_seconds: int = None) -> None:
        """
        Set cache value with TTL.

        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time-to-live in seconds (uses default if None)
        """
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
        with self._lock:
            self._cache[key] = (value, datetime.now() + timedelta(seconds=ttl))

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()

    def cleanup_expired(self) -> int:
        """
        Remove expired entries from cache.

        Returns:
            Number of entries removed
        """
        removed = 0
        now = datetime.now()
        with self._lock:
            expired_keys = [
                key for key, (_, expiry) in self._cache.items()
                if now >= expiry
            ]
            for key in expired_keys:
                del self._cache[key]
                removed += 1
        return removed

    def size(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self._cache)


class RateLimiter:
    """Rate limiter to prevent API abuse."""

    def __init__(self, calls_per_minute: int = 30, calls_per_day: int = None):
        """
        Initialize rate limiter.

        Args:
            calls_per_minute: Maximum calls allowed per minute
            calls_per_day: Maximum calls allowed per day (optional)
        """
        self._calls_per_minute = calls_per_minute
        self._calls_per_day = calls_per_day
        self._minute_calls: List[datetime] = []
        self._day_calls: List[datetime] = []
        self._lock = threading.RLock()

    def can_proceed(self) -> bool:
        """
        Check if a call can proceed without blocking.

        Returns:
            True if within rate limits
        """
        with self._lock:
            now = datetime.now()

            # Clean old minute calls
            minute_ago = now - timedelta(minutes=1)
            self._minute_calls = [c for c in self._minute_calls if c > minute_ago]

            # Check minute limit
            if len(self._minute_calls) >= self._calls_per_minute:
                return False

            # Check daily limit if set
            if self._calls_per_day:
                day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
                self._day_calls = [c for c in self._day_calls if c > day_start]
                if len(self._day_calls) >= self._calls_per_day:
                    return False

            return True

    def acquire(self, blocking: bool = True, timeout: float = 60.0) -> bool:
        """
        Acquire permission to make a call.

        Args:
            blocking: If True, wait until rate limit allows; if False, return immediately
            timeout: Maximum time to wait in seconds (only if blocking=True)

        Returns:
            True if permission acquired, False if timed out or non-blocking and limited
        """
        start_time = time.time()

        while True:
            with self._lock:
                now = datetime.now()

                # Clean old minute calls
                minute_ago = now - timedelta(minutes=1)
                self._minute_calls = [c for c in self._minute_calls if c > minute_ago]

                # Clean old day calls
                if self._calls_per_day:
                    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
                    self._day_calls = [c for c in self._day_calls if c > day_start]

                # Check limits
                minute_ok = len(self._minute_calls) < self._calls_per_minute
                day_ok = True
                if self._calls_per_day:
                    day_ok = len(self._day_calls) < self._calls_per_day

                if minute_ok and day_ok:
                    self._minute_calls.append(now)
                    if self._calls_per_day:
                        self._day_calls.append(now)
                    return True

            # Not blocking - return immediately
            if not blocking:
                return False

            # Check timeout
            if time.time() - start_time > timeout:
                return False

            # Wait a bit before retrying
            time.sleep(1.0)

    def get_remaining_minute(self) -> int:
        """Get remaining calls available this minute."""
        with self._lock:
            now = datetime.now()
            minute_ago = now - timedelta(minutes=1)
            self._minute_calls = [c for c in self._minute_calls if c > minute_ago]
            return max(0, self._calls_per_minute - len(self._minute_calls))

    def get_remaining_day(self) -> Optional[int]:
        """Get remaining calls available today (if daily limit set)."""
        if not self._calls_per_day:
            return None
        with self._lock:
            now = datetime.now()
            day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            self._day_calls = [c for c in self._day_calls if c > day_start]
            return max(0, self._calls_per_day - len(self._day_calls))


def cached(cache: SentimentCache, key_fn: Callable, ttl: int = None):
    """
    Decorator to cache function results.

    Args:
        cache: SentimentCache instance to use
        key_fn: Function to generate cache key from args/kwargs
        ttl: Time-to-live in seconds (uses cache default if None)

    Example:
        @cached(my_cache, lambda ticker: f"news_{ticker}", ttl=1800)
        def get_news(ticker: str) -> str:
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = key_fn(*args, **kwargs)
            result = cache.get(key)
            if result is not None:
                return result
            result = func(*args, **kwargs)
            cache.set(key, result, ttl)
            return result
        return wrapper
    return decorator


# Global cache instances for different data types
news_cache = SentimentCache(default_ttl_seconds=1800)  # 30 min for news
sentiment_cache = SentimentCache(default_ttl_seconds=900)  # 15 min for sentiment
social_cache = SentimentCache(default_ttl_seconds=1800)  # 30 min for social

# Global rate limiters for different APIs
newsapi_limiter = RateLimiter(calls_per_minute=10, calls_per_day=100)  # NewsAPI free tier
twitter_limiter = RateLimiter(calls_per_minute=15)  # Conservative for Nitter
reddit_limiter = RateLimiter(calls_per_minute=30)  # Reddit API
google_rss_limiter = RateLimiter(calls_per_minute=10)  # Google News RSS
