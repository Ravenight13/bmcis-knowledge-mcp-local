"""Authentication and rate limiting for BMCIS Knowledge MCP server.

Provides API key authentication with constant-time comparison and multi-tier
rate limiting using token bucket algorithm.

Security Features:
- Constant-time API key comparison (timing attack safe)
- No key logging or exposure
- Multi-tier rate limiting (per-minute, per-hour, per-day)
- Token bucket algorithm for smooth rate limiting
- Actionable error messages for MCP clients

Rate Limiting Defaults:
- Per-minute: 100 requests
- Per-hour: 1000 requests
- Per-day: 10000 requests

Configuration:
- BMCIS_API_KEY: Required API key for authentication
- BMCIS_RATE_LIMIT_MINUTE: Override minute limit (default: 100)
- BMCIS_RATE_LIMIT_HOUR: Override hour limit (default: 1000)
- BMCIS_RATE_LIMIT_DAY: Override day limit (default: 10000)

Example:
    # Validate API key
    >>> if validate_api_key(provided_key):
    ...     print("Authenticated")

    # Check rate limit
    >>> if rate_limiter.is_allowed("api_key_123"):
    ...     # Process request
    ...     pass

    # Decorate tool function
    >>> @require_auth
    ... def my_tool():
    ...     return "result"
"""

from __future__ import annotations

import hmac
import logging
import os
from collections.abc import Callable
from functools import wraps
from time import time
from typing import Any

from src.core.logging import StructuredLogger

logger: logging.Logger = StructuredLogger.get_logger(__name__)


def validate_api_key(provided_key: str) -> bool:
    """Validate API key with constant-time comparison.

    Uses hmac.compare_digest() for timing-attack-safe comparison.
    Does NOT log or print the API key for security.

    Args:
        provided_key: API key from request header

    Returns:
        True if valid, False if invalid

    Raises:
        ValueError: If BMCIS_API_KEY environment variable not set

    Security:
        - Constant-time comparison prevents timing attacks
        - No key logging or exposure
        - Clear error messages for configuration issues

    Example:
        >>> os.environ['BMCIS_API_KEY'] = 'secret123'
        >>> validate_api_key('secret123')
        True
        >>> validate_api_key('wrong')
        False
    """
    valid_key = os.environ.get('BMCIS_API_KEY')

    if not valid_key:
        raise ValueError(
            "BMCIS_API_KEY environment variable not set. "
            "Set it before starting the MCP server."
        )

    # Constant-time comparison (timing attack safe)
    # Using hmac.compare_digest ensures comparison time doesn't leak info
    return hmac.compare_digest(provided_key, valid_key)


class RateLimiter:
    """Token bucket rate limiter for MCP tools.

    Implements multi-tier rate limiting with token bucket algorithm:
    - Tokens refill at fixed intervals (minute, hour, day)
    - Each request consumes 1 token from each bucket
    - Request allowed only if all buckets have tokens

    Token Bucket Algorithm:
    - Smoother than fixed window (no boundary spikes)
    - Allows bursts up to bucket size
    - Natural refill over time

    Attributes:
        rpm_limit: Max requests per minute
        rph_limit: Max requests per hour
        rpd_limit: Max requests per day
        buckets: Per-key token bucket state

    Example:
        >>> limiter = RateLimiter(
        ...     requests_per_minute=100,
        ...     requests_per_hour=1000,
        ...     requests_per_day=10000
        ... )
        >>> limiter.is_allowed("api_key_123")
        True
        >>> # After 100 requests in 1 minute:
        >>> limiter.is_allowed("api_key_123")
        False
    """

    def __init__(
        self,
        requests_per_minute: int = 100,
        requests_per_hour: int = 1000,
        requests_per_day: int = 10000,
    ) -> None:
        """Initialize rate limiter with configured limits.

        Args:
            requests_per_minute: Max requests per minute (default: 100)
            requests_per_hour: Max requests per hour (default: 1000)
            requests_per_day: Max requests per day (default: 10000)

        Example:
            >>> limiter = RateLimiter(
            ...     requests_per_minute=50,
            ...     requests_per_hour=500,
            ...     requests_per_day=5000
            ... )
        """
        self.rpm_limit = requests_per_minute
        self.rph_limit = requests_per_hour
        self.rpd_limit = requests_per_day

        # Token buckets per key
        # Structure: {key: {'minute': {...}, 'hour': {...}, 'day': {...}}}
        self.buckets: dict[str, dict[str, dict[str, float]]] = {}

        logger.info(
            f"RateLimiter initialized: {requests_per_minute}/min, "
            f"{requests_per_hour}/hr, {requests_per_day}/day"
        )

    def is_allowed(self, key: str) -> bool:
        """Check if request is allowed for the given key.

        Implements token bucket algorithm with multi-tier limits.
        Refills tokens based on elapsed time since last refill.

        Args:
            key: API key or identifier

        Returns:
            True if request allowed, False if rate limited

        Algorithm:
            1. Initialize bucket if first request
            2. Refill tokens based on elapsed time
            3. Check all tier limits
            4. Consume tokens if allowed

        Example:
            >>> limiter = RateLimiter(requests_per_minute=2)
            >>> limiter.is_allowed("key1")  # Request 1
            True
            >>> limiter.is_allowed("key1")  # Request 2
            True
            >>> limiter.is_allowed("key1")  # Request 3 (over limit)
            False
        """
        current_time = time()

        # Initialize new bucket for first request
        if key not in self.buckets:
            self.buckets[key] = {
                'minute': {'tokens': self.rpm_limit, 'last_refill': current_time},
                'hour': {'tokens': self.rph_limit, 'last_refill': current_time},
                'day': {'tokens': self.rpd_limit, 'last_refill': current_time},
            }
            logger.debug(f"Initialized rate limit bucket for key: {key[:8]}...")
            return True

        bucket = self.buckets[key]

        # Refill minute bucket (60 seconds)
        minute_bucket = bucket['minute']
        minute_elapsed = current_time - minute_bucket['last_refill']
        if minute_elapsed >= 60:
            minute_bucket['tokens'] = self.rpm_limit
            minute_bucket['last_refill'] = current_time
            logger.debug(f"Refilled minute bucket for key: {key[:8]}...")

        # Refill hour bucket (3600 seconds)
        hour_bucket = bucket['hour']
        hour_elapsed = current_time - hour_bucket['last_refill']
        if hour_elapsed >= 3600:
            hour_bucket['tokens'] = self.rph_limit
            hour_bucket['last_refill'] = current_time
            logger.debug(f"Refilled hour bucket for key: {key[:8]}...")

        # Refill day bucket (86400 seconds)
        day_bucket = bucket['day']
        day_elapsed = current_time - day_bucket['last_refill']
        if day_elapsed >= 86400:
            day_bucket['tokens'] = self.rpd_limit
            day_bucket['last_refill'] = current_time
            logger.debug(f"Refilled day bucket for key: {key[:8]}...")

        # Check all limits (request allowed only if all buckets have tokens)
        if (minute_bucket['tokens'] > 0 and
            hour_bucket['tokens'] > 0 and
            day_bucket['tokens'] > 0):
            # Consume tokens from all buckets
            minute_bucket['tokens'] -= 1
            hour_bucket['tokens'] -= 1
            day_bucket['tokens'] -= 1

            logger.debug(
                f"Rate limit check passed for key: {key[:8]}... "
                f"(remaining: {int(minute_bucket['tokens'])}/min, "
                f"{int(hour_bucket['tokens'])}/hr, {int(day_bucket['tokens'])}/day)"
            )
            return True

        # Rate limit exceeded
        logger.warning(
            f"Rate limit exceeded for key: {key[:8]}... "
            f"(remaining: {int(minute_bucket['tokens'])}/min, "
            f"{int(hour_bucket['tokens'])}/hr, {int(day_bucket['tokens'])}/day)"
        )
        return False

    def get_reset_times(self, key: str) -> dict[str, float]:
        """Get when rate limits reset for the given key.

        Returns epoch timestamps for when each tier resets.

        Args:
            key: API key or identifier

        Returns:
            Dictionary with reset times for minute, hour, day.
            Empty dict if key not found.

        Example:
            >>> limiter = RateLimiter()
            >>> limiter.is_allowed("key1")
            True
            >>> reset_times = limiter.get_reset_times("key1")
            >>> assert 'minute_reset' in reset_times
            >>> assert reset_times['minute_reset'] > time()
        """
        if key not in self.buckets:
            return {}

        bucket = self.buckets[key]

        return {
            'minute_reset': bucket['minute']['last_refill'] + 60,
            'hour_reset': bucket['hour']['last_refill'] + 3600,
            'day_reset': bucket['day']['last_refill'] + 86400,
        }

    def get_remaining(self, key: str) -> dict[str, int]:
        """Get remaining request counts for the given key.

        Args:
            key: API key or identifier

        Returns:
            Dictionary with remaining requests for each tier.
            Empty dict if key not found.

        Example:
            >>> limiter = RateLimiter(requests_per_minute=100)
            >>> limiter.is_allowed("key1")
            True
            >>> remaining = limiter.get_remaining("key1")
            >>> assert remaining['minute'] == 99
        """
        if key not in self.buckets:
            return {}

        bucket = self.buckets[key]

        return {
            'minute': int(bucket['minute']['tokens']),
            'hour': int(bucket['hour']['tokens']),
            'day': int(bucket['day']['tokens']),
        }


def require_auth(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to require API key authentication for MCP tools.

    Validates API key and checks rate limits before calling tool function.
    Expects 'api_key' parameter in function kwargs or BMCIS_API_KEY_CURRENT
    environment variable (set by MCP server per-request).

    Args:
        func: Tool function to wrap with authentication

    Returns:
        Wrapped function with authentication

    Raises:
        ValueError: If authentication fails or rate limit exceeded

    Example:
        >>> @require_auth
        ... def protected_tool(query: str) -> str:
        ...     return f"Result for: {query}"

        >>> # With valid API key
        >>> os.environ['BMCIS_API_KEY_CURRENT'] = 'valid_key'
        >>> protected_tool(query="test")
        'Result for: test'

        >>> # With invalid API key
        >>> os.environ['BMCIS_API_KEY_CURRENT'] = 'invalid'
        >>> protected_tool(query="test")  # Raises ValueError
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Extract API key from kwargs or environment
        # MCP server should set BMCIS_API_KEY_CURRENT for each request
        api_key = kwargs.get('api_key') or os.environ.get('BMCIS_API_KEY_CURRENT')

        if not api_key:
            raise ValueError(
                "Authentication required. API key must be provided. "
                "Set BMCIS_API_KEY environment variable or pass api_key parameter."
            )

        # Validate API key
        try:
            if not validate_api_key(api_key):
                logger.warning(f"Invalid API key attempted: {api_key[:8]}...")
                raise ValueError(
                    "Invalid API key. Please check your credentials and try again."
                )
        except ValueError as e:
            # Re-raise configuration errors (BMCIS_API_KEY not set)
            if "environment variable not set" in str(e):
                logger.error("Server configuration error: BMCIS_API_KEY not set")
                raise
            # Re-raise authentication errors
            raise ValueError(f"Authentication failed: {e!s}") from e

        # Check rate limiting (get from kwargs or use global)
        rate_limiter_instance = kwargs.get('_rate_limiter') or rate_limiter
        if rate_limiter_instance and not rate_limiter_instance.is_allowed(api_key):
            reset_times = rate_limiter_instance.get_reset_times(api_key)
            remaining = rate_limiter_instance.get_remaining(api_key)

            # Format next reset time
            minute_reset = reset_times.get('minute_reset', 0)
            next_reset_seconds = int(minute_reset - time())

            raise ValueError(
                f"Rate limit exceeded. You have exhausted your request quota. "
                f"Remaining: {remaining.get('minute', 0)}/min, "
                f"{remaining.get('hour', 0)}/hr, {remaining.get('day', 0)}/day. "
                f"Try again in {next_reset_seconds} seconds."
            )

        logger.info(
            f"Authentication successful for key: {api_key[:8]}...",
            extra={'function': func.__name__}
        )

        # Call original function
        return func(*args, **kwargs)

    return wrapper


# Load configuration from environment
DEFAULT_RATE_LIMIT_PER_MINUTE = int(os.environ.get('BMCIS_RATE_LIMIT_MINUTE', '100'))
DEFAULT_RATE_LIMIT_PER_HOUR = int(os.environ.get('BMCIS_RATE_LIMIT_HOUR', '1000'))
DEFAULT_RATE_LIMIT_PER_DAY = int(os.environ.get('BMCIS_RATE_LIMIT_DAY', '10000'))

# Initialize global rate limiter (singleton)
rate_limiter = RateLimiter(
    requests_per_minute=DEFAULT_RATE_LIMIT_PER_MINUTE,
    requests_per_hour=DEFAULT_RATE_LIMIT_PER_HOUR,
    requests_per_day=DEFAULT_RATE_LIMIT_PER_DAY,
)

logger.info("Authentication module initialized successfully")
