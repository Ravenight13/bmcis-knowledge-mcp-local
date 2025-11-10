"""Circuit breaker pattern for handling embedding generation failures.

Implements a state machine (CLOSED → OPEN → HALF_OPEN) to prevent cascading
failures when embedding models fail. Tracks consecutive failures and automatically
recovers when the service becomes healthy again.

Thread-safe for concurrent access in multi-threaded embedding generation.

Module exports:
    - CircuitState: Enum of circuit breaker states
    - CircuitBreakerConfig: Configuration dataclass
    - CircuitBreaker: State machine implementation
    - CircuitBreakerError: Exception raised when circuit is open
"""

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Final

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states for handling embedding generation failures.

    States:
        CLOSED: Normal operation, requests pass through
        OPEN: Failures detected, requests fail fast
        HALF_OPEN: Testing if service recovered
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open and request is rejected."""

    def __init__(self, message: str) -> None:
        """Initialize with error message.

        Args:
            message: Error description.
        """
        self.message = message
        super().__init__(message)


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior.

    Attributes:
        failure_threshold: Consecutive failures before opening (default: 5).
        success_threshold: Consecutive successes before closing (default: 2).
        timeout_seconds: Time before attempting recovery from OPEN (default: 60).
        reset_interval_seconds: Automatic reset interval (default: 300).
    """

    failure_threshold: int = 5
    success_threshold: int = 2
    timeout_seconds: float = 60.0
    reset_interval_seconds: float = 300.0

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if self.success_threshold < 1:
            raise ValueError("success_threshold must be >= 1")
        if self.timeout_seconds < 0.1:
            raise ValueError("timeout_seconds must be >= 0.1")
        if self.reset_interval_seconds < 0.1:
            raise ValueError("reset_interval_seconds must be >= 0.1")


class CircuitBreaker:
    """Prevents cascading failures in embedding generation.

    Why it exists:
        - Embedding model can fail (out of memory, network issues)
        - Cascading requests worsen the problem
        - Need graceful degradation with automatic recovery

    What it does:
        - Tracks consecutive failures
        - Opens circuit after threshold to fail fast
        - Enters HALF_OPEN to test recovery
        - Automatically closes after timeout
        - Thread-safe for concurrent access

    State transitions:
        - CLOSED -> OPEN: failure_threshold consecutive failures
        - OPEN -> HALF_OPEN: timeout_seconds elapsed
        - HALF_OPEN -> CLOSED: success_threshold consecutive successes
        - HALF_OPEN -> OPEN: First failure
    """

    def __init__(self, config: CircuitBreakerConfig | None = None) -> None:
        """Initialize circuit breaker with configuration.

        Args:
            config: CircuitBreakerConfig. Uses defaults if None.
        """
        self.config: CircuitBreakerConfig = config or CircuitBreakerConfig()

        # State management
        self._state: CircuitState = CircuitState.CLOSED
        self._failure_count: int = 0
        self._success_count: int = 0
        self._last_failure_time: datetime | None = None
        self._last_state_change: datetime = datetime.now()

        # Thread safety
        self._lock: threading.RLock = threading.RLock()

        logger.info(
            "CircuitBreaker initialized",
            extra={
                "failure_threshold": self.config.failure_threshold,
                "success_threshold": self.config.success_threshold,
                "timeout_seconds": self.config.timeout_seconds,
            },
        )

    def is_open(self) -> bool:
        """Check if circuit is open (rejecting requests).

        Returns:
            True if circuit is OPEN, False otherwise.
        """
        with self._lock:
            # Check if OPEN state should transition to HALF_OPEN
            if self._state == CircuitState.OPEN:
                if self._should_attempt_recovery():
                    self._transition_to_half_open()
                    return False

            return self._state == CircuitState.OPEN

    def record_success(self) -> None:
        """Record successful embedding generation.

        Increments success counter when in HALF_OPEN state.
        Transitions to CLOSED when success_threshold reached.
        In CLOSED state, resets failure counter.
        """
        with self._lock:
            if self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0
                self._success_count = 0
                logger.debug("CircuitBreaker: success in CLOSED state")

            elif self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                logger.debug(
                    f"CircuitBreaker: success in HALF_OPEN state "
                    f"({self._success_count}/{self.config.success_threshold})"
                )

                if self._success_count >= self.config.success_threshold:
                    self._transition_to_closed()

            elif self._state == CircuitState.OPEN:
                # Ignore successes in OPEN state
                logger.debug("CircuitBreaker: ignoring success in OPEN state")

    def record_failure(self) -> None:
        """Record failed embedding generation.

        Increments failure counter and transitions to OPEN when
        failure_threshold reached (in CLOSED state).
        Transitions back to OPEN if failure occurs in HALF_OPEN state.
        """
        with self._lock:
            self._last_failure_time = datetime.now()

            if self._state == CircuitState.CLOSED:
                self._failure_count += 1
                logger.warning(
                    f"CircuitBreaker: failure in CLOSED state "
                    f"({self._failure_count}/{self.config.failure_threshold})"
                )

                if self._failure_count >= self.config.failure_threshold:
                    self._transition_to_open()

            elif self._state == CircuitState.HALF_OPEN:
                logger.warning("CircuitBreaker: failure in HALF_OPEN state, reopening")
                self._transition_to_open()

            elif self._state == CircuitState.OPEN:
                # Just track another failure
                logger.debug("CircuitBreaker: failure in OPEN state")

    def reset(self) -> None:
        """Manually reset circuit breaker to CLOSED state.

        Clears all failure/success counters and timestamps.
        Useful for testing or manual recovery.
        """
        with self._lock:
            old_state = self._state
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
            self._last_state_change = datetime.now()

            logger.info(f"CircuitBreaker reset from {old_state.value} to CLOSED")

    def get_state(self) -> CircuitState:
        """Get current circuit state.

        Returns:
            Current CircuitState.
        """
        with self._lock:
            return self._state

    def get_metrics(self) -> dict[str, str | int | float]:
        """Get circuit breaker metrics for monitoring.

        Returns:
            Dictionary with state, counters, and timing information.
        """
        with self._lock:
            now = datetime.now()
            time_in_state = (now - self._last_state_change).total_seconds()

            return {
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "last_failure_time": (
                    self._last_failure_time.isoformat()
                    if self._last_failure_time
                    else None
                ),
                "time_in_state_seconds": time_in_state,
            }

    def _should_attempt_recovery(self) -> bool:
        """Check if enough time has passed to attempt recovery.

        Returns:
            True if timeout has elapsed, False otherwise.
        """
        if self._last_failure_time is None:
            return False

        elapsed = (datetime.now() - self._last_failure_time).total_seconds()
        return elapsed >= self.config.timeout_seconds

    def _transition_to_open(self) -> None:
        """Transition circuit state from CLOSED to OPEN.

        Logs transition and updates state change timestamp.
        """
        old_state = self._state
        self._state = CircuitState.OPEN
        self._last_state_change = datetime.now()
        self._failure_count = 0
        self._success_count = 0

        logger.warning(
            f"CircuitBreaker transitioned from {old_state.value} to OPEN",
            extra={"timeout_seconds": self.config.timeout_seconds},
        )

    def _transition_to_half_open(self) -> None:
        """Transition circuit state from OPEN to HALF_OPEN.

        Tests if service has recovered. Logs transition and resets counters.
        """
        self._state = CircuitState.HALF_OPEN
        self._last_state_change = datetime.now()
        self._failure_count = 0
        self._success_count = 0

        logger.info(
            "CircuitBreaker transitioned from OPEN to HALF_OPEN, testing recovery"
        )

    def _transition_to_closed(self) -> None:
        """Transition circuit state from HALF_OPEN to CLOSED.

        Service is healthy again. Logs transition and resets counters.
        """
        self._state = CircuitState.CLOSED
        self._last_state_change = datetime.now()
        self._failure_count = 0
        self._success_count = 0

        logger.info("CircuitBreaker transitioned from HALF_OPEN to CLOSED")
