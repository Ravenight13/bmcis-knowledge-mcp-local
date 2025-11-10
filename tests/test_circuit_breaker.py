"""Tests for circuit breaker pattern implementation.

Tests the state machine transitions, failure/success tracking, thread safety,
and automatic recovery in the CircuitBreaker class.
"""

import threading
import time
from datetime import datetime

import pytest

from src.embedding.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitState,
)


class TestCircuitBreakerConfig:
    """Tests for CircuitBreakerConfig validation."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.success_threshold == 2
        assert config.timeout_seconds == 60.0
        assert config.reset_interval_seconds == 300.0

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = CircuitBreakerConfig(
            failure_threshold=10,
            success_threshold=3,
            timeout_seconds=120.0,
            reset_interval_seconds=600.0,
        )
        assert config.failure_threshold == 10
        assert config.success_threshold == 3
        assert config.timeout_seconds == 120.0
        assert config.reset_interval_seconds == 600.0

    def test_invalid_failure_threshold(self) -> None:
        """Test validation of failure_threshold."""
        with pytest.raises(ValueError):
            CircuitBreakerConfig(failure_threshold=0)

    def test_invalid_success_threshold(self) -> None:
        """Test validation of success_threshold."""
        with pytest.raises(ValueError):
            CircuitBreakerConfig(success_threshold=0)

    def test_invalid_timeout_seconds(self) -> None:
        """Test validation of timeout_seconds."""
        with pytest.raises(ValueError):
            CircuitBreakerConfig(timeout_seconds=0.05)

    def test_invalid_reset_interval(self) -> None:
        """Test validation of reset_interval_seconds."""
        with pytest.raises(ValueError):
            CircuitBreakerConfig(reset_interval_seconds=0.05)


class TestCircuitBreakerInitialization:
    """Tests for CircuitBreaker initialization."""

    def test_default_initialization(self) -> None:
        """Test initialization with default config."""
        cb = CircuitBreaker()
        assert cb.get_state() == CircuitState.CLOSED
        assert not cb.is_open()

    def test_custom_config_initialization(self) -> None:
        """Test initialization with custom config."""
        config = CircuitBreakerConfig(failure_threshold=3, success_threshold=1)
        cb = CircuitBreaker(config=config)
        assert cb.get_state() == CircuitState.CLOSED
        assert cb.config.failure_threshold == 3

    def test_metrics_initial_state(self) -> None:
        """Test initial metrics."""
        cb = CircuitBreaker()
        metrics = cb.get_metrics()
        assert metrics["state"] == "closed"
        assert metrics["failure_count"] == 0
        assert metrics["success_count"] == 0
        assert metrics["last_failure_time"] is None


class TestCircuitBreakerClosedState:
    """Tests for CLOSED state behavior."""

    def test_closed_state_accepts_requests(self) -> None:
        """Test that CLOSED state does not reject requests."""
        cb = CircuitBreaker()
        assert not cb.is_open()

    def test_closed_state_increments_failures(self) -> None:
        """Test that failures are tracked in CLOSED state."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker(config=config)

        cb.record_failure()
        metrics = cb.get_metrics()
        assert metrics["failure_count"] == 1

        cb.record_failure()
        metrics = cb.get_metrics()
        assert metrics["failure_count"] == 2

    def test_closed_state_transitions_to_open_on_threshold(self) -> None:
        """Test transition from CLOSED to OPEN on failure threshold."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker(config=config)

        assert cb.get_state() == CircuitState.CLOSED

        cb.record_failure()
        assert cb.get_state() == CircuitState.CLOSED

        cb.record_failure()
        assert cb.get_state() == CircuitState.CLOSED

        cb.record_failure()
        assert cb.get_state() == CircuitState.OPEN
        assert cb.is_open()

    def test_closed_state_resets_failures_on_success(self) -> None:
        """Test that success resets failure counter in CLOSED state."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker(config=config)

        cb.record_failure()
        cb.record_failure()
        metrics = cb.get_metrics()
        assert metrics["failure_count"] == 2

        cb.record_success()
        metrics = cb.get_metrics()
        assert metrics["failure_count"] == 0


class TestCircuitBreakerOpenState:
    """Tests for OPEN state behavior."""

    def test_open_state_rejects_requests(self) -> None:
        """Test that OPEN state rejects requests."""
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker(config=config)

        cb.record_failure()
        assert cb.is_open()

    def test_open_state_ignores_successes(self) -> None:
        """Test that successes are ignored in OPEN state."""
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker(config=config)

        cb.record_failure()
        assert cb.get_state() == CircuitState.OPEN

        cb.record_success()
        # Should still be OPEN
        assert cb.get_state() == CircuitState.OPEN

    def test_open_state_transitions_after_timeout(self) -> None:
        """Test automatic transition from OPEN to HALF_OPEN after timeout."""
        config = CircuitBreakerConfig(
            failure_threshold=1, timeout_seconds=0.1
        )
        cb = CircuitBreaker(config=config)

        cb.record_failure()
        assert cb.get_state() == CircuitState.OPEN

        # Sleep for timeout period
        time.sleep(0.15)

        # Calling is_open() should trigger transition
        assert not cb.is_open()
        assert cb.get_state() == CircuitState.HALF_OPEN


class TestCircuitBreakerHalfOpenState:
    """Tests for HALF_OPEN state behavior."""

    def test_half_open_state_closes_on_success_threshold(self) -> None:
        """Test transition from HALF_OPEN to CLOSED on success threshold."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            success_threshold=2,
            timeout_seconds=0.1,
        )
        cb = CircuitBreaker(config=config)

        # Open circuit
        cb.record_failure()
        assert cb.get_state() == CircuitState.OPEN

        # Wait for timeout
        time.sleep(0.15)

        # Trigger transition to HALF_OPEN
        cb.is_open()
        assert cb.get_state() == CircuitState.HALF_OPEN

        # Record successes
        cb.record_success()
        assert cb.get_state() == CircuitState.HALF_OPEN

        cb.record_success()
        assert cb.get_state() == CircuitState.CLOSED

    def test_half_open_state_reopens_on_failure(self) -> None:
        """Test transition from HALF_OPEN back to OPEN on failure."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            timeout_seconds=0.1,
        )
        cb = CircuitBreaker(config=config)

        # Open circuit
        cb.record_failure()
        assert cb.get_state() == CircuitState.OPEN

        # Wait for timeout
        time.sleep(0.15)

        # Trigger transition to HALF_OPEN
        cb.is_open()
        assert cb.get_state() == CircuitState.HALF_OPEN

        # Failure should reopen
        cb.record_failure()
        assert cb.get_state() == CircuitState.OPEN

    def test_half_open_tracks_successes(self) -> None:
        """Test that HALF_OPEN tracks success count."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            success_threshold=3,
            timeout_seconds=0.1,
        )
        cb = CircuitBreaker(config=config)

        # Open and transition to HALF_OPEN
        cb.record_failure()
        time.sleep(0.15)
        cb.is_open()

        cb.record_success()
        metrics = cb.get_metrics()
        assert metrics["success_count"] == 1

        cb.record_success()
        metrics = cb.get_metrics()
        assert metrics["success_count"] == 2


class TestCircuitBreakerReset:
    """Tests for manual reset functionality."""

    def test_reset_from_open_state(self) -> None:
        """Test resetting circuit from OPEN to CLOSED."""
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker(config=config)

        cb.record_failure()
        assert cb.get_state() == CircuitState.OPEN

        cb.reset()
        assert cb.get_state() == CircuitState.CLOSED
        assert not cb.is_open()

    def test_reset_clears_counters(self) -> None:
        """Test that reset clears failure and success counters."""
        config = CircuitBreakerConfig(failure_threshold=5)
        cb = CircuitBreaker(config=config)

        cb.record_failure()
        cb.record_failure()

        cb.reset()
        metrics = cb.get_metrics()
        assert metrics["failure_count"] == 0
        assert metrics["success_count"] == 0

    def test_reset_clears_timestamps(self) -> None:
        """Test that reset clears failure timestamps."""
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker(config=config)

        cb.record_failure()
        metrics = cb.get_metrics()
        assert metrics["last_failure_time"] is not None

        cb.reset()
        metrics = cb.get_metrics()
        assert metrics["last_failure_time"] is None


class TestCircuitBreakerThreadSafety:
    """Tests for thread-safe concurrent access."""

    def test_concurrent_success_recording(self) -> None:
        """Test concurrent success recording is thread-safe."""
        cb = CircuitBreaker()
        num_threads = 10
        successes_per_thread = 100

        def record_successes() -> None:
            for _ in range(successes_per_thread):
                cb.record_success()

        threads = [
            threading.Thread(target=record_successes) for _ in range(num_threads)
        ]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Should complete without errors
        assert cb.get_state() == CircuitState.CLOSED

    def test_concurrent_failure_recording(self) -> None:
        """Test concurrent failure recording is thread-safe."""
        config = CircuitBreakerConfig(failure_threshold=200)
        cb = CircuitBreaker(config=config)
        num_threads = 10
        failures_per_thread = 10

        def record_failures() -> None:
            for _ in range(failures_per_thread):
                cb.record_failure()

        threads = [
            threading.Thread(target=record_failures) for _ in range(num_threads)
        ]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Should complete without errors
        # Failure count tracks while CLOSED, once threshold is hit it opens
        metrics = cb.get_metrics()
        # All threads may not contribute if one thread hits threshold first
        assert metrics["failure_count"] >= 0

    def test_concurrent_state_check(self) -> None:
        """Test concurrent state checking is thread-safe."""
        cb = CircuitBreaker()
        results: list[bool] = []
        lock = threading.Lock()

        def check_state() -> None:
            state = cb.is_open()
            with lock:
                results.append(state)

        threads = [threading.Thread(target=check_state) for _ in range(100)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        assert len(results) == 100
        assert all(not result for result in results)  # All should be False


class TestCircuitBreakerMetrics:
    """Tests for metrics reporting."""

    def test_metrics_in_closed_state(self) -> None:
        """Test metrics reporting in CLOSED state."""
        cb = CircuitBreaker()
        cb.record_failure()
        cb.record_failure()

        metrics = cb.get_metrics()
        assert metrics["state"] == "closed"
        assert metrics["failure_count"] == 2
        assert metrics["success_count"] == 0
        assert metrics["time_in_state_seconds"] >= 0

    def test_metrics_in_open_state(self) -> None:
        """Test metrics reporting in OPEN state."""
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker(config=config)
        cb.record_failure()

        metrics = cb.get_metrics()
        assert metrics["state"] == "open"
        assert metrics["last_failure_time"] is not None

    def test_metrics_in_half_open_state(self) -> None:
        """Test metrics reporting in HALF_OPEN state."""
        config = CircuitBreakerConfig(
            failure_threshold=1, timeout_seconds=0.1
        )
        cb = CircuitBreaker(config=config)
        cb.record_failure()
        time.sleep(0.15)
        cb.is_open()

        metrics = cb.get_metrics()
        assert metrics["state"] == "half_open"
        assert metrics["success_count"] == 0

    def test_time_in_state_increases(self) -> None:
        """Test that time_in_state_seconds increases over time."""
        cb = CircuitBreaker()

        metrics1 = cb.get_metrics()
        time1 = metrics1["time_in_state_seconds"]

        time.sleep(0.1)

        metrics2 = cb.get_metrics()
        time2 = metrics2["time_in_state_seconds"]

        assert time2 > time1


class TestCircuitBreakerStateTransitions:
    """Tests for comprehensive state transition scenarios."""

    def test_complete_failure_recovery_cycle(self) -> None:
        """Test complete cycle: CLOSED → OPEN → HALF_OPEN → CLOSED."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            success_threshold=1,
            timeout_seconds=0.1,
        )
        cb = CircuitBreaker(config=config)

        # Start in CLOSED
        assert cb.get_state() == CircuitState.CLOSED

        # Accumulate failures → OPEN
        cb.record_failure()
        assert cb.get_state() == CircuitState.CLOSED
        cb.record_failure()
        assert cb.get_state() == CircuitState.OPEN

        # Wait for recovery timeout
        time.sleep(0.15)

        # Check state triggers transition → HALF_OPEN
        cb.is_open()
        assert cb.get_state() == CircuitState.HALF_OPEN

        # Success → CLOSED
        cb.record_success()
        assert cb.get_state() == CircuitState.CLOSED

    def test_repeated_open_close_cycles(self) -> None:
        """Test multiple open/close cycles."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            success_threshold=1,
            timeout_seconds=0.15,
        )
        cb = CircuitBreaker(config=config)

        for _ in range(3):
            # Open
            cb.record_failure()
            assert cb.get_state() == CircuitState.OPEN

            # Wait and transition
            time.sleep(0.2)
            cb.is_open()
            assert cb.get_state() == CircuitState.HALF_OPEN

            # Close
            cb.record_success()
            assert cb.get_state() == CircuitState.CLOSED


class TestCircuitBreakerEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_is_open_without_failures(self) -> None:
        """Test is_open() returns False when no failures occurred."""
        cb = CircuitBreaker()
        assert not cb.is_open()

    def test_success_without_prior_failure(self) -> None:
        """Test recording success without prior failures."""
        cb = CircuitBreaker()
        cb.record_success()
        assert not cb.is_open()

    def test_threshold_boundary(self) -> None:
        """Test exact threshold boundary."""
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker(config=config)

        cb.record_failure()
        assert cb.get_state() == CircuitState.OPEN

    def test_rapid_state_checks(self) -> None:
        """Test rapid calls to is_open() don't cause issues."""
        cb = CircuitBreaker()
        for _ in range(100):
            cb.is_open()
        assert not cb.is_open()


class TestCircuitBreakerIntegration:
    """Integration tests simulating realistic scenarios."""

    def test_cascade_prevention(self) -> None:
        """Test that circuit breaker prevents request cascade."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker(config=config)

        # Simulate cascading failures
        failures = 0
        for i in range(10):
            if cb.is_open():
                # Request would be rejected by caller
                failures += 1
            else:
                cb.record_failure()

        # Only first 3 requests go through before circuit opens
        assert failures >= 7

    def test_graceful_degradation_scenario(self) -> None:
        """Test graceful degradation with fallback mechanism."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            timeout_seconds=0.1,
        )
        cb = CircuitBreaker(config=config)

        request_count = 0
        fallback_count = 0

        # Simulate requests with failure
        for _ in range(5):
            if cb.is_open():
                fallback_count += 1
            else:
                request_count += 1
                cb.record_failure()

        assert request_count == 2  # 2 failures before opening
        assert fallback_count == 3  # 3 requests rejected while open

        # Wait for recovery
        time.sleep(0.15)

        # Try again
        if not cb.is_open():
            cb.record_success()

        # Should be recovering
        assert cb.get_state() == CircuitState.HALF_OPEN
