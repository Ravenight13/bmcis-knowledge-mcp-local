#!/usr/bin/env python3
"""Minimal test to verify generator context manager semantics."""

from contextlib import contextmanager
from typing import Generator

# Simulate the current implementation
@contextmanager
def get_connection_current() -> Generator[str, None, None]:
    """Current implementation with inner finally that clears tracking."""
    yielded_conn: str | None = None
    conn: str | None = None

    try:
        conn = "connection_from_pool"
        print(f"1. Acquired connection: {conn}")

        yielded_conn = conn
        print(f"2. Set yielded_conn = {yielded_conn}")

        try:
            print("3. About to yield...")
            yield conn
            print("4. Yield completed normally")
        finally:
            print("5. Inner finally: clearing yielded_conn")
            yielded_conn = None
            print(f"6. yielded_conn is now: {yielded_conn}")

        print("7. Returning from generator")
        return

    finally:
        print(f"8. Outer finally: yielded_conn = {yielded_conn}")
        if yielded_conn is not None:
            print(f"9. Returning connection to pool: {yielded_conn}")
        else:
            print(f"9. yielded_conn is None - NO CONNECTION RETURNED!")


# Test 1: Success case
print("=" * 60)
print("TEST 1: Success case (no exception)")
print("=" * 60)
try:
    with get_connection_current() as conn:
        print(f"  User code: received {conn}")
        print("  User code: no exception")
except Exception as e:
    print(f"  Exception: {e}")
print()

# Test 2: User exception case
print("=" * 60)
print("TEST 2: User exception case (RuntimeError)")
print("=" * 60)
try:
    with get_connection_current() as conn:
        print(f"  User code: received {conn}")
        print("  User code: raising RuntimeError")
        raise RuntimeError("Simulated user error")
except RuntimeError as e:
    print(f"  Exception caught: {e}")
print()

# Now test the fixed version
print("=" * 60)
print("FIXED IMPLEMENTATION")
print("=" * 60)

@contextmanager
def get_connection_fixed() -> Generator[str, None, None]:
    """Fixed implementation without clearing in inner finally."""
    yielded_conn: str | None = None
    conn: str | None = None

    try:
        conn = "connection_from_pool"
        print(f"1. Acquired connection: {conn}")

        yielded_conn = conn
        print(f"2. Set yielded_conn = {yielded_conn}")
        print("3. About to yield...")

        yield conn

        print("4. Yield completed normally")
        print("5. Returning from generator")
        return

    finally:
        print(f"6. Outer finally: yielded_conn = {yielded_conn}")
        if yielded_conn is not None:
            print(f"7. Returning connection to pool: {yielded_conn}")
        else:
            print(f"7. yielded_conn is None - NO CONNECTION RETURNED!")


# Test 3: Success case with fixed implementation
print()
print("=" * 60)
print("TEST 3: Fixed version - Success case")
print("=" * 60)
try:
    with get_connection_fixed() as conn:
        print(f"  User code: received {conn}")
        print("  User code: no exception")
except Exception as e:
    print(f"  Exception: {e}")
print()

# Test 4: User exception case with fixed implementation
print("=" * 60)
print("TEST 4: Fixed version - User exception case")
print("=" * 60)
try:
    with get_connection_fixed() as conn:
        print(f"  User code: received {conn}")
        print("  User code: raising RuntimeError")
        raise RuntimeError("Simulated user error")
except RuntimeError as e:
    print(f"  Exception caught: {e}")
