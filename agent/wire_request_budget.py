"""Bounded accounting for physical provider requests.

``IterationBudget`` limits logical loop iterations.  This budget is deliberately
separate: a retry, fallback, auxiliary request, or stream transport attempt is a
new physical request, while settling one attempt is idempotent.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any, Callable, Optional


@dataclass
class WireRequestLease:
    key: str
    _state: str = "reserved"


class WireRequestBudget:
    """Thread-safe bounded reserve/commit/refund ledger."""

    def __init__(self, max_requests: int):
        if isinstance(max_requests, bool) or not isinstance(max_requests, int) or max_requests < 0:
            raise ValueError("max_requests must be a non-negative integer")
        self.max_requests = max_requests
        self._reserved = 0
        self._committed = 0
        self._leases: dict[str, WireRequestLease] = {}
        self._lock = threading.Lock()

    def reserve(self, key: str) -> Optional[WireRequestLease]:
        """Reserve one request; an already-reserved/settled key is not charged again."""
        key = str(key)
        with self._lock:
            if key in self._leases:
                return None
            if self._reserved + self._committed >= self.max_requests:
                return None
            self._reserved += 1
            lease = WireRequestLease(key)
            self._leases[key] = lease
            return lease

    def commit(self, lease: WireRequestLease) -> bool:
        with self._lock:
            if lease is None or lease._state != "reserved":
                return False
            lease._state = "committed"
            self._reserved -= 1
            self._committed += 1
            return True

    def refund(self, lease: WireRequestLease) -> bool:
        with self._lock:
            if lease is None or lease._state != "reserved":
                return False
            lease._state = "refunded"
            self._reserved -= 1
            self._leases.pop(lease.key, None)
            return True

    @property
    def used(self) -> int:
        with self._lock:
            return self._committed

    @property
    def remaining(self) -> int:
        with self._lock:
            return max(0, self.max_requests - self._reserved - self._committed)


def execute_bounded_request(
    budget: WireRequestBudget, key: str, callback: Callable[[], Any], *,
    refund_on_error: bool = True,
) -> Any:
    """Run one provider callback.

    Standalone callers may refund callback errors (the callback can be a preflight
    operation). The provider integration sets ``refund_on_error=False`` because an
    exception from the external callback still represents a physical attempt.
    """
    lease = budget.reserve(key)
    if lease is None:
        raise RuntimeError("physical provider request budget exhausted")
    try:
        result = callback()
    except BaseException:
        if refund_on_error:
            budget.refund(lease)
        else:
            budget.commit(lease)
        raise
    budget.commit(lease)
    return result


__all__ = ["WireRequestBudget", "WireRequestLease", "execute_bounded_request"]
