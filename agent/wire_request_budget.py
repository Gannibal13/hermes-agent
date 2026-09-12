"""Bounded accounting for physical provider requests.

``IterationBudget`` limits logical loop iterations.  This budget is deliberately
separate: a retry, fallback, auxiliary request, or stream transport attempt is a
new physical request, while settling one attempt is idempotent.
"""

from __future__ import annotations

from dataclasses import dataclass
import contextvars
import inspect
import threading
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Optional


@dataclass
class _WireBudgetContext:
    budget: "WireRequestBudget"
    prefix: str
    next_attempt: int = 0


_CURRENT_WIRE_BUDGET: contextvars.ContextVar[Optional[_WireBudgetContext]] = contextvars.ContextVar(
    "hermes_wire_request_budget", default=None
)


@contextmanager
def wire_request_budget_scope(budget: "WireRequestBudget", prefix: str = "wire") -> Iterator[None]:
    """Bind one physical-request ledger to nested auxiliary/provider helpers."""
    state = _WireBudgetContext(budget=budget, prefix=str(prefix or "wire"))
    token = _CURRENT_WIRE_BUDGET.set(state)
    try:
        yield
    finally:
        _CURRENT_WIRE_BUDGET.reset(token)


def execute_scoped_wire_request(callback: Callable[[], Any], *, key_suffix: str = "request",
                                hold_until_close: bool = False) -> Any:
    """Charge a nested physical callback when a wire scope is active."""
    state = _CURRENT_WIRE_BUDGET.get()
    if state is None:
        return callback()
    state.next_attempt += 1
    key = f"{state.prefix}/{key_suffix}/{state.next_attempt}"
    return execute_bounded_request(
        state.budget, key, callback, refund_on_error=False,
        hold_until_close=hold_until_close,
    )


@dataclass
class WireRequestLease:
    key: str
    _state: str = "reserved"


class _LeaseBackedIterator:
    """Keep a physical-request lease open until a stream has a terminal outcome."""

    def __init__(self, source: Any, budget: "WireRequestBudget", lease: WireRequestLease) -> None:
        self._source = source
        self._budget = budget
        self._lease = lease
        self._started = False
        self._settled = False

    def _settle(self, *, commit: bool) -> None:
        if self._settled:
            return
        self._settled = True
        if commit:
            self._budget.commit(self._lease)
        else:
            self._budget.refund(self._lease)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            item = next(self._source)
        except StopIteration:
            self._settle(commit=True)
            raise
        except BaseException:
            # Once iteration started, the provider was reached and the attempt is billable.
            self._settle(commit=True)
            raise
        self._started = True
        return item

    def close(self) -> None:
        try:
            close = getattr(self._source, "close", None)
            if callable(close):
                close()
        finally:
            # Closing before the first chunk means cancellation happened before the
            # provider produced data; release the reservation for a retry.
            self._settle(commit=self._started)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._source, name)


class _AsyncLeaseBackedIterator:
    """Async counterpart of ``_LeaseBackedIterator``."""

    def __init__(self, source: Any, budget: "WireRequestBudget", lease: WireRequestLease) -> None:
        self._source = source
        self._budget = budget
        self._lease = lease
        self._started = False
        self._settled = False

    def _settle(self, *, commit: bool) -> None:
        if self._settled:
            return
        self._settled = True
        if commit:
            self._budget.commit(self._lease)
        else:
            self._budget.refund(self._lease)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            item = await self._source.__anext__()
        except StopAsyncIteration:
            self._settle(commit=True)
            raise
        except BaseException:
            self._settle(commit=True)
            raise
        self._started = True
        return item

    async def aclose(self) -> None:
        try:
            close = getattr(self._source, "aclose", None)
            if callable(close):
                await close()
        finally:
            self._settle(commit=self._started)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.aclose()
        return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._source, name)


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


async def execute_scoped_wire_request_async(callback: Callable[[], Any], *, key_suffix: str = "request",
                                      hold_until_close: bool = False) -> Any:
    """Async counterpart to ``execute_scoped_wire_request``."""
    state = _CURRENT_WIRE_BUDGET.get()
    if state is None:
        result = callback()
        return await result if inspect.isawaitable(result) else result
    state.next_attempt += 1
    key = f"{state.prefix}/{key_suffix}/{state.next_attempt}"
    lease = state.budget.reserve(key)
    if lease is None:
        raise RuntimeError("physical provider request budget exhausted")
    try:
        result = callback()
        if inspect.isawaitable(result):
            result = await result
    except BaseException:
        state.budget.commit(lease)
        raise
    if hold_until_close:
        try:
            result.__aiter__
        except AttributeError:
            state.budget.commit(lease)
            return result
        return _AsyncLeaseBackedIterator(result.__aiter__(), state.budget, lease)
    state.budget.commit(lease)
    return result


def hold_wire_request_lease(source: Any, budget: WireRequestBudget, lease: WireRequestLease) -> Any:
    """Attach an already-reserved lease to a stream returned after preflight."""
    try:
        iterator = iter(source)
    except TypeError:
        budget.commit(lease)
        return source
    return _LeaseBackedIterator(iterator, budget, lease)


def execute_bounded_request(
    budget: WireRequestBudget, key: str, callback: Callable[[], Any], *,
    refund_on_error: bool = True, hold_until_close: bool = False,
) -> Any:
    """Run one provider callback and settle one physical-request lease.

    ``hold_until_close`` wraps an iterator so a returned stream remains reserved until
    it is exhausted or closed.  A pre-provider/preflight exception should use the
    default refund behavior; an exception after the external callback was reached can
    opt into ``refund_on_error=False`` and remains billable.
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
    if hold_until_close:
        try:
            iterator = iter(result)
        except TypeError:
            budget.commit(lease)
            return result
        return _LeaseBackedIterator(iterator, budget, lease)
    budget.commit(lease)
    return result


__all__ = ["WireRequestBudget", "WireRequestLease", "execute_bounded_request", "hold_wire_request_lease",
           "wire_request_budget_scope", "execute_scoped_wire_request", "execute_scoped_wire_request_async"]
