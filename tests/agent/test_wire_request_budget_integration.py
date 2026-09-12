import pytest


def test_preflight_failure_refunds_before_provider_attempt():
    from agent.wire_request_budget import execute_bounded_request
    from agent.wire_request_budget import WireRequestBudget

    budget = WireRequestBudget(1)
    provider_calls = []

    def preflight_then_provider():
        raise ValueError("invalid preflight")

    with pytest.raises(ValueError, match="invalid preflight"):
        execute_bounded_request(
            budget,
            "turn/preflight",
            preflight_then_provider,
            refund_on_error=True,
        )

    assert provider_calls == []
    assert budget.used == 0
    assert budget.remaining == 1


def test_stream_lease_commits_only_when_stream_closes_or_exhausts():
    from agent.wire_request_budget import WireRequestBudget
    from agent.wire_request_budget import execute_bounded_request

    budget = WireRequestBudget(1)
    stream = execute_bounded_request(
        budget,
        "turn/stream",
        lambda: iter(["one", "two"]),
        hold_until_close=True,
    )

    assert budget.used == 0
    assert budget.remaining == 0
    assert next(stream) == "one"
    assert budget.used == 0
    assert next(stream) == "two"
    with pytest.raises(StopIteration):
        next(stream)
    assert budget.used == 1
    assert budget.remaining == 0


def test_stream_lease_refunds_when_cancelled_before_first_chunk():
    from agent.wire_request_budget import WireRequestBudget
    from agent.wire_request_budget import execute_bounded_request

    budget = WireRequestBudget(1)
    stream = execute_bounded_request(
        budget,
        "turn/cancelled-stream",
        lambda: iter(["late"]),
        hold_until_close=True,
    )
    stream.close()

    assert budget.used == 0
    assert budget.remaining == 1
