import pytest


def test_physical_request_budget_reserves_commits_and_bounds_requests():
    from agent.wire_request_budget import WireRequestBudget

    budget = WireRequestBudget(2)
    first = budget.reserve("turn-1/attempt-1")
    second = budget.reserve("turn-1/attempt-2")

    assert first is not None
    assert second is not None
    assert budget.reserve("turn-1/attempt-3") is None
    budget.commit(first)
    assert budget.used == 1
    budget.refund(second)
    assert budget.remaining == 1


def test_physical_request_budget_is_idempotent_for_repeated_settlement():
    from agent.wire_request_budget import WireRequestBudget

    budget = WireRequestBudget(1)
    lease = budget.reserve("same-attempt")
    budget.commit(lease)
    budget.commit(lease)
    budget.refund(lease)

    assert budget.used == 1
    assert budget.remaining == 0


def test_duplicate_attempt_key_cannot_reserve_a_second_physical_request():
    from agent.wire_request_budget import WireRequestBudget

    budget = WireRequestBudget(2)
    assert budget.reserve("same") is not None
    assert budget.reserve("same") is None
    assert budget.remaining == 1


def test_physical_request_budget_refunds_provider_failure():
    from agent.wire_request_budget import WireRequestBudget, execute_bounded_request

    budget = WireRequestBudget(1)

    with pytest.raises(RuntimeError):
        execute_bounded_request(budget, "failed", lambda: (_ for _ in ()).throw(RuntimeError("boom")))

    assert budget.used == 0
    assert budget.remaining == 1


def test_retry_and_stream_close_each_settle_one_physical_attempt():
    from agent.wire_request_budget import WireRequestBudget, execute_bounded_request

    budget = WireRequestBudget(2)
    with pytest.raises(RuntimeError):
        execute_bounded_request(budget, "retry-1", lambda: (_ for _ in ()).throw(RuntimeError("provider")),
                                refund_on_error=False)
    stream = execute_bounded_request(budget, "retry-2", lambda: object())
    assert stream is not None
    assert budget.used == 2
    assert budget.remaining == 0
