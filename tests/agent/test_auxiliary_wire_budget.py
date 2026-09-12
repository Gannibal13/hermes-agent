from types import SimpleNamespace

import pytest


def test_auxiliary_physical_attempts_share_wire_budget(monkeypatch):
    from agent import auxiliary_client as ac
    from agent.wire_request_budget import WireRequestBudget

    calls = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    monkeypatch.setattr(ac, "_client_streams_internally", lambda _client: True)
    budget = WireRequestBudget(2)

    with ac.wire_request_budget_scope(budget, "auxiliary"):
        ac._create_with_progress_once(client, {"model": "fixture"}, task="first")
        ac._create_with_progress_once(client, {"model": "fixture"}, task="retry")

    assert len(calls) == 2
    assert budget.used == 2
    assert budget.remaining == 0


@pytest.mark.asyncio
async def test_async_auxiliary_stream_holds_budget_until_exhaustion():
    from agent.wire_request_budget import WireRequestBudget
    from agent.wire_request_budget import execute_scoped_wire_request_async
    from agent.wire_request_budget import wire_request_budget_scope

    async def source():
        yield "chunk"

    budget = WireRequestBudget(1)
    with wire_request_budget_scope(budget, "async"):
        stream = await execute_scoped_wire_request_async(
            source, key_suffix="stream", hold_until_close=True
        )
        assert budget.used == 0
        assert budget.remaining == 0
        assert [item async for item in stream] == ["chunk"]

    assert budget.used == 1
