from contextlib import contextmanager


def test_call_llm_binds_wire_budget_scope_around_logical_aux_call(monkeypatch):
    from agent import auxiliary_client as ac

    entered = []

    @contextmanager
    def fake_scope(budget, prefix):
        entered.append((budget, prefix))
        yield

    monkeypatch.setattr(ac, "wire_request_budget_scope", fake_scope)
    monkeypatch.setattr(ac, "_call_llm_impl", lambda **_kwargs: "fixture-response")
    monkeypatch.setattr(ac, "_acquire_sync_aux_semaphore", lambda _task: None)

    assert ac.call_llm(task="fixture", messages=[]) == "fixture-response"
    assert len(entered) == 1
    assert entered[0][0].max_requests > 0
    assert entered[0][1] == "auxiliary/fixture"
