from types import SimpleNamespace

from agent.global_model_router import (
    GlobalModelRouter,
    ProviderCallResult,
    quota_from_headers,
)
from agent.turn_api_call import _record_router_result


def test_common_quota_headers_are_exact_and_unknown_is_honest():
    q = quota_from_headers({
        "Retry-After": "12",
        "X-RateLimit-Remaining-Requests": "7",
        "X-RateLimit-Remaining-Tokens": "9000",
        "X-RateLimit-Reset": "60",
    }, now=1000)
    assert q.confidence == "EXACT"
    assert q.source == "provider_header"
    assert q.remaining_requests == 7
    assert q.remaining_tokens == 9000
    assert q.reset_at == 1060
    unknown = quota_from_headers({})
    assert unknown.confidence == "UNKNOWN"
    assert unknown.source == "unknown"
    assert unknown.remaining_requests is None


def test_provider_call_ledger_persists_and_reads_back(tmp_path):
    db = tmp_path / "router.db"
    router = GlobalModelRouter(store_path=db)
    router.record_provider_result(ProviderCallResult(
        provider="free", model="small", route_id="lease-a", request_id="req-a",
        status="RATE_LIMITED", http_status=429, retry_after=17,
        remaining_requests=2, timestamp=1234, task_id="goal-a", session_id="session-a",
        reason="rate_limit_429",
    ))
    reopened = GlobalModelRouter(store_path=db)
    rows = reopened.provider_call_ledger(session_id="session-a")
    assert len(rows) == 1
    assert rows[0]["status"] == "RATE_LIMITED"
    assert rows[0]["http_status"] == 429
    assert rows[0]["quota_confidence"] == "EXACT"
    assert rows[0]["quota_source"] == "provider_header"
    assert rows[0]["task_id"] == "goal-a"


def test_production_adapter_records_synthetic_http_429(tmp_path):
    router = GlobalModelRouter(store_path=tmp_path / "router.db")
    lease = router.acquire_lease("free", "small", holder="goal-a")
    agent = SimpleNamespace(
        _global_model_router=router, provider="free", model="small",
        api_request_id="req-429", session_id="session-a",
        _fallback_activated=False, _global_router_task_class=None,
    )
    error = SimpleNamespace(
        status_code=429, retry_after=None, response=SimpleNamespace(
            status_code=429,
            headers={
                "retry-after": "23",
                "x-ratelimit-remaining-requests": "0",
            },
        ),
    )
    _record_router_result(agent, lease, status="RATE_LIMITED", error=error, task_id="goal-a")
    rows = router.provider_call_ledger(session_id="session-a")
    assert rows[0]["http_status"] == 429
    assert rows[0]["retry_after"] == 23
    assert rows[0]["status"] == "RATE_LIMITED"
    assert rows[0]["quota_confidence"] == "EXACT"
