def test_active_task_injection_benchmark_preserves_human_task():
    from evals.active_task_injection.runner import run_benchmark

    result = run_benchmark()

    assert result["legacy"]["active_task_preservation_rate"] == 0.0
    assert result["hardened"]["active_task_preservation_rate"] == 1.0
    assert result["hardened"]["service_substitution_rate"] == 0.0
    assert result["hardened"]["resume_recovery_rate"] == 1.0
    assert result["hardened"]["duplicate_replay_rate"] == 0.0
