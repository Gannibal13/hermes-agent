"""Offline token-economy benchmark. Same file runs in ANY worktree and measures
that tree's real behavior — no provider network calls, no mocked seams beyond
process-local counting. Emits JSON metrics for BASELINE vs OPTIMIZED comparison.

Run: python evals/token_accounting/e2e_benchmark.py --out <path>.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _m_request_sizes() -> dict:
    """Wire bytes kept for N oversized tool results under each tree's own
    persistence pipeline (layer2 maybe_persist_tool_result + layer3 turn budget)."""
    from tools import tool_result_storage as trs

    sizes = {}
    for n in (1, 5, 10):
        messages = []
        for i in range(n):
            messages.append({
                "role": "tool",
                "tool_call_id": f"bench_{n}_{i}",
                "content": "D" * 5_000,
            })
        for msg in messages:
            content = msg["content"]
            try:
                persisted = trs.maybe_persist_tool_result(
                    content=content, tool_name="bench_tool",
                    tool_use_id=msg["tool_call_id"], env=None, threshold=1_000,
                )
            except trs.ToolResultPersistenceError:
                persisted = "[STORAGE FAILURE] bounded preview only"
            msg["content"] = persisted
        try:
            trs.enforce_turn_budget(messages, env=None, threshold=2_000)
        except TypeError:
            pass  # older signature: budget-only
        sizes[f"tool_calls_{n}"] = sum(len(m["content"]) for m in messages)
    return sizes


def _m_background_review_default() -> dict:
    """Auto background-review default (aux provider calls fired per turn)."""
    try:
        from agent import background_review as br
    except Exception:
        try:
            import background_review as br  # older layout
        except Exception:
            return {"default_enabled": None, "error": "module not importable"}
    try:
        enabled, _task = br.load_background_review_settings()
    except Exception as exc:
        return {"default_enabled": None, "error": repr(exc)}
    return {"default_enabled": bool(enabled)}


def _m_prompt_build_file_reads() -> dict:
    """File reads consumed by two consecutive full system-prompt builds.

    The optimized tree caches a content-fingerprinted skills manifest, so the
    second build must not re-read skill files; the baseline re-reads them.
    """
    tmp = tempfile.mkdtemp(prefix="bench-prompt-")
    os.environ["HERMES_HOME"] = tmp
    Path(tmp, "skills").mkdir(exist_ok=True)
    Path(tmp, "skills", "bench-skill", "SKILL.md").parent.mkdir(exist_ok=True)
    Path(tmp, "skills", "bench-skill", "SKILL.md").write_text(
        "---\nname: bench-skill\ndescription: benchmark probe skill\n---\n\nBody.\n",
        encoding="utf-8",
    )
    from types import SimpleNamespace

    from agent.system_prompt import build_system_prompt

    agent = SimpleNamespace(
        load_soul_identity=False, skip_context_files=True, valid_tool_names=[],
        _task_completion_guidance=False, _tool_use_enforcement=False,
        _environment_probe=False, _kanban_worker_guidance="", _memory_store=None,
        _memory_manager=None, model="", provider="", platform="desktop",
        pass_session_id=False, session_id="", _emit_status=lambda *a: None,
    )
    reads = {"count": 0}
    orig = Path.read_text

    def counting(self, *a, **k):
        reads["count"] += 1
        return orig(self, *a, **k)

    first_chars = second_chars = None
    with patch.object(Path, "read_text", counting):
        prompt = build_system_prompt(agent)
        first_chars = len(prompt)
        reads["first_build"] = reads["count"]
        prompt2 = build_system_prompt(agent)
        second_chars = len(prompt2)
        reads["second_build"] = reads["count"] - reads["first_build"]
    return {
        "reads_first_build": reads["first_build"],
        "reads_second_build": reads["second_build"],
        "prompt_chars": first_chars,
        "second_equals_first": first_chars == second_chars,
    }


def _m_tool_schema_chars() -> dict:
    """Chars of the tools payload a default terminal+file agent would send."""
    from types import SimpleNamespace

    try:
        from run_agent import AIAgent
        agent = AIAgent(
            api_key="offline", base_url="http://127.0.0.1:9/v1",
            provider="openai-compat", model="bench", enabled_toolsets=["terminal", "file"],
            quiet_mode=True, skip_context_files=True, skip_memory=True,
            skip_background_review=True, save_trajectories=False, platform="cli",
            session_id="bench_schema_probe",
        )
        tools = agent.tools
    except Exception:
        # Older trees may refuse the kwarg set; fall back to registry description.
        from tools.registry import default_registry
        tools = []
        try:
            for name in ("terminal", "read_file", "write_file", "patch", "search_files"):
                entry = default_registry.get(name, metadata_only=True)
                if entry is not None:
                    tools.append({"function": {"name": name, "parameters": entry.get("parameters", {})}})
        except Exception as exc:
            return {"tools_chars": None, "error": repr(exc)}
    total = 0
    for t in tools:
        total += len(json.dumps(t, default=str))
    return {"tools_count": len(tools), "tools_chars": total}


def _m_noop_compactions() -> dict:
    """Local-compress calls under the native checkpoint preflight scenario."""
    preflight = ROOT / "evals" / "native_compaction" / "ab_checkpoint_preflight.py"
    if not preflight.exists():
        return {"available": False}
    import subprocess
    out = Path(tempfile.mkdtemp(prefix="bench-preflight-")) / "res.json"
    r = subprocess.run(
        [sys.executable, "-B", str(preflight), "--out", str(out)],
        cwd=str(ROOT), capture_output=True, text=True, timeout=900,
    )
    if r.returncode != 0 or not out.exists():
        return {"available": True, "rc": r.returncode, "tail": r.stdout[-400:]}
    data = json.loads(out.read_text(encoding="utf-8"))
    compact = {}
    for key in ("capture_cli_same_objects", "capture_gateway_reloaded_history", "restore"):
        scen = data.get(key) or {}
        compact[key] = {
            "local_compress_calls": (
                scen.get("local_compress_calls_turn2", 0)
                + scen.get("local_compress_calls_turn3", 0)
                + scen.get("local_compress_calls_resume", 0)
            ),
        }
    return {"available": True, "scenarios": compact}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    result = {
        "checkout": str(ROOT),
        "head": os.popen("git rev-parse HEAD").read().strip(),
        "request_sizes_after_n_tool_results": _m_request_sizes(),
        "background_review": _m_background_review_default(),
        "prompt_build_file_reads": _m_prompt_build_file_reads(),
        "tool_schema": _m_tool_schema_chars(),
        "no_op_compactions": _m_noop_compactions(),
    }
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
