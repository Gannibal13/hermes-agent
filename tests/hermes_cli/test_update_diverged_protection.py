"""Diverged-checkout protection for ``hermes update`` — incident regression (2026-09-15).

``hermes update`` ran ``merge --ff-only`` on a main that had 5 local commits while
origin/main had moved: ff-only failed, ``_reconcile_diverged_checkout`` classified the
common-ancestor divergence as an upstream force-push and ran ``reset --hard
origin/main`` — silently discarding committed local history (autostash only covers the
working tree). These tests pin the required semantics:

- local-ahead-only divergence must NOT reset;
- true divergence (local ahead AND remote ahead) must create a durable rescue ref
  pointing at the pre-update HEAD, then abort — never silently reset;
- an in-progress merge/rebase must stop the update before any mutation;
- the orphan rescue path (no common ancestor) keeps its backup;
- behind-only fast-forward and equal-tip updates keep working unchanged.

Every destructive scenario runs against a real temporary git repository and calls the
REAL reconciliation path (``update_cmd._reconcile_diverged_checkout`` / the
``_pull_updates`` wrapper), not a helper.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hermes_cli import update_cmd
from hermes_cli import main as hermes_main


GIT_IDENTITY = ("-c", "user.name=Test", "-c", "user.email=test@example.invalid")


def _git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["git", *GIT_IDENTITY, *args], cwd=str(cwd), capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    if check and result.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result


def _commit(cwd: Path, message: str) -> str:
    (cwd / "file.txt").write_text(message, encoding="utf-8")
    _git(cwd, "add", "-A")
    _git(cwd, "commit", "-m", message)
    return _git(cwd, "rev-parse", "HEAD").stdout.strip()


class _Repo:
    """Bare origin + working clone, both on ``main``."""

    def __init__(self, tmp_path: Path):
        self.tmp = tmp_path
        self.origin = tmp_path / "origin.git"
        self.origin.mkdir(parents=True)
        _git(self.origin, "init", "--bare", "-b", "main")

        self.clone = tmp_path / "checkout"
        _git(tmp_path, "clone", str(self.origin), str(self.clone))
        # The clone has no commits yet — create the root commit and push.
        self.base_sha = _commit(self.clone, "base")
        _git(self.clone, "push", "-q", "origin", "main")
        _git(self.clone, "fetch", "-q", "origin")
        self._upstream_work = None

    def local_commit(self, message: str) -> str:
        return _commit(self.clone, message)

    def upstream_commit(self, message: str) -> str:
        """Commit on a second clone (simulating the remote moving ahead)."""
        if self._upstream_work is None:
            second = self.tmp / "upstream-work"
            _git(self.tmp, "clone", str(self.origin), str(second))
            self._upstream_work = second
        sha = _commit(self._upstream_work, message)
        _git(self._upstream_work, "push", "-q", "origin", "main")
        _git(self.clone, "fetch", "-q", "origin")
        return sha

    @property
    def head(self) -> str:
        return _git(self.clone, "rev-parse", "HEAD").stdout.strip()

    @property
    def origin_head(self) -> str:
        return _git(self.clone, "rev-parse", "origin/main").stdout.strip()

    def rescue_refs(self) -> list[str]:
        out = _git(
            self.clone, "for-each-ref", "--format=%(refname)",
            "refs/hermes-update-backups/*")
        return [line.strip() for line in out.stdout.splitlines() if line.strip()]


@pytest.fixture()
def repo(tmp_path: Path) -> _Repo:
    return _Repo(tmp_path)


def _patch_root(monkeypatch, repo: _Repo) -> None:
    """Point the updater plumbing at the temp checkout."""
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", repo.clone)
    monkeypatch.setattr(update_cmd, "_m", lambda: hermes_main)


def _run_reconcile(repo: _Repo, branch: str = "main") -> None:
    """Invoke the REAL reconciliation entrypoint against the temp checkout."""
    git_cmd = ["git"]
    pre_pull_sha = update_cmd._capture_head_sha(git_cmd, repo.clone)
    update_cmd._reconcile_diverged_checkout(git_cmd, branch, pre_pull_sha)


def _run_pull_updates(repo: _Repo, monkeypatch, branch: str = "main", **kwargs) -> str:
    """Invoke the REAL ``_pull_updates`` wrapper (ff-only → reconcile → syntax guard)."""
    _patch_root(monkeypatch, repo)
    git_cmd = ["git"]
    return update_cmd._pull_updates(
        git_cmd, branch, kwargs.get("auto_stash_ref"), prompt_for_restore=False,
        gw_input_fn=None, discard_local_changes=False,
        keep_stash=kwargs.get("keep_stash", False))


# ---------------------------------------------------------------------------
# A — equal tips: nothing to do.
# ---------------------------------------------------------------------------

def test_equal_tips_is_noop(repo: _Repo, monkeypatch):
    _run_pull_updates(repo, monkeypatch)
    assert repo.head == repo.base_sha
    assert repo.rescue_refs() == []


# ---------------------------------------------------------------------------
# B — behind-only: fast-forward keeps local history intact.
# ---------------------------------------------------------------------------

def test_behind_only_fast_forwards(repo: _Repo, monkeypatch):
    remote_sha = repo.upstream_commit("remote work 1")
    _run_pull_updates(repo, monkeypatch)
    assert repo.head == remote_sha == repo.origin_head
    assert repo.rescue_refs() == []


# ---------------------------------------------------------------------------
# C — ahead-only: local commits, remote did not move. merge --ff-only reports
#     "Already up to date"; reconcile is never reached. Prove HEAD kept.
# ---------------------------------------------------------------------------

def test_ahead_only_preserves_local_commits(repo: _Repo, monkeypatch):
    local1 = repo.local_commit("local work 1")
    local2 = repo.local_commit("local work 2")

    _run_pull_updates(repo, monkeypatch)

    assert repo.head == local2
    assert local1 in _git(repo.clone, "rev-list", "HEAD").stdout
    assert repo.rescue_refs() == []


def test_ahead_only_direct_reconcile_also_preserves(repo: _Repo, capsys):
    """Even if reconcile is reached ahead-only (defensive), it must not reset."""
    local1 = repo.local_commit("ahead only 1")
    _run_reconcile(repo)
    assert repo.head == local1
    assert repo.rescue_refs() == []


# ---------------------------------------------------------------------------
# D — diverged (incident): MUST abort, MUST leave a rescue ref at pre-update
#     HEAD, MUST NOT reset.
# ---------------------------------------------------------------------------

def test_diverged_aborts_with_rescue_ref_and_no_reset(repo: _Repo, monkeypatch, capsys):
    repo.local_commit("local divergence 1")
    local_tip = repo.local_commit("local divergence 2")
    repo.upstream_commit("remote divergence 1")
    repo.upstream_commit("remote divergence 2")

    with pytest.raises(SystemExit) as excinfo:
        _run_pull_updates(repo, monkeypatch)

    assert excinfo.value.code == 1
    out = capsys.readouterr().out
    # Durable rescue ref exists, points at the pre-update HEAD, on-disk ref.
    refs = repo.rescue_refs()
    assert refs, "expected a durable rescue ref under refs/hermes-update-backups/"
    pre_diverged = [ref for ref in refs if ref.startswith("refs/hermes-update-backups/pre-diverged-")]
    assert pre_diverged, refs
    rescue_sha = _git(repo.clone, "rev-parse", pre_diverged[0]).stdout.strip()
    assert rescue_sha == local_tip
    assert _git(repo.clone, "show-ref").stdout.count(refs[0]) == 1
    # Local commits survived — HEAD never moved to origin/main.
    assert repo.head == local_tip
    assert repo.head != repo.origin_head
    assert "local divergence 1" in _git(repo.clone, "log", "--format=%s", "HEAD").stdout
    # The diagnostic told the user what happened and where the backup lives.
    assert refs[0] in out


# ---------------------------------------------------------------------------
# E — dirty tree + diverged: stash parked, commits kept, rescue ref, abort.
# ---------------------------------------------------------------------------

def test_dirty_diverged_preserves_tree_and_commits(repo: _Repo, monkeypatch, capsys):
    local_tip = repo.local_commit("dirty local 1")
    repo.upstream_commit("dirty remote 1")
    (repo.clone / "file.txt").write_text("uncommitted edit", encoding="utf-8")

    stash_ref = update_cmd._stash_local_changes_if_needed(["git"], repo.clone)
    assert stash_ref is not None

    with pytest.raises(SystemExit):
        _run_pull_updates(repo, monkeypatch, auto_stash_ref=stash_ref, keep_stash=True)

    refs = repo.rescue_refs()
    assert refs, "dirty+diverged must still leave a rescue ref"
    assert repo.head == local_tip
    # The stash still holds the uncommitted edit (parked, not destroyed).
    stash_show = _git(repo.clone, "stash", "show", "-p", "stash@{0}").stdout
    assert "uncommitted edit" in stash_show


# ---------------------------------------------------------------------------
# F — merge in progress: stop BEFORE pull/reconciliation, no reset.
# ---------------------------------------------------------------------------

def test_merge_in_progress_stops_update(repo: _Repo, monkeypatch, capsys):
    repo.local_commit("local merge base work")
    remote_tip = repo.upstream_commit("remote side")
    # Create a real in-progress merge that conflicts on file.txt.
    repo.local_commit("local conflicting edit")
    _git(repo.clone, "merge", "--no-commit", "--no-ff", f"{remote_tip}", check=False)
    assert (repo.clone / ".git" / "MERGE_HEAD").exists()

    with pytest.raises(SystemExit) as excinfo:
        _run_pull_updates(repo, monkeypatch)

    assert excinfo.value.code == 1
    assert (repo.clone / ".git" / "MERGE_HEAD").exists(), "merge state must remain untouched"
    out = capsys.readouterr().out
    assert "merge" in out.lower()


# ---------------------------------------------------------------------------
# G — rebase in progress: stop, rebase state remains, no reset.
# ---------------------------------------------------------------------------

def test_rebase_in_progress_stops_update(repo: _Repo, monkeypatch, capsys):
    repo.upstream_commit("remote before rebase")
    repo.local_commit("local to rebase")
    repo.upstream_commit("remote after local")
    # Simulate an interrupted rebase via its state marker: the guard keys off
    # .git/rebase-merge / .git/rebase-apply existing.
    rebase_dir = repo.clone / ".git" / "rebase-merge"
    rebase_dir.mkdir()
    (rebase_dir / "interactive").write_text("", encoding="utf-8")
    try:
        with pytest.raises(SystemExit) as excinfo:
            _run_pull_updates(repo, monkeypatch)
        assert excinfo.value.code == 1
        assert rebase_dir.exists(), "rebase state must remain untouched"
        out = capsys.readouterr().out
        assert "rebase" in out.lower()
    finally:
        import shutil as _shutil
        _shutil.rmtree(rebase_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# H — orphan divergence (no common ancestor): existing rescue behavior kept,
#     but the old silent reset is gone — it aborts with the backup in place.
# ---------------------------------------------------------------------------

def test_orphan_divergence_backs_up_and_resets(repo: _Repo, monkeypatch, capsys):
    """H — Orphan/no merge-base: the documented #87694/#53257 self-heal is
    preserved. The whole local graph is parked behind an orphan- rescue ref
    first, then the install resets to origin so it recovers. No regression in
    the existing rescue behavior — the backup keeps every local commit."""
    repo.upstream_commit("remote keeps moving")
    # Build an unrelated history: replace the clone's HEAD with a fresh root.
    _git(repo.clone, "checkout", "--orphan", "unrelated")
    _git(repo.clone, "rm", "-rf", "--cached", ".", check=False)
    (repo.clone / "unrelated.txt").write_text("x", encoding="utf-8")
    _git(repo.clone, "add", "-A")
    _git(repo.clone, "commit", "-m", "unrelated root")
    orphan_sha = repo.head
    _git(repo.clone, "branch", "-M", "main")

    _run_pull_updates(repo, monkeypatch)

    refs = repo.rescue_refs()
    assert refs, "orphan divergence must keep its backup ref"
    assert any("orphan-" in ref for ref in refs), refs
    # The rescue ref must point at the parked pre-reset history...
    pointed = {_git(repo.clone, "rev-parse", ref).stdout.strip() for ref in refs}
    assert orphan_sha in pointed, refs
    # ...and the install self-heals onto origin (documented #53257 behavior).
    origin_sha = _git(repo.clone, "rev-parse", "origin/main").stdout.strip()
    assert repo.head == origin_sha
    # The parked history stays fully recoverable from the rescue ref.
    assert "unrelated root" in _git(
        repo.clone, "log", "--format=%s", f"{refs[0]}").stdout


# ---------------------------------------------------------------------------
# I — Desktop semantics: --yes --force --gateway --branch main --keep-stash
#     must still not destroy local commits on divergence.
# ---------------------------------------------------------------------------

def test_desktop_force_flags_do_not_destroy_local_commits(repo: _Repo, monkeypatch, capsys):
    repo.local_commit("desktop local 1")
    local_tip = repo.local_commit("desktop local 2")
    repo.upstream_commit("desktop remote 1")

    rc = _run_real_update_argv(
        repo,
        ["update", "--yes", "--force", "--gateway", "--branch", "main", "--keep-stash"],
        monkeypatch)

    assert rc == 1, "diverged update under desktop flags must abort, not succeed"
    assert repo.head == local_tip
    refs = repo.rescue_refs()
    assert refs, "desktop-flag divergence must leave a rescue ref"
    assert any(ref.startswith("refs/hermes-update-backups/pre-diverged-") for ref in refs), refs
    assert "desktop local 1" in _git(repo.clone, "log", "--format=%s", "HEAD").stdout


def _run_real_update_argv(repo: _Repo, argv: list[str], monkeypatch) -> int:
    """Drive ``hermes_cli.main.cmd_update`` in-process against the temp repo.

    Only environment-dependent phases (managed-install gate, update lock, backup
    snapshot, receipt, gateway pause, venv holders) are stubbed; fetch/reconcile/
    stash run for real against the local bare origin — the divergence decision
    under test is NOT stubbed.
    """
    import hermes_cli.main as hm
    from hermes_cli import _parser as parser_mod

    monkeypatch.setattr(hm, "PROJECT_ROOT", repo.clone)
    monkeypatch.setattr(update_cmd, "_m", lambda: hm)
    monkeypatch.setattr(hm, "_update_preflight_handled", lambda args: False)
    monkeypatch.setattr(hm, "_run_pre_update_backup", lambda *a, **k: None)
    monkeypatch.setattr(update_cmd, "_begin_update_receipt_and_plan", lambda args: None)
    monkeypatch.setattr(hm, "_pause_windows_gateways_for_update", lambda: None)
    monkeypatch.setattr(
        update_cmd, "_clear_windows_venv_holders_or_exit", lambda *a, **k: None)
    monkeypatch.setattr(hm, "_sync_with_upstream_if_needed", lambda *a, **k: True)
    monkeypatch.setattr(update_cmd, "_repair_current_checkout", lambda **k: True)
    monkeypatch.setattr(update_cmd, "_apply_pulled_update", lambda *a, **k: None)
    monkeypatch.setattr(
        update_cmd, "_finish_already_up_to_date", lambda *a, **k: None)

    # Cross-process update lock: real lock lives in HERMES_HOME, irrelevant to a
    # temp repo and could collide with a concurrent run — stub acquisition.
    from hermes_cli.update_lock import UpdateLock
    monkeypatch.setattr(UpdateLock, "acquire", lambda self: True)
    monkeypatch.setattr(UpdateLock, "release", lambda self: None)

    parser, _sub = hm._build_cli_parser()
    args = parser.parse_args(argv)
    try:
        hm.cmd_update(args)
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


# ---------------------------------------------------------------------------
# Rescue ref durability: a fresh git process (no in-memory state) sees it.
# ---------------------------------------------------------------------------

def test_rescue_ref_survives_process_exit(repo: _Repo, monkeypatch):
    repo.local_commit("survive local")
    repo.upstream_commit("survive remote")
    with pytest.raises(SystemExit):
        _run_pull_updates(repo, monkeypatch)
    refs = repo.rescue_refs()
    assert refs
    probe = subprocess.run(
        ["git", "show-ref", refs[0]], cwd=str(repo.clone), capture_output=True, text=True)
    assert probe.returncode == 0
    assert refs[0] in probe.stdout
