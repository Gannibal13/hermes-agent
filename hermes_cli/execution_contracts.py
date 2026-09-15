"""Runtime execution contracts for substantial user tasks.

This module intentionally has no Hermes imports.  Contracts are evidence-class strict:
evidence from a cheaper or unrelated checker cannot satisfy another requirement.

Trust boundary (read this before wiring): ``record_evidence`` is a PURE
type-matrix function — it never runs anything and its PASS means "the claim is
well-typed", not "the work is verified".  The runtime path is ``submit_evidence``,
which matrix-checks and then EXECUTES the real checker via ``verify_item``
(pytest, command, file hash, UI state, review marker, composite subs).  Only
executed verification can PASS an item.  Claims alone never close anything.
"""

from __future__ import annotations

import hashlib
import re
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

TEST, COMMAND, RUNTIME, FILE, UI, REVIEW, COMPOSITE, UNKNOWN = (
    "TEST", "COMMAND", "RUNTIME", "FILE", "UI", "REVIEW", "COMPOSITE", "UNKNOWN"
)
EVIDENCE_TYPES = (TEST, COMMAND, RUNTIME, FILE, UI, REVIEW, COMPOSITE, UNKNOWN)
OPEN, PASS, WRONG_EVIDENCE_TYPE, FAILED = "OPEN", "PASS", "WRONG_EVIDENCE_TYPE", "FAILED"

#: Shared nudge budget for the contract stop-gate (both branches).
MAX_CONTRACT_NUDGES = 3


ACCEPT = {
    TEST: {TEST}, COMMAND: {COMMAND}, RUNTIME: {RUNTIME}, FILE: {FILE},
    UI: {UI}, REVIEW: {REVIEW}, COMPOSITE: {COMPOSITE}, UNKNOWN: set(),
}


@dataclass
class ExecItem:
    item_id: str
    requirement: str
    required: bool = True
    spec: Dict[str, Any] = field(default_factory=dict)
    actual_type: str = UNKNOWN
    actual_artifact: str = ""
    verdict: str = OPEN
    origin: str = "original"
    trace: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecItem":
        fields = {key: data.get(key) for key in cls.__dataclass_fields__}
        fields["spec"] = dict(fields.get("spec") or {})
        fields["actual_type"] = fields.get("actual_type") or UNKNOWN
        fields["actual_artifact"] = fields.get("actual_artifact") or ""
        fields["verdict"] = fields.get("verdict") or OPEN
        fields["origin"] = fields.get("origin") or "original"
        fields["trace"] = fields.get("trace") or ""
        fields["required"] = True if fields.get("required") is None else bool(fields["required"])
        return cls(**fields)


_CHAT_WORDS = re.compile(r"^(?:hi|hello|hey|thanks|thank you|ok|okay|good morning|привет|здравствуй|спасибо)\W*$", re.I)
_TYPE_RE = re.compile(r"\b(TEST|COMMAND|RUNTIME|FILE|UI|REVIEW|COMPOSITE)\b", re.I)


def is_substantial_task(text: str) -> bool:
    text = (text or "").strip()
    if len(text) < 35 or _CHAT_WORDS.match(text):
        return False
    return bool(re.search(r"\b(build|implement|create|fix|add|change|update|test|verify|реализ|созд|исправ|добав|сделай|провер)\w*\b", text, re.I)
                or len(text.split()) >= 12)


def _spec_for(requirement: str, kind: str) -> Dict[str, Any]:
    spec: Dict[str, Any] = {"type": kind}
    quoted = re.findall(r"[`\"']([^`\"']+)[`\"']", requirement)
    if kind in (COMMAND, RUNTIME, UI) and quoted:
        spec["command"] = quoted[0]
    if kind == FILE and quoted:
        spec["path"] = quoted[0]
    return spec


def extract_requirements(text: str) -> List[ExecItem]:
    """Extract explicit typed checklist lines, with a conservative task fallback.

    The fallback item is advisory (``required=False``): an untyped task must be
    visible in the checklist but must NEVER brick the goal — a required UNKNOWN
    item is unclosable by design (no checker accepts UNKNOWN evidence).
    """
    lines = [line.strip(" -*\t") for line in (text or "").splitlines() if line.strip()]
    found: List[ExecItem] = []
    for line in lines:
        matches = list(_TYPE_RE.finditer(line))
        for index, match in enumerate(matches):
            kind = match.group(1).upper()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(line)
            req = line[match.end():end].strip(" =:;,.-–—") or line
            found.append(ExecItem(f"R{len(found) + 1}", req, True, _spec_for(req, kind), trace=line))
    if not found and is_substantial_task(text):
        found.append(ExecItem("R1", (text or "").strip(), False, {"type": UNKNOWN}, trace=(text or "").strip()))
    return found


def _split_clauses(text: str) -> List[str]:
    parts = re.split(r"[;\n]+", text or "")
    clauses: List[str] = []
    for part in parts:
        clauses.extend(re.split(r"\.(?=\s)|,(?=\s*\S)|\s+и\s+|\s+then\s+", part, flags=re.I))
    return [c.strip(" -*\t.,;:!") for c in clauses if len(c.strip(" -*\t.,;:! ")) >= 4]


# Verb → evidence class, checked in order.  Generic build/implement/create with
# no concrete target stays UNKNOWN (advisory): demanding a specific evidence
# class for unspecified work would be noise, not verification.
_CLAUSE_VERBS = (
    (TEST, r"тест|test|regression|регресс"),
    (REVIEW, r"review|ревью|approve|согласу|проверь.*фина|final.*review|готов|done|complete\b|evidence|доказательств"),
    (COMMAND, r"typecheck|типизац|линт|lint|mypy|tsc|eslint|ruff|pytest|собери|build|запусти|запустить|\brun\b|команду|command"),
    (RUNTIME, r"production path|продакш|прод\b|deploy|депло|runtime|smoke|миграц|migrat"),
    (FILE, r"исправь|почини|fix\b|баг|bug|файл|file\b|патч|patch|код\b|code\b"),
    (UI, r"\bui\b|интерфейс|кнопка|button|экран|страниц|page\b|вёрстк|скриншот|screenshot"),
)


def _classify_clause(clause: str) -> str:
    lowered = clause.lower()
    for kind, pattern in _CLAUSE_VERBS:
        if re.search(pattern, lowered):
            return kind
    return UNKNOWN


def _extract_explicit_typed(text: str) -> List[ExecItem]:
    """Strict path, but only for EXPLICIT markers: a line counts when the type
    word is assigned with ``=`` (``A=TEST``) or leads the line (``TEST: ...``).
    Bare words like "test"/"runtime" inside natural prose are NOT markers —
    those go through clause classification instead.
    """
    lines = [line.strip(" -*\t") for line in (text or "").splitlines() if line.strip()]
    found: List[ExecItem] = []
    for line in lines:
        if not re.search(
            r"=\s*(TEST|COMMAND|RUNTIME|FILE|UI|REVIEW|COMPOSITE)\b"
            r"|\b(TEST|COMMAND|RUNTIME|FILE|UI|REVIEW|COMPOSITE)\s*="
            r"|^\s*(TEST|COMMAND|RUNTIME|FILE|UI|REVIEW|COMPOSITE)\b", line, re.I):
            continue
        matches = list(_TYPE_RE.finditer(line))
        for index, match in enumerate(matches):
            kind = match.group(1).upper()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(line)
            req = line[match.end():end].strip(" =:;,.-–—") or line
            found.append(ExecItem(f"R{len(found) + 1}", req, True, _spec_for(req, kind), trace=line))
    return found


def extract_auto_requirements(text: str) -> List[ExecItem]:
    """Checklist for a substantial task WITHOUT explicit /goal or typed lines.

    Explicit ``X=TYPE`` lines keep the strict path.  Otherwise each action
    clause of the natural-language request becomes its own item, classified by
    verb into an evidence class; clauses with no classifiable evidence signal
    stay advisory (required=False) and can never brick the goal.
    """
    typed = _extract_explicit_typed(text)
    if typed:
        return typed
    clauses = _split_clauses(text)
    out: List[ExecItem] = []
    seen = set()
    for clause in clauses[:8]:
        key = clause.lower()
        if key in seen:
            continue
        seen.add(key)
        kind = _classify_clause(clause)
        out.append(ExecItem(f"R{len(out) + 1}", clause, kind != UNKNOWN, {"type": kind}, trace=clause))
    if not any(item.required for item in out):
        # Nothing actionable — single advisory fallback (never blocking).
        return extract_requirements(text)
    return out


def merge_amendment(items: List[ExecItem], new_text: str) -> List[ExecItem]:
    """Append-only: amendments add items, never rewrite or drop existing ones.

    Untyped amendment lines are advisory (``required=False``) for the same
    reason as the extraction fallback — they must not brick the goal.
    """
    result = list(items)
    additions = extract_requirements(new_text)
    if not additions and (new_text or "").strip():
        additions = [ExecItem("", (new_text or "").strip(), False, {"type": UNKNOWN})]
    for item in additions:
        item.item_id = f"R{len(result) + 1}"
        item.origin = "amendment"
        item.trace = (new_text or "").strip()
        result.append(item)
    return result


def record_evidence(item: ExecItem, actual_type: str, artifact: str) -> str:
    """Pure type-matrix pre-check.  PASS here means "well-typed claim", NOT
    "verified work" — the runtime path must go through ``submit_evidence``."""
    item.actual_type = str(actual_type or UNKNOWN).upper()
    item.actual_artifact = artifact or ""
    accepted = item.spec.get("type", UNKNOWN).upper()
    if item.actual_type not in ACCEPT.get(accepted, set()):
        item.verdict = WRONG_EVIDENCE_TYPE
    else:
        item.verdict = PASS if artifact else FAILED
    return item.verdict


def _run(command: Any, cwd: Optional[str], regex: Optional[str] = None) -> bool:
    try:
        args = command if isinstance(command, (list, tuple)) else shlex.split(str(command))
        proc = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=300)
        output = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode == 0 and (not regex or re.search(regex, output, re.M) is not None)
    except (OSError, subprocess.SubprocessError, ValueError, re.error):
        return False


def verify_item(item: ExecItem, *, cwd: Optional[str] = None) -> bool:
    kind = str(item.spec.get("type", UNKNOWN)).upper()
    spec = item.spec
    ok = False
    if kind == TEST:
        nodeids = spec.get("nodeids") or ([spec["nodeid"]] if spec.get("nodeid") else [])
        ok = _run([sys.executable, "-m", "pytest", *map(str, nodeids), "-p", "no:cacheprovider"], cwd)
    elif kind in (COMMAND, RUNTIME):
        ok = _run(spec.get("command", ""), cwd, spec.get("expect_regex") or spec.get("regex"))
    elif kind == FILE:
        path = Path(cwd or ".") / str(spec.get("path", ""))
        try:
            if spec.get("sha256"):
                ok = hashlib.sha256(path.read_bytes()).hexdigest() == str(spec["sha256"])
            elif spec.get("content_regex"):
                ok = re.search(str(spec["content_regex"]), path.read_text(encoding="utf-8"), re.M) is not None
            else:
                # Type-only FILE: the named proof artifact must exist.  Presence
                # is the documented default; hash/regex tighten it when given.
                ok = bool(spec.get("path")) and path.is_file()
        except (OSError, UnicodeError, re.error):
            ok = False
    elif kind == UI:
        if spec.get("state_file"):
            try:
                ok = re.search(str(spec.get("state_regex", spec.get("content_regex", ".+"))),
                                Path(cwd or ".", str(spec["state_file"])).read_text(encoding="utf-8"), re.M) is not None
            except (OSError, UnicodeError, re.error):
                ok = False
        elif spec.get("command"):
            ok = _run(spec["command"], cwd, spec.get("expect_regex") or spec.get("regex"))
    elif kind == REVIEW:
        marker = str(spec.get("verdict_marker", "APPROVED"))
        artifact = Path(cwd or ".") / str(item.actual_artifact or spec.get("artifact", ""))
        try:
            ok = bool(item.actual_artifact) and artifact.is_file() and marker in artifact.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            ok = False
    elif kind == COMPOSITE:
        subs = spec.get("requires") or []
        ok = bool(subs) and all(verify_item(s if isinstance(s, ExecItem) else ExecItem.from_dict(s), cwd=cwd) for s in subs)
    # UNKNOWN and anything unrecognised: no checker exists — stays FAILED by design.
    item.verdict = PASS if ok else FAILED
    return ok


def _fill_spec_from_artifact(item: ExecItem, artifact: str) -> None:
    """Let the submitted artifact name the concrete target when the extracted
    spec carries only a type (e.g. TEST item + artifact ``tests/x.py::test_y``)."""
    spec = item.spec
    kind = str(spec.get("type", UNKNOWN)).upper()
    text = (artifact or "").strip()
    if not text:
        return
    if kind == TEST and not spec.get("nodeids") and not spec.get("nodeid"):
        spec["nodeids"] = [text]
    elif kind in (COMMAND, RUNTIME) and not spec.get("command"):
        spec["command"] = text
    elif kind == UI and not spec.get("command") and not spec.get("state_file"):
        spec["command"] = text
    elif kind == FILE and not spec.get("path"):
        spec["path"] = text
    elif kind == REVIEW and not spec.get("artifact"):
        spec["artifact"] = text


def submit_evidence(item: ExecItem, actual_type: str, artifact: str, *, cwd: Optional[str] = None) -> str:
    """Runtime evidence path: matrix pre-check, then REAL checker execution.

    A claim alone never PASSes — the verdict comes from executed verification
    (COMPOSITE verdicts come only from verified subs, never from the claim).
    """
    matrix = record_evidence(item, actual_type, artifact)
    if matrix != PASS:
        return matrix  # WRONG_EVIDENCE_TYPE, or FAILED on empty artifact — nothing runs
    _fill_spec_from_artifact(item, artifact)
    verify_item(item, cwd=cwd)
    return item.verdict


def contract_verdict(items: Iterable[ExecItem]) -> Tuple[bool, List[str]]:
    open_ids = [i.item_id for i in items if i.required and i.verdict != PASS]
    return not open_ids, open_ids


def render_compact(items: Iterable[ExecItem], *, budget: int = 1200) -> str:
    items = list(items)
    open_items = [i for i in items if i.required and i.verdict != PASS]
    if not open_items:
        return "Execution contract: all mandatory items PASS"[:budget]
    lines = [f"next: {open_items[0].item_id}", f"Execution contract: {len(open_items)} mandatory item(s) open"]
    for item in open_items:
        lines.append(f"{item.item_id} [{item.spec.get('type', UNKNOWN)}] {item.requirement}")
    return "\n".join(lines)[:budget]


def maybe_auto_activate(session_id: str, user_text: str) -> str:
    """GLOBAL admission: a substantial task gets an ACTIVE CONTRACT even when
    the user never typed /goal and never mentioned contracts.

    - No goal state at all → create one with origin="auto" and fill the
      checklist from the request text (typed lines or natural clauses).
    - Active manual /goal → fill the checklist only if empty (never clobber).
    - Active auto goal → merge newly-seen requirements append-only (cap 10).
    - Paused/done/cleared → never touch ("skipped-inactive").

    Auto goals skip the judge loop (see _maybe_continue / gateway) so ordinary
    chat still answers normally; enforcement lives at the turn-stop gate,
    bounded by MAX_CONTRACT_NUDGES.
    """
    if not is_substantial_task(user_text):
        return "skipped-chat"
    from hermes_cli.goals import GoalManager
    manager = GoalManager(session_id=session_id)
    state = manager.state
    if state is None:
        manager.set(user_text, origin="auto")
        manager.ensure_auto_contract(user_text)
        return "created"
    if state.status != "active":
        return "skipped-inactive"
    if not state.exec_items:
        manager.ensure_auto_contract(user_text)
        return "exists"
    if getattr(state, "origin", "manual") == "auto" and len(state.exec_items) < 10:
        fresh = extract_auto_requirements(user_text)
        known = {str(raw.get("requirement", "")).lower() for raw in state.exec_items}
        added = False
        for item in fresh:
            if item.requirement.lower() not in known and len(state.exec_items) < 10:
                state.exec_items.append(asdict(item))
                known.add(item.requirement.lower())
                added = True
        if added:
            manager._save()
            return "merged"
    return "exists"
