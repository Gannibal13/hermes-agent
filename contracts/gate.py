"""Global Execution Contract System — generic gate.

Any task contract enforces: every mandatory item has status + evidence,
evidence is EXECUTED in this run (never inherited green), requirements are
never collapsed or narrowed, and FINAL is forbidden while anything is open.

Contract file format (markdown):
  MinEvidenceItems: <N>            # anti-collapse floor (default 1)
  ## [ID] Title                    # ID = letters+digits, e.g. P1, M2, A12
  Status: CLOSED
  Evidence:
  - tests/...::Class::test_x      # executable pytest nodeids, OR
  - GATE:SELF                      # final-report item: passes iff rest pass
  - GATE:STRUCT                    # meta item: structural checks below
  - GATE:SELFTEST                  # meta item: negative control below
  Asserts: one-line behavior this evidence proves
  ## Amendments                     # required: new prompts APPEND here,
  - A1: ...                        # never rewrite scope from scratch

Exit 0 + FINAL ALLOWED  — all items closed (this output IS the report).
Exit 1 + FINAL FORBIDDEN — open items listed; final is prohibited.

Structural checks (GATE:STRUCT): every item has Status+Asserts; evidence
items have >=1 nodeid; evidence-item count >= floor; no two evidence items
share an identical nodeid set (anti-collapse); Amendments section present
with >=1 entry (anti-narrowing); union of nodeids executed here, none
skipped/missing (anti-stale-green).
Negative control (GATE:SELFTEST): re-runs self with --force-open on the
first evidence item and requires exit 1 + FINAL FORBIDDEN naming it.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ITEM_RE = re.compile(r"^## \[([A-Z]+\d*)\]\s+(.*)$")
MIN_RE = re.compile(r"^MinEvidenceItems:\s*(\d+)", re.MULTILINE)
# Keyed evidence line: "- SOME_REQUIREMENT_KEY: tests/...::test_x" or
# "- SOME_REQUIREMENT_KEY: DESKTOP_VITEST:apps/desktop/src/...test.tsx"
KEYED_RE = re.compile(r"^-\s+([A-Z][A-Z0-9_]+):\s+(tests/\S+|DESKTOP_VITEST:\S+)$")
RESULT_RE = re.compile(r"^(\S+::\S+) (PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)")


def parse_contract(path):
    text = path.read_text(encoding="utf-8")
    floor = 1
    m = MIN_RE.search(text)
    if m:
        floor = int(m.group(1))
    items = {}
    amendments = []
    current = None
    in_evidence = False
    in_amend = False
    for line in text.splitlines():
        s = line.strip()
        im = ITEM_RE.match(s)
        if im:
            current = im.group(1)
            items[current] = {"title": im.group(2), "status": "",
                              "nodeids": [], "gate": "", "asserts": "",
                              "keyed": set()}
            in_evidence = False
            in_amend = (current == "Amendments" or
                        s.lower().startswith("## [amend"))
            if current.lower().startswith("amend"):
                in_amend = True
            continue
        if re.match(r"^##\s+Amendments", s, re.IGNORECASE):
            current = "Amendments"
            in_amend = True
            in_evidence = False
            continue
        if s.startswith("## "):
            current = None
            in_evidence = in_amend = False
            continue
        if in_amend and s.startswith("- "):
            amendments.append(s[2:].strip())
            continue
        if current is None or current == "Amendments":
            continue
        if s.startswith("Status:"):
            items[current]["status"] = s.split(":", 1)[1].strip()
        elif s.startswith("Evidence:"):
            in_evidence = True
        elif s.startswith("Asserts:"):
            items[current]["asserts"] = s.split(":", 1)[1].strip()
            in_evidence = False
        elif in_evidence and s.startswith("- tests/"):
            items[current]["nodeids"].append(s[2:].strip())
        elif in_evidence and s.startswith("- GATE:"):
            items[current]["gate"] = s.split("GATE:", 1)[1].strip()
        elif in_evidence and KEYED_RE.match(s):
            # Keyed evidence: "- EVIDENCE_KEY: tests/..." — the key names the
            # requirement (e.g. LARGE_CONTEXT_COMPACTION_SWITCH_E2E); the
            # nodeid after the colon is executed like any other. Gate sections
            # REQUIRED_EVIDENCE_KEYS below fail closed when a listed key is
            # absent from the contract, so a unit test cannot silently stand in
            # for a keyed E2E requirement.
            items[current]["nodeids"].append(KEYED_RE.match(s).group(2))
            items[current]["keyed"].add(KEYED_RE.match(s).group(1))
        elif in_evidence and s and not s.startswith("-"):
            in_evidence = False
    return items, amendments, floor


# Keyed evidence that MUST exist in the contract (fail closed). Each key here
# binds a named acceptance requirement to a real executable nodeid; removing
# the key or its nodeid (e.g. replacing the P17 E2E with a selector unit test)
# makes the gate FAIL instead of quietly passing.
REQUIRED_EVIDENCE_KEYS = {
    "LARGE_CONTEXT_COMPACTION_SWITCH_E2E",
    "LARGE_CONTEXT_STILL_TOO_LARGE_NEXT_ROUTE",
    "FREE_MODEL_DESKTOP_PICKER_RENDER",
}


def check_required_keys(items):
    """Fail-closed keyed-evidence check. Returns a list of violations."""
    violations = []
    for key in sorted(REQUIRED_EVIDENCE_KEYS):
        owners = [iid for iid, it in items.items() if key in it.get("keyed", ())]
        if not owners:
            violations.append(
                f"missing keyed evidence: {key} (no item carries '- {key}: tests/...')")
    return violations


def run_pytest(nodeids):
    cmd = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider",
           "-v", "--tb=short", *sorted(set(nodeids))]
    proc = subprocess.run(cmd, cwd=str(REPO), capture_output=True,
                          text=True, encoding="utf-8", errors="replace")
    results = {}
    for line in proc.stdout.splitlines():
        m = RESULT_RE.match(line.strip())
        if m:
            results[m.group(1)] = m.group(2)
    return results


VITEST_RE = re.compile(r"^DESKTOP_VITEST:(.+)$")


def run_desktop_vitest(spec_paths):
    """Run the repo's desktop vitest specs; map file results into the gate.

    vitest reports per-file pass/fail (JSON reporter, --reporter=json --output).
    Each spec file is reported as one nodeid (the file itself). Returns
    {spec_file: PASSED|FAILED}; a runner crash yields FAILED for every spec.
    """
    specs = sorted(set(spec_paths))
    import shutil
    npx = shutil.which("npx") or shutil.which("npx.cmd") or "npx"
    # vitest 4 dropped --output for the JSON reporter; JSON goes to stdout.
    # Spec paths in the contract are repo-root-relative, so translate to
    # cwd-relative (cwd is apps/desktop) for the runner.
    desktop = REPO / "apps" / "desktop"
    def _spec_arg(spec: str) -> str:
        # Contract stores repo-root-relative paths; strip the apps/desktop/
        # prefix so the runner (cwd=apps/desktop) matches the file.
        prefix = "apps/desktop/"
        rel = spec[len(prefix):] if spec.startswith(prefix) else spec
        return rel.replace("/", "\\")
    cmd = [npx, "vitest", "run", "--reporter=json",
           *(_spec_arg(s) for s in specs)]
    proc = subprocess.run(cmd, cwd=str(desktop),
                          capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    results = {}
    verdict = "FAILED"
    try:
        data = json.loads(proc.stdout[proc.stdout.find("{"):])
        total = int(data.get("numTotalTests", 0))
        failed = int(data.get("numFailedTests", 0))
        passed = int(data.get("numPassedTests", 0))
        if proc.returncode == 0 and total > 0 and total == failed + passed:
            verdict = "PASSED" if failed == 0 else "FAILED"
    except Exception:
        verdict = "FAILED"
    for spec in specs:
        # Key with the contract's DESKTOP_VITEST: prefix so the gate's
        # union/missing bookkeeping (which stores prefixed nodeids)
        # matches these entries verbatim.
        results[f"DESKTOP_VITEST:{spec}"] = verdict
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract",
                    default="contracts/smart_router_reference.md")
    ap.add_argument("--force-open", default=None)
    ap.add_argument("--no-selftest", action="store_true")
    args = ap.parse_args()

    cpath = (REPO / args.contract) if not Path(args.contract).is_absolute() \
        else Path(args.contract)
    items, amendments, floor = parse_contract(cpath)
    assert items, "no items parsed"

    ev_ids = sorted([i for i in items
                     if items[i]["nodeids"] or items[i]["gate"] == "SELF"])
    self_ids = [i for i in items if items[i]["gate"] == "SELF"]
    struct_ids = [i for i in items if items[i]["gate"] == "STRUCT"]
    selftest_ids = [i for i in items if items[i]["gate"] == "SELFTEST"]

    # STRUCT-0 (fail-closed keyed evidence): named acceptance keys MUST be
    # present in the contract. Without this, P17 could pass with a selector
    # unit test standing in for the required E2E — the exact regression this
    # amendment closes.
    key_violations = check_required_keys(items)
    assert not key_violations, "; ".join(key_violations)

    # STRUCT-1: status + asserts everywhere; evidence where required.
    for iid, it in items.items():
        if iid == "Amendments":
            continue
        assert it["status"], f"{iid} has no Status"
        assert it["asserts"], f"{iid} has no Asserts"
        if not it["gate"]:
            assert it["nodeids"], f"{iid} has no Evidence"

    # STRUCT-2/3: anti-collapse floor + unique evidence sets.
    assert len(ev_ids) - len(self_ids) >= floor, \
        f"only {len(ev_ids) - len(self_ids)} evidence items (floor {floor})"
    seen = {}
    for iid in ev_ids:
        if items[iid]["gate"] == "SELF":
            continue
        key = tuple(sorted(items[iid]["nodeids"]))
        assert key not in seen, f"{iid} collapses into {seen[key]}"
        seen[key] = iid

    # STRUCT-4: anti-narrowing — amendments must exist and grow.
    assert amendments, "no ## Amendments section (new prompts must append)"

    if args.force_open and args.force_open in items:
        items[args.force_open]["nodeids"] = []

    union = [n for iid in ev_ids if iid not in self_ids
             for n in items[iid]["nodeids"]]
    assert union, "empty evidence set"
    # Split the union: pytest nodeids run via pytest; DESKTOP_VITEST: specs run
    # via the desktop vitest runner. Both report into the same results map so
    # verdicts stay uniform.
    pytest_nodeids = [n for n in union if not n.startswith("DESKTOP_VITEST:")]
    vitest_specs = [VITEST_RE.match(n).group(1)
                    for n in union if VITEST_RE.match(n)]
    results = run_pytest(pytest_nodeids) if pytest_nodeids else {}
    if vitest_specs:
        results.update(run_desktop_vitest(vitest_specs))
    executed = set(results)
    missing = [n for n in set(union) if n not in executed]
    skipped = [n for n, r in results.items() if r == "SKIPPED"]

    verdicts = {}
    for iid in ev_ids:
        if iid in self_ids:
            continue  # resolved after everything else
        nids = items[iid]["nodeids"]
        if not nids:
            verdicts[iid] = "OPEN (no evidence)"
        elif any(n in missing for n in nids):
            verdicts[iid] = "OPEN (evidence not executed)"
        elif any(results.get(n) != "PASSED" for n in nids):
            bad = [n for n in nids if results.get(n) != "PASSED"]
            verdicts[iid] = f"OPEN (not PASSED: {bad})"
        else:
            verdicts[iid] = "PASS"

    # STRUCT-5: coverage of THIS run (anti-stale-green).
    struct_ok = not missing and not skipped
    struct_msg = "PASS"
    if missing:
        struct_msg = f"OPEN (not executed: {missing})"
    elif skipped:
        struct_msg = f"OPEN (skipped: {skipped})"
    for iid in struct_ids:
        verdicts[iid] = struct_msg

    # SELFTEST: negative control must block.
    if not args.no_selftest and selftest_ids:
        probe = [sys.executable, str(Path(__file__)), "--contract",
                 args.contract, "--force-open", ev_ids[0], "--no-selftest"]
        pr = subprocess.run(probe, cwd=str(REPO), capture_output=True,
                            text=True, encoding="utf-8", errors="replace")
        neg_ok = (pr.returncode == 1 and "FINAL FORBIDDEN" in pr.stdout
                  and ev_ids[0] in pr.stdout)
        for iid in selftest_ids:
            verdicts[iid] = ("PASS" if neg_ok
                             else "OPEN (negative control did not block)")
    else:
        for iid in selftest_ids:
            verdicts[iid] = "PASS (selftest skipped by flag)"

    # SELF: final report passes iff everything else passes.
    rest = [v for k, v in verdicts.items() if k not in self_ids]
    for iid in self_ids:
        verdicts[iid] = ("PASS" if rest and all(v == "PASS" for v in rest)
                         else "OPEN (gate output incomplete)")

    order = ev_ids + struct_ids + selftest_ids
    print(f"contract: {cpath.name} | items: {len(items)} | "
          f"evidence tests executed: {len(executed)}")
    for iid in order:
        print(f"  [{iid}] {verdicts.get(iid)} — {items[iid]['title'][:72]}")

    blocked_by_flag = [i for i in order
                       if not verdicts.get(i, "").startswith("PASS")]
    if blocked_by_flag:
        print(f"FINAL FORBIDDEN — open: {blocked_by_flag}")
        return 1
    print("FINAL ALLOWED — all items closed with executed evidence")
    return 0


if __name__ == "__main__":
    sys.exit(main())
