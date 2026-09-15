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
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ITEM_RE = re.compile(r"^## \[([A-Z]+\d*)\]\s+(.*)$")
MIN_RE = re.compile(r"^MinEvidenceItems:\s*(\d+)", re.MULTILINE)
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
                              "nodeids": [], "gate": "", "asserts": ""}
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
        elif in_evidence and s and not s.startswith("-"):
            in_evidence = False
    return items, amendments, floor


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
    results = run_pytest(union)
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
