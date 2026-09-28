#!/usr/bin/env python3
"""Validate the provenance.yaml template embedded in AGENTS.md §0.3.

WHY THIS EXISTS
    The template lives inside a ```yaml fence in a markdown file, so nothing parses it. A
    typo in a field name, or an unbracketed flow sequence, stays invisible until an agent
    copies the template into a real arm and the arm's provenance fails to load — by which
    point the arm may have run. It also asserts the presence of the three return-edge
    fields (serves / tree_path / return_question), so that deleting the rule from the doc
    is a loud failure rather than a silent loss.

    Assertion note: this checks for the FIELD NAMES, which is the substantive rule here —
    the fields' absence is the failure. (The skill warns against token-assertions: a lint
    that greps for a string that also appears elsewhere stays green after the real rule is
    deleted. Field presence in a parsed mapping is not satisfiable by prose.)

LIMIT: this catches project-side regression only. It cannot detect a change in the skill,
so green means "AGENTS.md has not lost what it had", never "AGENTS.md is current".

USAGE
    .venv/bin/python scripts/check_agents_template_yaml.py [path]     # default AGENTS.md
"""
import re
import sys

import yaml

SRC = sys.argv[1] if len(sys.argv) > 1 else "AGENTS.md"
REQUIRED = ("serves", "tree_path", "return_question")

try:
    text = open(SRC, encoding="utf-8").read()
except FileNotFoundError:
    sys.exit(f"FAIL: {SRC} not found")

blocks = re.findall(r"```yaml\n(.*?)```", text, re.DOTALL)
print(f"{SRC}: {len(blocks)} yaml block(s) found")
if not blocks:
    sys.exit("FAIL: no ```yaml block found - is the template still fenced?")

bad = 0
for i, block in enumerate(blocks, 1):
    try:
        data = yaml.safe_load(block)
    except Exception as exc:  # noqa: BLE001
        bad += 1
        print(f"  block {i}: FAIL -> {type(exc).__name__}: {exc}")
        continue
    n = len(data) if isinstance(data, dict) else 0
    print(f"  block {i}: parses OK ({n} top-level keys)")
    for want in REQUIRED:
        if not isinstance(data, dict) or want not in data:
            print(f"    !! missing required return-edge field: {want}  <- "
                  f"the skill's 'Return Edge' rule has been lost from this doc")
            bad += 1

print("RESULT:", "FAIL" if bad else "OK")
sys.exit(1 if bad else 0)
