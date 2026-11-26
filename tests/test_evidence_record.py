"""The README's kernel tables must be the record, not a memory of one.

The numbers are allowed to move between runs; what is not allowed is the prose
drifting away from the run that produced it.  This is the guard that makes the
README's tables a citation rather than a claim.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "host_kernels.json"


def _record() -> dict:
    assert EVIDENCE.exists(), "evidence/host_kernels.json is missing"
    return json.loads(EVIDENCE.read_text())


def test_record_passed_its_gates():
    record = _record()
    assert record["verdict"] == "PASS", record["gates"]
    assert all(v for v in record["gates"].values() if v is not None)


def test_readme_tables_match_the_record():
    record = _record()
    if not all(record["arms"].values()):
        pytest.skip("record was generated without every arm")
    readme = (ROOT / "README.md").read_text()
    for name, lines in record["markdown"].items():
        block = "\n".join(lines)
        assert block in readme, f"README is out of date with the record: {name}"
