"""Immutable experiment records (PLAN section 15).

A record is evidence, not a log line: it names the hypothesis, the exact
configuration, the hardware, the objective and the compute consumed, and it
carries a content hash so two runs can be compared without trusting a filename.
Failed experiments are first-class: ``result_status="rejected"`` is a legitimate
record.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..core import FitResult
from ..benchmarks.timing import hardware_descriptor

VALID_STATUS = ("hypothesis", "replicated", "rejected", "inconclusive", "baseline")


@dataclass(frozen=True)
class ExperimentRecord:
    algorithm: str
    variant: str
    dataset: str
    n: int
    d: int
    k: int
    batch_size: int
    seed: int
    initialization: str
    state_update: str
    schedule_parameters: dict[str, Any]
    hardware: dict[str, Any]
    software_version: str
    objective: float
    examples_processed: int
    iterations: int
    wall_time: float
    memory: int
    final_metrics: dict[str, Any] = field(default_factory=dict)
    baseline_reference: str | None = None
    result_status: str = "hypothesis"
    hypothesis: str = ""
    notes: str = ""
    created: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        if self.result_status not in VALID_STATUS:
            raise ValueError(f"result_status must be one of {VALID_STATUS}")
        if self.n < 0 or self.d < 0 or self.k < 0:
            raise ValueError("n, d, k must be non-negative")

    # -- serialisation ------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["schedule_parameters"] = _jsonable(out["schedule_parameters"])
        out["final_metrics"] = _jsonable(out["final_metrics"])
        out["hardware"] = _jsonable(out["hardware"])
        return out

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @property
    def content_hash(self) -> str:
        """Stable identity of the measurement (excludes the creation timestamp)."""
        payload = self.to_dict()
        payload.pop("created", None)
        payload.pop("notes", None)
        blob = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]

    def with_status(self, status: str) -> "ExperimentRecord":
        return replace(self, result_status=status)

    def with_note(self, note: str) -> "ExperimentRecord":
        return replace(self, notes=(self.notes + " " + note).strip())


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return None
    return value


def record_from_fit(
    result: FitResult,
    *,
    algorithm: str,
    variant: str,
    dataset: str,
    k: int,
    d: int | None = None,
    initialization: str = "",
    hypothesis: str = "",
    baseline_reference: str | None = None,
    result_status: str = "hypothesis",
    final_metrics: dict[str, Any] | None = None,
    hardware: dict[str, Any] | None = None,
    notes: str = "",
) -> ExperimentRecord:
    """Build a record from a :class:`~emaopt.core.FitResult`."""
    n = int(result.extras.get("n", 0)) or 0
    if d is None:
        d = int(result.extras.get("d", 0))
    return ExperimentRecord(
        algorithm=algorithm,
        variant=variant,
        dataset=dataset,
        n=n,
        d=int(d),
        k=int(k),
        batch_size=int(result.extras.get("batch_size", 0)),
        seed=int(result.extras.get("seed", 0)),
        initialization=initialization or str(result.extras.get("init", "")),
        state_update=str(result.extras.get("updater", "")),
        schedule_parameters={
            key: value
            for key, value in result.extras.get("updater_config", {}).items()
            if key not in ("name",)
        },
        hardware=hardware or hardware_descriptor(),
        software_version=f"emaopt 0.1.0 (python {sys.version.split()[0]}, {platform.system()})",
        objective=float(result.objective),
        examples_processed=int(result.examples_processed),
        iterations=int(result.updates),
        wall_time=float(result.wall_time),
        memory=int(result.peak_memory),
        final_metrics=dict(final_metrics or {}),
        baseline_reference=baseline_reference,
        result_status=result_status,
        hypothesis=hypothesis,
        notes=notes,
    )


def write_jsonl(path: str | Path, records: Iterable[ExperimentRecord]) -> int:
    """Append records to a JSONL evidence file; returns the number written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
