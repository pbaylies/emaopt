"""T43: ``State`` keeps durable, run-scoped and live data apart, and serializes.

Before the split everything lived in ``meta``, so a shallow ``copy()`` could
alias a live accumulator and a saved state could only be read back by knowing
every private ``_final_*`` key.  These tests pin the three pockets, a deep copy
in all of them, and a versioned, JSON-safe snapshot.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from emaopt.core import STATE_VERSION, State


def _state() -> State:
    return State(
        values={"counts": np.array([2.0, 3.0]), "sums": np.zeros((2, 4))},
        step=7,
        meta={"fallback_means": np.ones((2, 4)), "note": "x", "flag": True},
        runtime={"final_epoch": 4},
        accumulators={
            "final_values": {"counts": np.array([1.0, 1.0])},
            "final_complete": True,
        },
    )


def test_a_snapshot_is_json_safe_and_round_trips():
    original = _state()
    payload = original.to_dict()
    assert payload["version"] == STATE_VERSION
    text = json.dumps(payload)          # arrays must not leak into JSON as objects
    back = State.from_dict(json.loads(text))
    assert back.step == 7
    assert np.array_equal(back["counts"], original["counts"])
    assert np.array_equal(back["sums"], original["sums"])
    assert back.meta["fallback_means"].dtype == np.float64
    assert np.array_equal(back.meta["fallback_means"],
                          original.meta["fallback_means"])
    assert back.meta["flag"] is True
    assert back.runtime == {"final_epoch": 4}
    assert back.accumulators["final_complete"] is True
    assert np.array_equal(back.accumulators["final_values"]["counts"],
                          np.array([1.0, 1.0]))


def test_copy_is_deep_in_every_pocket():
    original = _state()
    clone = original.copy()
    clone["counts"][0] = 99.0
    clone.meta["fallback_means"][0, 0] = -5.0
    clone.accumulators["final_values"]["counts"][0] = -1.0
    clone.runtime["final_epoch"] = 0
    assert original["counts"][0] == 2.0
    assert original.meta["fallback_means"][0, 0] == 1.0
    assert original.accumulators["final_values"]["counts"][0] == 1.0
    assert original.runtime["final_epoch"] == 4


def test_pockets_are_not_shared_between_states():
    a = State({"x": np.zeros(1)})
    b = State({"x": np.zeros(1)})
    a.runtime["k"] = 1
    a.accumulators["v"] = np.zeros(1)
    assert b.runtime == {}
    assert b.accumulators == {}


def test_a_future_version_is_refused():
    payload = _state().to_dict()
    payload["version"] = STATE_VERSION + 1
    with pytest.raises(ValueError, match="unsupported State version"):
        State.from_dict(payload)
