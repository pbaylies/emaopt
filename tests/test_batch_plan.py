"""T38: one batch plan, and every stream agrees with it.

The arithmetic was written six times with the same formula; this pins that the
six sites still produce the counts the plan says, on a grid that includes the
exact-off multiples, the partial trailing batch both ways, and a cap.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from emaopt.core import _iter_batches
from emaopt.plan import BatchPlan, plan_batches
from emaopt.stream import block_steps, iter_blocks

GRID = [(100, 32, 3), (400, 50, 2), (1000, 100, 1), (97, 10, 2), (5, 128, 4)]


@pytest.mark.parametrize("n,bs,epochs", GRID)
@pytest.mark.parametrize("drop_last", [False, True])
def test_the_plan_is_the_formula(n, bs, epochs, drop_last):
    plan = plan_batches(n, bs, epochs, drop_last=drop_last)
    expected = (n // bs if drop_last else math.ceil(n / bs))
    assert isinstance(plan, BatchPlan)
    assert plan.per_epoch == max(1, expected)
    assert plan.total_steps == plan.per_epoch * epochs
    # an over-wide batch is clamped to the data, never to a zero-length pass
    assert plan.batch_size == min(bs, n)


def test_a_cap_and_a_full_batch():
    assert plan_batches(1000, 100, 5, max_updates=7).total_steps == 7
    full = plan_batches(1000, None, 2)
    assert full.batch_size == 1000 and full.per_epoch == 1
    assert full.total_steps == 2
    # a batch wider than the data is clamped, never zero updates
    assert plan_batches(5, 1000).batch_size == 5


@pytest.mark.parametrize("n,bs,epochs", GRID)
@pytest.mark.parametrize("drop_last", [False, True])
def test_both_batch_streams_emit_the_planned_count(n, bs, epochs, drop_last):
    x = np.arange(n * 2, dtype=np.float64).reshape(n, 2)
    plan = plan_batches(n, bs, epochs, drop_last=drop_last)
    host = list(_iter_batches(x, bs, epochs, np.random.default_rng(0),
                              True, drop_last, None))
    assert len(host) == plan.total_steps
    blocks = list(iter_blocks(x, bs, nepochs=epochs, shuffle=True,
                              drop_last=drop_last))
    assert len(blocks) == plan.total_steps
    assert block_steps(n, bs, nepochs=epochs, drop_last=drop_last) == plan.total_steps


def test_the_device_stream_uses_the_same_count():
    pytest.importorskip("torch")
    from emaopt.gpu import _batch_slices

    x = np.zeros((400, 2))
    for drop_last in (False, True):
        plan = plan_batches(400, 50, 3, drop_last=drop_last)
        rows = list(_batch_slices(400, 50, 3, 0, True, drop_last, None))
        assert len(rows) == plan.total_steps
        # and the cap is honoured identically
        capped = list(_batch_slices(400, 50, 3, 0, True, drop_last, 4))
        assert len(capped) == 4
