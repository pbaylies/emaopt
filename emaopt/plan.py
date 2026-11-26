"""The library's one batch plan (T38).

``per_epoch``/``total_steps`` was written six times -- in ``core.fit``,
``core._iter_batches``, ``stream.iter_blocks``, ``stream.block_steps``,
``gpu._batch_slices`` and ``gpu.egmm_gpu_fit`` -- with the same arithmetic in each
and different ``drop_last`` defaults at the call sites.  Duplicated arithmetic is
where an off-by-one becomes a different number of updates in one arm and not
another: the measured instance was 120 vs 125 updates and a 1.40-nat endpoint gap
between ``core.fit`` and ``GaussianMixture.fit``.

This module is the single implementation.  It does **not** choose ``drop_last``:
that is a semantic decision of each entry point (a full-covariance fit drops the
short trailing batch because a short batch can push every component below its
identification floor; a streaming fold keeps it because discarding 18% of the
rows is worse than one small update).  Callers pass theirs, so unifying the
arithmetic moves no update count.  Flipping the defaults themselves would shift
every recorded fit and needs the objective gates re-run -- a separate change.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["BatchPlan", "plan_batches"]


@dataclass(frozen=True)
class BatchPlan:
    """How one run is cut into updates."""

    batch_size: int
    per_epoch: int
    total_steps: int
    drop_last: bool
    nepochs: int

    def as_dict(self) -> dict[str, int | bool]:
        return {
            "batch_size": self.batch_size,
            "per_epoch": self.per_epoch,
            "total_steps": self.total_steps,
            "drop_last": self.drop_last,
            "nepochs": self.nepochs,
        }


def plan_batches(
    n: int,
    batch_size: int | None,
    nepochs: int = 1,
    *,
    drop_last: bool = False,
    max_updates: int | None = None,
) -> BatchPlan:
    """Cut ``n`` rows into ``nepochs`` passes of ``batch_size``.

    ``batch_size=None`` means one full batch.  ``drop_last`` decides whether the
    trailing partial batch is an update (``False``) or is discarded (``True``);
    the caller owns that choice.  ``max_updates`` caps the total, which is what a
    device arm's captured loop and a saturating schedule both need.
    """
    n = max(0, int(n))
    bs = n if batch_size is None else int(batch_size)
    bs = max(1, min(bs, max(1, n)))
    epochs = max(1, int(nepochs))
    per_epoch = n // bs if drop_last else int(math.ceil(n / bs))
    per_epoch = max(1, per_epoch)
    total = per_epoch * epochs
    if max_updates is not None:
        total = min(total, max(0, int(max_updates)))
    return BatchPlan(batch_size=bs, per_epoch=per_epoch, total_steps=total,
                     drop_last=bool(drop_last), nepochs=epochs)
