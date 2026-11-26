"""The public surface is a contract: every name in ``__all__`` must resolve.

This is the test that would have caught the 0.1.0 defect where four sufficient
statistics were listed in ``__all__`` but only importable from ``emaopt.state``
-- ``from emaopt import *`` raised ``AttributeError`` while the suite stayed
green, because every test imported them from the subpackage.
"""

from __future__ import annotations

import pytest

import emaopt


def test_every_all_name_resolves() -> None:
    missing = []
    for name in emaopt.__all__:
        try:
            getattr(emaopt, name)
        except AttributeError:
            missing.append(name)
    assert not missing, f"names in __all__ that do not resolve: {missing}"


def test_all_has_no_duplicates() -> None:
    assert len(emaopt.__all__) == len(set(emaopt.__all__))


def test_star_import_executes() -> None:
    namespace: dict[str, object] = {}
    exec("from emaopt import *", namespace)  # noqa: S102 - the contract itself
    for name in emaopt.__all__:
        assert name in namespace, name


def test_documented_statistics_are_top_level() -> None:
    from emaopt.state import CentroidStat, GramianStat, OptimizerStat, SufficientStat

    assert emaopt.SufficientStat is SufficientStat
    assert emaopt.CentroidStat is CentroidStat
    assert emaopt.GramianStat is GramianStat
    assert emaopt.OptimizerStat is OptimizerStat


def test_quickstart_runs() -> None:
    import numpy as np

    x = np.random.default_rng(0).normal(size=(2000, 8))
    res = emaopt.emeans(x, k=8, nepochs=2)
    assert res.centroids.shape == (8, 8)
    assert np.isfinite(res.objective)

    model = emaopt.GaussianMixture(4, covariance="diag")
    fit = model.fit(x)
    assert np.isfinite(fit.objective)
    probs = model.predict_proba(x[:16], fit.params)
    assert probs.shape == (16, 4)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_version_is_importable() -> None:
    assert emaopt.__version__ and emaopt.__version__.count(".") >= 1


@pytest.mark.parametrize("name", ["gpu", "hostkernels", "levels"])
def test_optional_surfaces_import_cleanly(name: str) -> None:
    """Optional modules must fail with ImportError only when actually reached."""
    import importlib

    if name == "gpu":
        pytest.importorskip("torch")
    importlib.import_module(f"emaopt.{name}")
