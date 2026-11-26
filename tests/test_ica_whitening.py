"""Truncated whitening must keep the leading directions, not eigh's first rows."""

from __future__ import annotations

import numpy as np
import pytest

from emaopt.algorithms.ica import StatefulICA, _whiten, fastica, whitening_matrix
from emaopt.benchmarks import datasets, metrics


def _anisotropic(n=4000, d=10, seed=0):
    rng = np.random.default_rng(seed)
    scales = np.linspace(5.0, 0.2, d)  # strong top directions, weak tail
    return rng.normal(size=(n, d)) * scales


def test_truncated_whitening_keeps_the_leading_eigen_directions():
    x = _anisotropic()
    cov = (x.T @ x) / x.shape[0]
    xw, wz = _whiten(x, 3)

    assert xw.shape == (x.shape[0], 3)
    assert wz.shape == (3, x.shape[1])
    # whitened coordinates: unit variance and uncorrelated
    assert np.allclose((xw.T @ xw) / x.shape[0], np.eye(3), atol=1e-8)
    # the kept directions are the top three, so the bottom eigenvectors are nulled
    s, u = np.linalg.eigh(0.5 * (cov + cov.T))
    bottom = u[:, :2]  # eigh returns ascending eigenvalues
    assert np.allclose(wz @ bottom, 0.0, atol=1e-8)
    # and they are not what slicing the full whitening matrix's rows would give
    full = whitening_matrix(cov)
    assert not np.allclose(wz, full[:3])


def test_full_whitening_is_unchanged_and_still_whitens():
    x = _anisotropic()
    cov = (x.T @ x) / x.shape[0]
    wz = whitening_matrix(cov)
    assert wz.shape == (x.shape[1], x.shape[1])
    assert np.allclose(wz @ cov @ wz.T, np.eye(x.shape[1]), atol=1e-8)


def test_fastica_with_fewer_components_returns_unit_variance_sources():
    x = _anisotropic()
    result = fastica(x, n_components=3, niter=50)
    assert result.sources.shape == (x.shape[0], 3)
    variance = result.sources.var(axis=0)
    # whitened sources: the mixture then sees features of O(1), which is the whole
    # point -- the bug left them at O(10) and inflated every downstream NLL
    assert np.all(variance > 0.2)
    assert np.all(variance < 5.0)


def test_stateful_ica_decode_inverts_its_own_measurement():
    x = _anisotropic(n=600, d=6)
    model = StatefulICA(n_components=3, seed=0)
    state = model.initialize(x, None)
    params = model.decode(state)
    sources = (x - params["mean"]) @ params["unmixing"].T
    whitened = model._whiten(x, state)
    direct = whitened @ np.asarray(state["W"], dtype=np.float64).T
    assert np.allclose(sources, direct, atol=1e-9)


def test_fastica_is_translation_invariant():
    """A constant offset must not change the ICA solution.

    Whitening the raw second moment made the offset the leading direction: on
    this fixture a +100 shift dropped one recovered direction from 0.96 to 0.06
    alignment.  Centring the rows restores the fit and makes ``transform`` use
    the same frame.
    """
    sources = datasets.laplace_sources(n=4000, d=3, seed=0)
    mixed = datasets.linear_mix(sources, seed=1)
    x = mixed.x
    base = fastica(x, 3, niter=200, seed=0)
    shifted = fastica(x + 100.0, 3, niter=200, seed=0)

    assert np.allclose(shifted.mean, (x + 100.0).mean(axis=0))
    assert np.allclose(base.unmixing, shifted.unmixing, atol=1e-8)
    assert np.allclose(base.sources, shifted.sources, atol=1e-8)
    assert np.allclose(shifted.transform(x + 100.0), shifted.sources, atol=1e-10)
    assert metrics.best_source_correlation(sources.x, shifted.sources) > 0.99
