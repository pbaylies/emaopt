"""Phase-3 ports: PCA Gramian, NMF moments, LDA counts, dictionary moments."""

from __future__ import annotations

import json

import numpy as np

from emaopt.algorithms import DictionaryLearner, LDA, NMF, PCACovariance
from emaopt.algorithms.dictionary import DictionaryStat
from emaopt.algorithms.lda import LDAStat
from emaopt.algorithms.nmf import NMFStat
from emaopt.benchmarks import datasets
from emaopt.core import State
from emaopt.state.sufficient import GramianStat, merge_stats


def test_pca_from_blocks_matches_from_data():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(900, 8)) @ rng.normal(size=(8, 8)) + 2.0
    model, full = PCACovariance.from_data(x, 4)
    _, blocked = PCACovariance.from_blocks(
        [x[start : start + 100] for start in range(0, 900, 100)], 4
    )
    assert np.allclose(full["eigenvalues"], blocked["eigenvalues"], rtol=1e-10, atol=1e-12)
    assert np.allclose(full["mean"], blocked["mean"], atol=1e-12)


def test_pca_from_data_survives_a_large_offset():
    """Centring before the Gramian avoids the 1e16 cancellation."""
    rng = np.random.default_rng(9)
    x = rng.normal(size=(1200, 8)) @ rng.normal(size=(8, 2))
    offset = 1e8
    shifted = x + offset
    _, params = PCACovariance.from_data(shifted, 2)
    centred = shifted - shifted.mean(axis=0)
    cov = centred.T @ centred / shifted.shape[0]
    want = np.linalg.eigvalsh(cov)[::-1][:2]
    assert np.allclose(params["mean"], shifted.mean(axis=0))
    assert np.allclose(params["eigenvalues"], want, rtol=1e-9, atol=1e-9)
    # the old raw-moment form produced 36.97 and -0.97 here instead of 10.34
    # and 3.90, because a unit variance is below the 1e8 offset's rounding
    model = PCACovariance(2)
    assert model.objective(shifted, params) < 1e-12

    _, blocked = PCACovariance.from_blocks(
        [shifted[start : start + 300] for start in range(0, 1200, 300)],
        2,
        mean=shifted.mean(axis=0),
    )
    assert np.allclose(blocked["eigenvalues"], want, rtol=1e-9, atol=1e-9)


def test_pca_gramian_stat_matches_the_covariance_optimizer():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(400, 6))
    model = PCACovariance(3, seed=0)
    state = model.initialize(x)
    batch = model.decode(State({"C": (x.T @ x) / 400}, 0, state.meta))
    stat = model.stat(center=False)  # the optimizer's state is uncentred
    for block in np.array_split(x, 4):
        stat.update(block)
    decoded = stat.decode()
    assert np.allclose(decoded["eigenvalues"], batch["eigenvalues"], rtol=1e-10)
    assert np.allclose(decoded["covariance"], (x.T @ x) / 400, atol=1e-12)


def test_nmf_stat_matches_the_batch_measurement_and_decode():
    ds = datasets.nonneg_low_rank(n=300, d=8, rank=3, seed=2)
    model = NMF(3, seed=0)
    state = model.initialize(ds.x)
    basis = model.decode(state)["basis"]
    batch = model.measure(ds.x, state)

    stat = NMFStat(3, seed=0)
    stat.prepare(basis)
    stat.update(ds.x)
    expected = model.decode(State(dict(batch.values), 0, state.meta))
    assert np.allclose(stat.values["A"], batch["A"], atol=1e-12)
    assert np.allclose(stat.values["B"], batch["B"], atol=1e-12)
    assert np.allclose(stat.decode()["basis"], expected["basis"], atol=1e-12)
    assert json.dumps(stat.config())


def test_lda_stat_matches_the_batch_measurement_and_decode():
    ds = datasets.topics(n_docs=120, vocab=30, n_topics=2, words_per_topic=10,
                         doc_len=15, seed=3)
    model = LDA(2, seed=0)
    state = model.initialize(ds.x)
    batch = model.measure(ds.x, state)

    stat = LDAStat(2, 30, seed=0)
    stat.prepare(state["lam"])
    stat.update(ds.x)
    expected = model.decode(State(dict(batch.values), 0, state.meta))
    # the stat accumulates raw counts and applies eta once at decode, so the
    # smoothing mass does not depend on how the stream was blocked
    assert np.allclose(stat.values["lam"] + stat.eta, batch["lam"], atol=1e-12)
    assert np.allclose(stat.decode()["beta"], expected["beta"], atol=1e-12)
    assert json.dumps(stat.config())


def test_dictionary_stat_matches_the_batch_measurement_and_decode():
    ds = datasets.sparse_codes(n=160, d=12, m=6, sparsity=2, seed=4)
    model = DictionaryLearner(6, seed=0)
    state = model.initialize(ds.x)
    atoms = model.decode(state)["atoms"]
    batch = model.measure(ds.x, state)

    stat = DictionaryStat(6, seed=0)
    stat.prepare(atoms)
    stat.update(ds.x)
    expected = model.decode(State(dict(batch.values), 0, state.meta))
    assert np.allclose(stat.values["G"], batch["G"], atol=1e-12)
    assert np.allclose(stat.values["H"], batch["H"], atol=1e-12)
    assert np.allclose(stat.decode()["atoms"], expected["atoms"], atol=1e-12)
    assert json.dumps(stat.config())


def test_ports_accumulate_blocks_without_acting_into_the_full_batch_moments():
    ds = datasets.nonneg_low_rank(n=300, d=8, rank=3, seed=2)
    model = NMF(3, seed=0)
    state = model.initialize(ds.x)
    batch = model.measure(ds.x, state)
    stat = NMFStat(3, seed=0)
    stat.prepare(model.decode(state)["basis"])
    stat.act_every = 10 ** 9
    for block in np.array_split(ds.x, 5):
        stat.update(block)
    assert np.allclose(stat.values["A"], batch["A"], atol=1e-10)
    assert np.allclose(stat.values["B"], batch["B"], atol=1e-10)


def test_ports_merge_exactly_when_the_statistic_is_additive():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(400, 7))
    single = GramianStat(n_components=3)
    for block in np.array_split(x, 8):
        single.update(block)
    parts = []
    for partition in np.array_split(x, 4):
        stat = GramianStat(n_components=3)
        for block in np.array_split(partition, 2):
            stat.update(block)
        parts.append(stat)
    merged = merge_stats(parts)
    assert np.allclose(single.values["gram"], merged.values["gram"], atol=1e-9)
    assert np.allclose(single.decode()["eigenvalues"], merged.decode()["eigenvalues"],
                       atol=1e-9)
