"""The covariance floor follows the data's scale, in every family that has one.

``reg_covar`` used to be absolute: 1e-6 of variance wherever the data happened to
live.  On the LAION feature file the ICA half is stored at variance 1.0e-6 --
the old floor exactly -- so the block sat on the floor rather than on its own
estimate: the joint fit conditioned on the PCA half 112 nats worse than the
block's own density, and the block only helped after every column was rescaled
by hand.  These tests pin the properties that make the rescale unnecessary in
every family that carries a covariance scale -- ``tied``, ``diag``,
``spherical``, ``full`` and ``lowrank``: the floor is a fraction of the data's
own mean variance, one global rescale of the input cannot move the fit, and a
block whose variance is far below the old absolute floor is no longer clamped
to it.
"""

import numpy as np
import pytest

from emaopt.algorithms.gmm import GaussianMixture


def test_the_tied_floor_is_a_fraction_of_the_mean_variance():
    model = GaussianMixture(2, covariance="tied")
    cov = np.diag([4.0, 1.0, 0.0])
    assert model.reg_covar_floor(cov) == pytest.approx(1e-6 * 5.0 / 3.0)


def test_the_diag_floor_reads_a_variance_vector():
    model = GaussianMixture(2, covariance="diag")
    assert model.reg_covar_floor(np.array([4.0, 1.0, 0.0])) == pytest.approx(
        1e-6 * 5.0 / 3.0)


def test_the_absolute_floor_is_still_available():
    model = GaussianMixture(2, covariance="tied", reg_covar_relative=False)
    assert model.reg_covar_floor(np.diag([4.0, 1.0])) == pytest.approx(1e-6)


def test_describe_records_which_floor_was_used():
    info = GaussianMixture(2, covariance="diag").describe()
    assert info["reg_covar"] == 1e-6
    assert info["reg_covar_relative"] is True


def test_a_global_rescale_does_not_change_the_fitted_shape():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(800, 4)) @ np.diag([1.0, 2.0, 0.5, 3.0])
    base = GaussianMixture(3, covariance="tied", seed=0).fit(
        x, batch_size=None, nepochs=6, seed=0, tol=None
    )
    factor = 1000.0
    scaled = GaussianMixture(3, covariance="tied", seed=0).fit(
        factor * x, batch_size=None, nepochs=6, seed=0, tol=None
    )
    # a scale-equivariant floor means the same local optimum, in other units
    np.testing.assert_allclose(scaled.params["covariances"] / factor**2,
                               base.params["covariances"], rtol=1e-7, atol=1e-12)
    np.testing.assert_allclose(scaled.params["means"] / factor,
                               base.params["means"], rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(scaled.params["weights"],
                               base.params["weights"], rtol=1e-7, atol=1e-12)


@pytest.mark.parametrize("family", ["diag", "spherical"])
def test_a_global_rescale_does_not_change_the_fitted_shape_of_a_diagonal_family(family):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(600, 3)) * np.array([1.0, 4.0, 0.25])
    base = GaussianMixture(3, covariance=family, seed=0).fit(
        x, batch_size=None, nepochs=6, seed=0, tol=None
    )
    factor = 50.0
    scaled = GaussianMixture(3, covariance=family, seed=0).fit(
        factor * x, batch_size=None, nepochs=6, seed=0, tol=None
    )
    np.testing.assert_allclose(scaled.params["covariances"] / factor**2,
                               base.params["covariances"], rtol=1e-7, atol=1e-12)
    np.testing.assert_allclose(scaled.params["means"] / factor,
                               base.params["means"], rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(scaled.params["weights"],
                               base.params["weights"], rtol=1e-7, atol=1e-12)


def test_a_block_below_the_old_absolute_floor_is_no_longer_clamped():
    rng = np.random.default_rng(5)
    n = 2000
    x = np.hstack([rng.normal(size=(n, 4)) * 0.03,     # variance 9e-4
                   rng.normal(size=(n, 4)) * 1e-4])    # variance 1e-8

    def fit(**kw):
        model = GaussianMixture(4, covariance="tied", seed=0, **kw)
        return model.fit(x, batch_size=None, nepochs=5, seed=0, tol=None)

    relative = fit()
    absolute = fit(reg_covar_relative=False)
    lo_rel = float(np.linalg.eigvalsh(
        np.asarray(relative.params["covariances"]))[0])
    lo_abs = float(np.linalg.eigvalsh(
        np.asarray(absolute.params["covariances"]))[0])

    assert lo_abs >= 1e-6 * (1.0 - 1e-9)   # the absolute floor is on the block
    assert lo_rel < 1e-7                   # the relative floor leaves it alone
    assert relative.objective < absolute.objective


def test_a_diag_block_below_the_old_absolute_floor_is_no_longer_clamped():
    rng = np.random.default_rng(6)
    n = 2000
    x = np.hstack([rng.normal(size=(n, 4)) * 0.03,     # variance 9e-4
                   rng.normal(size=(n, 4)) * 1e-4])    # variance 1e-8

    def fit(**kw):
        model = GaussianMixture(4, covariance="diag", seed=0, **kw)
        return model.fit(x, batch_size=None, nepochs=5, seed=0, tol=None)

    relative = fit()
    absolute = fit(reg_covar_relative=False)
    small_rel = float(np.asarray(relative.params["covariances"])[:, 4:].min())
    small_abs = float(np.asarray(absolute.params["covariances"])[:, 4:].min())

    assert small_abs >= 1e-6 * (1.0 - 1e-9)   # clipped to the absolute floor
    assert small_rel < 1e-7                   # the relative floor leaves it alone
    assert relative.objective < absolute.objective


def test_the_vectorised_floor_matches_the_scalar_rule():
    model = GaussianMixture(2, covariance="diag")
    scales = np.array([4.0, 1.0, 0.0, np.nan])
    floors = model._reg_covar_floors(scales)
    assert floors[0] == pytest.approx(4e-6)
    assert floors[1] == pytest.approx(1e-6)
    # a degenerate scale falls back to the absolute number, exactly as the
    # scalar method does, so the two entry points cannot disagree
    assert floors[2] == pytest.approx(1e-6)
    assert floors[3] == pytest.approx(1e-6)
    absolute = GaussianMixture(2, covariance="diag", reg_covar_relative=False)
    np.testing.assert_array_equal(absolute._reg_covar_floors(scales),
                                  np.full(4, 1e-6))


def test_a_full_covariance_block_below_the_old_absolute_floor_is_no_longer_clamped():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(1500, 3)) * np.array([1e-3, 1e-3, 1e-4])
    labels = np.zeros(x.shape[0], dtype=int)
    _rel_model, rel = GaussianMixture.from_labels(x, labels, covariance="full")
    _abs_model, absolute = GaussianMixture.from_labels(
        x, labels, covariance="full", reg_covar_relative=False)
    lo_rel = float(np.linalg.eigvalsh(np.asarray(rel["covariances"])[0])[0])
    lo_abs = float(np.linalg.eigvalsh(np.asarray(absolute["covariances"])[0])[0])

    assert lo_abs >= 1e-6 * (1.0 - 1e-9)  # every eigenvalue lifted onto 1e-6
    assert lo_rel < 1e-7                  # the 1e-8 direction survives


def test_a_lowrank_psi_below_the_old_absolute_floor_is_no_longer_clamped():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(2000, 4)) * np.array([1e-3, 1e-3, 1e-3, 1e-4])

    def psi(**kw):
        model = GaussianMixture(2, covariance="lowrank", rank=2, seed=0, **kw)
        return np.asarray(model.decode(model.initialize(x))["covariances"])

    rel = psi()
    absolute = psi(reg_covar_relative=False)
    assert absolute[:, 3].min() >= 1e-6 * (1.0 - 1e-9)  # clipped to the floor
    assert rel[:, 3].min() < 1e-7                       # left on its estimate


def test_the_seed_and_from_labels_use_the_same_floor():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(3000, 3)) * 1e-3              # variance 1e-6
    labels = np.zeros(x.shape[0], dtype=int)           # component 1 never seen
    rel_model, rel = GaussianMixture.from_labels(
        x, labels, covariance="diag", n_components=2)
    _abs_model, absolute = GaussianMixture.from_labels(
        x, labels, covariance="diag", reg_covar_relative=False, n_components=2)

    assert rel_model.describe()["reg_covar_relative"] is True
    # the unseen component takes the global variance; the absolute reading adds
    # its 1e-6 floor on top of that same variance, so it is seeded twice as wide
    assert rel["covariances"][1].mean() == pytest.approx(1e-6, rel=0.05)
    assert absolute["covariances"][1].mean() == pytest.approx(2e-6, rel=0.05)


def test_the_seed_can_run_in_each_columns_own_units():
    model = GaussianMixture(2, covariance="diag", init_scale="std")
    sub = np.array([[0.0, 0.0], [1.0, 1e-9], [2.0, 2e-9]])
    rows, scale = model._seed_scaling(sub)
    np.testing.assert_allclose(rows, sub / scale)
    assert scale[1] == pytest.approx(sub[:, 1].std())
    # a zero-variance column keeps its units rather than dividing by zero
    _rows, flat = model._seed_scaling(np.array([[1.0, 5.0], [1.0, 6.0]]))
    assert flat[0] == 1.0


def test_the_raw_seed_is_the_historical_one():
    rng = np.random.default_rng(13)
    x = rng.normal(size=(400, 3))

    def fit(**kw):
        return GaussianMixture(3, covariance="diag", seed=0, **kw).fit(
            x, batch_size=None, nepochs=3, seed=0, tol=None)

    np.testing.assert_array_equal(fit().params["means"],
                                  fit(init_scale="raw").params["means"])


def test_std_seeding_gives_a_small_block_a_vote():
    rng = np.random.default_rng(14)
    n = 600
    big = rng.normal(size=(n, 2))
    # the small block is bimodal at three of *its own* standard deviations
    small = np.where(rng.random((n, 1)) < 0.5, -1.0, 1.0) * 3e-3
    x = np.hstack([big, small + rng.normal(size=(n, 2)) * 1e-3])

    def separated(**kw):
        model = GaussianMixture(2, covariance="diag", seed=0, **kw)
        seeded = model.initialize(x).meta["fallback_means"]
        return float(np.ptp(seeded[:, 2:], axis=0).max())

    assert separated(init_scale="std") > 1e-3   # the small block's split
    assert separated() < 1e-3                   # invisible in raw units


def test_a_full_covariance_fit_at_variance_one_e_minus_6_keeps_its_scale():
    rng = np.random.default_rng(12)
    n = 1200
    x = rng.normal(size=(n, 3)) * np.array([1e-3, 1e-3, 1e-4])
    x[n // 2:] += 2e-3

    def fit(**kw):
        model = GaussianMixture(2, covariance="full", seed=0, **kw)
        return model.fit(x, batch_size=None, nepochs=4, seed=0, tol=None)

    rel = fit()
    absolute = fit(reg_covar_relative=False)
    lo_rel = min(float(np.linalg.eigvalsh(np.asarray(c))[0])
                 for c in rel.params["covariances"])
    lo_abs = min(float(np.linalg.eigvalsh(np.asarray(c))[0])
                 for c in absolute.params["covariances"])

    assert lo_abs >= 1e-6 * (1.0 - 1e-9)
    assert lo_rel < 1e-7
    assert rel.objective < absolute.objective


def test_a_lowrank_fit_at_variance_one_e_minus_6_keeps_its_scale():
    rng = np.random.default_rng(12)
    n = 1200
    x = rng.normal(size=(n, 3)) * np.array([1e-3, 1e-3, 1e-4])
    x[n // 2:] += 2e-3

    def fit(**kw):
        model = GaussianMixture(2, covariance="lowrank", rank=2, seed=0, **kw)
        return model.fit(x, batch_size=None, nepochs=4, seed=0, tol=None)

    rel = fit()
    absolute = fit(reg_covar_relative=False)
    psi_rel = float(np.asarray(rel.params["covariances"])[:, 2].min())
    psi_abs = float(np.asarray(absolute.params["covariances"])[:, 2].min())

    assert psi_abs >= 1e-6 * (1.0 - 1e-9)
    assert psi_rel < 1e-7
    assert rel.objective < absolute.objective
