"""Acceptance tests for scalar quantization with orthogonal rotations.

The tests pin the measured ordering and, just as importantly, the control: on
Gaussian data the rotation must do nothing, because what it buys is tail
spreading, not compression.
"""

from __future__ import annotations

import numpy as np
import pytest

from emaopt.algorithms import ScalarQuantizer
from emaopt.algorithms.scalar_quant import hadamard_matrix


def heavy_tailed(n: int = 4000, d: int = 16, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_t(3, size=(n, d))


def test_hadamard_rotation_cuts_heavy_tail_distortion():
    x = heavy_tailed()
    plain = ScalarQuantizer(3, rotation="none").fit(x)
    rotated = ScalarQuantizer(3, rotation="hadamard").fit(x)
    assert rotated.relative_error(x) < plain.relative_error(x) / 2.5
    assert rotated.bits_per_vector() == 48.0  # 16 coordinates, 3 bits, no padding


def test_rotation_does_not_help_gaussian_data():
    x = np.random.default_rng(0).normal(size=(4000, 16))
    plain = ScalarQuantizer(3, rotation="none").fit(x).relative_error(x)
    rotated = ScalarQuantizer(3, rotation="hadamard").fit(x).relative_error(x)
    assert abs(rotated - plain) / plain < 0.15


def test_learned_rotation_beats_none_but_loses_to_hadamard():
    """Procrustes fits squared error, which the tails dominate; Hadamard spreads them."""
    x = heavy_tailed()
    plain = ScalarQuantizer(3, rotation="none").fit(x).relative_error(x)
    learned = ScalarQuantizer(3, rotation="learned").fit(x).relative_error(x)
    hadamard = ScalarQuantizer(3, rotation="hadamard").fit(x).relative_error(x)
    assert learned < plain / 1.5
    assert hadamard < learned


def test_error_decreases_with_bit_depth():
    x = heavy_tailed()
    errors = [ScalarQuantizer(bits, rotation="hadamard").fit(x).relative_error(x)
              for bits in (2, 3, 4)]
    assert errors[0] > errors[1] > errors[2]


def test_non_power_of_two_dimension_is_padded():
    x = heavy_tailed(d=12)
    model = ScalarQuantizer(3, rotation="hadamard").fit(x)
    codes = model.encode(x)
    assert codes.shape == (x.shape[0], 16)
    assert model.decode(codes).shape == x.shape
    assert model.bits_per_vector() == 48.0


def test_codes_stay_in_range_and_shapes_are_checked():
    x = heavy_tailed(d=16)
    model = ScalarQuantizer(3).fit(x)
    codes = model.encode(x)
    assert codes.min() >= 0
    assert codes.max() <= model.levels - 1
    with pytest.raises(ValueError):
        model.encode(np.zeros((3, 15)))
    with pytest.raises(ValueError):
        ScalarQuantizer(3, rotation="dct")
    with pytest.raises(ValueError):
        hadamard_matrix(6)


def test_auto_uses_min_range_for_a_variance_hierarchy():
    rng = np.random.default_rng(0)
    scales = np.logspace(0.0, 1.5, 16)
    z = rng.standard_t(4, size=(4000, 16)) * scales
    q, _ = np.linalg.qr(rng.normal(size=(16, 16)))
    x = z @ q.T
    model = ScalarQuantizer(3, rotation="auto").fit(x)
    assert model.describe()["resolved_rotation"] == "min_range"
    assert model.bits_per_vector() == 48.0
    assert model.relative_error(x) <= ScalarQuantizer(
        3, rotation="hadamard").fit(x).relative_error(x)


def test_auto_spread_is_translation_invariant():
    """A constant offset is not a scale hierarchy."""
    rng = np.random.default_rng(1)
    scales = np.logspace(0.0, 1.5, 16)
    x = rng.standard_t(4, size=(2000, 16)) * scales
    base = ScalarQuantizer._principal_scale_spread(x)
    shifted = ScalarQuantizer._principal_scale_spread(x + 100.0)
    assert shifted == pytest.approx(base, rel=1e-8)


def test_auto_keeps_hadamard_for_an_isotropic_problem():
    x = heavy_tailed()
    model = ScalarQuantizer(3, rotation="auto").fit(x)
    assert model.describe()["resolved_rotation"] == "hadamard"
    assert model.bits_per_vector() == 48.0


def test_auto_will_not_pay_the_search_cost_above_the_dimension_gate():
    rng = np.random.default_rng(0)
    scales = np.logspace(0.0, 1.5, 48)
    z = rng.standard_t(4, size=(500, 48)) * scales
    model = ScalarQuantizer(3, rotation="auto").fit(z)
    assert model.describe()["resolved_rotation"] == "hadamard"


def test_min_range_rotation_does_not_pad():
    x = heavy_tailed(d=12)
    model = ScalarQuantizer(3, rotation="min_range").fit(x)
    assert model.bits_per_vector() == 36.0
    assert model.decode(model.encode(x)).shape == x.shape


def test_min_range_is_close_to_hadamard_on_isotropic_heavy_tails():
    """The fixed basis is already a good local optimum; the search does not beat it
    (0.3488 against 0.3371 measured), so this pins near-parity rather than a win."""
    x = heavy_tailed()
    fixed = ScalarQuantizer(3, rotation="hadamard").fit(x).relative_error(x)
    searched = ScalarQuantizer(3, rotation="min_range").fit(x).relative_error(x)
    assert searched <= fixed * 1.05


def test_min_range_is_not_worse_on_a_variance_hierarchy():
    rng = np.random.default_rng(0)
    scales = np.logspace(0.0, 1.5, 16)
    z = rng.standard_t(4, size=(4000, 16)) * scales
    q, _ = np.linalg.qr(rng.normal(size=(16, 16)))
    x = z @ q.T
    plain = ScalarQuantizer(3, rotation="none").fit(x).relative_error(x)
    fixed = ScalarQuantizer(3, rotation="hadamard").fit(x).relative_error(x)
    searched = ScalarQuantizer(3, rotation="min_range").fit(x).relative_error(x)
    assert searched < plain
    assert searched <= fixed


def test_scalar_quantizer_is_exported_at_the_top_level():
    from emaopt import ScalarQuantizer as top_level

    assert top_level is ScalarQuantizer
    assert "ScalarQuantizer" in __import__("emaopt").__all__


def test_hadamard_matrix_is_orthonormal():
    h = hadamard_matrix(16)
    np.testing.assert_allclose(h @ h.T, np.eye(16), atol=1e-12)
