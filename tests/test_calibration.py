"""Probability calibration interface and eGMM integration."""

import numpy as np

from emaopt import GaussianMixture
from emaopt.calibration import (
    CalibratedProbabilities,
    PriorCorrector,
    TemperatureScaler,
    auto_calibrate,
    auroc_score,
    average_precision_score,
    base_rate_baseline,
    binary_reliability_curve,
    brier_score,
    calibration_report,
    classwise_expected_calibration_error,
    expected_calibration_error,
    negative_log_likelihood,
    reliability_curve,
)
from emaopt.state import EMA


def _softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    z = np.exp(z)
    return z / z.sum(axis=1, keepdims=True)


def _overconfident_case(n=12_000, k=4, seed=7):
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(n, k))
    true = _softmax(logits)
    labels = np.array([rng.choice(k, p=row) for row in true])
    # Deliberately sharpen the same scores; the correct temperature is ~4.
    return _softmax(logits * 4), labels


def test_temperature_scaling_improves_heldout_calibration():
    probabilities, labels = _overconfident_case()
    train = slice(0, 8_000)
    test = slice(8_000, None)
    scaler = TemperatureScaler().fit(probabilities[train], labels[train])
    calibrated = scaler.transform(probabilities[test])
    assert scaler.temperature_ > 1.5
    assert expected_calibration_error(calibrated, labels[test]) < expected_calibration_error(
        probabilities[test], labels[test]
    )
    assert brier_score(calibrated, labels[test]) <= brier_score(
        probabilities[test], labels[test]
    ) + 1e-12
    assert negative_log_likelihood(calibrated, labels[test]) < negative_log_likelihood(
        probabilities[test], labels[test]
    )
    report = calibration_report(probabilities[test], calibrated, labels[test])
    assert report["after"]["ece"] < report["before"]["ece"]


def test_temperature_scaling_can_optimize_ece():
    probabilities, labels = _overconfident_case()
    train = slice(0, 8_000)
    test = slice(8_000, None)
    nll_fit = TemperatureScaler(criterion="nll").fit(probabilities[train], labels[train])
    ece_fit = TemperatureScaler(criterion="ece").fit(probabilities[train], labels[train])
    # The ECE criterion minimizes ECE on its calibration sample.  It need not
    # generalize better than NLL on a different sample; the experiment runner
    # reports that held-out trade-off separately.
    assert ece_fit.ece_ <= expected_calibration_error(
        probabilities[train], labels[train]
    ) + 1e-12
    assert ece_fit.temperature_ is not None
    assert nll_fit.temperature_ is not None
    assert ece_fit.criterion == "ece"


def test_temperature_scaling_preserves_argmax():
    probabilities, labels = _overconfident_case(n=1_000)
    scaler = TemperatureScaler().fit(probabilities, labels)
    calibrated = scaler.transform(probabilities)
    np.testing.assert_array_equal(calibrated.argmax(1), probabilities.argmax(1))


def test_prior_correction_matches_target_marginal_without_labels():
    rng = np.random.default_rng(11)
    logits = rng.normal(size=(20_000, 3)) + np.log([0.6, 0.3, 0.1])
    probabilities = _softmax(logits)
    target = np.array([0.2, 0.3, 0.5])
    corrector = PriorCorrector(target).fit(probabilities)
    corrected = corrector.transform(probabilities)
    np.testing.assert_allclose(corrected.mean(0), target, atol=2e-6)
    np.testing.assert_allclose(corrector.source_prior_.sum(), 1.0, atol=1e-12)
    assert corrector.iterations_ > 0
    np.testing.assert_allclose(corrected.sum(1), 1.0, atol=1e-12)


def test_common_interface_and_validation():
    probabilities, labels = _overconfident_case(n=2_000)
    calibrated = CalibratedProbabilities(method="temperature").fit_transform(
        probabilities, labels
    )
    assert calibrated.shape == probabilities.shape
    prior = CalibratedProbabilities(
        method="prior", target_prior=[0.25, 0.25, 0.25, 0.25]
    ).fit_transform(probabilities)
    np.testing.assert_allclose(prior.mean(0), 0.25, atol=1e-6)
    try:
        CalibratedProbabilities(method="temperature").fit(probabilities)
    except ValueError as exc:
        assert "labels" in str(exc)
    else:
        raise AssertionError("temperature calibration without labels must fail")
    try:
        TemperatureScaler().fit(probabilities, labels[:-1])
    except ValueError as exc:
        assert "shape" in str(exc)
    else:
        raise AssertionError("mismatched labels must fail")


def test_discrimination_metrics_and_constant_baseline():
    scores = np.array([0.1, 0.2, 0.8, 0.9, 0.4, 0.4])
    labels = np.array([0, 0, 1, 1, 0, 1])
    assert np.isclose(auroc_score(scores, labels), 17.0 / 18.0)
    assert np.isclose(average_precision_score(scores, labels), 11.0 / 12.0)

    constant = np.full_like(scores, labels.mean())
    assert np.isclose(auroc_score(constant, labels), 0.5)
    assert np.isclose(average_precision_score(constant, labels), labels.mean())
    np.testing.assert_allclose(base_rate_baseline(labels), [0.5, 0.5])


    two_column = np.column_stack([1.0 - scores, scores])
    assert np.isclose(auroc_score(two_column, labels), 17.0 / 18.0)


def test_quantile_reliability_handles_saturated_scores():
    scores = np.full(100, 0.999)
    labels = np.array([0, 1] * 50)
    curve = binary_reliability_curve(scores, labels)
    assert curve["binning"] == "quantile"
    assert curve["nonempty_bins"] == 1
    assert curve["bins"][0]["count"] == 100
    assert np.isclose(curve["bins"][0]["mean_probability"], 0.999)
    assert np.isclose(curve["bins"][0]["observed_rate"], 0.5)


def test_classwise_ece_is_zero_for_perfect_probabilities():
    labels = np.array([0, 1, 2, 0, 1, 2])
    probabilities = np.eye(3)[labels]
    assert np.isclose(classwise_expected_calibration_error(probabilities, labels), 0.0)
    assert np.isclose(expected_calibration_error(probabilities, labels), 0.0)


def test_auto_calibration_requires_a_target_and_selects_by_input():
    probabilities, labels = _overconfident_case(n=2_000)
    fitted = CalibratedProbabilities().fit(probabilities, labels)
    assert fitted.method_ == "temperature"
    assert "labels" in fitted.reason_

    prior = CalibratedProbabilities(target_prior=[0.25, 0.25, 0.25, 0.25]).fit(
        probabilities
    )
    assert prior.method_ == "prior"
    assert "prior" in prior.reason_

    try:
        CalibratedProbabilities().fit(probabilities)
    except ValueError as exc:
        assert "labels" in str(exc) and "target_prior" in str(exc)
    else:
        raise AssertionError("auto calibration without labels or a prior must fail")

    direct = auto_calibrate(probabilities, labels)
    assert direct.method_ == "temperature"


def test_calibration_report_contains_scores_discrimination_and_control():
    rng = np.random.default_rng(5)
    logits = rng.normal(size=(4_000, 2))
    true = _softmax(logits)
    labels = (rng.random(4_000) > true[:, 0]).astype(np.int64)
    probabilities = _softmax(logits * 4)

    scaler = TemperatureScaler().fit(probabilities[:3_000], labels[:3_000])
    calibrated = scaler.transform(probabilities[3_000:])
    report = calibration_report(
        probabilities[3_000:],
        calibrated,
        labels[3_000:],
        baseline_labels=labels[:3_000],
    )
    assert report["settings"]["baseline_source"] == "baseline_labels"
    assert report["discrimination"]["before"]["kind"] == "binary"
    assert 0.0 <= report["discrimination"]["before"]["auroc"] <= 1.0
    assert report["baseline"]["ece"] >= 0.0
    assert "binary_positive" in report["reliability"]
    assert report["reliability"]["after"]["binning"] == "quantile"


def test_gmm_predict_proba_matches_responsibilities():
    rng = np.random.default_rng(3)
    x = np.vstack(
        [rng.normal(-1.5, 0.6, size=(300, 3)), rng.normal(1.5, 0.6, size=(300, 3))]
    )
    model = GaussianMixture(2, covariance="diag", updater=EMA(alpha=0.5), seed=0)
    result = model.fit(x, batch_size=300, nepochs=2, seed=0)
    proba = model.predict_proba(x, result.params)
    np.testing.assert_allclose(proba, model.responsibilities(x, result.params))
    np.testing.assert_allclose(proba.sum(1), 1.0, atol=1e-12)
    try:
        model.predict_proba(x)
    except ValueError as exc:
        assert "params" in str(exc)
    else:
        raise AssertionError("predict_proba without params must fail")
