

def test_projection_pair_scores_geometry():
    """Detector parity + behavior (T13/REPORT 12.2): separable blobs score
    same-cluster pairs above cross-cluster pairs; deterministic; diag 1."""
    import numpy as np
    from emaopt.benchmarks.metrics import projection_pair_scores

    rng = np.random.default_rng(0)
    centers = rng.normal(size=(4, 12)) * 4.0
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    x = np.vstack([c + rng.normal(scale=0.05, size=(80, 12)) for c in centers])
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    s1 = projection_pair_scores(x, x, width=4)
    s2 = projection_pair_scores(x, x, width=4)
    assert np.array_equal(s1, s2)                      # deterministic
    assert np.allclose(np.diag(s1), 1.0, atol=1e-10)   # self-similarity
    grp = np.repeat(np.arange(4), 80)
    same = s1[grp[:, None] == grp[None, :]]
    diff = s1[grp[:, None] != grp[None, :]]
    assert same.min() > diff.max()                     # geometry separates
