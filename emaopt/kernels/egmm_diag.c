/* Fused diagonal-GMM E-step for the host (EMAOPT T21, restructured T27).
 *
 * Why this exists
 * ---------------
 * The shipped NumPy E-step reads the data matrix three times: once for the
 * BLAS log-density, once for ``r.T @ x`` and once for ``r.T @ (x * x)``, and it
 * materialises an ``(n, k)`` logits array plus two ``(n, d)`` temporaries.  The
 * whole E-step is row-local, so one pass over ``x`` produces the same numbers.
 *
 * Why the parameters are transposed (this is the part that matters)
 * ----------------------------------------------------------------
 * The first version kept ``inv`` and ``mean_inv`` as ``(k, d)`` and summed
 * ``a += x[c]*x[c]*inv[j*d+c]`` over ``c``.  That ``c`` loop is a *reduction*,
 * so a compiler may not vectorise it without reassociating, and it runs as two
 * dependent scalar chains: measured **1.5x slower than the BLAS path on one
 * thread** (28.4 ms vs 19.2 ms for the log-density at n=8192, k=16, d=512).
 * All of the original 5x came from OpenMP threads, not from the loop structure.
 *
 * Here the parameters are stored ``(d, k)`` and the outer loop runs over ``c``
 * with ``j`` innermost.  Each ``a[j]`` is then an *independent* accumulator, so
 * the loop vectorises with no fast-math and no reassociation, and four rows are
 * processed together so one ``inv[c, :]``/``mean_inv[c, :]`` load feeds four
 * rows (parameters are the traffic that dominates at this shape: 2*k*d doubles
 * per row *pass*, versus d doubles of ``x``).
 *
 * Each (row, j) still accumulates over ``c`` in increasing order, so the answer
 * is bit-identical to the unblocked/untransposed form -- the transformation is
 * a layout and schedule change, not an arithmetic one.
 *
 * What the caller precomputes (``emaopt/hostkernels.py``)
 * ------------------------------------------------------
 *   invT[c*k+j]      = 1 / cov[j, c]
 *   miT[c*k+j]       = mean[j, c] / cov[j, c]
 *   quad_bias[j]     = sum_c mean[j,c]^2 / cov[j,c]
 *   off[j]           = -0.5*(d*log(2*pi) + logdet_j) + log(w_j)
 *
 * so the expanded logit is ``-0.5*(a - 2b + quad_bias[j]) + off[j]`` with the
 * quadratic clamped at zero exactly as ``component_log_prob`` clamps it.  The
 * default log-density instead uses the direct ``(x - mean)^2 / cov`` row below,
 * which avoids that cancellation.
 *
 * Arithmetic is transcribed from ``GaussianMixture.component_log_prob``, from
 * ``emaopt.core.softmax_rows`` (exp-shift, max replaced by 0 when non-finite)
 * and from ``statistics_from``'s hygiene (non-finite responsibilities zeroed,
 * row sums floored at 1e-300).  The expanded form remains the fused E-step and
 * the old-ABI fallback; it cancels badly when |mean|/sd is large.
 *
 * Build (python -m emaopt.kernels.build, or this line by hand):
 *   cc -O3 -march=native -fopenmp -fPIC -shared egmm_diag.c -o libegmm_diag.so -lm
 */

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200112L   /* posix_memalign */
#endif

#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define EMAOPT_FLOOR 1e-300
/*: Rows that share one parameter load.  Four rows of 16 components keep the
 *: eight accumulator vectors in registers on AVX2 (4 doubles/vector). */
#define EMAOPT_BLOCK 4

static inline double emaopt_clamp_min(double v, double lo) {
    return v < lo ? lo : v;
}

/* 64-byte aligned scratch: per-thread slots are then a cache line apart, so two
 * threads writing their own accumulators never share a line. */
static void *emaopt_align64(size_t bytes) {
    void *p = NULL;
    if (posix_memalign(&p, 64, bytes) == 0) return p;
    return malloc(bytes);
}

static inline size_t emaopt_pad(size_t n) {
    return ((n + 7u) / 8u) * 8u;
}

static inline double emaopt_quad(double a, double b, double bias) {
    double quad = a - 2.0 * b + bias;
    return quad < 0.0 ? 0.0 : quad;      /* round-off, not geometry */
}

/* One row, transposed parameters, no blocking: the tail path and the scalar
 * reference for the blocked form. */
static inline void emaopt_row1(const double *restrict xi,
                               const double *restrict invT,
                               const double *restrict miT,
                               const double *restrict quad_bias,
                               const double *restrict off,
                               int64_t d, int64_t k, double *restrict lg) {
    for (int64_t j = 0; j < k; ++j) {
        double a = 0.0;
        double b = 0.0;
        for (int64_t c = 0; c < d; ++c) {
            const double v = xi[c];
            a += v * v * invT[(size_t)c * (size_t)k + (size_t)j];
            b += v * miT[(size_t)c * (size_t)k + (size_t)j];
        }
        lg[j] = -0.5 * emaopt_quad(a, b, quad_bias[j]) + off[j];
    }
}

/* Four rows at once: ``c`` outer, ``j`` inner, one parameter load per four
 * rows.  ``acc`` is 8*k doubles (a0,b0,a1,b1,a2,b2,a3,b3). */
static inline void emaopt_row4(const double *restrict x0,
                               const double *restrict x1,
                               const double *restrict x2,
                               const double *restrict x3,
                               const double *restrict invT,
                               const double *restrict miT,
                               const double *restrict quad_bias,
                               const double *restrict off,
                               int64_t d, int64_t k, double *restrict acc,
                               double *restrict lg0, double *restrict lg1,
                               double *restrict lg2, double *restrict lg3) {
    double *a0 = acc;
    double *b0 = acc + k;
    double *a1 = acc + 2 * k;
    double *b1 = acc + 3 * k;
    double *a2 = acc + 4 * k;
    double *b2 = acc + 5 * k;
    double *a3 = acc + 6 * k;
    double *b3 = acc + 7 * k;
    for (int64_t j = 0; j < k; ++j) {
        a0[j] = 0.0; b0[j] = 0.0;
        a1[j] = 0.0; b1[j] = 0.0;
        a2[j] = 0.0; b2[j] = 0.0;
        a3[j] = 0.0; b3[j] = 0.0;
    }
    for (int64_t c = 0; c < d; ++c) {
        const double *restrict iv = invT + (size_t)c * (size_t)k;
        const double *restrict mi = miT + (size_t)c * (size_t)k;
        const double v0 = x0[c], v1 = x1[c], v2 = x2[c], v3 = x3[c];
        const double q0 = v0 * v0, q1 = v1 * v1, q2 = v2 * v2, q3 = v3 * v3;
#pragma omp simd
        for (int64_t j = 0; j < k; ++j) {
            const double civ = iv[j];
            const double cmi = mi[j];
            a0[j] += q0 * civ;
            b0[j] += v0 * cmi;
            a1[j] += q1 * civ;
            b1[j] += v1 * cmi;
            a2[j] += q2 * civ;
            b2[j] += v2 * cmi;
            a3[j] += q3 * civ;
            b3[j] += v3 * cmi;
        }
    }
#pragma omp simd
    for (int64_t j = 0; j < k; ++j) {
        lg0[j] = -0.5 * emaopt_quad(a0[j], b0[j], quad_bias[j]) + off[j];
        lg1[j] = -0.5 * emaopt_quad(a1[j], b1[j], quad_bias[j]) + off[j];
        lg2[j] = -0.5 * emaopt_quad(a2[j], b2[j], quad_bias[j]) + off[j];
        lg3[j] = -0.5 * emaopt_quad(a3[j], b3[j], quad_bias[j]) + off[j];
    }
}

/* Direct-difference row: ``(x - mean)^2 / cov`` with the output row as the
 * accumulator.  This is the tiled form, and it removes the second
 * accumulator and the ``miT``/``quad_bias`` traffic of the expanded form.  One
 * row owns its ``out`` range, so the outer row loop is thread-safe without
 * scratch. */
static inline void emaopt_row_direct(const double *restrict xi,
                                     const double *restrict meanT,
                                     const double *restrict invT,
                                     const double *restrict off,
                                     int64_t d, int64_t k,
                                     double *restrict lg) {
    for (int64_t j = 0; j < k; ++j) lg[j] = 0.0;
    for (int64_t c = 0; c < d; ++c) {
        const double v = xi[c];
        const double *restrict mu = meanT + (size_t)c * (size_t)k;
        const double *restrict iv = invT + (size_t)c * (size_t)k;
#pragma omp simd
        for (int64_t j = 0; j < k; ++j) {
            const double diff = v - mu[j];
            lg[j] += diff * diff * iv[j];
        }
    }
#pragma omp simd
    for (int64_t j = 0; j < k; ++j) lg[j] = off[j] - 0.5 * lg[j];
}

/* Direct-difference log-density.  Kept separate from ``egmm_diag_logits`` so
 * an older artifact without this symbol still serves the expanded ABI. */
int egmm_diag_logits_direct(const double *x, const double *meanT,
                            const double *invT, const double *off,
                            double *out, int64_t n, int64_t d, int64_t k,
                            int threads) {
    if (n < 0 || d < 1 || k < 1 || x == NULL || meanT == NULL ||
        invT == NULL || off == NULL || out == NULL) return 1;
    if (n == 0) return 0;
    int nt = 1;
#ifdef _OPENMP
    omp_set_dynamic(0);
    nt = threads > 0 ? threads : omp_get_max_threads();
#else
    (void)threads;
#endif
    if (nt < 1) nt = 1;
    if ((int64_t)nt > n) nt = (int)n;
#pragma omp parallel for schedule(static) num_threads(nt)
    for (int64_t i = 0; i < n; ++i) {
        emaopt_row_direct(x + (size_t)i * (size_t)d, meanT, invT, off, d, k,
                          out + (size_t)i * (size_t)k);
    }
    return 0;
}

/* (n, k) log-density only -- the ``component_log_prob`` fast path.
 * Returns 0 on success, non-zero on a shape or allocation error. */
int egmm_diag_logits(const double *x, const double *invT, const double *miT,
                     const double *quad_bias, const double *off, double *out,
                     int64_t n, int64_t d, int64_t k, int threads) {
    if (n < 0 || d < 1 || k < 1 || x == NULL || out == NULL) return 1;
    if (n == 0) return 0;
    int nt = 1;
#ifdef _OPENMP
    omp_set_dynamic(0);                  /* the team is exactly nt threads */
    nt = threads > 0 ? threads : omp_get_max_threads();
#else
    (void)threads;
#endif
    if (nt < 1) nt = 1;
    if ((int64_t)nt > n) nt = (int)n;

    const size_t k_pad = emaopt_pad((size_t)k);
    double *scratch = (double *)emaopt_align64(
        (size_t)nt * 8u * k_pad * sizeof(double));
    if (scratch == NULL) return 2;

    /* Written straight into the caller's (n, k) answer: the per-thread logit
     * row the first version malloc'd was unnecessary, and its failure path
     * could return success with ``out`` untouched. */
#pragma omp parallel num_threads(nt)
    {
        int t = 0;
#ifdef _OPENMP
        t = omp_get_thread_num();
#endif
        double *acc = scratch + (size_t)t * 8u * k_pad;
#pragma omp for schedule(static)
        for (int64_t i = 0; i < n; i += EMAOPT_BLOCK) {
            if (n - i >= EMAOPT_BLOCK) {
                emaopt_row4(x + (size_t)i * (size_t)d,
                            x + (size_t)(i + 1) * (size_t)d,
                            x + (size_t)(i + 2) * (size_t)d,
                            x + (size_t)(i + 3) * (size_t)d,
                            invT, miT, quad_bias, off, d, k, acc,
                            out + (size_t)i * (size_t)k,
                            out + (size_t)(i + 1) * (size_t)k,
                            out + (size_t)(i + 2) * (size_t)k,
                            out + (size_t)(i + 3) * (size_t)k);
            } else {
                for (int64_t r = i; r < n; ++r)
                    emaopt_row1(x + (size_t)r * (size_t)d, invT, miT,
                                quad_bias, off, d, k,
                                out + (size_t)r * (size_t)k);
            }
        }
    }
    free(scratch);
    return 0;
}

/* The whole E-step in one pass: logits, exp-shift softmax, responsibilities and
 * the sufficient statistics (counts, sums, sq).  ``nll_sum`` receives
 * ``sum_i -log_norm_i`` in float64; the caller divides by n.
 *
 * ``resp`` and ``log_norm`` may be NULL.  Per-thread accumulators are reduced
 * in a fixed thread order, so the answer does not depend on scheduling. */
int egmm_diag_estep(const double *x, const double *invT, const double *miT,
                    const double *quad_bias, const double *off,
                    int64_t n, int64_t d, int64_t k, int threads,
                    double *counts, double *sums, double *sq,
                    double *resp, double *log_norm, double *nll_sum) {
    if (n < 0 || d < 1 || k < 1) return 1;
    if (x == NULL || counts == NULL || sums == NULL || sq == NULL) return 1;
    const size_t kd = (size_t)k * (size_t)d;
    for (size_t i = 0; i < (size_t)k; ++i) counts[i] = 0.0;
    for (size_t i = 0; i < kd; ++i) { sums[i] = 0.0; sq[i] = 0.0; }
    if (nll_sum != NULL) *nll_sum = 0.0;
    if (n == 0) return 0;

    int nt = 1;
#ifdef _OPENMP
    omp_set_dynamic(0);                  /* the team is exactly nt threads */
    nt = threads > 0 ? threads : omp_get_max_threads();
#else
    (void)threads;
#endif
    if (nt < 1) nt = 1;
    if ((int64_t)nt > n) nt = (int)n;

    /* Padded to 64-byte slots so neighbouring threads never share a line that
     * every row writes (accumulators, and the per-thread logit rows). */
    const size_t k_pad = emaopt_pad((size_t)k);
    const size_t kd_pad = emaopt_pad(kd);
    const size_t d_pad = emaopt_pad((size_t)d);
    const size_t per_stats = 2 * kd_pad + k_pad;
    const size_t per_scratch = 8 * k_pad + 4 * k_pad + d_pad;
    if ((size_t)nt > SIZE_MAX / (per_stats * sizeof(double))) return 1;

    double *stats = (double *)calloc((size_t)nt * per_stats, sizeof(double));
    if (stats == NULL) return 1;
    double *scratch = (double *)emaopt_align64(
        (size_t)nt * per_scratch * sizeof(double));
    if (scratch == NULL) { free(stats); return 1; }
    double *thread_nll = (double *)calloc((size_t)nt, sizeof(double));
    if (thread_nll == NULL) { free(stats); free(scratch); return 1; }

#pragma omp parallel num_threads(nt)
    {
        int t = 0;
#ifdef _OPENMP
        t = omp_get_thread_num();
#endif
        double *tsums = stats + (size_t)t * per_stats;
        double *tsq = tsums + kd_pad;
        double *tc = tsq + kd_pad;
        double *block_acc = scratch + (size_t)t * per_scratch;
        double *lg4 = block_acc + 8 * k_pad;
        double *xsq = lg4 + 4 * k_pad;
        double local_nll = 0.0;

#pragma omp for schedule(static)
        for (int64_t i = 0; i < n; i += EMAOPT_BLOCK) {
            const int64_t rows = (n - i >= EMAOPT_BLOCK) ? EMAOPT_BLOCK : (n - i);
            if (rows == EMAOPT_BLOCK) {
                emaopt_row4(x + (size_t)i * (size_t)d,
                            x + (size_t)(i + 1) * (size_t)d,
                            x + (size_t)(i + 2) * (size_t)d,
                            x + (size_t)(i + 3) * (size_t)d,
                            invT, miT, quad_bias, off, d, k, block_acc,
                            lg4, lg4 + k_pad, lg4 + 2 * k_pad,
                            lg4 + 3 * k_pad);
            } else {
                for (int64_t r = 0; r < rows; ++r)
                    emaopt_row1(x + (size_t)(i + r) * (size_t)d, invT, miT,
                                quad_bias, off, d, k, lg4 + (size_t)r * k_pad);
            }
            for (int64_t r = 0; r < rows; ++r) {
                const size_t row = (size_t)(i + r);
                const double *restrict xi = x + row * (size_t)d;
                double *restrict lg = lg4 + (size_t)r * k_pad;
                double m = -INFINITY;
                for (int64_t j = 0; j < k; ++j) if (lg[j] > m) m = lg[j];
                if (!isfinite(m)) m = 0.0;     /* softmax_rows' exp-shift */
                double s = 0.0;
                for (int64_t j = 0; j < k; ++j) {
                    lg[j] = exp(lg[j] - m);
                    s += lg[j];
                }
                const double inv_s = 1.0 / emaopt_clamp_min(s, EMAOPT_FLOOR);
                double rs = 0.0;
                for (int64_t j = 0; j < k; ++j) {
                    double r2 = lg[j] * inv_s;
                    if (!isfinite(r2)) r2 = 0.0;
                    lg[j] = r2;
                    rs += r2;
                }
                const double inv_rs = 1.0 / emaopt_clamp_min(rs, EMAOPT_FLOOR);
                const double ln = m + log(s);
                local_nll -= ln;
                if (log_norm != NULL) log_norm[row] = ln;

                /* x^2 once per row: the two statistic updates then vectorise
                 * over ``d`` instead of repeating the multiply per component. */
                for (int64_t c = 0; c < d; ++c) xsq[c] = xi[c] * xi[c];
                for (int64_t j = 0; j < k; ++j) {
                    const double rj = lg[j] * inv_rs;
                    if (resp != NULL) resp[row * (size_t)k + (size_t)j] = rj;
                    tc[j] += rj;
                    double *restrict ts = tsums + (size_t)j * (size_t)d;
                    double *restrict tq = tsq + (size_t)j * (size_t)d;
#pragma omp simd
                    for (int64_t c = 0; c < d; ++c) {
                        ts[c] += rj * xi[c];
                        tq[c] += rj * xsq[c];
                    }
                }
            }
        }
        thread_nll[t] = local_nll;
    }
    /* fixed thread order, so the reduction does not depend on scheduling */
    double nll_total = 0.0;
    for (int t = 0; t < nt; ++t) nll_total += thread_nll[t];
    if (nll_sum != NULL) *nll_sum = nll_total;
    for (int t = 0; t < nt; ++t) {
        const double *tsums = stats + (size_t)t * per_stats;
        const double *tsq = tsums + kd_pad;
        const double *tc = tsq + kd_pad;
        for (size_t j = 0; j < (size_t)k; ++j) counts[j] += tc[j];
        for (size_t i = 0; i < kd; ++i) { sums[i] += tsums[i]; sq[i] += tsq[i]; }
    }
    free(stats);
    free(scratch);
    free(thread_nll);
    return 0;
}

/* Row log-sum-exp of an (n, k) logits matrix: one output per row.
 *
 * This is the eGMM scoring pass (`emaopt.core.logsumexp_rows`), and the shape
 * is what makes it threadable: one scalar per row, no cross-row reduction.  The
 * NumPy form issues max, a broadcast subtract, exp, sum and log over an (n, k)
 * temporary; this is one pass per row into the caller's `out`.
 *
 * Non-finite handling matches the NumPy path: a non-finite row max is shifted by
 * 0 instead of by itself, so an all--inf row returns -inf and a NaN row returns
 * NaN.  The sum runs in increasing j where NumPy sums pairwise, so the two agree
 * to rounding (measured <= 9e-16 at k <= 512), not necessarily bit for bit.
 */
int egmm_logsumexp_rows(const double *logits, double *out,
                        int64_t n, int64_t k, int threads) {
    if (n < 0 || k < 1 || logits == NULL || out == NULL) return 1;
    if (n == 0) return 0;
    int nt = 1;
#ifdef _OPENMP
    omp_set_dynamic(0);
    nt = threads > 0 ? threads : omp_get_max_threads();
#else
    (void)threads;
#endif
    if (nt < 1) nt = 1;
    if ((int64_t)nt > n) nt = (int)n;
#pragma omp parallel for schedule(static) num_threads(nt)
    for (int64_t i = 0; i < n; ++i) {
        const double *restrict lg = logits + (size_t)i * (size_t)k;
        double m = lg[0];
        for (int64_t j = 1; j < k; ++j) if (lg[j] > m) m = lg[j];
        if (!isfinite(m)) m = 0.0;
        double s = 0.0;
        for (int64_t j = 0; j < k; ++j) s += exp(lg[j] - m);
        out[i] = m + log(s);
    }
    return 0;
}
