/* Production twin of ``egmm_diag.c`` for a CUDA device, with no torch.

``emaopt/gpu.py`` is the torch arm: it is the right tool when torch is installed.
This file is the same fused diagonal-GMM E-step as the host kernel -- same maths,
same parameter layout, same outputs -- compiled into a plain shared object that
``emaopt.devicekernels`` loads with ctypes.  Keeping a torch-free twin means the
kernel can be gated line by line against the C host arm, and it runs wherever
the driver exists even when installing torch is not wanted.  It is opt-in
(``EMAOPT_DEVICE_ESTEP``) and never imported by default.

Two entry points per precision, because the precision is the whole question:

  f32   logits, exp and responsibilities in float, statistics and NLL still
        summed in double.  This is the arm the device is built for: kernel-only
        it is 9.6-25.7x the local 16-thread CPU on the harness shapes, and
        through the single-shot ctypes API (casts, upload and download paid) it
        is 3.3-12.9x from 41M updates up.  Against the float64 host kernel its
        counts agree to 1.1e-6 and its nll to 1.5e-9
        (``scripts/run_device_gate.py``).
  f64   parity with the host kernel (3e-14 on counts/sums/sq, 5e-16 on nll), for
        gating and for callers who cannot spend the rounding.

Kernel shape: one block owns 32 rows, 128 threads cover the k components.  A
row's logits and exponentials live in dynamic shared memory; the block's
per-component statistics accumulate there too and flush once per block, so the
atomics a row would otherwise pay are amortised 32:1.  That is also why the
block, not the thread, owns the row loop.

Two APIs per precision: the single-shot wrappers allocate, upload, launch and
download on every call -- the honest cost of a stateless ctypes API; and
``emaopt_device_session_*``, which keeps x, the packed parameters and the
statistics buffers resident so a step is one small upload, one launch and one
download.  The harness at ``scripts/bench_device_estep.cu`` measures the same
kernel with the copies separated out, including a resident loop
(``--resident``): on a 4090 the resident step is 5.96 ms against the kernel's
5.89 at 200000x8 k=256, so the stats download is ~1% of a step there.

Build:
  python -m emaopt.kernels.build --backend cuda
or by hand:
  nvcc -O3 -arch=native -Xcompiler -fPIC -shared egmm_diag_device.cu \
       -o libegmm_diag_cuda.so -lm
*/
#include <cuda_runtime.h>

#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define EMAOPT_TPB 128
#define EMAOPT_ROWS_PER_BLOCK 32

static char g_error[512] = {0};

static void set_error(const char *what, cudaError_t code) {
    snprintf(g_error, sizeof(g_error), "%s: %s", what,
             cudaGetErrorString(code));
}

extern "C" const char *emaopt_device_last_error(void) { return g_error; }

template <typename T>
__device__ __forceinline__ T dexp(T v);
template <>
__device__ __forceinline__ double dexp<double>(double v) { return exp(v); }
template <>
__device__ __forceinline__ float dexp<float>(float v) { return expf(v); }

template <typename T>
__global__ void emaopt_estep_kernel(const T *__restrict__ x,
                                    const T *__restrict__ invT,
                                    const T *__restrict__ miT,
                                    const T *__restrict__ bias,
                                    const T *__restrict__ offv, long n, int d,
                                    int k, long rows_per_block,
                                    double *__restrict__ counts,
                                    double *__restrict__ sums,
                                    double *__restrict__ sq,
                                    double *__restrict__ nll_sum) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    /* The double accumulation lives first: a float-typed T array would leave it
     * 4-byte aligned and every store to it would be misaligned. */
    double *nll_red = reinterpret_cast<double *>(smem_raw);
    T *lg = reinterpret_cast<T *>(nll_red + EMAOPT_TPB);
    T *ex = lg + k;
    T *red = ex + k;
    T *sc = red + EMAOPT_TPB;
    T *ss = sc + k;
    T *sqs = ss + (size_t)k * (size_t)d;

    for (int j = threadIdx.x; j < k; j += blockDim.x) sc[j] = (T)0.0;
    for (int i = threadIdx.x; i < k * d; i += blockDim.x) {
        ss[i] = (T)0.0;
        sqs[i] = (T)0.0;
    }
    __syncthreads();

    double nll_local = 0.0;
    const long row0 = (long)blockIdx.x * rows_per_block;
    const long row1 = row0 + rows_per_block < n ? row0 + rows_per_block : n;
    for (long r = row0; r < row1; ++r) {
        const T *xi = x + r * (long)d;
        T tmax = -INFINITY;
        for (int j = threadIdx.x; j < k; j += blockDim.x) {
            T a = (T)0.0, b = (T)0.0;
            for (int c = 0; c < d; ++c) {
                const T v = xi[c];
                a += v * v * invT[(size_t)c * (size_t)k + (size_t)j];
                b += v * miT[(size_t)c * (size_t)k + (size_t)j];
            }
            T q = a - (T)2.0 * b + bias[j];
            if (q < (T)0.0) q = (T)0.0;
            const T v = (T)-0.5 * q + offv[j];
            lg[j] = v;
            if (v > tmax) tmax = v;
        }
        red[threadIdx.x] = tmax;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s)
                red[threadIdx.x] = fmax(red[threadIdx.x], red[threadIdx.x + s]);
            __syncthreads();
        }
        T m = red[0];
        if (!isfinite((double)m)) m = (T)0.0;
        T tsum = (T)0.0;
        for (int j = threadIdx.x; j < k; j += blockDim.x) {
            const T e = dexp(lg[j] - m);
            ex[j] = e;
            tsum += e;
        }
        red[threadIdx.x] = tsum;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
            __syncthreads();
        }
        const T ssum = red[0];
        if (ssum > (T)0.0) nll_local += -((double)m + log((double)ssum));
        for (int j = threadIdx.x; j < k; j += blockDim.x) {
            const T resp = ex[j] / ssum;
            atomicAdd(&sc[j], resp);
            for (int c = 0; c < d; ++c) {
                const T v = xi[c];
                atomicAdd(&ss[(size_t)j * (size_t)d + (size_t)c], resp * v);
                atomicAdd(&sqs[(size_t)j * (size_t)d + (size_t)c],
                          resp * v * v);
            }
        }
        __syncthreads(); /* next row rewrites lg/ex */
    }

    for (int j = threadIdx.x; j < k; j += blockDim.x)
        atomicAdd(&counts[j], (double)sc[j]);
    for (int i = threadIdx.x; i < k * d; i += blockDim.x) {
        atomicAdd(&sums[i], (double)ss[i]);
        atomicAdd(&sq[i], (double)sqs[i]);
    }
    nll_red[threadIdx.x] = nll_local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) nll_red[threadIdx.x] += nll_red[threadIdx.x + s];
        __syncthreads();
    }
    /* Every thread walks every row (the row loop is not split across threads),
     * so one thread's total is the block total, not TPB copies of it. */
    if (threadIdx.x == 0)
        atomicAdd(nll_sum, nll_red[0] / (double)blockDim.x);
}

template <typename T>
static size_t shared_bytes(int d, int k) {
    return sizeof(double) * EMAOPT_TPB
           + sizeof(T) * (size_t)(2 * k + EMAOPT_TPB + k + 2 * k * d);
}

template <typename T>
static int launch_estep(const T *d_x, const T *d_invT, const T *d_miT,
                        const T *d_bias, const T *d_off, long n, int d, int k,
                        double *d_counts, double *d_sums, double *d_sq,
                        double *d_nll) {
    static int probed = 0;
    static int shared_ok = 0;
    static int optin = 0;
    if (!probed) {
        int dev = 0;
        cudaDeviceProp prop;
        cudaError_t rc = cudaGetDevice(&dev);
        if (rc == cudaSuccess) rc = cudaGetDeviceProperties(&prop, dev);
        if (rc != cudaSuccess) {
            set_error("cudaGetDeviceProperties", rc);
            return (int)rc;
        }
        shared_ok = prop.sharedMemPerBlock;
        optin = prop.sharedMemPerBlockOptin;
        probed = 1;
    }
    const size_t smem = shared_bytes<T>(d, k);
    if ((int)smem > optin) {
        snprintf(g_error, sizeof(g_error),
                 "kernel needs %zu bytes of shared memory; this device caps "
                 "opt-in dynamic shared memory at %d (reduce k or d)",
                 smem, optin);
        return (int)cudaErrorInvalidValue;
    }
    if ((int)smem > shared_ok) {
        cudaError_t rc = cudaFuncSetAttribute(
            emaopt_estep_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)smem);
        if (rc != cudaSuccess) {
            set_error("cudaFuncSetAttribute", rc);
            return (int)rc;
        }
    }
    const long blocks = (n + EMAOPT_ROWS_PER_BLOCK - 1) / EMAOPT_ROWS_PER_BLOCK;
    emaopt_estep_kernel<T><<<(unsigned)blocks, EMAOPT_TPB, smem>>>(
        d_x, d_invT, d_miT, d_bias, d_off, n, d, k, EMAOPT_ROWS_PER_BLOCK,
        d_counts, d_sums, d_sq, d_nll);
    cudaError_t rc = cudaGetLastError();
    if (rc != cudaSuccess) set_error("emaopt_estep_kernel launch", rc);
    return (int)rc;
}

template <typename T>
static int estep_typed(const T *hx, const T *hinv, const T *hmi,
                       const T *hbias, const T *hoff, int64_t n, int64_t d,
                       int64_t k, double *h_counts, double *h_sums,
                       double *h_sq, double *h_nll) {
    if (!hx || !hinv || !hmi || !hbias || !hoff || !h_counts || !h_sums ||
        !h_sq || !h_nll || n <= 0 || d <= 0 || k <= 0) {
        snprintf(g_error, sizeof(g_error), "invalid argument (n=%lld d=%lld "
                 "k=%lld)", (long long)n, (long long)d, (long long)k);
        return (int)cudaErrorInvalidValue;
    }
    const size_t nx = (size_t)n * (size_t)d * sizeof(T);
    const size_t nk = (size_t)k * (size_t)d * sizeof(T);
    const size_t kb = (size_t)k * sizeof(T);
    const size_t stat = (size_t)k * sizeof(double);
    const size_t stat2 = (size_t)k * (size_t)d * sizeof(double);
    T *d_x = NULL, *d_inv = NULL, *d_mi = NULL, *d_bias = NULL, *d_off = NULL;
    double *d_counts = NULL, *d_sums = NULL, *d_sq = NULL, *d_nll = NULL;
    cudaError_t rc = cudaSuccess;
#define ALLOC(ptr, bytes)                                     \
    do {                                                      \
        rc = cudaMalloc((void **)&(ptr), (bytes));            \
        if (rc != cudaSuccess) {                              \
            set_error("cudaMalloc", rc);                      \
            goto cleanup;                                     \
        }                                                     \
    } while (0)
    ALLOC(d_x, nx);
    ALLOC(d_inv, nk);
    ALLOC(d_mi, nk);
    ALLOC(d_bias, kb);
    ALLOC(d_off, kb);
    ALLOC(d_counts, stat);
    ALLOC(d_sums, stat2);
    ALLOC(d_sq, stat2);
    ALLOC(d_nll, sizeof(double));
#undef ALLOC
    rc = cudaMemcpy(d_x, hx, nx, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess) { set_error("cudaMemcpy x", rc); goto cleanup; }
    rc = cudaMemcpy(d_inv, hinv, nk, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess) { set_error("cudaMemcpy invT", rc); goto cleanup; }
    rc = cudaMemcpy(d_mi, hmi, nk, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess) { set_error("cudaMemcpy miT", rc); goto cleanup; }
    rc = cudaMemcpy(d_bias, hbias, kb, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess) { set_error("cudaMemcpy bias", rc); goto cleanup; }
    rc = cudaMemcpy(d_off, hoff, kb, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess) { set_error("cudaMemcpy off", rc); goto cleanup; }
    rc = cudaMemset(d_counts, 0, stat);
    if (rc == cudaSuccess) rc = cudaMemset(d_sums, 0, stat2);
    if (rc == cudaSuccess) rc = cudaMemset(d_sq, 0, stat2);
    if (rc == cudaSuccess) rc = cudaMemset(d_nll, 0, sizeof(double));
    if (rc != cudaSuccess) { set_error("cudaMemset", rc); goto cleanup; }
    rc = (cudaError_t)launch_estep<T>(d_x, d_inv, d_mi, d_bias, d_off,
                                      (long)n, (int)d, (int)k, d_counts,
                                      d_sums, d_sq, d_nll);
    if (rc != cudaSuccess) goto cleanup;
    rc = cudaMemcpy(h_counts, d_counts, stat, cudaMemcpyDeviceToHost);
    if (rc == cudaSuccess)
        rc = cudaMemcpy(h_sums, d_sums, stat2, cudaMemcpyDeviceToHost);
    if (rc == cudaSuccess)
        rc = cudaMemcpy(h_sq, d_sq, stat2, cudaMemcpyDeviceToHost);
    if (rc == cudaSuccess)
        rc = cudaMemcpy(h_nll, d_nll, sizeof(double), cudaMemcpyDeviceToHost);
    if (rc != cudaSuccess) set_error("cudaMemcpy statistics back", rc);

cleanup:
    cudaFree(d_x);
    cudaFree(d_inv);
    cudaFree(d_mi);
    cudaFree(d_bias);
    cudaFree(d_off);
    cudaFree(d_counts);
    cudaFree(d_sums);
    cudaFree(d_sq);
    cudaFree(d_nll);
    return (int)rc;
}

/* ---- Resident sessions ----------------------------------------------------
 * The single-shot wrappers pay allocation, the float32 cast and the upload of
 * x on every call.  A session keeps x, the packed parameter block and the four
 * statistics buffers on the device for the life of a fit, so a step is one
 * small upload, one launch and one download.  ``d_params`` layout is
 * [invT (d*k)][miT (d*k)][bias (k)][off (k)] in ``precision``'s type.
 */
typedef struct {
    int precision;                 /* 0 = float32, 1 = float64 */
    int64_t n;
    int d, k;
    void *d_x;
    void *d_params;
    double *d_counts, *d_sums, *d_sq, *d_nll;
} emaopt_session;

static size_t session_elem_size(int precision) {
    return precision ? sizeof(double) : sizeof(float);
}

static size_t session_param_count(int d, int k) {
    return 2 * (size_t)d * (size_t)k + 2 * (size_t)k;
}

extern "C" void emaopt_device_session_destroy(void *handle);

extern "C" void *emaopt_device_session_create(int64_t n, int64_t d, int64_t k,
                                              int precision, char *err,
                                              int err_cap) {
    if (err && err_cap > 0) err[0] = '\0';
    if (n <= 0 || d <= 0 || k <= 0 || (precision != 0 && precision != 1)) {
        snprintf(g_error, sizeof(g_error),
                 "invalid session (n=%lld d=%lld k=%lld precision=%d)",
                 (long long)n, (long long)d, (long long)k, precision);
        if (err && err_cap > 0) snprintf(err, err_cap, "%s", g_error);
        return NULL;
    }
    emaopt_session *s = (emaopt_session *)calloc(1, sizeof(emaopt_session));
    if (!s) {
        snprintf(g_error, sizeof(g_error), "out of host memory for session");
        if (err && err_cap > 0) snprintf(err, err_cap, "%s", g_error);
        return NULL;
    }
    s->precision = precision;
    s->n = n;
    s->d = (int)d;
    s->k = (int)k;
    const size_t esz = session_elem_size(precision);
    cudaError_t rc = cudaSuccess;
#define SALLOC(field, bytes)                                                \
    do {                                                                    \
        rc = cudaMalloc((void **)&(s->field), (bytes));                     \
        if (rc != cudaSuccess) {                                            \
            set_error("cudaMalloc session " #field, rc);                    \
            goto fail;                                                      \
        }                                                                   \
    } while (0)
    SALLOC(d_x, (size_t)n * (size_t)d * esz);
    SALLOC(d_params, session_param_count((int)d, (int)k) * esz);
    SALLOC(d_counts, (size_t)k * sizeof(double));
    SALLOC(d_sums, (size_t)k * (size_t)d * sizeof(double));
    SALLOC(d_sq, (size_t)k * (size_t)d * sizeof(double));
    SALLOC(d_nll, sizeof(double));
#undef SALLOC
    rc = cudaMemset(s->d_counts, 0, (size_t)k * sizeof(double));
    if (rc == cudaSuccess)
        rc = cudaMemset(s->d_sums, 0, (size_t)k * (size_t)d * sizeof(double));
    if (rc == cudaSuccess)
        rc = cudaMemset(s->d_sq, 0, (size_t)k * (size_t)d * sizeof(double));
    if (rc == cudaSuccess) rc = cudaMemset(s->d_nll, 0, sizeof(double));
    if (rc != cudaSuccess) {
        set_error("cudaMemset session statistics", rc);
        goto fail;
    }
    return s;
fail:
    emaopt_device_session_destroy(s);
    if (err && err_cap > 0) snprintf(err, err_cap, "%s", g_error);
    return NULL;
}

extern "C" int emaopt_device_session_set_data(void *handle, const void *x) {
    emaopt_session *s = (emaopt_session *)handle;
    if (!s || !x) {
        snprintf(g_error, sizeof(g_error), "set_data: null session or data");
        return (int)cudaErrorInvalidValue;
    }
    cudaError_t rc = cudaMemcpy(
        s->d_x, x,
        (size_t)s->n * (size_t)s->d * session_elem_size(s->precision),
        cudaMemcpyHostToDevice);
    if (rc != cudaSuccess) set_error("cudaMemcpy session x", rc);
    return (int)rc;
}

template <typename T>
static int session_estep_typed(emaopt_session *s, const T *params,
                               double *h_counts, double *h_sums, double *h_sq,
                               double *h_nll) {
    const size_t dk = (size_t)s->d * (size_t)s->k;
    const size_t stat = (size_t)s->k * sizeof(double);
    const size_t stat2 = dk * sizeof(double);
    cudaError_t rc = cudaMemcpy(s->d_params, params,
                                session_param_count(s->d, s->k) * sizeof(T),
                                cudaMemcpyHostToDevice);
    if (rc != cudaSuccess) {
        set_error("cudaMemcpy session parameters", rc);
        return (int)rc;
    }
    rc = cudaMemset(s->d_counts, 0, stat);
    if (rc == cudaSuccess) rc = cudaMemset(s->d_sums, 0, stat2);
    if (rc == cudaSuccess) rc = cudaMemset(s->d_sq, 0, stat2);
    if (rc == cudaSuccess) rc = cudaMemset(s->d_nll, 0, sizeof(double));
    if (rc != cudaSuccess) {
        set_error("cudaMemset session statistics", rc);
        return (int)rc;
    }
    const T *d_inv = (const T *)s->d_params;
    const T *d_mi = d_inv + dk;
    const T *d_bias = d_mi + dk;
    const T *d_off = d_bias + s->k;
    rc = (cudaError_t)launch_estep<T>((const T *)s->d_x, d_inv, d_mi, d_bias,
                                      d_off,
                                      (long)s->n, s->d, s->k, s->d_counts,
                                      s->d_sums, s->d_sq, s->d_nll);
    if (rc != cudaSuccess) return (int)rc;
    rc = cudaMemcpy(h_counts, s->d_counts, stat, cudaMemcpyDeviceToHost);
    if (rc == cudaSuccess)
        rc = cudaMemcpy(h_sums, s->d_sums, stat2, cudaMemcpyDeviceToHost);
    if (rc == cudaSuccess)
        rc = cudaMemcpy(h_sq, s->d_sq, stat2, cudaMemcpyDeviceToHost);
    if (rc == cudaSuccess)
        rc = cudaMemcpy(h_nll, s->d_nll, sizeof(double),
                        cudaMemcpyDeviceToHost);
    if (rc != cudaSuccess) set_error("cudaMemcpy session statistics back", rc);
    return (int)rc;
}

extern "C" int emaopt_device_session_estep(
    void *handle, const void *params, double *counts, double *sums, double *sq,
    double *nll_sum) {
    emaopt_session *s = (emaopt_session *)handle;
    if (!s || !params || !counts || !sums || !sq || !nll_sum) {
        snprintf(g_error, sizeof(g_error), "session_estep: null argument");
        return (int)cudaErrorInvalidValue;
    }
    if (s->precision == 0)
        return session_estep_typed<float>(s, (const float *)params, counts,
                                          sums, sq, nll_sum);
    return session_estep_typed<double>(s, (const double *)params, counts, sums,
                                       sq, nll_sum);
}

extern "C" void emaopt_device_session_destroy(void *handle) {
    emaopt_session *s = (emaopt_session *)handle;
    if (!s) return;
    cudaFree(s->d_x);
    cudaFree(s->d_params);
    cudaFree(s->d_counts);
    cudaFree(s->d_sums);
    cudaFree(s->d_sq);
    cudaFree(s->d_nll);
    free(s);
}

extern "C" int emaopt_device_estep_f32(
    const float *x, const float *invT, const float *miT, const float *bias,
    const float *off, int64_t n, int64_t d, int64_t k, double *counts,
    double *sums, double *sq, double *nll_sum) {
    return estep_typed<float>(x, invT, miT, bias, off, n, d, k, counts, sums,
                              sq, nll_sum);
}

extern "C" int emaopt_device_estep_f64(
    const double *x, const double *invT, const double *miT, const double *bias,
    const double *off, int64_t n, int64_t d, int64_t k, double *counts,
    double *sums, double *sq, double *nll_sum) {
    return estep_typed<double>(x, invT, miT, bias, off, n, d, k, counts, sums,
                               sq, nll_sum);
}

/* Device description plus a 2x1x1 self-check.  A shared object built for a
 * different architecture loads fine and fails at launch ("no kernel image is
 * available"), so the loader needs the launch to run, not just cudaGetDevice. */
extern "C" int emaopt_device_probe(char *name, int name_cap, int *major,
                                   int *minor, int *sms, char *err,
                                   int err_cap) {
    if (name && name_cap > 0) name[0] = '\0';
    if (err && err_cap > 0) err[0] = '\0';
    int dev = 0;
    cudaError_t rc = cudaGetDevice(&dev);
    if (rc != cudaSuccess) {
        set_error("cudaGetDevice", rc);
        if (err && err_cap > 0) snprintf(err, err_cap, "%s", g_error);
        return (int)rc;
    }
    cudaDeviceProp prop;
    rc = cudaGetDeviceProperties(&prop, dev);
    if (rc != cudaSuccess) {
        set_error("cudaGetDeviceProperties", rc);
        if (err && err_cap > 0) snprintf(err, err_cap, "%s", g_error);
        return (int)rc;
    }
    *major = prop.major;
    *minor = prop.minor;
    *sms = prop.multiProcessorCount;
    snprintf(name, name_cap, "%s", prop.name);
    const float x[2] = {0.10f, -0.30f};
    const float inv[2] = {1.0f, 1.0f};
    const float mi[2] = {0.0f, 0.0f};
    const float bias[2] = {0.0f, 0.0f};
    const float off[2] = {-1.6120857f, -1.6120857f};
    double counts[2] = {0.0, 0.0}, sums[2] = {0.0, 0.0};
    double sq[2] = {0.0, 0.0}, nll = 0.0;
    rc = (cudaError_t)emaopt_device_estep_f32(x, inv, mi, bias, off, 2, 1, 2,
                                              counts, sums, sq, &nll);
    if (rc != cudaSuccess) {
        if (err && err_cap > 0) snprintf(err, err_cap, "%s", g_error);
        return (int)rc;
    }
    const double total = counts[0] + counts[1];
    if (!isfinite(total) || fabs(total - 2.0) > 1e-3 || !isfinite(nll)) {
        snprintf(g_error, sizeof(g_error),
                 "self-check failed on %s (counts sum %.6f, nll %.6f); rebuild "
                 "with `python -m emaopt.kernels.build --backend cuda`",
                 prop.name, total, nll);
        if (err && err_cap > 0) snprintf(err, err_cap, "%s", g_error);
        return (int)cudaErrorUnknown;
    }
    return 0;
}
