/*
 * stabkernel — hand-tuned assembly kernels for GF(2) / stabilizer
 * bit-linear-algebra.
 *
 * Bit-packed convention: a length-`nbits` GF(2) vector is stored as
 * `nwords = ceil(nbits/64)` little-endian uint64 words. Bit j lives in
 * word j/64, bit position j%64. Unused trailing bits must be zero for
 * inner-product / weight results to be meaningful.
 *
 * All kernels are pure functions over raw bit-packed rows. The public
 * entry points below are resolved once at load time to the fastest
 * implementation the running CPU supports (AVX-512 > AVX2 > scalar on
 * x86-64; NEON > scalar on aarch64).
 */
#ifndef STABKERNEL_GF2KERNEL_H
#define STABKERNEL_GF2KERNEL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
#define SYSV_ABI __attribute__((sysv_abi))
#else
#define SYSV_ABI
#endif

/* dst[i] ^= src[i] for i in [0,nwords). The GF(2) row-add / tableau
 * update / Pauli-frame propagation primitive. */
typedef void (SYSV_ABI *gf2_xor_fn)(uint64_t *dst, const uint64_t *src, size_t nwords);

/* Returns the GF(2) inner product <a,b> = parity( popcount(a[i] & b[i]) ),
 * i.e. 0 or 1. The symplectic/stabilizer commutation primitive. */
typedef uint64_t (SYSV_ABI *gf2_inner_fn)(const uint64_t *a, const uint64_t *b,
                                 size_t nwords);

/* Returns the full Hamming weight sum_i popcount(a[i]). Used for Pauli
 * weight and stabilizer-Renyi / magic statistics. */
typedef uint64_t (SYSV_ABI *gf2_weight_fn)(const uint64_t *a, size_t nwords);

/* Resolved function pointers (set by stabkernel_init, also lazily on first
 * use through the wrapper functions below). */
extern gf2_xor_fn    gf2_xor;
extern gf2_inner_fn  gf2_inner;
extern gf2_weight_fn gf2_weight;

/* Name of the active backend, e.g. "avx512", "avx2", "neon", "scalar". */
const char *stabkernel_backend(void);

/* Force (re)detection of the best backend. Safe to call multiple times.
 * Set env STABKERNEL_BACKEND=scalar|avx2|avx512|neon to override. */
void stabkernel_init(void);

/* ---- Higher-level GF(2) linear algebra, built on the kernels above ---- */

/* In-place Gaussian elimination (row echelon) of an m x n bit-matrix stored
 * as `m` rows of `nwords` words each (row stride = nwords). Returns the rank.
 * `rows` is modified. */
size_t gf2_rank(uint64_t *rows, size_t m, size_t nwords);

/* Stabilizer-state / product-state overlap magnitude-squared exponent.
 * Given the check matrix of a stabilizer group (m = n independent rows over
 * 2n symplectic columns) this returns the GF(2) rank used by the Gauss-sum
 * overlap |<psi_S|phi>|^2 evaluation (Corollary-1 style). Convenience wrapper
 * around gf2_rank kept separate so the linalg entry points are discoverable. */
size_t gf2_solve_rank(uint64_t *rows, size_t m, size_t nwords);

/* ---- Multi-core layer (OpenMP across every CPU core) ------------------- */

/* Max threads OpenMP will use (== logical cores unless OMP_NUM_THREADS set). */
int stabkernel_num_threads(void);

/* GF(2) matrix multiply  C(m x p) = A(m x k) * B(k x p), bit-packed rows.
 *   Aw = ceil(k/64) words per A row;  Bw = ceil(p/64) words per B / C row.
 * C is overwritten. Method of the Four Russians + SIMD row-XOR, parallel
 * over output rows across all cores. */
void gf2_matmul(const uint64_t *A, size_t Aw,
                const uint64_t *B, size_t Bw,
                uint64_t *C, size_t m, size_t k);

/* out[i] = popcount(rows[i]) for m rows of nwords each (parallel). */
void gf2_weight_many(const uint64_t *rows, size_t m, size_t nwords,
                     uint64_t *out);

/* rows[i] ^= vec for all m rows of nwords each (parallel). */
void gf2_xor_many(uint64_t *rows, size_t m, size_t nwords,
                  const uint64_t *vec);

#ifdef __cplusplus
}
#endif

#endif /* STABKERNEL_GF2KERNEL_H */
