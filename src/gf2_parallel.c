/* Multi-core layer: uses every CPU core via OpenMP, on top of the per-core
 * SIMD kernels. The showcase is gf2_matmul — GF(2) matrix multiply via the
 * Method of the Four Russians (Gray-code table) + SIMD row-XOR, parallelised
 * over output rows so throughput scales with core count.
 *
 * If the compiler has no OpenMP, everything still builds and runs correctly
 * single-threaded (the pragmas are simply ignored). */
#include "gf2kernel.h"
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

int stabkernel_num_threads(void) {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

/* out[i] = popcount(rows[i]) for m rows of nwords each; parallel over rows. */
void gf2_weight_many(const uint64_t *rows, size_t m, size_t nwords,
                     uint64_t *out) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long i = 0; i < (long)m; i++)
        out[i] = gf2_weight(rows + (size_t)i * nwords, nwords);
}

/* rows[i] ^= vec for all m rows; parallel over rows. */
void gf2_xor_many(uint64_t *rows, size_t m, size_t nwords,
                  const uint64_t *vec) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long i = 0; i < (long)m; i++)
        gf2_xor(rows + (size_t)i * nwords, vec, nwords);
}

/* GF(2) matrix multiply  C(m x p) = A(m x k) * B(k x p), bit-packed rows.
 *   Aw = ceil(k/64) words per A row;  Bw = ceil(p/64) words per B and C row.
 * C is overwritten (zeroed internally).
 *
 * Method of the Four Russians: process A's columns in blocks of TR bits. For
 * each block, precompute a table of all 2^TR XOR-combinations of the
 * corresponding TR rows of B (built in O(2^TR) row-XORs via the lowest-set-bit
 * recurrence), then for every output row add the single table entry selected
 * by that row's TR bits. This turns the naive O(m*k) row-XORs into
 * O(m*k/TR + (k/TR)*2^TR), independent of bit density. The per-row add loop is
 * parallelised across all cores; each core owns a disjoint set of output rows. */
#define TR 8
void gf2_matmul(const uint64_t *A, size_t Aw,
                const uint64_t *B, size_t Bw,
                uint64_t *C, size_t m, size_t k) {
    memset(C, 0, m * Bw * sizeof(uint64_t));
    const size_t ntabmax = (size_t)1 << TR;
    uint64_t *table = (uint64_t *)malloc(ntabmax * Bw * sizeof(uint64_t));
    if (!table) return;

    for (size_t col0 = 0; col0 < k; col0 += TR) {
        size_t tb   = (k - col0 < TR) ? (k - col0) : TR;   /* bits this block */
        size_t ntab = (size_t)1 << tb;
        /* Build the Gray/lowest-bit table: table[idx] = XOR of B-rows in idx. */
        memset(table, 0, Bw * sizeof(uint64_t));           /* table[0] = 0 */
        for (size_t idx = 1; idx < ntab; idx++) {
            size_t low = (size_t)__builtin_ctzll((unsigned long long)idx);
            size_t src = idx ^ ((size_t)1 << low);
            uint64_t *dst = table + idx * Bw;
            memcpy(dst, table + src * Bw, Bw * sizeof(uint64_t));
            gf2_xor(dst, B + (col0 + low) * Bw, Bw);       /* SIMD row-XOR */
        }
        /* col0 is a multiple of TR (=8) which divides 64, so the TR bits never
         * straddle a word boundary. */
        size_t word = col0 >> 6, off = col0 & 63;
        uint64_t mask = ntab - 1;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (long i = 0; i < (long)m; i++) {
            uint64_t bits = (A[(size_t)i * Aw + word] >> off) & mask;
            if (bits)
                gf2_xor(C + (size_t)i * Bw, table + bits * Bw, Bw);
        }
    }
    free(table);
}
