/* Portable C reference kernels. Ground truth for correctness tests and the
 * universal fallback when no SIMD backend is available. */
#include "gf2kernel.h"

void SYSV_ABI gf2_xor_scalar(uint64_t *dst, const uint64_t *src, size_t nwords) {
    for (size_t i = 0; i < nwords; i++) dst[i] ^= src[i];
}

uint64_t SYSV_ABI gf2_inner_scalar(const uint64_t *a, const uint64_t *b, size_t nwords) {
    uint64_t acc = 0;
    for (size_t i = 0; i < nwords; i++)
        acc += (uint64_t)__builtin_popcountll(a[i] & b[i]);
    return acc & 1u;
}

uint64_t SYSV_ABI gf2_weight_scalar(const uint64_t *a, size_t nwords) {
    uint64_t acc = 0;
    for (size_t i = 0; i < nwords; i++)
        acc += (uint64_t)__builtin_popcountll(a[i]);
    return acc;
}
