/* Runtime backend selection. Picks the fastest kernel the current CPU
 * supports; honours STABKERNEL_BACKEND for testing/benchmark pinning. */
#include "gf2kernel.h"
#include <stdlib.h>
#include <string.h>

/* ---- scalar (always present) ---- */
void     gf2_xor_scalar(uint64_t *, const uint64_t *, size_t);
uint64_t gf2_inner_scalar(const uint64_t *, const uint64_t *, size_t);
uint64_t gf2_weight_scalar(const uint64_t *, size_t);

#if defined(__x86_64__)
void     gf2_xor_avx2(uint64_t *, const uint64_t *, size_t);
void     gf2_xor_avx512(uint64_t *, const uint64_t *, size_t);
uint64_t gf2_inner_avx2(const uint64_t *, const uint64_t *, size_t);
uint64_t gf2_inner_avx512(const uint64_t *, const uint64_t *, size_t);
uint64_t gf2_weight_avx2(const uint64_t *, size_t);
uint64_t gf2_weight_avx512(const uint64_t *, size_t);
#elif defined(__aarch64__)
/* --- ARM64 / NEON backend TEMPORARILY DISABLED ---------------------------
 * The NEON kernels exist in src/gf2_arm.S but are not wired in: they were
 * written but could NOT be executed/validated in the x86 build sandbox
 * (no ARM emulator / cross-toolchain available). Until validated on real
 * ARM hardware, aarch64 builds run the portable scalar backend.
 *
 * To RE-ENABLE NEON:
 *   1. Uncomment the three extern declarations below.
 *   2. Uncomment the NEON selection branch further down in stabkernel_init().
 *   3. In the Makefile, restore `ASM_SRC := src/gf2_arm.S` for aarch64.
 *   4. Run `make test` on-device and confirm ALL TESTS PASSED.
 *
 * void     gf2_xor_neon(uint64_t *, const uint64_t *, size_t);
 * uint64_t gf2_inner_neon(const uint64_t *, const uint64_t *, size_t);
 * uint64_t gf2_weight_neon(const uint64_t *, size_t);
 */
#endif

gf2_xor_fn    gf2_xor    = gf2_xor_scalar;
gf2_inner_fn  gf2_inner  = gf2_inner_scalar;
gf2_weight_fn gf2_weight = gf2_weight_scalar;

static const char *g_backend = "scalar";

#if defined(__x86_64__)
#include <cpuid.h>
/* AVX-512F + AVX-512 VPOPCNTDQ needed for the *_avx512 tier. */
static int has_avx512_vpopcnt(void) {
    unsigned a, b, c, d;
    if (!__get_cpuid_count(7, 0, &a, &b, &c, &d)) return 0;
    int avx512f       = (b >> 16) & 1;   /* EBX bit 16 */
    int avx512_vpopcnt = (c >> 14) & 1;  /* ECX bit 14 (VPOPCNTDQ) */
    return avx512f && avx512_vpopcnt;
}
static int has_avx2(void) {
    unsigned a, b, c, d;
    if (!__get_cpuid_count(7, 0, &a, &b, &c, &d)) return 0;
    return (b >> 5) & 1;                 /* EBX bit 5 (AVX2) */
}
#endif

void stabkernel_init(void) {
    const char *forced = getenv("STABKERNEL_BACKEND");

    /* default: scalar */
    gf2_xor = gf2_xor_scalar; gf2_inner = gf2_inner_scalar;
    gf2_weight = gf2_weight_scalar; g_backend = "scalar";

#if defined(__x86_64__)
    int want512 = has_avx512_vpopcnt();
    int want2   = has_avx2();
    if (forced) { want512 = !strcmp(forced, "avx512");
                  want2   = !strcmp(forced, "avx2"); }
    if (want512) {
        gf2_xor = gf2_xor_avx512; gf2_inner = gf2_inner_avx512;
        gf2_weight = gf2_weight_avx512; g_backend = "avx512";
    } else if (want2) {
        gf2_xor = gf2_xor_avx2; gf2_inner = gf2_inner_avx2;
        gf2_weight = gf2_weight_avx2; g_backend = "avx2";
    }
#elif defined(__aarch64__)
    /* ARM64 NEON backend DISABLED for now — falls through to scalar.
     * See the note near the extern declarations above to re-enable.
     *
     * int want_neon = 1;                     // NEON is baseline on aarch64
     * if (forced) want_neon = !strcmp(forced, "neon");
     * if (want_neon) {
     *     gf2_xor = gf2_xor_neon; gf2_inner = gf2_inner_neon;
     *     gf2_weight = gf2_weight_neon; g_backend = "neon";
     * }
     */
    (void)forced;
#endif
    if (forced && !strcmp(forced, "scalar")) {
        gf2_xor = gf2_xor_scalar; gf2_inner = gf2_inner_scalar;
        gf2_weight = gf2_weight_scalar; g_backend = "scalar";
    }
}

const char *stabkernel_backend(void) {
    if (gf2_xor == NULL) stabkernel_init();
    return g_backend;
}

/* Auto-init at load time (GCC/Clang). */
__attribute__((constructor))
static void stabkernel_ctor(void) { stabkernel_init(); }
