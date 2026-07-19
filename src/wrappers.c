/* Stable C entry points that call through the resolved backend pointers.
 * These give language bindings (ctypes/cffi) a fixed symbol to bind to,
 * independent of which SIMD tier was selected at load time. */
#include "gf2kernel.h"

void     sk_xor(uint64_t *d, const uint64_t *s, size_t n) { gf2_xor(d, s, n); }
uint64_t sk_inner(const uint64_t *a, const uint64_t *b, size_t n) { return gf2_inner(a, b, n); }
uint64_t sk_weight(const uint64_t *a, size_t n) { return gf2_weight(a, n); }
const char *sk_backend(void) { return stabkernel_backend(); }
