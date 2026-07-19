/* Throughput benchmark. Pin a backend with STABKERNEL_BACKEND to compare
 * scalar vs avx2 vs avx512 (or neon) on the same machine. */
#include "gf2kernel.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
static uint64_t rs = 0x9E3779B97F4A7C15ull;
static uint64_t xr(void){ rs^=rs<<13; rs^=rs>>7; rs^=rs<<17; return rs; }

int main(int argc, char **argv) {
    stabkernel_init();
    size_t nwords = (argc > 1) ? (size_t)strtoull(argv[1],0,10) : 4096; /* 256 Kbit rows */
    long iters    = (argc > 2) ? strtol(argv[2],0,10) : 200000;

    uint64_t *a = malloc(nwords*8), *b = malloc(nwords*8);
    for (size_t i=0;i<nwords;i++){ a[i]=xr(); b[i]=xr(); }

    double bytes = (double)nwords * 8.0;
    printf("backend=%-7s  row=%zu words (%.0f KiB)  iters=%ld\n",
           stabkernel_backend(), nwords, bytes/1024.0, iters);

    /* xor */
    double t0 = now_s();
    for (long k=0;k<iters;k++) gf2_xor(a, b, nwords);
    double t = now_s()-t0;
    printf("  gf2_xor    %8.2f ms   %7.2f GB/s\n", t*1e3,
           (bytes*2*iters)/t/1e9);

    /* weight */
    volatile uint64_t sink=0;
    t0 = now_s();
    for (long k=0;k<iters;k++) sink ^= gf2_weight(a, nwords);
    t = now_s()-t0;
    printf("  gf2_weight %8.2f ms   %7.2f GB/s\n", t*1e3, (bytes*iters)/t/1e9);

    /* inner */
    t0 = now_s();
    for (long k=0;k<iters;k++) sink ^= gf2_inner(a, b, nwords);
    t = now_s()-t0;
    printf("  gf2_inner  %8.2f ms   %7.2f GB/s\n", t*1e3,
           (bytes*2*iters)/t/1e9);
    (void)sink;
    free(a); free(b);
    return 0;
}
