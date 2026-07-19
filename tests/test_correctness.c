/* Correctness: every SIMD backend must match the scalar reference exactly,
 * across all length classes (including sub-vector tails). */
#include "gf2kernel.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void     SYSV_ABI gf2_xor_scalar(uint64_t *, const uint64_t *, size_t);
uint64_t SYSV_ABI gf2_inner_scalar(const uint64_t *, const uint64_t *, size_t);
uint64_t SYSV_ABI gf2_weight_scalar(const uint64_t *, size_t);

static uint64_t rng_state = 0x243F6A8885A308D3ull;
static uint64_t xrng(void) {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17; return rng_state;
}

int main(void) {
    stabkernel_init();
    printf("backend under test: %s\n", stabkernel_backend());

    int fails = 0;
    /* cover 0..40 words (spans tails for 256b=4w and 512b=8w blocks) plus big */
    size_t sizes[] = {0,1,2,3,4,5,7,8,9,15,16,17,31,32,33,40,63,64,65,127,128,
                      129,255,256,257,1000,4096,65537};
    for (size_t si = 0; si < sizeof(sizes)/sizeof(sizes[0]); si++) {
        size_t n = sizes[si];
        uint64_t *a = calloc(n?n:1, 8), *b = calloc(n?n:1, 8);
        uint64_t *d1 = calloc(n?n:1, 8), *d2 = calloc(n?n:1, 8);
        for (size_t i = 0; i < n; i++) { a[i] = xrng(); b[i] = xrng(); }

        /* xor */
        memcpy(d1, a, n*8); memcpy(d2, a, n*8);
        gf2_xor_scalar(d1, b, n);
        gf2_xor(d2, b, n);
        if (memcmp(d1, d2, n*8)) { printf("FAIL xor n=%zu\n", n); fails++; }

        /* weight */
        uint64_t ws = gf2_weight_scalar(a, n), wk = gf2_weight(a, n);
        if (ws != wk) { printf("FAIL weight n=%zu (%llu vs %llu)\n", n,
            (unsigned long long)ws,(unsigned long long)wk); fails++; }

        /* inner */
        uint64_t is = gf2_inner_scalar(a, b, n), ik = gf2_inner(a, b, n);
        if (is != ik) { printf("FAIL inner n=%zu (%llu vs %llu)\n", n,
            (unsigned long long)is,(unsigned long long)ik); fails++; }

        free(a); free(b); free(d1); free(d2);
    }

    /* gf2_rank sanity: identity has full rank; duplicated rows drop rank */
    {
        size_t m = 200, nw = 4;
        uint64_t *rows = calloc(m*nw, 8);
        for (size_t r = 0; r < m; r++)
            rows[r*nw + (r/64 % nw)] |= (1ull << (r % 64));
        size_t rk = gf2_rank(rows, m, nw);
        size_t expect = (m < nw*64) ? m : nw*64;
        if (rk != expect) { printf("FAIL rank identity (%zu vs %zu)\n", rk, expect); fails++; }
        free(rows);
    }
    {
        size_t m = 10, nw = 2;
        uint64_t *rows = calloc(m*nw, 8);
        for (size_t r = 0; r < m; r++) { rows[r*nw] = 0xF0F0F0F0ull; } /* all equal */
        size_t rk = gf2_rank(rows, m, nw);
        if (rk != 1) { printf("FAIL rank duplicate (%zu vs 1)\n", rk); fails++; }
        free(rows);
    }

    /* matmul: four-Russians + parallel vs naive reference, odd dims + tail */
    {
        size_t m=133, k=201, Aw=(k+63)/64, Bw=5;
        uint64_t *A=calloc(m*Aw,8), *B=calloc(k*Bw,8),
                 *C=calloc(m*Bw,8), *R=calloc(m*Bw,8);
        for (size_t i=0;i<m*Aw;i++) A[i]=xrng();
        for (size_t i=0;i<k*Bw;i++) B[i]=xrng();
        if (k%64) for (size_t i=0;i<m;i++) A[i*Aw+Aw-1] &= ((1ull<<(k%64))-1);
        gf2_matmul(A,Aw,B,Bw,C,m,k);
        for (size_t i=0;i<m;i++)
            for (size_t l=0;l<k;l++)
                if ((A[i*Aw+(l>>6)]>>(l&63))&1)
                    for (size_t w=0;w<Bw;w++) R[i*Bw+w]^=B[l*Bw+w];
        if (memcmp(C,R,m*Bw*8)) { printf("FAIL matmul\n"); fails++; }
        free(A); free(B); free(C); free(R);
    }
    /* gf2_weight_many parallel vs per-row */
    {
        size_t m=257, nw=9; uint64_t *rows=calloc(m*nw,8), *out=calloc(m,8);
        for (size_t i=0;i<m*nw;i++) rows[i]=xrng();
        gf2_weight_many(rows,m,nw,out);
        for (size_t i=0;i<m;i++)
            if (out[i]!=gf2_weight_scalar(rows+i*nw,nw)) { printf("FAIL weight_many\n"); fails++; break; }
        free(rows); free(out);
    }
    printf("cores available: %d\n", stabkernel_num_threads());

    if (fails == 0) printf("ALL TESTS PASSED\n");
    else            printf("%d TEST(S) FAILED\n", fails);
    return fails ? 1 : 0;
}
