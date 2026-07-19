/* Multi-core scaling benchmark for gf2_matmul. Shows time / throughput /
 * speedup as the thread count climbs to every core on the machine. */
#include "gf2kernel.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static double now_s(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_sec + ts.tv_nsec*1e-9; }
static uint64_t rs = 0xD1B54A32D192ED03ull;
static uint64_t xr(void){ rs^=rs<<13; rs^=rs>>7; rs^=rs<<17; return rs; }

int main(int argc, char **argv) {
    stabkernel_init();
    size_t m  = (argc>1)?strtoull(argv[1],0,10):4096;
    size_t k  = (argc>2)?strtoull(argv[2],0,10):4096;
    size_t Bw = (argc>3)?strtoull(argv[3],0,10):64;   /* p = Bw*64 bits */
    size_t Aw = (k+63)/64, p = Bw*64;

    uint64_t *A=malloc(m*Aw*8), *B=malloc(k*Bw*8), *C=malloc(m*Bw*8);
    for (size_t i=0;i<m*Aw;i++) A[i]=xr();
    for (size_t i=0;i<k*Bw;i++) B[i]=xr();
    if (k%64) for (size_t i=0;i<m;i++) A[i*Aw+Aw-1] &= ((1ull<<(k%64))-1);

    int maxth = stabkernel_num_threads();
    double bitops = (double)m*(double)k*(double)p;   /* AND+XOR bit operations */
    printf("GF(2) matmul  A[%zux%zu] * B[%zux%zu]  backend=%s  cores=%d\n",
           m,k,k,p, stabkernel_backend(), maxth);
    printf("  four-Russians + SIMD row-XOR, parallel over output rows\n");
    printf("  bit-ops / matmul = %.3g\n", bitops);
    printf("  %-8s %11s %12s %9s\n","threads","time(ms)","bit-GOP/s","speedup");

    int tlist[32]; int nt=0;
    for (int t=1;t<=maxth;t*=2) tlist[nt++]=t;
    if (nt==0 || tlist[nt-1]!=maxth) tlist[nt++]=maxth;

    double base=0;
    for (int ti=0; ti<nt; ti++) {
        int t=tlist[ti];
#ifdef _OPENMP
        omp_set_num_threads(t);
#endif
        gf2_matmul(A,Aw,B,Bw,C,m,k);                   /* warmup */
        int reps=3; double t0=now_s();
        for (int r=0;r<reps;r++) gf2_matmul(A,Aw,B,Bw,C,m,k);
        double el=(now_s()-t0)/reps;
        if (t==1) base=el;
        printf("  %-8d %11.2f %12.2f %8.2fx\n",
               t, el*1e3, bitops/el/1e9, base>0?base/el:1.0);
    }
    free(A); free(B); free(C);
    return 0;
}
