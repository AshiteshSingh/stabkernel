/* Hand-written x86-64 kernels (GAS, Intel syntax) for GF(2) bit-linear-algebra.
 * Two microarchitecture tiers:
 *   *_avx512 : Skylake-X / Ice Lake / Zen4 / Zen5 (AVX-512F + VPOPCNTDQ)
 *   *_avx2   : Haswell+ / Zen1-3 (AVX2, software popcount via PSHUFB LUT)
 * System V AMD64 ABI: args rdi, rsi, rdx; return rax.
 *
 * Portable across ELF (Linux) and Mach-O (macOS): C symbol names are emitted
 * with the platform's mangling via CSYM(), the read-only data section is
 * selected per object format, and the ELF-only non-executable-stack note is
 * guarded out on Apple.
 */
#if defined(__x86_64__)

#if defined(__APPLE__)
#  define CSYM(x) _##x
#  define RODATA_SECTION .section __TEXT,__const
#elif defined(_WIN32)
#  define CSYM(x) x
#  define RODATA_SECTION .section .rdata,"dr"
#else
#  define CSYM(x) x
#  define RODATA_SECTION .section .rodata
#endif

.intel_syntax noprefix
.text

/* ================= gf2_xor: dst[i] ^= src[i] ================= */

.globl CSYM(gf2_xor_avx512)
.p2align 4
CSYM(gf2_xor_avx512):           /* rdi=dst rsi=src rdx=nwords */
    xor     rax, rax
    mov     rcx, rdx
    shr     rcx, 3              /* blocks of 8 qwords (512b) */
    jz      .Lx5_tail
.Lx5_loop:
    vmovdqu64 zmm0, [rdi + rax*8]
    vpxorq    zmm0, zmm0, [rsi + rax*8]
    vmovdqu64 [rdi + rax*8], zmm0
    add     rax, 8
    dec     rcx
    jnz     .Lx5_loop
.Lx5_tail:
    mov     rcx, rdx
    and     rcx, 7
    jz      .Lx5_done
.Lx5_t:
    mov     r8, [rsi + rax*8]
    xor     [rdi + rax*8], r8
    inc     rax
    dec     rcx
    jnz     .Lx5_t
.Lx5_done:
    vzeroupper
    ret

.globl CSYM(gf2_xor_avx2)
.p2align 4
CSYM(gf2_xor_avx2):             /* rdi=dst rsi=src rdx=nwords */
    xor     rax, rax
    mov     rcx, rdx
    shr     rcx, 2              /* blocks of 4 qwords (256b) */
    jz      .Lx2_tail
.Lx2_loop:
    vmovdqu ymm0, [rdi + rax*8]
    vpxor   ymm0, ymm0, [rsi + rax*8]
    vmovdqu [rdi + rax*8], ymm0
    add     rax, 4
    dec     rcx
    jnz     .Lx2_loop
.Lx2_tail:
    mov     rcx, rdx
    and     rcx, 3
    jz      .Lx2_done
.Lx2_t:
    mov     r8, [rsi + rax*8]
    xor     [rdi + rax*8], r8
    inc     rax
    dec     rcx
    jnz     .Lx2_t
.Lx2_done:
    vzeroupper
    ret

/* ================= gf2_weight: sum popcount(a[i]) ================= */

.globl CSYM(gf2_weight_avx512)
.p2align 4
CSYM(gf2_weight_avx512):       /* rdi=a rsi=nwords -> rax */
    vpxorq  zmm1, zmm1, zmm1
    xor     r9, r9
    mov     rcx, rsi
    shr     rcx, 3
    jz      .Lw5_after
.Lw5_loop:
    vmovdqu64 zmm0, [rdi + r9*8]
    vpopcntq  zmm0, zmm0
    vpaddq    zmm1, zmm1, zmm0
    add     r9, 8
    dec     rcx
    jnz     .Lw5_loop
.Lw5_after:
    vextracti64x4 ymm2, zmm1, 1
    vpaddq  ymm1, ymm1, ymm2
    vextracti128  xmm2, ymm1, 1
    vpaddq  xmm1, xmm1, xmm2
    vpsrldq xmm2, xmm1, 8
    vpaddq  xmm1, xmm1, xmm2
    vmovq   rax, xmm1
    mov     rcx, rsi
    and     rcx, 7
    jz      .Lw5_done
.Lw5_t:
    popcnt  r8, qword ptr [rdi + r9*8]
    add     rax, r8
    inc     r9
    dec     rcx
    jnz     .Lw5_t
.Lw5_done:
    vzeroupper
    ret

.globl CSYM(gf2_weight_avx2)
.p2align 4
CSYM(gf2_weight_avx2):         /* rdi=a rsi=nwords -> rax */
    vmovdqa ymm7, [rip + .Lpopc_lut]
    vpbroadcastb ymm6, byte ptr [rip + .Lmask0f]
    vpxor   ymm5, ymm5, ymm5        /* qword accumulator */
    vpxor   ymm4, ymm4, ymm4        /* zero for psadbw */
    xor     r9, r9
    mov     rcx, rsi
    shr     rcx, 2
    jz      .Lwa_after
.Lwa_loop:
    vmovdqu ymm0, [rdi + r9*8]
    vpsrlw  ymm1, ymm0, 4
    vpand   ymm0, ymm0, ymm6
    vpand   ymm1, ymm1, ymm6
    vpshufb ymm2, ymm7, ymm0
    vpshufb ymm3, ymm7, ymm1
    vpaddb  ymm2, ymm2, ymm3
    vpsadbw ymm2, ymm2, ymm4
    vpaddq  ymm5, ymm5, ymm2
    add     r9, 4
    dec     rcx
    jnz     .Lwa_loop
.Lwa_after:
    vextracti128 xmm0, ymm5, 1
    vpaddq  xmm5, xmm5, xmm0
    vpsrldq xmm0, xmm5, 8
    vpaddq  xmm5, xmm5, xmm0
    vmovq   rax, xmm5
    mov     rcx, rsi
    and     rcx, 3
    jz      .Lwa_done
.Lwa_t:
    popcnt  r8, qword ptr [rdi + r9*8]
    add     rax, r8
    inc     r9
    dec     rcx
    jnz     .Lwa_t
.Lwa_done:
    vzeroupper
    ret

/* ================= gf2_inner: parity(popcount(a[i]&b[i])) ================= */

.globl CSYM(gf2_inner_avx512)
.p2align 4
CSYM(gf2_inner_avx512):        /* rdi=a rsi=b rdx=nwords -> rax(0/1) */
    vpxorq  zmm1, zmm1, zmm1
    xor     r9, r9
    mov     rcx, rdx
    shr     rcx, 3
    jz      .Li5_after
.Li5_loop:
    vmovdqu64 zmm0, [rdi + r9*8]
    vpandq    zmm0, zmm0, [rsi + r9*8]
    vpopcntq  zmm0, zmm0
    vpaddq    zmm1, zmm1, zmm0
    add     r9, 8
    dec     rcx
    jnz     .Li5_loop
.Li5_after:
    vextracti64x4 ymm2, zmm1, 1
    vpaddq  ymm1, ymm1, ymm2
    vextracti128  xmm2, ymm1, 1
    vpaddq  xmm1, xmm1, xmm2
    vpsrldq xmm2, xmm1, 8
    vpaddq  xmm1, xmm1, xmm2
    vmovq   rax, xmm1
    mov     rcx, rdx
    and     rcx, 7
    jz      .Li5_done
.Li5_t:
    mov     r8, [rdi + r9*8]
    and     r8, [rsi + r9*8]
    popcnt  r8, r8
    add     rax, r8
    inc     r9
    dec     rcx
    jnz     .Li5_t
.Li5_done:
    and     rax, 1
    vzeroupper
    ret

.globl CSYM(gf2_inner_avx2)
.p2align 4
CSYM(gf2_inner_avx2):          /* rdi=a rsi=b rdx=nwords -> rax(0/1) */
    vmovdqa ymm7, [rip + .Lpopc_lut]
    vpbroadcastb ymm6, byte ptr [rip + .Lmask0f]
    vpxor   ymm5, ymm5, ymm5
    vpxor   ymm4, ymm4, ymm4
    xor     r9, r9
    mov     rcx, rdx
    shr     rcx, 2
    jz      .Lia_after
.Lia_loop:
    vmovdqu ymm0, [rdi + r9*8]
    vpand   ymm0, ymm0, [rsi + r9*8]
    vpsrlw  ymm1, ymm0, 4
    vpand   ymm0, ymm0, ymm6
    vpand   ymm1, ymm1, ymm6
    vpshufb ymm2, ymm7, ymm0
    vpshufb ymm3, ymm7, ymm1
    vpaddb  ymm2, ymm2, ymm3
    vpsadbw ymm2, ymm2, ymm4
    vpaddq  ymm5, ymm5, ymm2
    add     r9, 4
    dec     rcx
    jnz     .Lia_loop
.Lia_after:
    vextracti128 xmm0, ymm5, 1
    vpaddq  xmm5, xmm5, xmm0
    vpsrldq xmm0, xmm5, 8
    vpaddq  xmm5, xmm5, xmm0
    vmovq   rax, xmm5
    mov     rcx, rdx
    and     rcx, 3
    jz      .Lia_done
.Lia_t:
    mov     r8, [rdi + r9*8]
    and     r8, [rsi + r9*8]
    popcnt  r8, r8
    add     rax, r8
    inc     r9
    dec     rcx
    jnz     .Lia_t
.Lia_done:
    and     rax, 1
    vzeroupper
    ret

RODATA_SECTION
.p2align 5
.Lpopc_lut:                     /* popcount of each nibble, replicated per 128b lane */
    .byte 0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4
    .byte 0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4
.Lmask0f:
    .byte 0x0f

#if defined(__ELF__)
/* Mark stack non-executable (ELF only). */
.section .note.GNU-stack,"",@progbits
#endif
#endif /* __x86_64__ */
