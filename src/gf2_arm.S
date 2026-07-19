/* Hand-written AArch64 / Apple-Silicon / Android-arm64 (NEON) kernels for GF(2)
 * bit-linear-algebra. AAPCS64: args x0, x1, x2; return x0. Uses the NEON CNT
 * byte-popcount path (identical instruction sequence on Apple M1-M4, AWS
 * Graviton, and Qualcomm Snapdragon / Android arm64-v8a).
 *
 * Portable across ELF (Linux / Android) and Mach-O (macOS): symbol names are
 * emitted with the platform's C name-mangling via CSYM(), and the ELF-only
 * non-executable-stack note is guarded out on Apple.
 */
#if defined(__aarch64__)

/* C symbol mangling: Mach-O prefixes C symbols with an underscore, ELF does not. */
#if defined(__APPLE__)
#  define CSYM(x) _##x
#else
#  define CSYM(x) x
#endif

.text

/* ================= gf2_xor_neon: dst[i] ^= src[i] ================= */
.globl CSYM(gf2_xor_neon)
.p2align 4
CSYM(gf2_xor_neon):             // x0=dst x1=src x2=nwords
    lsr     x3, x2, #1          // blocks of 2 qwords (128b)
    cbz     x3, 1f
0:  ldr     q0, [x0]
    ldr     q1, [x1], #16
    eor     v0.16b, v0.16b, v1.16b
    str     q0, [x0], #16
    subs    x3, x3, #1
    b.ne    0b
1:  tst     x2, #1              // trailing single qword
    b.eq    2f
    ldr     x4, [x1]
    ldr     x5, [x0]
    eor     x5, x5, x4
    str     x5, [x0]
2:  ret

/* ================= gf2_weight_neon: sum popcount(a[i]) ================= */
.globl CSYM(gf2_weight_neon)
.p2align 4
CSYM(gf2_weight_neon):          // x0=a x1=nwords -> x0
    movi    v2.16b, #0          // qword accumulator
    mov     x2, x0              // ptr
    lsr     x3, x1, #1          // 2-qword blocks
    cbz     x3, 1f
0:  ldr     q0, [x2], #16
    cnt     v0.16b, v0.16b      // per-byte popcount
    uaddlp  v0.8h, v0.16b       // pairwise widen add -> halfwords
    uaddlp  v0.4s, v0.8h
    uaddlp  v0.2d, v0.4s        // -> 2 qword partial sums
    add     v2.2d, v2.2d, v0.2d
    subs    x3, x3, #1
    b.ne    0b
1:  addp    d2, v2.2d           // horizontal add of the 2 lanes
    fmov    x0, d2
    tst     x1, #1
    b.eq    2f
    ldr     x4, [x2]
    // scalar popcount of x4 via NEON
    fmov    d0, x4
    cnt     v0.8b, v0.8b
    uaddlv  h0, v0.8b
    fmov    w5, s0
    add     x0, x0, x5
2:  ret

/* ================= gf2_inner_neon: parity(popcount(a[i]&b[i])) ========= */
.globl CSYM(gf2_inner_neon)
.p2align 4
CSYM(gf2_inner_neon):           // x0=a x1=b x2=nwords -> x0 (0/1)
    movi    v2.16b, #0
    mov     x3, x0
    mov     x4, x1
    lsr     x5, x2, #1
    cbz     x5, 1f
0:  ldr     q0, [x3], #16
    ldr     q1, [x4], #16
    and     v0.16b, v0.16b, v1.16b
    cnt     v0.16b, v0.16b
    uaddlp  v0.8h, v0.16b
    uaddlp  v0.4s, v0.8h
    uaddlp  v0.2d, v0.4s
    add     v2.2d, v2.2d, v0.2d
    subs    x5, x5, #1
    b.ne    0b
1:  addp    d2, v2.2d
    fmov    x0, d2
    tst     x2, #1
    b.eq    2f
    ldr     x6, [x3]
    ldr     x7, [x4]
    and     x6, x6, x7
    fmov    d0, x6
    cnt     v0.8b, v0.8b
    uaddlv  h0, v0.8b
    fmov    w8, s0
    add     x0, x0, x8
2:  and     x0, x0, #1
    ret

#if defined(__ELF__)
.section .note.GNU-stack,"",%progbits
#endif
#endif /* __aarch64__ */
