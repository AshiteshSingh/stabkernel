"""stabkernel — Python bindings for the hand-tuned GF(2) / stabilizer
bit-linear-algebra kernels.

Bit-packed convention: a GF(2) vector of `nbits` is a numpy uint64 array of
ceil(nbits/64) little-endian words. These bindings are zero-copy over numpy
buffers.

    import numpy as np, stabkernel as sk
    a = np.random.randint(0, 2**64, size=64, dtype=np.uint64)
    b = np.random.randint(0, 2**64, size=64, dtype=np.uint64)
    print(sk.backend())          # 'avx512' | 'avx2' | 'neon' | 'scalar'
    print(sk.weight(a))          # total Hamming weight
    print(sk.inner(a, b))        # GF(2) symplectic inner product (0/1)
    sk.xor_into(a, b)            # a ^= b  (in place)
    print(sk.rank(M))            # GF(2) rank of a 2D uint64 bit-matrix
"""
import ctypes, os, glob
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))

def _find_lib():
    for cand in [os.path.join(_here, "..", "libstabkernel.so"),
                 os.path.join(_here, "libstabkernel.so"),
                 "libstabkernel.so"]:
        if os.path.exists(cand):
            return os.path.abspath(cand)
    hits = glob.glob(os.path.join(_here, "..", "**", "libstabkernel*"), recursive=True)
    if hits:
        return hits[0]
    raise OSError("libstabkernel shared library not found — run `make lib` first")

_lib = ctypes.CDLL(_find_lib())

_u64p = ctypes.POINTER(ctypes.c_uint64)
_lib.sk_xor.argtypes    = [_u64p, _u64p, ctypes.c_size_t]
_lib.sk_xor.restype     = None
_lib.sk_inner.argtypes  = [_u64p, _u64p, ctypes.c_size_t]
_lib.sk_inner.restype   = ctypes.c_uint64
_lib.sk_weight.argtypes = [_u64p, ctypes.c_size_t]
_lib.sk_weight.restype  = ctypes.c_uint64
_lib.gf2_rank.argtypes  = [_u64p, ctypes.c_size_t, ctypes.c_size_t]
_lib.gf2_rank.restype   = ctypes.c_size_t
_lib.sk_backend.restype = ctypes.c_char_p
_lib.stabkernel_num_threads.restype = ctypes.c_int
_lib.gf2_matmul.argtypes = [_u64p, ctypes.c_size_t, _u64p, ctypes.c_size_t,
                            _u64p, ctypes.c_size_t, ctypes.c_size_t]
_lib.gf2_matmul.restype = None

def _ptr(a):
    a = np.ascontiguousarray(a, dtype=np.uint64)
    return a, a.ctypes.data_as(_u64p)

def backend() -> str:
    return _lib.sk_backend().decode()

def weight(a) -> int:
    a, p = _ptr(a)
    return int(_lib.sk_weight(p, a.size))

def inner(a, b) -> int:
    a, pa = _ptr(a); b, pb = _ptr(b)
    assert a.size == b.size, "length mismatch"
    return int(_lib.sk_inner(pa, pb, a.size))

def xor_into(dst, src):
    """dst ^= src, in place. dst must be a contiguous uint64 numpy array."""
    assert dst.dtype == np.uint64 and dst.flags['C_CONTIGUOUS'], \
        "dst must be contiguous uint64"
    src, ps = _ptr(src)
    assert dst.size == src.size, "length mismatch"
    _lib.sk_xor(dst.ctypes.data_as(_u64p), ps, dst.size)
    return dst

def num_threads() -> int:
    """Number of CPU cores the multi-core kernels will use."""
    return int(_lib.stabkernel_num_threads())

def matmul(A, B, k):
    """GF(2) matmul C(m x p) = A(m x k) * B(k x p), bit-packed uint64 rows.
    A: (m, ceil(k/64)) uint64;  B: (k, Bw) uint64  ->  C: (m, Bw) uint64.
    Runs four-Russians + SIMD row-XOR across all cores."""
    A = np.ascontiguousarray(A, dtype=np.uint64)
    B = np.ascontiguousarray(B, dtype=np.uint64)
    m, Aw = A.shape
    _, Bw = B.shape
    C = np.zeros((m, Bw), dtype=np.uint64)
    _lib.gf2_matmul(A.ctypes.data_as(_u64p), Aw, B.ctypes.data_as(_u64p), Bw,
                    C.ctypes.data_as(_u64p), m, k)
    return C

def rank(matrix) -> int:
    """GF(2) rank of a 2D uint64 bit-packed matrix (m rows x nwords)."""
    m = np.ascontiguousarray(matrix, dtype=np.uint64).copy()
    if m.ndim != 2:
        raise ValueError("expected a 2D (m x nwords) uint64 array")
    rows, nwords = m.shape
    return int(_lib.gf2_rank(m.ctypes.data_as(_u64p), rows, nwords))

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    a = rng.integers(0, 2**64, size=100, dtype=np.uint64)
    b = rng.integers(0, 2**64, size=100, dtype=np.uint64)
    print("backend:", backend())
    print("weight :", weight(a), "(numpy check:",
          int(sum(int(x).bit_count() for x in a)), ")")
    print("inner  :", inner(a, b), "(numpy check:",
          sum(int(x & y).bit_count() for x, y in zip(a, b)) & 1, ")")
    M = rng.integers(0, 2**64, size=(50, 4), dtype=np.uint64)
    print("rank   :", rank(M))
    print("cores  :", num_threads())
    # GF(2) matmul cross-check against a slow numpy reference
    m, k, Bw = 40, 130, 3
    A = rng.integers(0, 2**64, size=(m, (k + 63) // 64), dtype=np.uint64)
    if k % 64:
        A[:, -1] &= np.uint64((1 << (k % 64)) - 1)
    B = rng.integers(0, 2**64, size=(k, Bw), dtype=np.uint64)
    C = matmul(A, B, k)
    R = np.zeros((m, Bw), dtype=np.uint64)
    for i in range(m):
        for l in range(k):
            if (int(A[i, l >> 6]) >> (l & 63)) & 1:
                R[i] ^= B[l]
    print("matmul :", "OK" if np.array_equal(C, R) else "MISMATCH")
