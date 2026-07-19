"""stabkernel — SIMD + multi-core CPU kernels for GF(2) / stabilizer
bit-linear-algebra.

Bit-packed convention: a GF(2) vector of ``nbits`` is a numpy ``uint64`` array of
``ceil(nbits/64)`` little-endian words. These bindings are zero-copy over numpy
buffers and dispatch at load time to the fastest kernel your CPU supports
(AVX-512 > AVX2 > scalar on x86-64; scalar elsewhere until NEON is re-enabled).

    import numpy as np, stabkernel as sk
    a = np.random.randint(0, 2**64, size=64, dtype=np.uint64)
    b = np.random.randint(0, 2**64, size=64, dtype=np.uint64)
    print(sk.backend())          # 'avx512' | 'avx2' | 'neon' | 'scalar'
    print(sk.weight(a))          # total Hamming weight
    print(sk.inner(a, b))        # GF(2) symplectic inner product (0/1)
    sk.xor_into(a, b)            # a ^= b  (in place)
    print(sk.rank(M))            # GF(2) rank of a 2D uint64 bit-matrix
    C = sk.matmul(A, B, k)       # GF(2) matmul across all cores
"""
import ctypes
import glob
import os

import numpy as np

__version__ = "0.1.0"
__all__ = [
    "backend", "weight", "inner", "xor_into", "rank", "matmul",
    "num_threads", "__version__",
]

_here = os.path.dirname(os.path.abspath(__file__))


def _find_lib():
    # 1) bundled shared library inside the installed package (wheel case)
    patterns = [
        "libstabkernel*.so", "libstabkernel*.dylib", "stabkernel*.dll",
        "_native*.so", "_native*.pyd", "_native*.dylib",
    ]
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(_here, pat)))
        if hits:
            return hits[0]
    # 2) dev fallback: a `make`-built library at the repo root
    for cand in (
        os.path.join(_here, "..", "libstabkernel.so"),
        os.path.join(_here, "..", "libstabkernel.dylib"),
        "libstabkernel.so",
    ):
        if os.path.exists(cand):
            return os.path.abspath(cand)
    raise OSError(
        "stabkernel native library not found. If installing from source, "
        "ensure a C compiler is available; for development run `make` at the "
        "repo root."
    )


_lib = ctypes.CDLL(_find_lib())

_u64p = ctypes.POINTER(ctypes.c_uint64)
_lib.sk_xor.argtypes = [_u64p, _u64p, ctypes.c_size_t]
_lib.sk_xor.restype = None
_lib.sk_inner.argtypes = [_u64p, _u64p, ctypes.c_size_t]
_lib.sk_inner.restype = ctypes.c_uint64
_lib.sk_weight.argtypes = [_u64p, ctypes.c_size_t]
_lib.sk_weight.restype = ctypes.c_uint64
_lib.gf2_rank.argtypes = [_u64p, ctypes.c_size_t, ctypes.c_size_t]
_lib.gf2_rank.restype = ctypes.c_size_t
_lib.sk_backend.restype = ctypes.c_char_p
_lib.stabkernel_num_threads.restype = ctypes.c_int
_lib.gf2_matmul.argtypes = [_u64p, ctypes.c_size_t, _u64p, ctypes.c_size_t,
                            _u64p, ctypes.c_size_t, ctypes.c_size_t]
_lib.gf2_matmul.restype = None


def _ptr(a):
    a = np.ascontiguousarray(a, dtype=np.uint64)
    return a, a.ctypes.data_as(_u64p)


def backend() -> str:
    """Name of the active backend: 'avx512', 'avx2', 'neon', or 'scalar'."""
    return _lib.sk_backend().decode()


def weight(a) -> int:
    """Total Hamming weight sum_i popcount(a[i])."""
    a, p = _ptr(a)
    return int(_lib.sk_weight(p, a.size))


def inner(a, b) -> int:
    """GF(2) symplectic inner product parity(sum popcount(a & b)) -> 0/1."""
    a, pa = _ptr(a)
    b, pb = _ptr(b)
    if a.size != b.size:
        raise ValueError("length mismatch")
    return int(_lib.sk_inner(pa, pb, a.size))


def xor_into(dst, src):
    """dst ^= src, in place. dst must be a contiguous uint64 numpy array."""
    if not (dst.dtype == np.uint64 and dst.flags["C_CONTIGUOUS"]):
        raise ValueError("dst must be a contiguous uint64 array")
    src, ps = _ptr(src)
    if dst.size != src.size:
        raise ValueError("length mismatch")
    _lib.sk_xor(dst.ctypes.data_as(_u64p), ps, dst.size)
    return dst


def num_threads() -> int:
    """Number of CPU cores the multi-core kernels will use."""
    return int(_lib.stabkernel_num_threads())


def matmul(A, B, k):
    """GF(2) matmul C(m x p) = A(m x k) * B(k x p), bit-packed uint64 rows.

    A: (m, ceil(k/64)) uint64;  B: (k, Bw) uint64  ->  C: (m, Bw) uint64.
    Runs four-Russians + SIMD row-XOR across all cores.
    """
    A = np.ascontiguousarray(A, dtype=np.uint64)
    B = np.ascontiguousarray(B, dtype=np.uint64)
    m, _Aw = A.shape
    _, Bw = B.shape
    C = np.zeros((m, Bw), dtype=np.uint64)
    _lib.gf2_matmul(A.ctypes.data_as(_u64p), _Aw, B.ctypes.data_as(_u64p), Bw,
                    C.ctypes.data_as(_u64p), m, k)
    return C


def rank(matrix) -> int:
    """GF(2) rank of a 2D uint64 bit-packed matrix (m rows x nwords)."""
    m = np.ascontiguousarray(matrix, dtype=np.uint64).copy()
    if m.ndim != 2:
        raise ValueError("expected a 2D (m x nwords) uint64 array")
    rows, nwords = m.shape
    return int(_lib.gf2_rank(m.ctypes.data_as(_u64p), rows, nwords))
