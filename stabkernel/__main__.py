"""`python -m stabkernel` — quick self-test / smoke check."""
import numpy as np

import stabkernel as sk


def main() -> None:
    rng = np.random.default_rng(0)
    a = rng.integers(0, 2**64, size=100, dtype=np.uint64)
    b = rng.integers(0, 2**64, size=100, dtype=np.uint64)
    print("stabkernel", sk.__version__)
    print("backend :", sk.backend())
    print("cores   :", sk.num_threads())
    w_ref = int(sum(int(x).bit_count() for x in a))
    print("weight  :", sk.weight(a), "(ref", w_ref, ")")
    i_ref = sum(int(x & y).bit_count() for x, y in zip(a, b)) & 1
    print("inner   :", sk.inner(a, b), "(ref", i_ref, ")")
    M = rng.integers(0, 2**64, size=(50, 4), dtype=np.uint64)
    print("rank    :", sk.rank(M))

    m, k, Bw = 40, 130, 3
    A = rng.integers(0, 2**64, size=(m, (k + 63) // 64), dtype=np.uint64)
    if k % 64:
        A[:, -1] &= np.uint64((1 << (k % 64)) - 1)
    B = rng.integers(0, 2**64, size=(k, Bw), dtype=np.uint64)
    C = sk.matmul(A, B, k)
    R = np.zeros((m, Bw), dtype=np.uint64)
    for row in range(m):
        for col in range(k):
            if (int(A[row, col >> 6]) >> (col & 63)) & 1:
                R[row] ^= B[col]
    print("matmul  :", "OK" if np.array_equal(C, R) else "MISMATCH")


if __name__ == "__main__":
    main()
