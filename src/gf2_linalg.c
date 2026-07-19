/* GF(2) linear algebra built on the hot assembly primitives (BLIS-style:
 * tiny hand-tuned kernels + portable C orchestration). */
#include "gf2kernel.h"

static inline int get_bit(const uint64_t *row, size_t j) {
    return (int)((row[j >> 6] >> (j & 63)) & 1u);
}

/* In-place row echelon reduction; returns rank. Row stride = nwords words.
 * Pivot search + elimination; every row XOR goes through the gf2_xor kernel. */
size_t gf2_rank(uint64_t *rows, size_t m, size_t nwords) {
    size_t nbits = nwords * 64;
    size_t rank = 0;
    for (size_t col = 0; col < nbits && rank < m; col++) {
        /* find a pivot row at/below `rank` with a 1 in this column */
        size_t piv = m;
        for (size_t r = rank; r < m; r++) {
            if (get_bit(rows + r * nwords, col)) { piv = r; break; }
        }
        if (piv == m) continue;
        /* swap pivot into place */
        if (piv != rank) {
            uint64_t *ra = rows + rank * nwords;
            uint64_t *rb = rows + piv * nwords;
            for (size_t w = 0; w < nwords; w++) {
                uint64_t t = ra[w]; ra[w] = rb[w]; rb[w] = t;
            }
        }
        /* eliminate this column from all other rows */
        const uint64_t *pv = rows + rank * nwords;
        for (size_t r = 0; r < m; r++) {
            if (r == rank) continue;
            if (get_bit(rows + r * nwords, col))
                gf2_xor(rows + r * nwords, pv, nwords);
        }
        rank++;
    }
    return rank;
}

size_t gf2_solve_rank(uint64_t *rows, size_t m, size_t nwords) {
    return gf2_rank(rows, m, nwords);
}
