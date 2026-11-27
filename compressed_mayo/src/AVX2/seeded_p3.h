// SPDX-License-Identifier: Apache-2.0
/* Kernels of the seeded-P3 key generation (match_P3_targets in mayo.c) for the x86
   vector backends, avx2 and avx2_gfni. They only use AVX2 and the multiplication tables
   of arithmetic_common.h, and read P1 and P2 in the layout both backends share: P1
   packed (PINV), P2 spread. */
#ifndef SEEDED_P3_H
#define SEEDED_P3_H

#include <immintrin.h>
#include <stdalign.h>
#include <stdint.h>
#include <string.h>
#include <arithmetic_common.h>

/* an m-vec in 4-limb groups, one register each; with 4n + 1 limbs (MAYO_1, MAYO_5) the
   last limb is a piece of its own, otherwise the last group is loaded at
   M_VEC_LIMBS_MAX - 4 and masked to the limbs the earlier groups left */
#define SEEDED_TAIL (M_VEC_LIMBS_MAX % 4 == 1 && M_VEC_LIMBS_MAX > 1)
#define SEEDED_GROUPS (SEEDED_TAIL ? M_VEC_LIMBS_MAX / 4 : (M_VEC_LIMBS_MAX + 3) / 4)

static inline size_t seeded_group_off(size_t g) {
    return 4 * g + 4 <= M_VEC_LIMBS_MAX ? 4 * g : M_VEC_LIMBS_MAX - 4;
}

static inline __m256i seeded_group_mask(size_t g) {
    const int64_t lo = 0x0f0f0f0f0f0f0f0f;
    const size_t skip = 4 * g - seeded_group_off(g);
    return _mm256_set_epi64x(lo, skip < 3 ? lo : 0, skip < 2 ? lo : 0, skip < 1 ? lo : 0);
}

#define MAYO_HAVE_SEEDED_P3_KERNELS 1
// one multiplication table per entry of x, in the multab format of a single column
static inline void seeded_x_multabs(const unsigned char *x, __m128i *tabs) {
    for (size_t r = 0; r < V_MAX; r++) {
        tabs[r] = _mm256_castsi256_si128(tbl32_gf16_multab2(x[r]));
    }
}

#ifdef MAYO_GFNI
/* acc[r] += sum_c (P1 + P1^t)[r][c] * x[c]. One pass over packed P1: the entry p at
   (r, c), r < c, adds p x[c] to row r, kept in a register, and p x[r] to row c, kept in
   colacc until the pass reaches row c; each product is one affine with the matrix of
   x[c] or x[r]. Registers of 4 limbs; the last one is loaded at M_VEC_LIMBS_MAX - 4 and
   stores only the limbs the earlier ones did not. */
static
inline void P1P1t_times_vec(const uint64_t *P1, const unsigned char *x, uint64_t *acc){
    uint64_t xt[V_MAX];
    mayo_gfni_tabs(x, xt, V_MAX);
    for (size_t g = 0; g * 4 < M_VEC_LIMBS_MAX; g++) {
        const size_t off = 4 * g + 4 <= M_VEC_LIMBS_MAX ? 4 * g : M_VEC_LIMBS_MAX - 4;
        const size_t skip = 4 * g - off;
        __m256i colacc[V_MAX];
        for (size_t r = 0; r < V_MAX; r++) {
            colacc[r] = _mm256_setzero_si256();
        }
        size_t pos = 0;
        for (size_t r = 0; r < V_MAX; r++) {
            const __m256i mr = _mm256_set1_epi64x((long long) xt[r]);
            __m256i t = colacc[r];
            pos++; /* skip the diagonal */
            for (size_t c = r + 1; c < V_MAX; c++, pos++) {
                const __m256i in = _mm256_loadu_si256((const __m256i *) (PINV(P1, pos) + off));
                t ^= _mm256_gf2p8affine_epi64_epi8(in, _mm256_set1_epi64x((long long) xt[c]), 0);
                colacc[c] ^= _mm256_gf2p8affine_epi64_epi8(in, mr, 0);
            }
            /* row r is complete: earlier rows have added all its transposed terms */
            alignas(32) uint64_t out[4];
            _mm256_store_si256((__m256i *) out, t);
            for (size_t l = skip; l < 4; l++) {
                acc[r * M_VEC_LIMBS_MAX + off + l] ^= out[l];
            }
        }
        mayo_secure_clear(colacc, sizeof(colacc));
    }
    mayo_secure_clear(xt, sizeof(xt));
}

#else
/* acc[r] += sum_c (P1 + P1^t)[r][c] * x[c]. One pass over packed P1, each entry unpacked
   once: the entry p at (r, c), r < c, adds p x[c] to row r, kept in registers, and
   p x[r] to row c, kept in colacc until the pass reaches row c. Each group of 4 limbs
   goes separately, with the low and high nibble products in colacc[.][0], [1] until
   they are recombined, then, for a 4n+1-limb m-vec, the tail limb, with both its
   nibbles in one lane. */
static
inline void P1P1t_times_vec(const uint64_t *P1, const unsigned char *x, uint64_t *acc){
    __m128i x_multabs[V_MAX];
    seeded_x_multabs(x, x_multabs);
    const __m256i lo = _mm256_set1_epi8(0x0f);
    for (size_t g = 0; g < SEEDED_GROUPS; g++) {
        const size_t goff = seeded_group_off(g);
        const __m256i gmask = seeded_group_mask(g);
        /* the transposed contributions, in the unpacked (lo/hi) form per output row */
        __m256i colacc[V_MAX][2];
        for (size_t r = 0; r < V_MAX; r++) colacc[r][0] = colacc[r][1] = _mm256_setzero_si256();
        size_t pos = 0;
        for (size_t r = 0; r < V_MAX; r++) {
            const __m256i tr = _mm256_broadcastsi128_si256(x_multabs[r]);
            __m256i t0 = colacc[r][0], t1 = colacc[r][1];
            pos++; /* skip the diagonal */
            for (size_t c = r + 1; c < V_MAX; c++, pos++) {
                __m256i in_odd = _mm256_loadu_si256((const __m256i *) (PINV(P1, pos) + goff));
                __m256i in_even = _mm256_srli_epi16(in_odd, 4) & gmask;
                in_odd &= gmask;
                const __m256i tc = _mm256_broadcastsi128_si256(x_multabs[c]);
                t0 ^= _mm256_shuffle_epi8(tc, in_odd);
                t1 ^= _mm256_shuffle_epi8(tc, in_even);
                colacc[c][0] ^= _mm256_shuffle_epi8(tr, in_odd);
                colacc[c][1] ^= _mm256_shuffle_epi8(tr, in_even);
            }
            /* row r is complete: earlier rows have added all its transposed terms */
            __m256i *a0 = (__m256i *) (acc + r * M_VEC_LIMBS_MAX + goff);
            const __m256i t = (t1 ^ _mm256_srli_epi16(t0, 4)) & lo;
            _mm256_storeu_si256(a0, _mm256_loadu_si256(a0) ^ t0 ^ _mm256_slli_epi16(t, 4));
        }
        mayo_secure_clear(colacc, sizeof(colacc));
    }
#if SEEDED_TAIL
    {   /* the tail limb: its low and high nibbles in the two qwords of a lane, one table
           per shuffle (only lane 0 of each product is used) */
        __m256i colacc[V_MAX];
        for (size_t r = 0; r < V_MAX; r++) colacc[r] = _mm256_setzero_si256();
        size_t pos = 0;
        for (size_t r = 0; r < V_MAX; r++) {
            const __m256i tr = _mm256_broadcastsi128_si256(x_multabs[r]);
            __m256i t0 = colacc[r];
            pos++;
            for (size_t c = r + 1; c < V_MAX; c++, pos++) {
                const __m256i xx = _mm256_broadcastq_epi64(_mm_loadu_si64((const void *) (PINV(P1, pos) + M_VEC_LIMBS_MAX - 1)));
                const __m256i in = _mm256_srlv_epi64(xx, _mm256_set_epi64x(4, 0, 4, 0)) & lo;
                t0 ^= _mm256_shuffle_epi8(_mm256_broadcastsi128_si256(x_multabs[c]), in);
                colacc[c] ^= _mm256_shuffle_epi8(tr, in);
            }
            const __m256i bq = _mm256_unpackhi_epi64(t0, t0);
            const __m256i t = (bq ^ _mm256_srli_epi16(t0, 4)) & lo;
            alignas(32) uint64_t col[4];
            _mm256_store_si256((__m256i *) col, t0 ^ _mm256_slli_epi16(t, 4));
            acc[r * M_VEC_LIMBS_MAX + M_VEC_LIMBS_MAX - 1] ^= col[0];
        }
        mayo_secure_clear(colacc, sizeof(colacc));
    }
#endif
    mayo_secure_clear(x_multabs, sizeof(x_multabs));
}
#endif

// rows of the match_P3_targets systems: at most v columns of A and o - 1 right-hand sides
#define SEEDED_REGS ((V_MAX + O_MAX + 31) / 32)
#define SEEDED_STRIDE (SEEDED_REGS * 32)
// rows of [A^T | I] for the two-block case (match_P3_targets): m + v columns
#define SEEDED_T_REGS ((M_MAX + V_MAX + 31) / 32)
#define SEEDED_T_STRIDE (SEEDED_T_REGS * 32)
#define SEEDED_MAXREGS (SEEDED_REGS > SEEDED_T_REGS ? SEEDED_REGS : SEEDED_T_REGS)
// a v-vector, zero after its v entries
#define SEEDED_V_REGS ((V_MAX + 31) / 32)

// 16 nibbles of a limb, one per byte, nibble k in byte k
static inline __m128i seeded_spread_limb(uint64_t w) {
    const __m128i x = _mm_cvtsi64_si128((long long) w);
    const __m128i lo = _mm_set1_epi8(0x0f);
    return _mm_unpacklo_epi8(_mm_and_si128(x, lo), _mm_and_si128(_mm_srli_epi16(x, 4), lo));
}

// 16x16 byte transpose: four rounds, each interleaving row i with row i + 8
static inline void seeded_transpose16(__m128i *a) {
    __m128i t[16];
    for (int round = 0; round < 4; round++) {
        for (int i = 0; i < 8; i++) {
            t[2 * i] = _mm_unpacklo_epi8(a[i], a[i + 8]);
            t[2 * i + 1] = _mm_unpackhi_epi8(a[i], a[i + 8]);
        }
        for (int i = 0; i < 16; i++) {
            a[i] = t[i];
        }
    }
}

/* rows[(k - k0) * SEEDED_STRIDE + r] = nibble k of the m-vec L[r], r < v, k0 <= k < k1:
   L transposed in 16 x 16 blocks of nibbles, so both sides are read and written in
   whole limbs and rows */
static inline void seeded_rows_from_L(const uint64_t *L, unsigned char *rows, int v, int mvl,
                                      int k0, int k1) {
    alignas(16) unsigned char blk[16][16];
    for (int kb = k0 / 16 * 16; kb < k1; kb += 16) {
        for (int rb = 0; rb < v; rb += 16) {
            __m128i a[16];
            for (int i = 0; i < 16; i++) {
                a[i] = rb + i < v ? seeded_spread_limb(L[(size_t) (rb + i) * mvl + kb / 16]) : _mm_setzero_si128();
            }
            seeded_transpose16(a);
            for (int i = 0; i < 16; i++) {
                _mm_store_si128((__m128i *) blk[i], a[i]);
            }
            /* a whole block where the row has room: the columns from v on get zeros,
               before the caller writes its right-hand sides there */
            for (int k = MAYO_MAX(kb, k0); k < MAYO_MIN(kb + 16, k1); k++) {
                unsigned char *dst = rows + (size_t) (k - k0) * SEEDED_STRIDE + rb;
                if (rb + 16 <= SEEDED_STRIDE) {
                    _mm_storeu_si128((__m128i *) dst, _mm_load_si128((const __m128i *) blk[k - kb]));
                } else {
                    memcpy(dst, blk[k - kb], (size_t) MAYO_MIN(16, v - rb));
                }
            }
        }
    }
    mayo_secure_clear(blk, sizeof(blk));
}

// the pivot search looks this many rows below the diagonal (cf. EF)
#define SEEDED_SEARCH 32

#ifdef MAYO_GFNI
/* The solvers work in the copy of GF(16) inside GF(256), a -> sum_i a_i b^i with
   b = 0x5c, a root of x^4 + x + 1 there, where a product is one vgf2p8mulb: in the
   vgf2p8affineqb encoding, SEEDED_EMBED maps GF(16) there and SEEDED_UNEMBED back. Both
   fix 0 and 1. */
#define SEEDED_EMBED 0x010002020a040e04LL
#define SEEDED_UNEMBED 0x0104201400000000LL
static inline void seeded_map(unsigned char *M, int nregs, long long mat) {
    __m256i *R = (__m256i *) M;
    for (int l = 0; l < nregs; l++) {
        R[l] = _mm256_gf2p8affine_epi64_epi8(R[l], _mm256_set1_epi64x(mat), 0);
    }
}
// multiplying by c: c broadcast, then one vgf2p8mulb
static inline __m256i seeded_coef(unsigned char c) {
    return _mm256_set1_epi8((char) c);
}
// the GF(256) inverse, 0 for 0, is that of GF(16) in the copy
static inline __m256i seeded_coef_inv(unsigned char c) {
    return _mm256_gf2p8affineinv_epi64_epi8(seeded_coef(c), _mm256_set1_epi64x(0x0102040810204080LL), 0);
}
static inline __m256i seeded_mulc(__m256i coef, __m256i x) {
    return _mm256_gf2p8mul_epi8(x, coef);
}
#else
static inline void seeded_map(unsigned char *M, int nregs, long long mat) {
    (void) M, (void) nregs, (void) mat;
}
#define SEEDED_EMBED 0
#define SEEDED_UNEMBED 0
// multiplying by c: its table, then one shuffle
static inline __m256i seeded_coef(unsigned char c) {
    return tbl32_gf16_multab2(c);
}
static inline __m256i seeded_coef_inv(unsigned char c) {
    return tbl32_gf16_multab2(inverse_f(c));
}
static inline __m256i seeded_mulc(__m256i coef, __m256i x) {
    return _mm256_shuffle_epi8(coef, x);
}
#endif

/* Gauss-Jordan steps c0..c1-1 on the n rows of M (one nibble per byte, rows of regs
   registers, zero past the last column), with columns 0..c0-1 already pivoted, i.e. rows
   0..c0-1 of the square part the identity there and rows c0..n-1 zero there. Steps 0..n-1
   in one go turn [A | B] into [I | A^-1 B]. The pivots stay on the diagonal, a zero one is
   first made nonzero by adding the rows below it, so addresses and trip counts are
   public. Returns 0 iff a pivot is zero, i.e. the leading c1 x c1 block of the square
   part is singular, or one is not found within SEEDED_SEARCH rows, 0xff if not. */
/* one step on column col of rows of regs registers: the pivot row, from rows col..n-1,
   is written to row col and cleared from column col of rows r0..r1-1 other than col. It
   is zero outside registers creg = col / 32 .. hreg - 1; the caller passes both as
   constants, so the register loops have fixed trip counts (cf. EF_PCR) */
static inline __attribute__((always_inline))
unsigned char seeded_solve_step(unsigned char *M, int n, int r0, int r1, int col, const int creg,
                                const int hreg, const int regs) {
    __m256i *R = (__m256i *) M;
    const int stride = 32 * regs;
    /* the pivot row is row col plus the rows after it up to the first one that is
       nonzero in column col, at most SEEDED_SEARCH of them: row r is in iff rows
       col..r-1 are zero there. The masks go through memory, so that each is broadcast
       by a load, and the additions do not depend on each other. */
    const int last = MAYO_MIN(n - 1, col + SEEDED_SEARCH);
    int64_t sel[SEEDED_SEARCH + 1];
    int64_t zero = ((int64_t) M[col * stride + col] - 1) >> 63;
    for (int r = col + 1; r <= last; r++) {
        sel[r - col] = zero;
        zero &= ((int64_t) M[r * stride + col] - 1) >> 63;
    }
    __m256i pr[SEEDED_MAXREGS] = {0};
    for (int l = creg; l < hreg; l++) {
        pr[l] = R[col * regs + l];
    }
    for (int r = col + 1; r <= last; r++) {
        const __m256i mask = _mm256_set1_epi64x(sel[r - col]);
        for (int l = creg; l < hreg; l++) {
            pr[l] ^= mask & R[r * regs + l];
        }
    }
    const unsigned char piv = ((const unsigned char *) pr)[col];
    const unsigned char ok = ct_compare_8(piv, 0);
    const __m256i inv = seeded_coef_inv(piv);
    for (int l = creg; l < hreg; l++) {
        pr[l] = seeded_mulc(inv, pr[l]);
    }
    /* clear column col; the pivot row itself is written back after. Rows col..n-1 are
       zero left of col, so pr is too. */
    for (int r = r0; r < r1; r++) {
        const __m256i coef = seeded_coef(M[r * stride + col]);
        for (int l = creg; l < hreg; l++) {
            R[r * regs + l] ^= seeded_mulc(coef, pr[l]);
        }
    }
    for (int l = creg; l < hreg; l++) {
        R[col * regs + l] = pr[l];
    }
    mayo_secure_clear(pr, sizeof(pr));
    return ok;
}

#define SEEDED_BAND(B, R0, R1, HREG, REGS) \
    if ((B) < (REGS)) for (; col < c1 && col < 32 * ((B) + 1); col++) ok &= seeded_solve_step(M, n, (R0), (R1), col, (B), (HREG), (REGS));
#define SEEDED_SBAND(B) SEEDED_BAND(B, 0, n, SEEDED_REGS, SEEDED_REGS)
// SEEDED_REGS <= 6 (V_MAX + O_MAX <= 154 + 12)
static inline unsigned char seeded_solve(unsigned char *M, int n, int c0, int c1) {
    unsigned char ok = 0xff;
    int col = c0;
    seeded_map(M, n * SEEDED_REGS, SEEDED_EMBED);
    SEEDED_SBAND(0) SEEDED_SBAND(1) SEEDED_SBAND(2) SEEDED_SBAND(3) SEEDED_SBAND(4) SEEDED_SBAND(5)
    seeded_map(M, n * SEEDED_REGS, SEEDED_UNEMBED);
    return ok; // secret until the caller declassifies it
}

/* Steps 0..c1-1 of the same on the n rows of [A^T | I], A^T n x c1, followed by ne - n
   more rows that are only reduced, SEEDED_T_REGS <= 9 (M_MAX + V_MAX <= 2 * 142). Each
   step clears its column only below the pivot (the rows above are never read again, so
   rows col + 1..ne - 1 come out as Gauss-Jordan elimination leaves them), and returns
   the same. The pivot row of step col is a sum of rows <= col + SEEDED_SEARCH, which the
   earlier pivot rows changed only in the same identity columns, so it is zero from
   identity column col + SEEDED_SEARCH + 1 on, and band B stops at the register holding
   column c1 + min(n - 1, 32 B + 31 + SEEDED_SEARCH). */
#define SEEDED_TBAND(B) \
    SEEDED_BAND(B, col + 1, ne, MAYO_MIN(SEEDED_T_REGS, (c1 + MAYO_MIN(n - 1, 32 * (B) + 31 + SEEDED_SEARCH)) / 32 + 1), SEEDED_T_REGS)
static inline unsigned char seeded_tsolve(unsigned char *M, int n, int ne, int c1) {
    unsigned char ok = 0xff;
    int col = 0;
    seeded_map(M, ne * SEEDED_T_REGS, SEEDED_EMBED);
    SEEDED_TBAND(0) SEEDED_TBAND(1) SEEDED_TBAND(2) SEEDED_TBAND(3) SEEDED_TBAND(4)
    SEEDED_TBAND(5) SEEDED_TBAND(6) SEEDED_TBAND(7) SEEDED_TBAND(8)
    seeded_map(M, ne * SEEDED_T_REGS, SEEDED_UNEMBED);
    return ok; // secret until the caller declassifies it
}
#undef SEEDED_TBAND
#undef SEEDED_SBAND
#undef SEEDED_BAND

// a * b per byte, for nibbles a and b: the log tables give 0 whenever a or b is 0
static inline __m256i seeded_mul_vec(__m256i a, __m256i b) {
    const __m256i logt = _mm256_load_si256((__m256i const *) mayo_gf16_log);
    __m256i idx = _mm256_add_epi8(_mm256_shuffle_epi8(logt, a), _mm256_shuffle_epi8(logt, b));
    idx = _mm256_min_epu8(idx, _mm256_sub_epi8(idx, _mm256_set1_epi8(15)));
    return _mm256_shuffle_epi8(_mm256_load_si256((__m256i const *) mayo_gf16_alog), idx);
}

// the dot product of two v-vectors
static inline unsigned char seeded_dot(const unsigned char *a, const unsigned char *b) {
    __m256i acc = _mm256_setzero_si256();
    for (int l = 0; l < SEEDED_V_REGS; l++) {
        acc ^= seeded_mul_vec(_mm256_load_si256((const __m256i *) a + l), _mm256_load_si256((const __m256i *) b + l));
    }
    __m128i t = _mm_xor_si128(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
    t = _mm_xor_si128(t, _mm_srli_si128(t, 8));
    t = _mm_xor_si128(t, _mm_srli_si128(t, 4));
    t = _mm_xor_si128(t, _mm_srli_si128(t, 2));
    t = _mm_xor_si128(t, _mm_srli_si128(t, 1));
    return (unsigned char) (_mm_cvtsi128_si32(t) & 0xf);
}

// out += c * a, for v-vectors
static inline void seeded_axpy(unsigned char *out, const unsigned char *a, unsigned char c) {
    const __m256i tab = tbl32_gf16_multab2(c);
    for (int l = 0; l < SEEDED_V_REGS; l++) {
        __m256i *o = (__m256i *) out + l;
        _mm256_store_si256(o, _mm256_load_si256(o) ^ _mm256_shuffle_epi8(tab, _mm256_load_si256((const __m256i *) a + l)));
    }
}

// the m nibbles of an m-vector, one per byte
static inline void seeded_spread_mvec(const uint64_t *vec, int m, unsigned char *out) {
    for (int l = 0; l < m / 16; l++) {
        _mm_storeu_si128((__m128i *) (out + 16 * l), seeded_spread_limb(vec[l]));
    }
    for (int k = m / 16 * 16; k < m; k++) {
        out[k] = (vec[k / 16] >> (4 * (k % 16))) & 0xf;
    }
}

// acc[c] += sum_r x[r] * P2[r][c], c < o: one row of O^t P2, P2 spread
static
inline void xt_times_P2(const uint64_t *P2, const unsigned char *x, uint64_t *acc){
    __m128i x_multabs[V_MAX];
    seeded_x_multabs(x, x_multabs);
    const __m256i lo = _mm256_set1_epi8(0x0f);
    for (size_t c = 0; c < O_MAX; c++) {
        for (size_t g = 0; g < SEEDED_GROUPS; g++) {
            const size_t goff = seeded_group_off(g);
            const __m256i gmask = seeded_group_mask(g);
            __m256i t0 = _mm256_setzero_si256(), t1 = _mm256_setzero_si256();
            for (size_t r = 0; r < V_MAX; r++) {
                __m256i in_odd = _mm256_loadu_si256((const __m256i *) (P2 + (r * O_MAX + c) * M_VEC_LIMBS_MAX + goff));
                __m256i in_even = _mm256_srli_epi16(in_odd, 4) & gmask;
                in_odd &= gmask;
                const __m256i tr = _mm256_broadcastsi128_si256(x_multabs[r]);
                t0 ^= _mm256_shuffle_epi8(tr, in_odd);
                t1 ^= _mm256_shuffle_epi8(tr, in_even);
            }
            __m256i *a0 = (__m256i *) (acc + c * M_VEC_LIMBS_MAX + goff);
            const __m256i t = (t1 ^ _mm256_srli_epi16(t0, 4)) & lo;
            _mm256_storeu_si256(a0, _mm256_loadu_si256(a0) ^ t0 ^ _mm256_slli_epi16(t, 4));
        }
#if SEEDED_TAIL
        {
            __m256i t0 = _mm256_setzero_si256();
            for (size_t r = 0; r < V_MAX; r++) {
                const __m256i xx = _mm256_broadcastq_epi64(_mm_loadu_si64((const void *) (P2 + (r * O_MAX + c) * M_VEC_LIMBS_MAX + M_VEC_LIMBS_MAX - 1)));
                const __m256i in = _mm256_srlv_epi64(xx, _mm256_set_epi64x(4, 0, 4, 0)) & lo;
                t0 ^= _mm256_shuffle_epi8(_mm256_broadcastsi128_si256(x_multabs[r]), in);
            }
            const __m256i bq = _mm256_unpackhi_epi64(t0, t0);
            const __m256i t = (bq ^ _mm256_srli_epi16(t0, 4)) & lo;
            alignas(32) uint64_t col[4];
            _mm256_store_si256((__m256i *) col, t0 ^ _mm256_slli_epi16(t, 4));
            acc[c * M_VEC_LIMBS_MAX + M_VEC_LIMBS_MAX - 1] ^= col[0];
        }
#endif
    }
    mayo_secure_clear(x_multabs, sizeof(x_multabs));
}

#endif
