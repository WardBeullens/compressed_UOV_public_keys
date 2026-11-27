// SPDX-License-Identifier: Apache-2.0

#include <mem.h>
#include <mayo.h>
#include <randombytes.h>
#include <aes_ctr.h>
#include <arithmetic.h>
#include <simple_arithmetic.h>
#include <fips202.h>
#include <stdlib.h>
#include <string.h>
#include <stdalign.h>
#ifdef ENABLE_CT_TESTING
#include <valgrind/memcheck.h>
#endif

#define PK_PRF AES_128_CTR

static void decode(const unsigned char *m, unsigned char *mdec, int mdeclen) {
    int i;
    for (i = 0; i < mdeclen / 2; ++i) {
        *mdec++ = m[i] & 0xf;
        *mdec++ = m[i] >> 4;
    }

    if (mdeclen % 2 == 1) {
        *mdec++ = m[i] & 0x0f;
    }
}

static void encode(const unsigned char *m, unsigned char *menc, int mlen) {
    int i;
    for (i = 0; i < mlen / 2; ++i, m += 2) {
        menc[i] = (*m) | (*(m + 1) << 4);
    }

    if (mlen % 2 == 1) {
        menc[i] = (*m);
    }
}

#ifdef TARGET_BIG_ENDIAN
// convert n limbs between their in-memory byte order and the little-endian one
static void bswap_limbs(uint64_t *limbs, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        limbs[i] = BSWAP64(limbs[i]);
    }
}
#endif

/* y = t - (the whipped sum of vPv), with t nibble-packed as SHAKE outputs it: element 2j
   in the low and 2j+1 in the high nibble of byte j, the layout temp's bytes have too. */
static void compute_rhs(const mayo_params_t *p, const uint64_t *vPv, const unsigned char *t, unsigned char *y){
    (void) p;

    const size_t top_pos = ((PARAM_m(p) - 1) % 16) * 4;
    const size_t m_vec_limbs = PARAM_m_vec_limbs(p);

    uint64_t temp[M_VEC_LIMBS_MAX] = {0};
    unsigned char *temp_bytes = (unsigned char *) temp;
    uint64_t ftail = 0; // tap jj in nibble jj of limb 0
    for (int jj = 0; jj < F_TAIL_LEN; jj++) {
        ftail |= (uint64_t) PARAM_f_tail(p)[jj] << (4 * jj);
    }
    // Horner in z over the whipping schedule: visiting the labels from the
    // highest exponent down leaves the term of label l multiplied by z^l
    const unsigned char *whip_positions = PARAM_whip_pos(p);
    for (int l = PARAM_k(p) * (PARAM_k(p) + 1) / 2 - 1; l >= 0; l--) {
        const int row = whip_positions[2*l], col = whip_positions[2*l + 1];
        // multiply by X (shift up 4 bits)
        unsigned char top = (temp[m_vec_limbs-1] >> top_pos) % 16;
        temp[m_vec_limbs-1] <<= 4;
        for(int k = m_vec_limbs - 2; k>=0; k--){
            temp[k+1] ^= temp[k] >> 60;
            temp[k] <<= 4;
        }
        // reduce mod f(X): all taps in one nibble-sliced multiply
        vec_mul_add_u64(1, &ftail, top, temp);

        // extract from vPv and add; row and col are public
        for(size_t k=0; k < m_vec_limbs; k ++){
            temp[k] ^= vPv[( row * PARAM_k(p) + col )* m_vec_limbs + k];
        }
        if (row != col) {
            for(size_t k=0; k < m_vec_limbs; k ++){
                temp[k] ^= vPv[( col * PARAM_k(p) + row )* m_vec_limbs + k];
            }
        }
    }

    // compute y
    for (int i = 0; i < PARAM_m(p); i+=2)
    {
        const unsigned char b = t[i/2] ^ temp_bytes[MAYO_LIMB_BYTE(i/2)];
        y[i]   = b & 0xF;
        y[i+1] = b >> 4;
    }
    mayo_secure_clear(temp, sizeof(temp));   /* from vPv: secret when signing */
}


#define MAYO_M_OVER_8 ((M_MAX + 7) / 8)
static void compute_A(const mayo_params_t *p, uint64_t *VtL, unsigned char *A_out) {
    (void) p;

    const int m_vec_limbs = PARAM_m_vec_limbs(p);
    uint64_t A[(((O_MAX*K_MAX+15)/16)*16)*MAYO_M_OVER_8] = {0};
    size_t A_width = ((PARAM_o(p)*PARAM_k(p) + 15)/16)*16;

    /* zero the unused nibbles above m in each m_vec: the vector backends leave garbage
       there, and the rows it would be shifted into are folded into the valid rows by
       the f_tail reduction below */
    if(PARAM_m(p) % 16 != 0){
        uint64_t mask = 1;
        mask <<= (PARAM_m(p) % 16)*4;
        mask -= 1;
        for (int i = 0; i < PARAM_o(p)*PARAM_k(p); i++)
        {
            VtL[i*m_vec_limbs + m_vec_limbs - 1] &= mask;
        }
    }

    /* the whipping schedule puts the pair of label l at row offset l, i.e. a shift
       of l nibbles: l/16 whole limb-rows plus 4*(l%16) bits within a limb */
    /* label l adds M_col at columns o*row and (off the diagonal) M_row at o*col, so
       collect each block's uses and hand them over together */
    const unsigned char *whip_positions = PARAM_whip_pos(p);
    int use_row0[K_MAX][K_MAX], use_col[K_MAX][K_MAX], use_bits[K_MAX][K_MAX], nuse[K_MAX] = {0};
    for (int l = 0; l < PARAM_k(p) * (PARAM_k(p) + 1) / 2; l++) {
        const int row = whip_positions[2*l], col = whip_positions[2*l + 1];
        for (int e = 0; e < (row != col ? 2 : 1); e++) {
            const int j = e ? row : col, at = e ? col : row;
            use_row0[j][nuse[j]] = l / 16;
            use_col[j][nuse[j]] = PARAM_o(p) * at;
            use_bits[j][nuse[j]++] = 4 * (l % 16);
        }
    }
    for (int j = 0; j < PARAM_k(p); j++) {
        m_compute_A_block(A, A_width, VtL + (size_t) j * m_vec_limbs * PARAM_o(p), PARAM_o(p),
                          m_vec_limbs, nuse[j], use_row0[j], use_col[j], use_bits[j]);
    }

    for (size_t c = 0; c < A_width*((PARAM_m(p) + (PARAM_k(p)+1)*PARAM_k(p)/2 +15)/16) ; c+= 16)
    {
        transpose_16x16_nibbles(A + c);
    }

    /* the tail rows reduce into the rows below them, one GF(16) multiply per tap. A's
       rows sit sixteen to a block, so copy the sources out and the destinations back, in
       whole blocks, and let each tap run over contiguous words. */
    #define FTAIL_ROWS ((K_MAX + 1) * K_MAX / 2)
    #define FT_BLOCKS(n) (((n) + 15) / 16 * 16)
    uint64_t ft_src[FT_BLOCKS(FTAIL_ROWS + 15)];        // from row m rounded down to a block
    uint64_t ft_dst[FT_BLOCKS(FTAIL_ROWS + F_TAIL_LEN)];
    const int ftail_rows = (PARAM_k(p) + 1) * PARAM_k(p) / 2;
    const int src_off = PARAM_m(p) % 16, src_block = PARAM_m(p) / 16;
    const int src_words = FT_BLOCKS(src_off + ftail_rows);
    const int dst_words = FT_BLOCKS(ftail_rows + F_TAIL_LEN - 1); // the extra rows get zeros
    /* only the table-driven backends read this; gfni folds its matrix from the
       coefficient, and building it there costs more than the loop it feeds */
    unsigned char ftab[F_TAIL_LEN][16] = {{0}};
#ifdef MAYO_GF16_MULC_NEEDS_TABLE
    for (int t = 0; t < F_TAIL_LEN; t++) {
        for (int j = 0; j < 16; j++) {
            ftab[t][j] = mul_f(PARAM_f_tail(p)[t], (unsigned char) j);
        }
    }
#endif
    for (size_t c = 0; c < A_width; c+= 16)
    {
        for (int b = 0; b < src_words; b += 16) {
            memcpy(ft_src + b, A + (size_t) (src_block + b / 16) * A_width + c, 16 * sizeof(uint64_t));
        }
        memset(ft_dst, 0, dst_words * sizeof(uint64_t));
        for (int t = 0; t < F_TAIL_LEN; t++) {
            if (PARAM_f_tail(p)[t] != 0) { // public: a zero tap adds nothing
                m_gf16_mulc_xor(ft_dst + t, ft_src + src_off, ftail_rows, PARAM_f_tail(p)[t], ftab[t]);
            }
        }
        for (int b = 0; b < dst_words; b += 16) {
            for (int i = 0; i < 16; i++) {
                A[(size_t) (b / 16) * A_width + c + i] ^= ft_dst[b + i];
            }
        }
    }
    mayo_secure_clear(ft_src, sizeof(ft_src));
    mayo_secure_clear(ft_dst, sizeof(ft_dst));
    #undef FTAIL_ROWS
    #undef FT_BLOCKS

#ifdef TARGET_BIG_ENDIAN
    bswap_limbs(A, A_width * MAYO_M_OVER_8);
#endif


    /* A is blocked sixteen rows to a chunk; walk the rows and pick the chunk */
    for (int row = 0; row < PARAM_m(p); row++)
    {
        const size_t base = (size_t) (row & ~15) * A_width / 16 + (size_t) (row & 15);
        int c = 0;
        for (; c + 16 <= PARAM_A_cols(p)-1; c += 16)
        {
            decode16( (unsigned char *) &A[base + c], A_out + PARAM_A_cols(p)*row + c);
        }
        if (c < PARAM_A_cols(p)-1)
        {
            decode( (unsigned char *) &A[base + c], A_out + PARAM_A_cols(p)*row + c, PARAM_A_cols(p)-1-c);
        }
    }
    mayo_secure_clear(A, sizeof(A));   /* M, shifted: secret */
}

static void unpack_m_vecs(const unsigned char *in, uint64_t *out, int vecs, int m){
    const int m_vec_limbs = (m + 15) / 16;
    const size_t padded = m_vec_limbs * sizeof(uint64_t);
    unsigned char *_out = (unsigned char *) out;
    // back to front: in and out may be the same buffer, and out is the wider layout
    for (int i = vecs-1; i >= 0; i--)
    {
        memmove(_out + i*padded, in + i*m/2, m/2);
        memset(_out + i*padded + m/2, 0, padded - m/2);
    }
}
/* ------------------------------------------------------------------ *
 * lambda: the hash-derived linear term                                *
 * ------------------------------------------------------------------ *
 * The verification equation is P*(s) + L(s) = t with
 *
 *     L(s) = sum_i E_i0 * Lambda(s_i),   Lambda(x) = sum_r x_r lambda_r,
 *
 * so the whole term is n elements of R, that is n*m/2 bytes. It is derived from
 * MAYO_LAMBDA_SEED_BYTES taken from the same SHAKE stream that produces t: 16
 * bytes of AES-128 key and 16 bytes of IV, so all 256 bits feed
 * every byte of the keystream.
 *
 * Block j of the keystream is AES-128(key, IV xor j), with the 32-bit block index
 * written big-endian into the last four bytes of the IV. It is built on ECB rather
 * than the native CTR because the CTR entry points take no counter argument, and
 * the portable and AES-NI implementations lay out nonce and counter differently --
 * driving ECB directly keeps lambda identical on every backend. */
#define MAYO_LAMBDA_SEED_BYTES 32

static void derive_lambda(const mayo_params_t *p, const unsigned char *seed,
                          uint64_t *lambda) {
    (void) p;
    const int param_n = PARAM_n(p);
    const int param_m = PARAM_m(p);
    const size_t nbytes = (size_t) param_n * (size_t) param_m / 2;
    unsigned char *ks = (unsigned char *) lambda;   /* expanded in place, then unpacked */

    /* whole blocks straight into lambda: the at most 15 bytes past nbytes land in its
       MAYO_MVEC_SLACK tail, which the unpack rewrites and MAYO_ZERO_SLACK clears */
    unsigned char in[16 * 256];
    const unsigned char *iv = seed + 16;
    for (size_t b = 0; b < 256; b++) {   /* the IV in every block; the counters vary below */
        memcpy(in + 16 * b, iv, 16);
    }
    uint32_t iv_ctr, ctr = 0;
    memcpy(&iv_ctr, iv + 12, 4);
    for (size_t done = 0; done < nbytes; ) {
        size_t nb = (nbytes - done + 15) / 16;
        if (nb > 256) {
            nb = 256;
        }
        for (size_t b = 0; b < nb; b++, ctr++) {   /* bytes 12..15: iv ^ the big-endian counter */
#ifdef TARGET_BIG_ENDIAN
            const uint32_t c = iv_ctr ^ ctr;
#else
            const uint32_t c = iv_ctr ^ BSWAP32(ctr);
#endif
            memcpy(in + 16 * b + 12, &c, 4);
        }
        AES_128_ECB(ks + done, in, nb, seed);
        done += 16 * nb;
    }
    unpack_m_vecs(ks, lambda, param_n, param_m);
#ifdef TARGET_BIG_ENDIAN
    bswap_limbs(lambda, (size_t) param_n * PARAM_m_vec_limbs(p));
#endif
    MAYO_ZERO_SLACK(lambda, (size_t) param_n * PARAM_m_vec_limbs(p));
}

#ifndef MAYO_SEEDED_P3
static void pack_m_vecs(const uint64_t *in, unsigned char *out, int vecs, int m){
    const int m_vec_limbs = (m + 15) / 16;
    unsigned char *_in = (unsigned char *) in;
    for (int i = 0; i < vecs; i++)
    {
        memmove(out + (i*m/2), _in + i*m_vec_limbs*sizeof(uint64_t), m/2);
    }
}
#endif
#ifdef MAYO_SEEDED_P3
static int seeded_in_col(const mayo_params_t *p, int j) {
    return MAYO_MIN(j * PARAM_m(p), PARAM_v(p));
}

// where column j's targets start in the target stream: columns 1..o-1 in order
static size_t seeded_offset(const mayo_params_t *p, int j) {
    size_t off = 0;
    for (int c = 1; c < j; c++) {
        off += seeded_in_col(p, c);
    }
    return off;
}

// the P3 targets as a nibble stream, plus 16 bytes so whole words can be read
#define P3_TARGET_BYTES (((O_MAX - 1) * V_MAX + 1) / 2 + 16)
#define P3_TARGET_LIMBS_MAX ((P3_TARGET_BYTES + 7) / 8)
static size_t target_bytes(const mayo_params_t *p) {
    return (seeded_offset(p, PARAM_o(p)) + 1) / 2;
}
#endif

/* Expand P1 and P2 into P. On the vector backends P1 stays in the packed keystream
   layout (its readers use PINV) and P2 is spread out, unless keep_p2_packed. With
   seeded P3 and T != NULL the keystream continues past P2 with the P3 targets, which
   are copied to T (P3_TARGET_BYTES) before P2 is spread over them; P then needs
   P3_TARGET_LIMBS_MAX more limbs. */
static void expand_P1_P2(const mayo_params_t *p, uint64_t *P, const unsigned char *seed_pk, int keep_p2_packed,
                         unsigned char *T) {
    (void) p;
#ifdef MAYO_SEEDED_P3
    const size_t tb = T ? target_bytes(p) : 0;
    PK_PRF((unsigned char *)P, PARAM_P1_bytes(p) + PARAM_P2_bytes(p) + tb, seed_pk, PARAM_pk_seed_bytes(p));
    if (T) {
        memset(T, 0, P3_TARGET_BYTES);
        memcpy(T, (unsigned char *)P + PARAM_P1_bytes(p) + PARAM_P2_bytes(p), tb);
    }
#else
    (void) T;
    PK_PRF((unsigned char *)P, PARAM_P1_bytes(p) + PARAM_P2_bytes(p), seed_pk, PARAM_pk_seed_bytes(p));
#endif
    // m % 16 == 0: packed stride == padded stride, the keystream already is the layout.
    if (PARAM_m(p) % 16 == 0) {
        return;
    }
#ifdef MAYO_PACKED_P1P2
    if (!keep_p2_packed) { // P1 stays packed; spread only P2 into its padded slot
        unpack_m_vecs((unsigned char *)P + PARAM_P1_bytes(p), P + PARAM_P1_limbs(p),
                      PARAM_P2_limbs(p) / PARAM_m_vec_limbs(p), PARAM_m(p));
    }
#else
    (void) keep_p2_packed;
    unpack_m_vecs((unsigned char *)P, P, (PARAM_P1_limbs(p) + PARAM_P2_limbs(p)) / PARAM_m_vec_limbs(p), PARAM_m(p));
#endif
}

// eval = t - P*(s), with t nibble-packed (see compute_rhs)
static void eval_public_map(const mayo_params_t *p, const unsigned char *s, const uint64_t *P1, const uint64_t *P2, const uint64_t *P3, const uint64_t *lambda, const unsigned char *t, unsigned char *eval){
    alignas(64) uint64_t SPS[K_MAX * K_MAX * M_VEC_LIMBS_MAX + MAYO_MVEC_SLACK];
    MAYO_ZERO_SLACK(SPS, PARAM_k(p) * PARAM_k(p) * PARAM_m_vec_limbs(p));
    MAYO_ACC_INIT(SPS); // the products below accumulate into it
    /* combine computing PS and SPS in a single step. lambda rides along inside:
       the backend xors it into column 0 of PS, and the S^t * PS half of this product
       then turns it into L(s) = sum_i E_i0 * Lambda(s_i) at SPS[i][0], which
       compute_rhs weights by E_i0 along with everything else. */
    m_calculate_PS_SPS(p, P1, P2, P3, s, lambda, SPS);

    compute_rhs(p, SPS, t, eval);
}

#ifdef MAYO_SEEDED_P3
/* Seed-derived entries of P3. Entry (i, k) of column j > i of Upper(P3), k < m, is
       L[:,i]^(k) . o_j + (O^t P2)[i][j]^(k),   L = (P1 + P1^t) O + P2,
   with o_j column j of O: linear in o_j once columns < j are fixed. Keygen makes the
   first min(j*m, v) of these, i-major, equal to targets expanded from seed_pk, which
   costs v unknowns per column at most, and the public key leaves them out. The targets
   are the PK_PRF keystream right after P2, see expand_P1_P2. */

// the seeded nibbles of P3_upper's entry (r, c) are its nibbles below this
static int seeded_prefix(const mayo_params_t *p, int r, int c) {
    return MAYO_MAX(0, MAYO_MIN(PARAM_m(p), seeded_in_col(p, c) - r * PARAM_m(p)));
}

static unsigned char mvec_nibble(const uint64_t *vec, int k) {
    return (vec[k / 16] >> (4 * (k % 16))) & 0xf;
}

static unsigned char stream_nibble(const unsigned char *buf, size_t s) {
    return (buf[s / 2] >> (4 * (s % 2))) & 0xf;
}

/* Overwrite oil columns 1..o-1 so the seeded entries come out equal to their targets T;
   column 0 stays as sampled. Row t = (i, k) of the system for column j is L[:,i]^(k),
   the same for every j > i, so the system for column j consists of the first n_j rows of
   one v x v matrix M whose row blocks arrive one per column. W is M, augmented with one
   right-hand side per later column.

   If n_1 = v (m >= v), all columns solve the same square system, and W is eliminated
   once. Otherwise m < v <= 2m, and M = [A; N] has two row blocks: A, from column 0, and
   the first v - m rows N from column 1. Column 1 must be a uniformly random solution of
   A x = b. Gauss-Jordan elimination of [A^T | I] with the pivots on the diagonal gives
   E with E A^T = [I; 0], so x = E^T (b, z) solves A x = b for every z, and is uniformly
   random for z uniformly random; it fails exactly when A does not have full row rank.
   For the later columns, with x = x0 + K y, x0 = E^T (b, 0) and K = E^T (0, I), the
   system M x = (b, c) becomes the (v - m) x (v - m) system (N K) y = c - N x0, which is
   singular exactly when M is.
   Neither needs E itself: with U the span of the identity parts of the pivot rows,
   x0 = E^T (b, 0) is the one x in U with A x = b, and column s of K the one x in
   e_(m+s) + U with A x = 0. The elimination finds x0 as the identity part of a row
   [b^T | 0] appended to [A^T | I] and reduced along with the rows of A^T, and K as the
   identity parts of the last v - m rows, so the rows above a pivot need not be reduced
   (see seeded_tsolve). Column 1 is x0 + K z.

   P2 is spread, P1 packed. Returns 0 if a system does not have full rank, in which case
   keygen picks another seed. */
#if defined(MAYO_VARIANT) && V_MAX > 2 * M_MAX
#error "match_P3_targets assumes v <= 2m"
#endif
#define SEEDED_F_MAX (V_MAX > M_MAX ? V_MAX - M_MAX : 1)  // free dimensions of column 1
#define SEEDED_T_ROWS (V_MAX > M_MAX ? V_MAX + O_MAX - 1 : 1)
#define SEEDED_V_BYTES (32 * SEEDED_V_REGS)
static int match_P3_targets(const mayo_params_t *p, const uint64_t *P1, const uint64_t *P2,
                            const unsigned char *T, unsigned char *O) {
    const int m = PARAM_m(p), v = PARAM_v(p), o = PARAM_o(p), mvl = PARAM_m_vec_limbs(p);
    // row (i, k): L[:,i]^(k), then a right-hand side for each column j = 1..o-1
    alignas(32) unsigned char W[V_MAX * SEEDED_STRIDE] = {0};
    alignas(32) uint64_t L[V_MAX * M_VEC_LIMBS_MAX + MAYO_MVEC_SLACK];
    uint64_t OtP2[O_MAX * M_VEC_LIMBS_MAX + MAYO_MVEC_SLACK];  // row i of O^t P2
    unsigned char x[V_MAX];
    unsigned char b[M_MAX];  // the nibbles of a row of O^t P2
    size_t off[O_MAX];
    unsigned char ok = 0xff;
    // the two-block case
    alignas(32) unsigned char At[SEEDED_T_ROWS * SEEDED_T_STRIDE];  // [A^T | I], then [b_c^T | 0] per c
    alignas(32) unsigned char K[SEEDED_F_MAX][SEEDED_V_BYTES];      // the columns of K
    alignas(32) unsigned char X[O_MAX][SEEDED_V_BYTES];             // x0, then x, per column
    alignas(32) unsigned char S[SEEDED_F_MAX * SEEDED_STRIDE];      // [N K | c - N x0]

    for (int c = 1; c < o; c++) {
        off[c] = seeded_offset(p, c);
    }

    for (int j = 1; j < o; j++) {
        const int i = j - 1, n = seeded_in_col(p, j), r0 = seeded_in_col(p, i);

        // the rows (i, k) that column j brings in: L[:,i]^(k) and (O^t P2)[i][c]^(k), c > i
        for (int r = 0; r < v; r++) {
            x[r] = O[r * o + i];
        }
        memset(L, 0, sizeof(L));
        memset(OtP2, 0, sizeof(OtP2));
        P1P1t_times_vec(P1, x, L);
        xt_times_P2(P2, x, OtP2);
        for (int r = 0; r < v; r++) {
            m_vec_add(mvl, P2 + (size_t) (r * o + i) * mvl, L + (size_t) r * mvl);
        }
        seeded_rows_from_L(L, W + (size_t) r0 * SEEDED_STRIDE, v, mvl, r0 - i * m, n - i * m);
        for (int c = j; c < o; c++) {
            seeded_spread_mvec(OtP2 + (size_t) c * mvl, m, b);
            for (int t = r0; t < n; t++) {
                W[t * SEEDED_STRIDE + v + c - 1] = b[t - i * m] ^ stream_nibble(T, off[c] + t);
            }
        }

        if (j == 1 && n == v) {
            // one block: columns 1..o-1 share the square system
            ok &= seeded_solve(W, v, 0, v);
            for (int c = 1; c < o; c++) {
                for (int r = 0; r < v; r++) {
                    O[r * o + c] = W[r * SEEDED_STRIDE + v + c - 1];
                }
            }
            break;
        }

        const int f = v - m;
        if (j == 1) {
            // [A^T | I]: row r is L[r]'s m nibbles, then e_r; then [b_c^T | 0], c = 1..o-1
            memset(At, 0, sizeof(At));
            for (int r = 0; r < v; r++) {
                seeded_spread_mvec(L + (size_t) r * mvl, m, At + r * SEEDED_T_STRIDE);
                At[r * SEEDED_T_STRIDE + m + r] = 1;
            }
            for (int c = 1; c < o; c++) {
                for (int t = 0; t < m; t++) {
                    At[(v + c - 1) * SEEDED_T_STRIDE + t] = W[t * SEEDED_STRIDE + v + c - 1];
                }
            }
            ok &= seeded_tsolve(At, v, v + o - 1, m);
            for (int s = 0; s < f; s++) {
                memset(K[s], 0, SEEDED_V_BYTES);
                memcpy(K[s], At + (m + s) * SEEDED_T_STRIDE + m, (size_t) v);
            }
            for (int c = 1; c < o; c++) {
                memset(X[c], 0, SEEDED_V_BYTES);
                memcpy(X[c], At + (v + c - 1) * SEEDED_T_STRIDE + m, (size_t) v);
            }
            // column 1: x0 + K z, z the last v - m sampled entries of column 1
            for (int s = 0; s < f; s++) {
                seeded_axpy(X[1], K[s], O[(m + s) * o + 1]);
            }
            for (int r = 0; r < v; r++) {
                O[r * o + 1] = X[1][r];
            }
            continue;
        }

        // j == 2: rows m..v-1 of W are N; columns 2..o-1 through (N K) y = c - N x0
        memset(S, 0, sizeof(S));
        for (int s = 0; s < f; s++) {
            const unsigned char *row = W + (m + s) * SEEDED_STRIDE;
            for (int s2 = 0; s2 < f; s2++) {
                S[s * SEEDED_STRIDE + s2] = seeded_dot(row, K[s2]);
            }
            for (int c = 2; c < o; c++) {
                S[s * SEEDED_STRIDE + f + c - 2] = row[v + c - 1] ^ seeded_dot(row, X[c]);
            }
        }
        ok &= seeded_solve(S, f, 0, f);
        for (int c = 2; c < o; c++) {
            for (int s = 0; s < f; s++) {
                seeded_axpy(X[c], K[s], S[s * SEEDED_STRIDE + f + c - 2]);
            }
            for (int r = 0; r < v; r++) {
                O[r * o + c] = X[c][r];
            }
        }
        break;
    }

    mayo_secure_clear(W, sizeof(W));
    mayo_secure_clear(L, sizeof(L));
    mayo_secure_clear(OtP2, sizeof(OtP2));
    mayo_secure_clear(x, sizeof(x));
    mayo_secure_clear(b, sizeof(b));
    mayo_secure_clear(At, sizeof(At));
    mayo_secure_clear(K, sizeof(K));
    mayo_secure_clear(X, sizeof(X));
    mayo_secure_clear(S, sizeof(S));
#ifdef ENABLE_CT_TESTING
    VALGRIND_MAKE_MEM_DEFINED(&ok, 1); // whether to pick another seed is public
#endif
    return ok == 0xff;
}

// Upper(P3) without the seeded nibbles, as one nibble stream
static void pack_P3(const mayo_params_t *p, const uint64_t *P3_upper, unsigned char *out) {
    const int m = PARAM_m(p), o = PARAM_o(p), mvl = PARAM_m_vec_limbs(p);
    size_t w = 0;
    for (int r = 0; r < o; r++) {
        for (int c = r; c < o; c++, P3_upper += mvl) {
            for (int k = seeded_prefix(p, r, c); k < m; k++, w++) {
                const unsigned char nib = mvec_nibble(P3_upper, k);
                out[w / 2] = (w % 2) ? (unsigned char) (out[w / 2] | nib << 4) : nib;
            }
        }
    }
}

// 16 nibbles of a nibble stream from nibble s on (little-endian only, like all of match_P3_targets)
static uint64_t nibbles_at(const unsigned char *buf, size_t s) {
    uint64_t w;
    memcpy(&w, buf + s / 2, 8);
    return (s % 2) ? (w >> 4) | ((uint64_t) buf[s / 2 + 8] << 60) : w;
}

// inverse of pack_P3, with the seeded nibbles taken from the targets T; P3_upper is m-vec strided
static void unpack_P3(const mayo_params_t *p, const unsigned char *in, const unsigned char *T, uint64_t *P3_upper) {
    const int m = PARAM_m(p), o = PARAM_o(p), mvl = PARAM_m_vec_limbs(p);
    const size_t in_bytes = PARAM_cpk_bytes(p) - PARAM_pk_seed_bytes(p);
    unsigned char buf[CPK_BYTES_MAX + 16] = {0}; // padded, so whole words can be read
    memcpy(buf, in, in_bytes);
    size_t w = 0;
    for (int r = 0; r < o; r++) {
        for (int c = r; c < o; c++, P3_upper += mvl) {
            const int pp = seeded_prefix(p, r, c);
            const size_t s = seeded_offset(p, c) + (size_t) r * m; // nibble 0 of the entry's targets
            for (int l = 0; l < mvl; l++) {
                // nibbles 16l .. 16l+15 of the entry, the first pp of which are targets
                const int t = 16 * l - pp;
                uint64_t limb = (t >= 0) ? nibbles_at(buf, w + t)
                              : (t > -16) ? nibbles_at(buf, w) << (-4 * t) : 0;
                if (t < 0) {
                    const uint64_t tmask = (t > -16) ? ((uint64_t) 1 << (-4 * t)) - 1 : ~(uint64_t) 0;
                    limb |= nibbles_at(T, s + 16 * l) & tmask;
                }
                if (m - 16 * l < 16) {
                    limb &= ((uint64_t) 1 << (4 * (m - 16 * l))) - 1;
                }
                P3_upper[l] = limb;
            }
            w += m - pp;
        }
    }
}
#endif

// Public API

int mayo_keypair(const mayo_params_t *p, unsigned char *pk, unsigned char *sk) {
    return mayo_keypair_compact(p, pk, sk);
}

static int expand_sk_internal(const mayo_params_t *p, const unsigned char *csk,
                              sk_t *sk, int materialize_l) {
    int ret = MAYO_OK;
    unsigned char S[PK_SEED_BYTES_MAX + O_BYTES_MAX];
    uint64_t *P = sk->p;
    unsigned char *O = sk->O;

    const int param_o = PARAM_o(p);
    const int param_v = PARAM_v(p);
    const int param_O_bytes = PARAM_O_bytes(p);
    const int param_pk_seed_bytes = PARAM_pk_seed_bytes(p);
    const int param_sk_seed_bytes = PARAM_sk_seed_bytes(p);

    const unsigned char *seed_sk = csk;
    unsigned char *seed_pk = S;

    shake256(S, param_pk_seed_bytes + param_O_bytes, seed_sk,
             param_sk_seed_bytes);
#ifdef MAYO_SEEDED_P3
    // O is not the one seed_sk derives but the one keygen solved for, stored after seed_sk
    decode(csk + param_sk_seed_bytes, O, param_v * param_o);
#else
    decode(S + param_pk_seed_bytes, O, param_v * param_o);
#endif

    // keep the seed in the expanded key so Sign(esk) can derive the salt
    memcpy(sk->sk_seed, seed_sk, param_sk_seed_bytes);

#ifdef ENABLE_CT_TESTING
    VALGRIND_MAKE_MEM_DEFINED(seed_pk, param_pk_seed_bytes);
#endif

    expand_P1_P2(p, P, seed_pk, 0, NULL);
    MAYO_ZERO_SLACK(P, PARAM_P1_limbs(p) + PARAM_P2_limbs(p));
    if (materialize_l) {
        uint64_t *P2 = P + PARAM_P1_limbs(p);
        uint64_t *P1 = P;

        // compute L_i = (P1 + P1^t) * O + P2, on native-order limbs
#ifdef TARGET_BIG_ENDIAN
        bswap_limbs(P, PARAM_P1_limbs(p) + PARAM_P2_limbs(p));
#endif
        uint64_t *L = P2;
        P1P1t_times_O(p, P1, O, L);
#ifdef TARGET_BIG_ENDIAN
        bswap_limbs(P, PARAM_P1_limbs(p) + PARAM_P2_limbs(p));
#endif
    }
#ifdef MAYO_SIGN_HALF_L
    else { // the fused sign keeps L implicit, from L' = P1 * O + P2
        compute_L_half(P, P + PARAM_P1_limbs(p), O);
    }
#endif

    mayo_secure_clear(S, PK_SEED_BYTES_MAX + O_BYTES_MAX);
    return ret;
}

int mayo_expand_sk(const mayo_params_t *p, const unsigned char *csk,
                   sk_t *sk) {
    // L = (P1 + P1^t) O + P2 goes into the expanded key; Sign(esk) reads it back
    return expand_sk_internal(p, csk, sk, 1);
}

static int sign_signature_internal(const mayo_params_t *p, unsigned char *sig,
              size_t *siglen, const unsigned char *m,
              size_t mlen, const sk_t *sk, int use_l) {
    int ret = MAYO_OK;
    unsigned char tenc[M_BYTES_MAX + MAYO_LAMBDA_SEED_BYTES]; // t, nibble-packed, then the lambda seed; no secret data
    alignas(32) uint64_t lambda[N_MAX * M_VEC_LIMBS_MAX + MAYO_MVEC_SLACK];
    unsigned char y[M_MAX];                    // secret data
    unsigned char salt[SALT_BYTES_MAX];        // not secret data
    unsigned char V[K_MAX * V_BYTES_MAX + R_BYTES_MAX], Vdec[V_MAX * K_MAX];                 // secret data
    unsigned char A[((M_MAX+7)/8*8) * (K_MAX * O_MAX + 1) + 32] = { 0 };   // secret data
    unsigned char x[K_MAX * N_MAX];                       // secret until it is the solution
    unsigned char r[K_MAX * O_MAX + 1] = { 0 };           // secret data
    unsigned char s[K_MAX * N_MAX];                       // not secret data
    const unsigned char *seed_sk;
    unsigned char tmp[DIGEST_BYTES_MAX + SALT_BYTES_MAX + SK_SEED_BYTES_MAX + 1];
    unsigned char *ctrbyte;
    int sol_found = 0;

    const int param_n = PARAM_n(p);
    const int param_o = PARAM_o(p);
    const int param_k = PARAM_k(p);
    const int param_v = PARAM_v(p);
    const int param_m_bytes = PARAM_m_bytes(p);
    const int param_v_bytes = PARAM_v_bytes(p);
    const int param_r_bytes = PARAM_r_bytes(p);
    const int param_sig_bytes = PARAM_sig_bytes(p);
    const int param_digest_bytes = PARAM_digest_bytes(p);
    const int param_sk_seed_bytes = PARAM_sk_seed_bytes(p);
    const int param_salt_bytes = PARAM_salt_bytes(p);

    seed_sk = sk->sk_seed;

    // hash message
    shake256(tmp, param_digest_bytes, m, mlen);

    // sk is read-only on little-endian; on big-endian the swaps below convert it in
    // place and the err: block converts it back, leaving the caller's key unchanged.
    uint64_t *P1 = (uint64_t *) sk->p;
    uint64_t *P2_or_L = P1 + PARAM_P1_limbs(p);
    uint64_t Mtmp[K_MAX * O_MAX * M_VEC_LIMBS_MAX + MAYO_MVEC_SLACK] = {0};

#ifdef TARGET_BIG_ENDIAN
    bswap_limbs(P1, PARAM_P1_limbs(p) + PARAM_P2_limbs(p)); // P1 and P2_or_L are adjacent
#endif

    // choose the randomizer
    #if defined(PQM4) || defined(HAVE_RANDOMBYTES_NORETVAL)
    randombytes(tmp + param_digest_bytes, param_salt_bytes);
    #else
    if (randombytes(tmp + param_digest_bytes, param_salt_bytes) != MAYO_OK) {
        ret = MAYO_ERR;
        goto err;
    }
    #endif

    // hashing to salt
    memcpy(tmp + param_digest_bytes + param_salt_bytes, seed_sk,
           param_sk_seed_bytes);
    shake256(salt, param_salt_bytes, tmp,
             param_digest_bytes + param_salt_bytes + param_sk_seed_bytes);

#ifdef ENABLE_CT_TESTING
    VALGRIND_MAKE_MEM_DEFINED(salt, SALT_BYTES_MAX); // Salt is not secret
#endif

    // hashing to t
    memcpy(tmp + param_digest_bytes, salt, param_salt_bytes);
    ctrbyte = tmp + param_digest_bytes + param_salt_bytes + param_sk_seed_bytes;

    shake256(tenc, param_m_bytes + MAYO_LAMBDA_SEED_BYTES, tmp,
             param_digest_bytes + param_salt_bytes);

    derive_lambda(p, tenc + param_m_bytes, lambda);

    for (int ctr = 0; ctr <= 255; ++ctr) {
        *ctrbyte = (unsigned char)ctr;

        shake256(V, param_k * param_v_bytes + param_r_bytes, tmp,
                 param_digest_bytes + param_salt_bytes + param_sk_seed_bytes + 1);

        // decode the v_i vectors
        for (int i = 0; i < param_k; ++i) {
            decode(V + i * param_v_bytes, Vdec + i * param_v, param_v);
        }

        // compute M_i matrices and all v_i*P1*v_j, with lambda folded in
        m_calculate_M_and_VPV(p, P1, P2_or_L, sk->O, Vdec, lambda, use_l,
                              Mtmp, (uint64_t *) A);

        compute_rhs(p, (uint64_t*) A, tenc, y);
        compute_A(p, Mtmp, A);

        decode(V + param_k * param_v_bytes, r, param_k * param_o);

        if (sample_solution(p, A, y, r, x)) {
            sol_found = 1;
            break;
        } else {
            memset(Mtmp, 0, sizeof(Mtmp));
            memset(A, 0, sizeof(A));
        }
    }
    if (!sol_found) {
        ret = MAYO_ERR;
        goto err;
    }

    m_compute_s(p, sk->O, Vdec, x, s);
    encode(s, sig, param_n * param_k);

    memcpy(sig + param_sig_bytes - param_salt_bytes, salt, param_salt_bytes);
    *siglen = param_sig_bytes;

err:
#ifdef TARGET_BIG_ENDIAN
    // convert the caller's key back
    bswap_limbs(P1, PARAM_P1_limbs(p) + PARAM_P2_limbs(p));
#endif
    mayo_secure_clear(V, sizeof(V));
    mayo_secure_clear(Vdec, sizeof(Vdec));
    mayo_secure_clear(A, sizeof(A));
    mayo_secure_clear(r, sizeof(r));
    mayo_secure_clear(tmp, sizeof(tmp));
    mayo_secure_clear(Mtmp, sizeof(Mtmp));
    mayo_secure_clear(y, sizeof(y));
    mayo_secure_clear(x, sizeof(x));   /* r of the last attempt if none succeeded */
    return ret;
}

// Sign with a pre-expanded key. L is materialized in esk, matching mayo_expand_sk:
// the pair is the expand-once-sign-many path, so it pays for L once and reads it
// back on every signature. esk is unchanged on return, but big-endian converts it
// in place, so concurrent signers cannot share one.
int mayo_sign_signature_esk(const mayo_params_t *p, unsigned char *sig,
              size_t *siglen, const unsigned char *m,
              size_t mlen, const sk_t *esk) {
    return sign_signature_internal(p, sig, siglen, m, mlen, esk, 1);
}   

// Fused expandSK + Sign (NIST-style, compact key). All backends keep L implicit
// (expand without L, sign on the fly) for best single-shot latency.
int mayo_sign_signature(const mayo_params_t *p, unsigned char *sig,
              size_t *siglen, const unsigned char *m,
              size_t mlen, const unsigned char *csk) {
    const int use_l = 0; // all backends expand without L and sign implicitly
    alignas(32) sk_t sk;
    int ret = expand_sk_internal(p, csk, &sk, use_l);
    if (ret == MAYO_OK) {
        ret = sign_signature_internal(p, sig, siglen, m, mlen, &sk, use_l);
    }
    // sk.p holds P1 and P2, expanded from the public seed; secret are O, the seed and,
    // with MAYO_SIGN_HALF_L, the L' = P1 * O + P2 that replaced P2
#ifdef MAYO_SIGN_HALF_L
    mayo_secure_clear(sk.p + PARAM_P1_limbs(p), PARAM_P2_limbs(p) * sizeof(uint64_t));
#endif
    mayo_secure_clear(sk.O, sizeof(sk.O));
    mayo_secure_clear(sk.sk_seed, sizeof(sk.sk_seed));
    return ret;
}

/* Signed message sm = sig || m, signed with csk if it is given and with esk otherwise.
   m is moved behind the signature first, so it may overlap sm. */
static int sign_attached(const mayo_params_t *p, unsigned char *sm,
                         size_t *smlen, const unsigned char *m, size_t mlen,
                         const unsigned char *csk, const sk_t *esk) {
    const int param_sig_bytes = PARAM_sig_bytes(p);
    size_t siglen = param_sig_bytes;
    memmove(sm + param_sig_bytes, m, mlen);
    int ret = csk ? mayo_sign_signature(p, sm, &siglen, sm + param_sig_bytes, mlen, csk)
                  : mayo_sign_signature_esk(p, sm, &siglen, sm + param_sig_bytes, mlen, esk);
    if (ret != MAYO_OK || siglen != (size_t) param_sig_bytes) {
        memset(sm, 0, siglen + mlen);
        return ret;
    }
    *smlen = siglen + mlen;
    return ret;
}

int mayo_sign_esk(const mayo_params_t *p, unsigned char *sm,
                  size_t *smlen, const unsigned char *m,
                  size_t mlen, const sk_t *esk) {
    return sign_attached(p, sm, smlen, m, mlen, NULL, esk);
}

int mayo_sign(const mayo_params_t *p, unsigned char *sm,
              size_t *smlen, const unsigned char *m,
              size_t mlen, const unsigned char *csk) {
    return sign_attached(p, sm, smlen, m, mlen, csk, NULL);
}

int mayo_open(const mayo_params_t *p, unsigned char *m,
              size_t *mlen, const unsigned char *sm,
              size_t smlen, const unsigned char *pk) {
    // reject a too short sm before the costly key expansion
    if (smlen < (size_t) PARAM_sig_bytes(p)) {
        return MAYO_ERR;
    }
    alignas(64) uint64_t epk[P1_LIMBS_MAX + P2_LIMBS_MAX + P3_LIMBS_MAX + MAYO_MVEC_SLACK];
    if (mayo_expand_pk(p, pk, epk) != MAYO_OK) {
        return MAYO_ERR;
    }
    return mayo_open_epk(p, m, mlen, sm, smlen, epk);
}

int mayo_keypair_compact(const mayo_params_t *p, unsigned char *cpk,
                         unsigned char *csk) {
    int ret = MAYO_OK;
    unsigned char *seed_sk = csk;
    unsigned char S[PK_SEED_BYTES_MAX + O_BYTES_MAX];
#ifdef MAYO_SEEDED_P3
    uint64_t P[P1_LIMBS_MAX + P2_LIMBS_MAX + P3_TARGET_LIMBS_MAX + MAYO_MVEC_SLACK];
    unsigned char T[P3_TARGET_BYTES];
#else
    uint64_t P[P1_LIMBS_MAX + P2_LIMBS_MAX + MAYO_MVEC_SLACK];
    unsigned char *T = NULL;
#endif
    uint64_t P3[O_MAX*O_MAX*M_VEC_LIMBS_MAX + MAYO_MVEC_SLACK] = {0};

    unsigned char *seed_pk;
    unsigned char O[(V_MAX)*O_MAX];

    const int m_vec_limbs = PARAM_m_vec_limbs(p);
    const int param_m = PARAM_m(p);
    const int param_v = PARAM_v(p);
    const int param_o = PARAM_o(p);
    const int param_O_bytes = PARAM_O_bytes(p);
    const int param_P1_limbs = PARAM_P1_limbs(p);
    const int param_P3_limbs = PARAM_P3_limbs(p);
    const int param_pk_seed_bytes = PARAM_pk_seed_bytes(p);
    const int param_sk_seed_bytes = PARAM_sk_seed_bytes(p);

    uint64_t *P1 = P;
    uint64_t *P2 = P + param_P1_limbs;

#ifdef MAYO_SEEDED_P3
retry:
#endif
    // seed_sk $←- B^(sk_seed bytes)
    #if defined(PQM4) || defined(HAVE_RANDOMBYTES_NORETVAL)
    randombytes(seed_sk, param_sk_seed_bytes);
    #else
    if (randombytes(seed_sk, param_sk_seed_bytes) != MAYO_OK) {
        ret = MAYO_ERR;
        goto err;
    }
    #endif

    // S ← shake256(seedsk, pk seed bytes + O bytes)
    shake256(S, param_pk_seed_bytes + param_O_bytes, seed_sk,
             param_sk_seed_bytes);
    // seed_pk ← s[0 : pk_seed_bytes]
    seed_pk = S;

    // o ← Decode_o(s[pk_seed_bytes : pk_seed_bytes + o_bytes])
    decode(S + param_pk_seed_bytes, O, param_v * param_o);

#ifdef ENABLE_CT_TESTING
    VALGRIND_MAKE_MEM_DEFINED(seed_pk, param_pk_seed_bytes);
#endif

    expand_P1_P2(p, P, seed_pk, 0, T);
    MAYO_ZERO_SLACK(P, param_P1_limbs + PARAM_P2_limbs(p));

#ifdef MAYO_SEEDED_P3
    if (!match_P3_targets(p, P1, P2, T, O)) {
        goto retry;
    }
    // csk = seed_sk || Encode(O), so expanding it needs no solve
    encode(O, csk + param_sk_seed_bytes, param_v * param_o);
#endif

    // compute P3 (modifies P2 in the process)
    compute_P3(p, P1, P2, O, P3);

    // store seed_pk in cpk
    memcpy(cpk, seed_pk, param_pk_seed_bytes);

    uint64_t P3_upper[P3_LIMBS_MAX + MAYO_MVEC_SLACK];

    // compute Upper(P3) and store in cpk
    MAYO_ZERO_SLACK(P3_upper, param_P3_limbs);
    m_upper(p, P3, P3_upper, param_o);
#ifdef MAYO_SEEDED_P3
    (void) m_vec_limbs;
    (void) param_m;
    pack_P3(p, P3_upper, cpk + param_pk_seed_bytes);
#else
    pack_m_vecs(P3_upper, cpk + param_pk_seed_bytes, param_P3_limbs/m_vec_limbs, param_m);
#endif

#if !defined(PQM4) && !defined(HAVE_RANDOMBYTES_NORETVAL)
    err:
#endif
    mayo_secure_clear(S, PK_SEED_BYTES_MAX + O_BYTES_MAX);
    mayo_secure_clear(O, sizeof(O));
    mayo_secure_clear(P2, PARAM_P2_limbs(p)*sizeof(uint64_t));
    mayo_secure_clear(P3, sizeof(P3));
    return ret;
}

int mayo_expand_pk(const mayo_params_t *p, const unsigned char *cpk,
                   uint64_t *pk) {
#ifdef MAYO_SEEDED_P3
    unsigned char T[P3_TARGET_BYTES];
#else
    unsigned char *T = NULL;
#endif
    expand_P1_P2(p, pk, cpk, 1, T); // verify reads P1 and P2 in place: keep P2 packed too
    // P3 follows P1||P2, whose end depends on whether those stayed packed.
#ifdef MAYO_PACKED_P1P2
    unsigned char *P3 = (unsigned char *)pk + PARAM_P1_bytes(p) + PARAM_P2_bytes(p);
#else
    unsigned char *P3 = (unsigned char *)(pk + PARAM_P1_limbs(p) + PARAM_P2_limbs(p));
#endif
#ifdef MAYO_SEEDED_P3
    unpack_P3(p, cpk + PARAM_pk_seed_bytes(p), T, (uint64_t *)P3);
#else
    unpack_m_vecs(cpk + PARAM_pk_seed_bytes(p), (uint64_t *)P3,
                  PARAM_P3_limbs(p) / PARAM_m_vec_limbs(p), PARAM_m(p));
#endif
    MAYO_ZERO_SLACK(P3, PARAM_P3_limbs(p));

    return MAYO_OK;
}

int mayo_verify_epk(const mayo_params_t *p, const unsigned char *m,
                    size_t mlen, const unsigned char *sig,
                    const uint64_t *epk) {
    unsigned char tEnc[M_BYTES_MAX + MAYO_LAMBDA_SEED_BYTES];
    alignas(32) uint64_t lambda[N_MAX * M_VEC_LIMBS_MAX + MAYO_MVEC_SLACK];
    unsigned char y[M_MAX];
    unsigned char s[K_MAX * N_MAX];
    unsigned char tmp[DIGEST_BYTES_MAX + SALT_BYTES_MAX];

    const int param_m = PARAM_m(p);
    const int param_n = PARAM_n(p);
    const int param_k = PARAM_k(p);
    const int param_m_bytes = PARAM_m_bytes(p);
    const int param_sig_bytes = PARAM_sig_bytes(p);
    const int param_digest_bytes = PARAM_digest_bytes(p);
    const int param_salt_bytes = PARAM_salt_bytes(p);

    const uint64_t *P1 = epk;
#ifdef MAYO_PACKED_P1P2
    // P1 and P2 are packed in the expanded key, so P2/P3 start at byte offsets
    const uint64_t *P2 = (const uint64_t *)((const unsigned char *)P1 + PARAM_P1_bytes(p));
    const uint64_t *P3 = (const uint64_t *)((const unsigned char *)P1 + PARAM_P1_bytes(p) + PARAM_P2_bytes(p));
#else
    const uint64_t *P2 = P1 + PARAM_P1_limbs(p);
    const uint64_t *P3 = P2 + PARAM_P2_limbs(p);
#endif

#if defined(MAYO_PACKED_P1P2) && defined(TARGET_BIG_ENDIAN)
#error "packed P1/P2 is little-endian only: the swap below needs limb-aligned m-vecs"
#endif

#ifdef TARGET_BIG_ENDIAN
    uint64_t pk_be[P1_LIMBS_MAX + P2_LIMBS_MAX + P3_LIMBS_MAX];
    memcpy(pk_be, epk, (PARAM_P1_limbs(p) + PARAM_P2_limbs(p) + PARAM_P3_limbs(p)) * sizeof(uint64_t));
    P1 = pk_be;
    P2 = pk_be + PARAM_P1_limbs(p);
    P3 = pk_be + PARAM_P1_limbs(p) + PARAM_P2_limbs(p);
    bswap_limbs(pk_be, PARAM_P1_limbs(p) + PARAM_P2_limbs(p) + PARAM_P3_limbs(p));
#endif

    // hash m
    shake256(tmp, param_digest_bytes, m, mlen);

    // compute t
    memcpy(tmp + param_digest_bytes, sig + param_sig_bytes - param_salt_bytes,
           param_salt_bytes);
    /* SHAKE is a stream, so the first param_m_bytes are the same t as before;
       the extra MAYO_LAMBDA_SEED_BYTES seed the lambda expansion. */
    shake256(tEnc, param_m_bytes + MAYO_LAMBDA_SEED_BYTES, tmp,
             param_digest_bytes + param_salt_bytes);
    derive_lambda(p, tEnc + param_m_bytes, lambda);

    // decode s
    decode(sig, s, param_k * param_n);

    // the signature is good iff P*(s) = t, i.e. y = t - P*(s) is zero
    eval_public_map(p, s, P1, P2, P3, lambda, tEnc, y);

    unsigned char diff = 0;
    for (int i = 0; i < param_m; i++) {
        diff |= y[i];
    }
    return diff == 0 ? MAYO_OK : MAYO_ERR;
}

int mayo_open_epk(const mayo_params_t *p, unsigned char *m,
                  size_t *mlen, const unsigned char *sm,
                  size_t smlen, const uint64_t *epk) {
    const int param_sig_bytes = PARAM_sig_bytes(p);
    if (smlen < (size_t)param_sig_bytes) {
        return MAYO_ERR;
    }
    int result = mayo_verify_epk(p, sm + param_sig_bytes, smlen - param_sig_bytes,
                                 sm, epk);
    if (result == MAYO_OK) {
        *mlen = smlen - param_sig_bytes;
        memmove(m, sm + param_sig_bytes, *mlen);
    }
    return result;
}

int mayo_verify(const mayo_params_t *p, const unsigned char *m,
                size_t mlen, const unsigned char *sig,
                const unsigned char *cpk) {
    alignas(64) uint64_t pk[P1_LIMBS_MAX + P2_LIMBS_MAX + P3_LIMBS_MAX + MAYO_MVEC_SLACK];

    int ret = mayo_expand_pk(p, cpk, pk);
    if (ret != MAYO_OK) {
        return MAYO_ERR;
    }

    return mayo_verify_epk(p, m, mlen, sig, pk);
}
