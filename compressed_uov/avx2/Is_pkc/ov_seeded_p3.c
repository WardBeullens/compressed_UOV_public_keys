// SPDX-License-Identifier: CC0 OR Apache-2.0
/// @file ov_seeded_p3.c
/// @brief Implementations for functions in ov_seeded_p3.h
///
#include "ov_seeded_p3.h"

#if defined(_OV_SEEDED_P3)

#include "blas_comm.h"
#include "blas.h"
#include "ov_blas.h"
#include "gf16.h"
#include "utils_malloc.h"

#include <stdint.h>
#include <string.h>
#if defined(_VALGRIND_)
#include "valgrind/memcheck.h"
#endif

#if defined(_USE_GF16)
#define gf_mul gf16_mul
#define gf_inv gf16_inv
#define gfv_set_ele gf16v_set_ele
#else
#define gf_mul gf256_mul
#define gf_inv gf256_inv
#define gfv_set_ele gf256v_set_ele
#endif

/// clear a buffer of secret data in a way the compiler does not remove: memset, called
/// through a volatile pointer
static void *(*const volatile memset_v)( void *, int, size_t ) = memset;
static void secure_clear( void *p, size_t len ) {
    memset_v( p, 0, len );
}

/// where column j's targets start: columns 1..o-1 in order. Column 1 seeds m and the
/// later ones v (m <= v <= 2m, see N_SEEDED), so the sum has a closed form.
static unsigned seeded_offset( unsigned j ) {
    return (j <= 1) ? 0 : _PUB_M + (j - 2) * _V;
}
#if (_PUB_M > _V) || (_V > 2 * _PUB_M)
#error "seeded_offset assumes m <= v <= 2m"
#endif

/// index of the m-vector (i,j), i <= j, in a batched upper-triangular o x o matrix
static unsigned p3_idx( unsigned i, unsigned j ) {
    return i * _O - i * (i - 1) / 2 + (j - i);
}

/// the seeded coordinates of entry (r,c) of P3 are its first seeded_prefix(r,c)
static unsigned seeded_prefix( unsigned r, unsigned c ) {
    if (r >= c) {
        return 0;
    }
    int left = (int) _SEEDED_IN_COL(c) - (int) (r * _PUB_M);
    return left <= 0 ? 0 : (left > _PUB_M ? _PUB_M : (unsigned) left);
}

/// constant-time: 0xff if a != 0, else 0
static uint8_t ct_nonzero( uint8_t a ) {
    return (uint8_t) - (uint8_t) ((a | (uint8_t) - a) >> 7);
}

#if defined(_USE_GF16)
#define ELE_BYTE(i) ((i) / 2)
#define ELE_BYTE_UP(i) (((i) + 1) / 2)
#else
#define ELE_BYTE(i) (i)
#define ELE_BYTE_UP(i) (i)
#endif
/// a row of [A^T | I]: m + v elements
#define AT_ELE (_PUB_M + _V)
#define AT_STRIDE ((ELE_BYTE_UP(AT_ELE) + 31) / 32 * 32)
/// the dimension of the solutions for column 1
#define _F (_V - _PUB_M)
#define _F_BYTE ELE_BYTE(_F)
/// a row of N^T: v - m entries; for GF(16) padded to 32 bytes, the column length the
/// vectorized gf16mat_prod is fast on
#if defined(_USE_GF16)
#define NT_BYTE ((_F_BYTE + 31) / 32 * 32)
#else
#define NT_BYTE _F_BYTE
#endif
#if (NT_BYTE > _O_BYTE) || (NT_BYTE > _V_BYTE)
#error "NT_BYTE must fit in a row of L and in u"
#endif
/// a row of [N K | right-hand sides for columns 2..o-1]
#define S_ELE (_F + _O - 2)
#define S_STRIDE ((ELE_BYTE_UP(S_ELE) + 31) / 32 * 32)
/// the pivot search looks this many rows below the diagonal
#define SEARCH 32
/// columns EP.. of E are those of the identity (see solve_steps), so only the first EP are
/// stored and multiplied
#define EP ((_PUB_M + SEARCH < _V) ? _PUB_M + SEARCH : _V)
#define EP_BYTE ELE_BYTE(EP)

/* Gauss-Jordan steps 0..steps-1 on the n rows of W (rows of stride bytes, len elements).
   The pivots stay on the diagonal; a zero one is made nonzero by adding the rows below it,
   selected in constant time. Columns id0.. hold an identity block (id0 = len if none):
   once made nonzero, the pivot row of step col holds identity entries only of rows up to
   col + SEARCH, so it is zero outside columns col..id0+col+SEARCH and a row operation
   covers only these. Returns 0xff if every pivot is nonzero. */
static uint8_t solve_steps( uint8_t *W, unsigned n, unsigned steps, unsigned stride, unsigned id0, unsigned len ) {
    uint8_t ok = 0xff;
    for (unsigned col = 0; col < steps; col++) {
        uint8_t *pr = W + col * stride;
        const unsigned end = (id0 + col + SEARCH + 1 < len) ? id0 + col + SEARCH + 1 : len;
        const unsigned b0 = ELE_BYTE(col), nb = ELE_BYTE_UP(end) - b0;
        unsigned last = (n - 1 < col + SEARCH) ? n - 1 : col + SEARCH;
        for (unsigned r = col + 1; r <= last; r++) {
            uint8_t mask = (uint8_t) ~ct_nonzero( gfv_get_ele( pr, col ) );
            for (unsigned l = b0; l < b0 + nb; l++) {
                pr[l] ^= mask & W[r * stride + l];
            }
        }
        const uint8_t piv = gfv_get_ele( pr, col );
        ok &= ct_nonzero( piv );
        gfv_mul_scalar( pr + b0, gf_inv( piv ), nb );
        for (unsigned r = 0; r < n; r++) {
            if (r == col) {
                continue;
            }
            gfv_madd( W + r * stride + b0, pr + b0, gfv_get_ele( W + r * stride, col ), nb );
        }
    }
    return ok;
}

/* L = (P1 + P1^T) o + P2 e_i. Row r of the upper triangular P1 is contiguous from its
   diagonal entry, so it gives (P1 o)_r as one product and its share of P1^T o as one madd
   into L[r..v-1]; the diagonal is counted twice and cancels. */
static void calc_L( uint8_t *L, const uint8_t *P1, const uint8_t *P2, const uint8_t *o, unsigned i ) {
    PQOV_ALIGN uint8_t t[_O_BYTE];
    #if defined(_MUL_WITH_MULTAB_)
    PQOV_ALIGN uint8_t multabs[_V * 32];       // the multiplication tables of the entries of o
    gfv_generate_multabs( multabs, o, _V );
    #elif defined(_USE_GF16)
    PQOV_ALIGN uint8_t o_odd[_V_BYTE] = {0};   // o from entry 1 on: (P1 o)_r for odd r
    for (unsigned r = 1; r < _V; r++) {
        gfv_set_ele( o_odd, r - 1, gfv_get_ele( o, r ) );
    }
    #endif
    for (unsigned r = 0; r < _V; r++) {
        memcpy( L + r * _O_BYTE, P2 + (r * _O + i) * _O_BYTE, _O_BYTE );
    }
    for (unsigned r = 0; r < _V; r++) {
        #if defined(_MUL_WITH_MULTAB_)
        gfmat_prod_multab( t, P1, _O_BYTE, _V - r, multabs + r * 32 );
        gf256v_add( L + r * _O_BYTE, t, _O_BYTE );
        gfv_madd_multab( L + r * _O_BYTE, P1, multabs + r * 32, (_V - r) * _O_BYTE );
        #else
        #if defined(_USE_GF16)
        const uint8_t *o_r = (r & 1) ? o_odd + (r - 1) / 2 : o + r / 2;
        #else
        const uint8_t *o_r = o + r;
        #endif
        gfmat_prod( t, P1, _O_BYTE, _V - r, o_r );
        gf256v_add( L + r * _O_BYTE, t, _O_BYTE );
        gfv_madd( L + r * _O_BYTE, P1, gfv_get_ele( o, r ), (_V - r) * _O_BYTE );
        #endif
        P1 += (_V - r) * _O_BYTE;
    }
    secure_clear( t, sizeof(t) );
    #if defined(_MUL_WITH_MULTAB_)
    secure_clear( multabs, sizeof(multabs) );
    #elif defined(_USE_GF16)
    secure_clear( o_odd, sizeof(o_odd) );
    #endif
}

/* Row t = (i, k) of the system for column j is L_i^(k), the same for every j > i, so the
   system for column j consists of the first n_j rows of one v x v matrix M = [A; N]: A,
   m rows from column 0, and N, the remaining v - m rows from column 1 (m < v <= 2m).
   Column 1 must be a uniformly random solution of A x = b. Gauss-Jordan elimination of
   [A^T | I] with the pivots on the diagonal gives E with E A^T = [I; 0], so x = E^T (b, z)
   solves A x = b for every z, and is uniformly random for z uniformly random; it fails
   exactly when A does not have full row rank. For the later columns, with x = x0 + K y,
   x0 = E^T (b, 0) and K = E^T (0, I), the system M x = (b, c) becomes the (v - m) x (v - m)
   system (N K) y = c - N x0, which is singular exactly when M is. Entries EP.. of E^T w
   are those of w.
   Row (i, k) has right-hand side (o_i^T P2 e_c)_k + target for column c, so the b and c of
   a column, and the rows of A^T and N^T (the first m and v - m entries of L_0[r] and
   L_1[r]), are contiguous vectors, and every product is a gfmat_prod. */
#if (_V <= _PUB_M) || (_V > 2 * _PUB_M)
#error "ov_match_P3_targets assumes m < v <= 2m"
#endif
int ov_match_P3_targets( unsigned char *O, const unsigned char *P1, const unsigned char *P2,
                         const unsigned char *targets ) {
    PQOV_ALIGN uint8_t L[_V * _O_BYTE];        // L_i: v m-vectors, then N^T (v - m entries each)
    PQOV_ALIGN uint8_t B[_O * _O_BYTE];        // column c: b = (o_0^T P2 e_c) + targets
    PQOV_ALIGN uint8_t OtP2[_O * _O_BYTE];     // o_1^T P2: o m-vectors
    PQOV_ALIGN uint8_t At[_V * AT_STRIDE];     // [A^T | I], then [E A^T | E], then E[:, 0..EP-1]
    PQOV_ALIGN uint8_t NK[_F * NT_BYTE];       // the columns of N K
    PQOV_ALIGN uint8_t S[_F * S_STRIDE];       // [N K | c - N x0 for columns 2..o-1]
    PQOV_ALIGN uint8_t u[_V_BYTE];
    uint8_t ok = 0xff;

    // L_0, and b = o_0^T P2 e_c + targets for every column c
    calc_L( L, P1, P2, O, 0 );
    gfmat_prod( B, P2, _O * _O_BYTE, _V, O );
    gf256v_add( B + _O_BYTE, targets, _O_BYTE );
    for (unsigned c = 2; c < _O; c++) {
        gf256v_add( B + c * _O_BYTE, targets + ELE_BYTE( seeded_offset(c) ), _O_BYTE );
    }

    // [A^T | I]: row r is L_0[r], then e_r; E is the right half after m steps
    memset( At, 0, sizeof(At) );
    for (unsigned r = 0; r < _V; r++) {
        memcpy( At + r * AT_STRIDE, L + r * _O_BYTE, _O_BYTE );
        gfv_set_ele( At + r * AT_STRIDE, _PUB_M + r, 1 );
    }
    ok &= solve_steps( At, _V, _PUB_M, AT_STRIDE, _PUB_M, AT_ELE );
    uint8_t *E = At;                           // the first EP entries of the rows of E, in place
    for (unsigned r = 0; r < _V; r++) {
        memmove( E + r * EP_BYTE, At + r * AT_STRIDE + ELE_BYTE(_PUB_M), EP_BYTE );
    }

    // column 1: x = E^T (b, z), z the last v - m sampled entries of column 1 (and so
    // already entries EP.. of x)
    uint8_t *o1 = O + _V_BYTE;
    memcpy( u, o1, _V_BYTE );
    memcpy( u, B + _O_BYTE, _O_BYTE );
    gfmat_prod( o1, E, EP_BYTE, _V, u );

    // L_1 and o_1^T P2; N^T: row r is the first v - m entries of L_1[r]
    calc_L( L, P1, P2, o1, 1 );
    gfmat_prod( OtP2, P2, _O * _O_BYTE, _V, o1 );
    uint8_t *NT = L;
    for (unsigned r = 0; r < _V; r++) {
        memmove( NT + r * NT_BYTE, L + r * _O_BYTE, NT_BYTE );
    }

    // columns 2..o-1: x = x0 + K y with (N K) y = c - N x0; K is rows m..v-1 of E
    memset( S, 0, sizeof(S) );
    for (unsigned s = 0; s < _F; s++) {
        gfmat_prod( NK + s * NT_BYTE, NT, NT_BYTE, EP, E + (_PUB_M + s) * EP_BYTE );
        if (_PUB_M + s >= EP) {
            gf256v_add( NK + s * NT_BYTE, NT + (_PUB_M + s) * NT_BYTE, NT_BYTE );
        }
    }
    for (unsigned s = 0; s < _F; s++) {
        for (unsigned s2 = 0; s2 < _F; s2++) {
            gfv_set_ele( S + s * S_STRIDE, s2, gfv_get_ele( NK + s2 * NT_BYTE, s ) );
        }
    }
    for (unsigned c = 2; c < _O; c++) {     // x0 into column c of O, and c - N x0
        uint8_t *oc = O + c * _V_BYTE;
        gfmat_prod( oc, E, EP_BYTE, _PUB_M, B + c * _O_BYTE );
        gfmat_prod( u, NT, NT_BYTE, EP, oc );
        gf256v_add( u, OtP2 + c * _O_BYTE, _F_BYTE );
        gf256v_add( u, targets + ELE_BYTE( seeded_offset(c) + _PUB_M ), _F_BYTE );
        for (unsigned s = 0; s < _F; s++) {
            gfv_set_ele( S + s * S_STRIDE, _F + c - 2, gfv_get_ele( u, s ) );
        }
    }
    ok &= solve_steps( S, _F, _F, S_STRIDE, S_ELE, S_ELE );
    for (unsigned c = 2; c < _O; c++) {
        uint8_t *oc = O + c * _V_BYTE;
        PQOV_ALIGN uint8_t y[_F_BYTE] = {0};
        for (unsigned s = 0; s < _F; s++) {
            gfv_set_ele( y, s, gfv_get_ele( S + s * S_STRIDE, _F + c - 2 ) );
        }
        gfmat_prod( u, E + _PUB_M * EP_BYTE, EP_BYTE, _F, y );
        gf256v_add( oc, u, EP_BYTE );
        memcpy( oc + EP_BYTE, y + ELE_BYTE(EP - _PUB_M), _V_BYTE - EP_BYTE );
        secure_clear( y, sizeof(y) );
    }

    secure_clear( L, sizeof(L) );
    secure_clear( B, sizeof(B) );
    secure_clear( OtP2, sizeof(OtP2) );
    secure_clear( At, sizeof(At) );
    secure_clear( NK, sizeof(NK) );
    secure_clear( S, sizeof(S) );
    secure_clear( u, sizeof(u) );
    #if defined(_VALGRIND_)
    VALGRIND_MAKE_MEM_DEFINED( &ok, sizeof(ok) );  // whether to pick another seed is public
    #endif
    return (ok == 0xff) ? 0 : -1;
}

/* Every seeded prefix, and with it every offset into the targets and into the stored
   stream, is a whole number of bytes: for GF(16), m, v and hence all of them are even. */
#if defined(_USE_GF16) && ((_PUB_M % 2) || (_V % 2))
#error "ov_pack_P3 assumes byte-aligned seeded prefixes"
#endif
#if defined(_USE_GF16)
#define ELE_BYTES(n) ((n) / 2)
#else
#define ELE_BYTES(n) (n)
#endif

/* Only rows 0 and 1 of P3 have seeded entries (column j seeds at most v <= 2m coordinates,
   i.e. rows 0 and 1), so they are copied entry by entry and rows 2..o-1, contiguous both
   in P3 and in the stream, in one go. */
#define SEEDED_ROWS 2

void ov_pack_P3( unsigned char *packed, const unsigned char *P3 ) {
    for (unsigned r = 0; r < SEEDED_ROWS; r++) {
        for (unsigned c = r; c < _O; c++) {
            const unsigned pb = ELE_BYTES( seeded_prefix(r, c) );
            memcpy( packed, P3 + p3_idx(r, c) * _O_BYTE + pb, _O_BYTE - pb );
            packed += _O_BYTE - pb;
        }
    }
    memcpy( packed, P3 + p3_idx(SEEDED_ROWS, SEEDED_ROWS) * _O_BYTE, _PK_P3_BYTE - p3_idx(SEEDED_ROWS, SEEDED_ROWS) * _O_BYTE );
}

void ov_unpack_P3( unsigned char *P3, const unsigned char *packed, const unsigned char *targets ) {
    for (unsigned r = 0; r < SEEDED_ROWS; r++) {
        for (unsigned c = r; c < _O; c++) {
            unsigned char *e = P3 + p3_idx(r, c) * _O_BYTE;
            const unsigned pb = ELE_BYTES( seeded_prefix(r, c) );
            memcpy( e, targets + ELE_BYTES( seeded_offset(c) + r * _PUB_M ), pb );
            memcpy( e + pb, packed, _O_BYTE - pb );
            packed += _O_BYTE - pb;
        }
    }
    memcpy( P3 + p3_idx(SEEDED_ROWS, SEEDED_ROWS) * _O_BYTE, packed, _PK_P3_BYTE - p3_idx(SEEDED_ROWS, SEEDED_ROWS) * _O_BYTE );
}

#endif
