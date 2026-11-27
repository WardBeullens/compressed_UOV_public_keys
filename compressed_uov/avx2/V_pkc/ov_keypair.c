// SPDX-License-Identifier: CC0 OR Apache-2.0
/// @file ov_keypair.c
/// @brief implementations of functions in ov_keypair.h
///
#include "ov_keypair.h"
#include "ov.h"
#include "ov_keypair_computation.h"

#include "blas_comm.h"
#include "blas.h"
#include "ov_blas.h"

#include <stdlib.h>
#include <stdint.h>
#include <string.h>


#include "utils_prng.h"
#include "utils_hash.h"
#include "utils_malloc.h"
#include "ov_seeded_p3.h"

#if !defined(PQM4)
#define _MALLOC_
#endif



///////////////////  Classic ov  //////////////////////////////////

int generate_keypair( pk_t *rpk, sk_t *sk, const unsigned char *sk_seed ) {
    memcpy( sk->sk_seed, sk_seed, LEN_SKSEED );

    // pk_seed || O
    unsigned char buf[LEN_PKSEED + sizeof(sk->O)];
    unsigned char *pk_seed = buf;
    unsigned char *O = buf + LEN_PKSEED;

    // prng for sk
    hash_ctx hctx;
    hash_init(&hctx);
    hash_update(&hctx, sk_seed, LEN_SKSEED );
    hash_final_digest( buf, sizeof(buf), &hctx );
    memcpy(sk->O, O, sizeof(sk->O));

    #if defined(_VALGRIND_)
    // mark pk_seed as public
    VALGRIND_MAKE_MEM_DEFINED(pk_seed, LEN_PKSEED );
    #endif

    // prng for pk
    prng_publicinputs_t prng1;
    prng_set_publicinputs(&prng1, pk_seed );
    // P1 and P2
    prng_gen_publicinputs(&prng1, rpk->pk, sizeof(sk->P1) + sizeof(sk->S) );
    memcpy( sk->P1, rpk->pk, sizeof(sk->P1) );
    prng_release_publicinputs(&prng1);

    // S and P3
    unsigned char *rpk_P2 = rpk->pk + sizeof(sk->P1);
    unsigned char *rpk_P3 = rpk->pk + sizeof(sk->P1) + sizeof(sk->S);
    calculate_F2_P3( sk->S, rpk_P3, sk->P1, rpk_P2, sk->O );
    return 0;
}


int expand_pk( pk_t *rpk, const cpk_t *cpk ) {
    prng_publicinputs_t prng0;
    prng_set_publicinputs(&prng0, cpk->pk_seed);
    // P1 and P2
    prng_gen_publicinputs(&prng0, rpk->pk, _PK_P1_BYTE + _PK_P2_BYTE );
    #if defined(_OV_SEEDED_P3)
    // the targets follow P2 in the same stream; P3 is the stored part plus the targets
    unsigned char targets[_SEEDED_P3_BYTE];
    prng_gen_publicinputs(&prng0, targets, sizeof(targets) );
    prng_release_publicinputs(&prng0);
    ov_unpack_P3( rpk->pk + _PK_P1_BYTE + _PK_P2_BYTE, cpk->P3, targets );
    #else
    prng_release_publicinputs(&prng0);
    // P3
    memcpy( rpk->pk + _PK_P1_BYTE + _PK_P2_BYTE, cpk->P3, sizeof(cpk->P3) );
    #endif

    return 0;
}

int expand_pk_predicate( pk_t *rpk, const cpk_t *cpk, const unsigned char *predicate) {
    prng_publicinputs_t prng0;
    prng_set_publicinputs(&prng0, cpk->pk_seed);

    // P1
    uint8_t *p1p2 = rpk->pk;
    for (unsigned i = 0; i < _V; i++) {
        unsigned long len_i = _O_BYTE * (_V - i);
        if ( 0 == predicate[i] ) {
            prng_skip_publicinputs(&prng0, len_i );
        } else                    {
            prng_gen_publicinputs(&prng0, p1p2, len_i );
        }
        p1p2 += len_i;
    }
    // P2
    for (unsigned i = 0; i < _V; i++) {
        unsigned long len_i = _O_BYTE * _O;
        if ( 0 == predicate[i] ) {
            prng_skip_publicinputs(&prng0, len_i );
        } else                    {
            prng_gen_publicinputs(&prng0, p1p2, len_i );
        }
        p1p2 += len_i;
    }
    prng_release_publicinputs(&prng0);
    // P3
    memcpy( rpk->pk + _PK_P1_BYTE + _PK_P2_BYTE, cpk->P3, sizeof(cpk->P3) );
    return 0;
}


// Derive O, P1 and P2 from the seed.  P2 lands in sk->S, where calculate_F2()
// overwrites it with S in place.
static
void derive_sk_from_seed( sk_t *sk, const unsigned char *sk_seed ) {
    memcpy( sk->sk_seed, sk_seed, LEN_SKSEED );

    // pk_seed || O
    unsigned char buf[LEN_PKSEED + sizeof(sk->O)];
    unsigned char *pk_seed = buf;
    unsigned char *O = buf + LEN_PKSEED;

    // prng for sk
    hash_ctx hctx;
    hash_init(&hctx);
    hash_update(&hctx, sk_seed, LEN_SKSEED );
    hash_final_digest( buf, sizeof(buf), &hctx );
    memcpy(sk->O, O, sizeof(sk->O));

    #if defined(_VALGRIND_)
    // mark pk_seed as public
    VALGRIND_MAKE_MEM_DEFINED(pk_seed, LEN_PKSEED );
    #endif

    // prng for pk
    prng_publicinputs_t prng1;
    prng_set_publicinputs(&prng1, pk_seed );
    // P1
    prng_gen_publicinputs(&prng1, sk->P1, sizeof(sk->P1) );
    // P2
    prng_gen_publicinputs(&prng1, sk->S, sizeof(sk->S) );

    prng_release_publicinputs(&prng1);
}


#if defined(_OV_PKC_SKC)
int expand_sk_implicit_s( sk_t *sk, const unsigned char *sk_seed ) {
    derive_sk_from_seed( sk, sk_seed );
    return 0;
}
#else
int expand_sk( sk_t *sk, const unsigned char *sk_seed ) {
    derive_sk_from_seed( sk, sk_seed );
    // calcuate the parts of sk according to pk.
    #if defined(_BLAS_M4F_)
    ov_pkc_calculate_F_from_Q( sk );
    #else
    calculate_F2( sk->S, sk->P1, sk->S, sk->O );
    #endif
    return 0;
}
#endif



////////////////////////////////////////////////////////////////////////////////////




int generate_keypair_pkc( cpk_t *pk, sk_t *sk, const unsigned char *sk_seed ) {
    memcpy( sk->sk_seed, sk_seed, LEN_SKSEED );

    // pk_seed || O
    unsigned char buf[LEN_PKSEED + sizeof(sk->O)];
    unsigned char *pk_seed = buf;
    unsigned char *O = buf + LEN_PKSEED;

    // prng for sk
    hash_ctx hctx;
    hash_init(&hctx);
    hash_update(&hctx, sk_seed, LEN_SKSEED );
    hash_final_digest( buf, sizeof(buf), &hctx );
    memcpy(sk->O, O, sizeof(sk->O));

    #if defined(_VALGRIND_)
    // mark pk_seed as public
    VALGRIND_MAKE_MEM_DEFINED(pk_seed, LEN_PKSEED );
    #endif

    memcpy( pk->pk_seed, pk_seed, LEN_PKSEED );

    // prng for pk
    prng_publicinputs_t prng1;
    prng_set_publicinputs( &prng1, pk_seed );
    // P1
    prng_gen_publicinputs(&prng1, sk->P1, sizeof(sk->P1) );
    // P2
    prng_gen_publicinputs(&prng1, sk->S, sizeof(sk->S) );
    #if defined(_OV_SEEDED_P3)
    // the targets for P3, and an O that meets them; sk->O is then no longer the hash
    // output of sk_seed, and the secret key stores it
    unsigned char targets[_SEEDED_P3_BYTE];
    prng_gen_publicinputs(&prng1, targets, sizeof(targets) );
    prng_release_publicinputs(&prng1);
    if (0 != ov_match_P3_targets( sk->O, sk->P1, sk->S, targets )) {
        return -1;  // a singular system: the caller picks another sk_seed
    }
    #if defined(_VALGRIND_)
    VALGRIND_MAKE_MEM_UNDEFINED(sk->O, sizeof(sk->O) );
    #endif
    PQOV_ALIGN unsigned char P3[_PK_P3_BYTE];
    calculate_F2_P3( sk->S, P3, sk->P1, sk->S, sk->O );
    ov_pack_P3( pk->P3, P3 );
    return 0;
    #else
    prng_release_publicinputs(&prng1);
    #endif

    #if defined(_BLAS_M4F_)
    calculate_P3( pk->P3, sk->P1, sk->S, sk->O );
    ov_pkc_calculate_F_from_Q( sk );     // calcuate the rest parts of secret key from Qs and S,T
    #else
    calculate_F2_P3( sk->S, pk->P3, sk->P1, sk->S, sk->O );
    #endif
    return 0;
}



int generate_keypair_pkc_skc( cpk_t *pk, csk_t *rsk, const unsigned char *sk_seed ) {
    memcpy( rsk->sk_seed, sk_seed, LEN_SKSEED );

    #if defined(_MALLOC_)
    sk_t *sk = ov_malloc(sizeof(sk_t));
    if (NULL == sk) {
        return -1;
    }
    #else
    sk_t _sk;
    sk_t *sk = &_sk;
    #endif
    int r = generate_keypair_pkc( pk, sk, sk_seed );

    #if defined(_MALLOC_)
    ov_free(sk, sizeof(sk_t));
    #endif
    return r;
}


