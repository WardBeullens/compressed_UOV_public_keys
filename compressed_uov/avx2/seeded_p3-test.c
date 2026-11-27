// SPDX-License-Identifier: CC0 OR Apache-2.0
/// Checks the seeded-P3 public key: P3 recomputed from the O in the secret key equals the
/// expanded public key, its seeded entries equal the targets, and their number is N_SEEDED.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "api.h"
#include "params.h"
#include "ov_keypair.h"
#include "ov.h"
#include "ov_keypair_computation.h"
#include "ov_seeded_p3.h"
#include "blas_comm.h"
#include "ov_blas.h"
#include "utils_prng.h"

int main(void) {
#if !defined(_OV_SEEDED_P3)
    printf("%s: not the seeded-P3 format, nothing to check\n", OV_ALGNAME);
    return 0;
#else
    static unsigned char cpk[CRYPTO_PUBLICKEYBYTES];
    static sk_t sk;
    static pk_t epk;
    static unsigned char P2[_PK_P2_BYTE], S[_PK_P2_BYTE], P3[_PK_P3_BYTE];
    static unsigned char targets[_SEEDED_P3_BYTE];
    int bad = 0;
    for (int it = 0; it < 20; it++) {
        if (crypto_sign_keypair(cpk, (unsigned char *) &sk) != 0) {
            printf("keygen failed\n");
            return 1;
        }
        const cpk_t *c = (const cpk_t *) cpk;
        expand_pk(&epk, c);
        // P1, P2 and the targets from pk_seed
        prng_publicinputs_t prng;
        prng_set_publicinputs(&prng, c->pk_seed);
        prng_skip_publicinputs(&prng, _PK_P1_BYTE);
        prng_gen_publicinputs(&prng, P2, sizeof(P2));
        prng_gen_publicinputs(&prng, targets, sizeof(targets));
        prng_release_publicinputs(&prng);
        // P3 from the stored O, independently of the key generation
        calculate_F2_P3(S, P3, epk.pk, P2, sk.O);
        if (memcmp(P3, epk.pk + _PK_P1_BYTE + _PK_P2_BYTE, sizeof(P3))) {
            bad++;
        }
        unsigned seeded = 0, off = 0;
        for (unsigned j = 1; j < _O; j++) {
            for (unsigned t = 0; t < _SEEDED_IN_COL(j); t++, seeded++) {
                const unsigned i = t / _PUB_M, k = t % _PUB_M;
                const unsigned idx = i * _O - i * (i - 1) / 2 + (j - i);
                if (gfv_get_ele(P3 + idx * _O_BYTE, k) != gfv_get_ele(targets, off + t)) {
                    bad++;
                }
            }
            off += _SEEDED_IN_COL(j);
        }
        if (seeded != N_SEEDED) {
            bad++;
        }
    }
    printf("%s: seeded P3 check %s (%d seeded field elements)\n", OV_ALGNAME, bad ? "FAILED" : "ok", N_SEEDED);
    return bad != 0;
#endif
}
