/* Median TSC ticks of UOV keygen, sign and verify; bench_uov.sh links it against one
   parameter set and turns the ticks into core cycles. usage: bench_uov <iterations> */
#include <stdio.h>
#include <stdlib.h>
#include <x86intrin.h>
#include "api.h"

static unsigned long long now(void) {
    unsigned aux;
    return __rdtscp(&aux);
}

static int cmp(const void *a, const void *b) {
    unsigned long long x = *(const unsigned long long *) a, y = *(const unsigned long long *) b;
    return (x > y) - (x < y);
}

static unsigned long long median(unsigned long long *t, int n) {
    qsort(t, n, sizeof *t, cmp);
    return t[n / 2];
}

int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 40;
    static unsigned char pk[CRYPTO_PUBLICKEYBYTES], sk[CRYPTO_SECRETKEYBYTES];
    unsigned char m[32] = {0}, sm[sizeof m + CRYPTO_BYTES], m2[sizeof sm];
    unsigned long long smlen, mlen, *tk = malloc(n * sizeof *tk), *ts = malloc(n * sizeof *ts),
                       *tv = malloc(n * sizeof *tv);
    for (int i = 0; i < n; i++) {
        unsigned long long t0 = now();
        if (crypto_sign_keypair(pk, sk)) return 1;
        unsigned long long t1 = now();
        m[0] = (unsigned char) i;
        if (crypto_sign(sm, &smlen, m, sizeof m, sk)) return 1;
        unsigned long long t2 = now();
        if (crypto_sign_open(m2, &mlen, sm, smlen, pk)) return 1;
        unsigned long long t3 = now();
        tk[i] = t1 - t0, ts[i] = t2 - t1, tv[i] = t3 - t2;
    }
    printf("keygen=%llu sign=%llu verify=%llu\n", median(tk, n), median(ts, n), median(tv, n));
    return 0;
}
