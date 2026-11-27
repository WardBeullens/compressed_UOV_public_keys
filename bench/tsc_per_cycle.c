/* TSC ticks per core cycle, printed to stdout; bench_cycles.sh uses it to turn the TSC
   counts of mayo_bench into core cycles. Times a chain of dependent 64-bit imul, which
   take 3 cycles each on current Intel P- and E-cores (a renamer cannot eliminate them,
   unlike dependent adds). Best of 20 runs. */
#include <stdio.h>
#include <x86intrin.h>
int main(void) {
    unsigned long long best = ~0ull;
    const long N = 2000000; /* iterations of 8 imuls */
    for (int rep = 0; rep < 20; rep++) {
        unsigned long long x = 3 + rep, m = 0x9E3779B97F4A7C15ull, a = __rdtsc();
        for (long i = 0; i < N; i++)
            __asm__ volatile("imul %1,%0\n\timul %1,%0\n\timul %1,%0\n\timul %1,%0\n\t"
                             "imul %1,%0\n\timul %1,%0\n\timul %1,%0\n\timul %1,%0" : "+r"(x) : "r"(m));
        unsigned long long t = __rdtsc() - a;
        if (t < best) best = t;
    }
    printf("%.4f\n", best / (N * 8.0 * 3.0)); /* ticks per cycle */
    return 0;
}
