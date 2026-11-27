# Compressing Public Keys for Free for UOV and its Variants

Implementations and benchmark scripts for the paper. Paper: TBA

In UOV and MAYO the public map is (P1, P2, P3), where P1 and P2 are expanded from a public
seed and P3 is computed from them and the secret oil space O. Here key generation chooses
O so that part of P3 equals pseudorandom targets that are also expanded from the public
seed, so those entries are not stored in the public key. Signing and verification do not
change. Key generation gets slower (MAYO 1.6–2.4×, UOV 6–11%), and the secret key stores O.
Key generation restarts when a square linear system is singular, which happens about 6.6% of
the time for q = 16 and 0.4% for q = 256.

## Public key sizes (bytes)

| Scheme  | standard | this work | saving |
|:--------|---------:|----------:|-------:|
| MAYO_1  |  1,456 |  1,176 | 19.2% |
| MAYO_2  |  2,928 |  2,495 | 14.8% |
| MAYO_3  |  2,986 |  2,500 | 16.3% |
| MAYO_5  |  5,554 |  4,773 | 14.1% |
| uov-Is (pkc)  |  66,576 |  63,568 | 4.5% |
| uov-Ip (pkc)  |  46,591 |  43,364 | 6.9% |
| uov-III (pkc) | 189,232 | 180,690 | 4.5% |
| uov-V (pkc)   | 446,992 | 431,574 | 3.4% |

## Layout

- `compressed_mayo/` – MAYO (round-3 parameters). The modified key generation is enabled by
  `#define MAYO_SEEDED_P3 1` in `include/mayo_constants.h`; remove it for standard MAYO.
- `compressed_uov/avx2/` – UOV round-3 AVX2 implementation. The compressed-public-key
  variants (`Is_pkc`, `Ip_pkc`, `III_pkc`, `V_pkc`) use the modified key generation;
  compile with `-D_OV_NO_SEEDED_P3` for the standard one. Other variants are unchanged.
- `bench/` – cycle-count scripts for Table 2 of the paper.

Each implementation keeps its upstream license: `compressed_mayo/LICENSE` (Apache-2.0, see
also `compressed_mayo/NOTICE`), and CC0 / public domain for UOV (see
`compressed_uov/avx2/README.md`).

## Build and test

MAYO (KAT and functional tests):

    cd compressed_mayo && mkdir build && cd build
    cmake -DMAYO_BUILD_TYPE=avx2_gfni -DENABLE_AESNI=ON .. && make -j && ctest

UOV (needs OpenSSL's libcrypto):

    cd compressed_uov/avx2
    make PROJ=Ip_pkc sign_api-test seeded_p3-test && ./sign_api-test && ./seeded_p3-test
    ./check_seeded_kat.sh    # regenerates the NIST KATs of the four pkc variants, checks seeded_kat.sha256

Pass extra defines through `CC` (e.g. `make "CC=gcc -D_OV_NO_SEEDED_P3" PROJ=Ip_pkc`), not
`CFLAGS`, which replaces the Makefile's flags.

## Reproducing the benchmarks

    bench/bench_cycles.sh 4 mayo_raw.txt avx2_gfni   # MAYO, AVX2+GFNI
    bench/bench_uov.sh 4 uov_raw.txt                  # UOV, AVX2, pkc variants

Arguments: core to pin to (default 4), optional file for the raw per-run results, and for
MAYO the build type (default `avx2`). Each script builds the committed `HEAD` twice in a
temporary directory (standard and modified key generation), runs the benchmarks pinned to
the core with `taskset`, and prints median core cycles for KeyGen, Sign and Verify. Since the
benchmarks read the TSC, every run is bracketed by `tsc_per_cycle`, which measures TSC ticks
per core cycle; counts are divided by the mean of the two measurements, and runs where it
moved by more than 2% are dropped. The paper's numbers were measured on an Intel Core Ultra
9 185H with gcc 13.3. Requires gcc, cmake, python3, taskset and libcrypto.
