#!/bin/bash
# Core cycle counts for Table 2: standard and seeded-P3 key generation, AVX2+GFNI MAYO.
#
# mayo_bench reads the TSC, which ticks at a fixed rate whatever the core clock does
# (turbo, power limits, battery). Every run is therefore bracketed by tsc_per_cycle,
# which measures TSC ticks per core cycle, and the counts are divided by the mean of the
# two; runs during which the clock moved by more than 2% are dropped.
#
# usage: bench/bench_cycles.sh [core] [raw-output-file] [build-type]
#        (builds HEAD into a temporary directory; build type avx2_gfni by default)
set -euo pipefail
core=${1:-4}
raw=${2:-}
btype=${3:-avx2_gfni}
here=$(cd "$(dirname "$0")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
work=$(mktemp -d)
echo "building in $work" >&2

gcc -O2 -o "$work/tsc_per_cycle" "$here/tsc_per_cycle.c"
for b in base seeded; do
    git -C "$repo" archive HEAD compressed_mayo | tar -x -C "$work"
    mv "$work/compressed_mayo" "$work/$b"
done
sed -i 's/^#define MAYO_SEEDED_P3 1$/\/\/ standard key generation/' "$work/base/include/mayo_constants.h"
grep -q "standard key generation" "$work/base/include/mayo_constants.h"
for b in base seeded; do
    mkdir "$work/$b/build"
    (cd "$work/$b/build" && cmake -DMAYO_BUILD_TYPE="$btype" -DENABLE_AESNI=ON .. >/dev/null &&
     make -j"$(nproc)" 'mayo_bench_MAYO_1' 'mayo_bench_MAYO_2' 'mayo_bench_MAYO_3' 'mayo_bench_MAYO_5' >/dev/null)
done

cal() { taskset -c "$core" "$work/tsc_per_cycle"; }
for v in 1 2 3 5; do
    for rep in 1 2 3 4 5 6 7 8; do
        for b in base seeded; do
            r1=$(cal)
            line=$(taskset -c "$core" "$work/$b/build/test/mayo_bench_MAYO_$v" 1500 2>&1 |
                   awk '/->/ {gsub(",","",$4); printf "%s=%s ", $1, $4}')
            r2=$(cal)
            echo "MAYO_$v $b cal=$r1,$r2 $line"
        done
    done
done > "$work/results.txt"

python3 - "$work/results.txt" <<'EOF'
import collections, statistics as st, sys
runs = collections.defaultdict(list)
dropped = 0
for line in open(sys.argv[1]):
    p, b, cal, *kv = line.split()
    c1, c2 = map(float, cal.split('=')[1].split(','))
    if abs(c1 - c2) / ((c1 + c2) / 2) > 0.02:
        dropped += 1
        continue
    r = (c1 + c2) / 2
    runs[(p, b)].append({k.replace('mayo_', ''): int(v) / r for k, v in (x.split('=') for x in kv)})
print(f"dropped {dropped} runs during which the clock changed")
print(f"{'':7s}" + "".join(f"{k:>22s}" for k in ['keypair', 'expand_pk', 'sign', 'verify']))
for p in ['MAYO_1', 'MAYO_2', 'MAYO_3', 'MAYO_5']:
    cells = []
    for k in ['keypair', 'expand_pk', 'sign', 'verify']:
        a = st.median(x[k] for x in runs[(p, 'base')]) / 1000
        s = st.median(x[k] for x in runs[(p, 'seeded')]) / 1000
        cells.append(f"{a:9.1f}k -> {s:7.1f}k")
    print(f"{p:7s}" + "".join(f"{c:>22s}" for c in cells))
EOF
[ -n "$raw" ] && cp "$work/results.txt" "$raw"
rm -rf "$work"
