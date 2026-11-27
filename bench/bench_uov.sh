#!/bin/bash
# Core cycle counts for the UOV table: standard and seeded-P3 key generation, AVX2 UOV,
# compressed-public-key format. Same method as bench_cycles.sh: TSC counts divided by
# the TSC ticks per core cycle measured before and after every run; runs during which
# the clock moved by more than 2% are dropped.
#
# usage: bench/bench_uov.sh [core] [raw-output-file]   (builds HEAD into a temp directory)
set -euo pipefail
core=${1:-4}
raw=${2:-}
here=$(cd "$(dirname "$0")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
work=$(mktemp -d)
echo "building in $work" >&2

gcc -O2 -o "$work/tsc_per_cycle" "$here/tsc_per_cycle.c"
git -C "$repo" archive HEAD compressed_uov/avx2 | tar -x -C "$work"
src=$work/compressed_uov/avx2
for p in Is Ip III V; do
    for b in std seeded; do
        extra=""; [ $b = std ] && extra="-D_OV_NO_SEEDED_P3"
        gcc -O3 -std=c99 -mavx2 -maes $extra -I"$src/${p}_pkc" "$src/${p}_pkc"/*.c \
            "$here/bench_uov.c" -o "$work/b_${p}_$b" -lcrypto
    done
done

cal() { taskset -c "$core" "$work/tsc_per_cycle"; }
for p in Is Ip III V; do
    for rep in 1 2 3 4 5 6 7 8; do
        for b in std seeded; do
            r1=$(cal); line=$(taskset -c "$core" "$work/b_${p}_$b" 40); r2=$(cal)
            echo "$p $b cal=$r1,$r2 $line"
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
    runs[(p, b)].append({k: int(v) / r for k, v in (x.split('=') for x in kv)})
print(f"dropped {dropped} runs during which the clock changed")
for p in ['Is', 'Ip', 'III', 'V']:
    cells = []
    for k in ['keygen', 'sign', 'verify']:
        a = st.median(x[k] for x in runs[(p, 'std')]) / 1e6
        s = st.median(x[k] for x in runs[(p, 'seeded')]) / 1e6
        cells.append(f"{k} {a:6.3f}M -> {s:6.3f}M")
    print(f"uov-{p:4s}" + "   ".join(cells))
EOF
[ -n "$raw" ] && cp "$work/results.txt" "$raw"
rm -rf "$work"
