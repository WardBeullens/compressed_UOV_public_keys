#!/bin/sh
# Regenerates the NIST KATs of the seeded-P3 pkc variants and checks them against the
# SHA-256 hashes in seeded_kat.sha256 (the files themselves are 60 MB to 620 MB each).
# Run from this directory; builds in a temporary copy.
set -e
here=$(pwd)
work=$(mktemp -d)
cp -r . "$work/src"
cd "$work/src"
status=0
while read -r hash proj; do
    make clean >/dev/null 2>&1 || true
    rm -f PQCsignKAT_* PQCgenKAT_sign
    make CC="${CC:-gcc}" PROJ="$proj" KAT=1 PQCgenKAT_sign >/dev/null
    ./PQCgenKAT_sign >/dev/null
    got=$(sha256sum PQCsignKAT_*.rsp | cut -c1-64)
    if [ "$got" = "$hash" ]; then echo "$proj: KAT ok"; else echo "$proj: KAT MISMATCH"; status=1; fi
done < "$here/seeded_kat.sha256"
cd "$here"
rm -rf "$work"
exit $status
