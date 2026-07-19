#!/usr/bin/env bash
# Runs the benchmark across row sizes and all backends, emits CSV to stdout.
set -e
cd "$(dirname "$0")/.."
make bench >/dev/null 2>&1
echo "backend,words,kib,kernel,gbps"
for words in 64 512 4096 65536 1048576; do
  # keep total work ~constant: iters scaled by size
  iters=$(( 800000000 / (words>64?words:64) ))
  for b in scalar avx2 avx512; do
    STABKERNEL_BACKEND=$b ./bench_run $words $iters | awk -v b=$b -v w=$words '
      /KiB/ { match($0,/\(([0-9.]+) KiB\)/,m); kib=m[1] }
      /gf2_/ { print b "," w "," kib "," $1 "," $(NF-1) }'
  done
done
