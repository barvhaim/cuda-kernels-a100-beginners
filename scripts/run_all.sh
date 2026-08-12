#!/usr/bin/env bash
set -euo pipefail
build_dir="${1:-build}"
for lesson in 00_device_query 01_vector_add 02_grid_stride 03_reduction 04_tiled_matmul; do
  printf '
== %s ==
' "$lesson"
  "$build_dir/$lesson"
done
