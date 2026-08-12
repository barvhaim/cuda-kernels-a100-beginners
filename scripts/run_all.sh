#!/usr/bin/env bash
set -euo pipefail
build_dir="${1:-build}"
track="${2:-all}"

foundations=(
  00_device_query 01_hello_kernel 02_one_block_index 03_global_index
  04_bounds_check 05_memory_roundtrip 06_vector_add
)
patterns=(07_grid_stride 08_reduction 09_tiled_matmul)
llm=(
  llm_01_token_embedding llm_02_residual_add llm_03_silu_activation
  llm_04_rmsnorm llm_05_causal_mask llm_06_attention_softmax
  llm_07_linear_projection llm_08_mini_transformer_step
)

case "$track" in
  foundations) lessons=("${foundations[@]}") ;;
  patterns) lessons=("${patterns[@]}") ;;
  basics) lessons=("${foundations[@]}" "${patterns[@]}") ;;
  llm) lessons=("${llm[@]}") ;;
  all) lessons=("${foundations[@]}" "${patterns[@]}" "${llm[@]}") ;;
  *) echo "Usage: $0 [build-dir] [foundations|patterns|basics|llm|all]" >&2; exit 2 ;;
esac

for lesson in "${lessons[@]}"; do
  printf '\n== %s ==\n' "$lesson"
  "$build_dir/$lesson"
done
