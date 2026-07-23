#!/usr/bin/env bash
# Prepare runnable GGUFs from Thireus SPECIAL_SPLIT distribution repos for ik_llama.cpp.
# Requires: ik_llama.cpp built with -DGGML_MAX_CONTEXTS>=2048 (for >64-shard merges),
#           conda env with gguf-py (see scripts/assemble_mtp_model.py).
# Usage: prepare_models.sh <ik_build_bin_dir> <general_first_shard> <mtp_first_shard> <out_dir>
set -euo pipefail
BIN="$1"; GEN1="$2"; MTP1="$3"; OUT="${4:-./.models}"
mkdir -p "$OUT"
export LD_LIBRARY_PATH="$(find "$BIN/.." -name '*.so' -printf '%h\n' 2>/dev/null | sort -u | tr '\n' ':')${LD_LIBRARY_PATH:-}"
echo "[1/3] merge general split -> single gguf"
"$BIN/llama-gguf-split" --merge "$GEN1" "$OUT/general.gguf"
echo "[2/3] merge mtp split -> single gguf"
"$BIN/llama-gguf-split" --merge "$MTP1" "$OUT/mtp-half.gguf"
echo "[3/3] assemble combined NextN model (needs gguf-py env active)"
python "$(dirname "$0")/assemble_mtp_model.py" "$OUT/general.gguf" "$OUT/mtp-half.gguf" "$OUT/mtp-combined.gguf"
echo "done: $OUT/general.gguf  $OUT/mtp-combined.gguf"
