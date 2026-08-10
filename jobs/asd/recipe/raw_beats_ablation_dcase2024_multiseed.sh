#!/bin/bash

# Run the frozen raw-BEATs ablation for seeds 0--4. Seed 0 performs extraction;
# later seeds reuse those exact deterministic embeddings and vary backend RNG.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/../../.." && pwd)
cd "${repo_root}"

name="${ASDKIT_ABLATION_NAME:-dcase2024_raw_beats_ablation}"
result_dir="${ASDKIT_RESULT_DIR:-./results}"
seeds=(0 1 2 3 4)
seed_zero_summary="${result_dir}/${name}/dcase2024/raw_beats_ablation_summary/0/last/summary.csv"

if [[ ! -f "${seed_zero_summary}" ]]; then
    echo "[multi-seed] seed 0 results are missing; running the full seed 0 study"
    ASDKIT_ABLATION_SEED=0 \
        bash jobs/asd/recipe/raw_beats_ablation_dcase2024.sh all all
else
    echo "[multi-seed] using existing seed 0 study: ${seed_zero_summary}"
fi

for seed in 1 2 3 4; do
    echo "[multi-seed] seed ${seed}; reusing deterministic seed 0 extraction"
    ASDKIT_ABLATION_SEED="${seed}" \
    ASDKIT_ABLATION_REUSE_SEED=0 \
        bash jobs/asd/recipe/raw_beats_ablation_dcase2024.sh all all
done

python -m asdkit.bin.summarize_raw_beats_ablation_dcase2024_multiseed \
    --result-dir "${result_dir}" --name "${name}" --seeds "${seeds[@]}"
