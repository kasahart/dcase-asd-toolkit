#!/bin/bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/../../.." && pwd)

cd "${repo_root}"
bash jobs/asd/recipe/raw_beats_ablation_dcase2024.sh "${1:-all}" "${2:-all}"
