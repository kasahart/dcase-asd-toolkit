#!/bin/bash

# Deterministic DCASE 2024 follow-up: A6, coupled AP, and coupled RDP.
# Usage: bash jobs/asd/recipe/raw_beats_ablation_dcase2024_followup.sh [machine|all]

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/../../.." && pwd)
cd "${repo_root}"

machine_selector="${1:-all}"
dcase="dcase2024"
seed="${ASDKIT_ABLATION_SEED:-0}"
name="${ASDKIT_ABLATION_NAME:-dcase2024_raw_beats_ablation}"
infer_ver="${ASDKIT_ABLATION_INFER_VER:-last}"
result_dir="${ASDKIT_RESULT_DIR:-./results}"
conditions=(A6 CAP CRDP)

mapfile -t all_machines < <(
    python - <<'PY'
from asdkit.utils.dcase_utils import MACHINE_DICT
print(*(MACHINE_DICT["dcase2024-dev"] + MACHINE_DICT["dcase2024-eval"]), sep="\n")
PY
)
if [[ "${machine_selector}" == "all" ]]; then
    machines=("${all_machines[@]}")
elif printf '%s\n' "${all_machines[@]}" | grep -Fxq -- "${machine_selector}"; then
    machines=("${machine_selector}")
else
    echo "Unknown DCASE 2024 machine: ${machine_selector}" >&2
    exit 2
fi

version_for() { printf 'raw_beats_ablation_%s' "$1"; }
source_for() {
    case "$1" in
        A6|CAP) printf 'B1' ;;
        CRDP) printf 'B2' ;;
    esac
}
backend_for() {
    case "$1" in
        A6) printf 'beam_varmin4' ;;
        CAP|CRDP) printf 'beam_coupled_raw' ;;
    esac
}
output_dir_for() {
    printf '%s/%s/%s/%s/%s/output/%s/%s' \
        "${result_dir}" "${name}" "${dcase}" "$(version_for "$1")" \
        "${seed}" "${infer_ver}" "$2"
}

link_extraction() {
    local target_id=$1 machine=$2 source_id source_dir target_dir artifact
    source_id=$(source_for "${target_id}")
    source_dir=$(output_dir_for "${source_id}" "${machine}")
    target_dir=$(output_dir_for "${target_id}" "${machine}")
    mkdir -p "${target_dir}"
    for artifact in train_extract.npz test_extract.npz .hydra_extract extract.log; do
        if [[ ! -e "${source_dir}/${artifact}" ]]; then
            echo "Missing primary extraction: ${source_dir}/${artifact}" >&2
            exit 1
        fi
        if [[ -e "${target_dir}/${artifact}" || -L "${target_dir}/${artifact}" ]]; then
            echo "Refusing to overwrite ${target_dir}/${artifact}" >&2
            exit 1
        fi
        ln -s "$(realpath "${source_dir}/${artifact}")" "${target_dir}/${artifact}"
    done
    for artifact in train_extract.npz test_extract.npz; do
        if [[ ! "${source_dir}/${artifact}" -ef "${target_dir}/${artifact}" ]]; then
            echo "Extraction sharing failed for ${target_id} ${machine}" >&2
            exit 1
        fi
    done
}

for machine in "${machines[@]}"; do
    for condition_id in "${conditions[@]}"; do
        echo "[follow-up] ${machine}: ${condition_id}"
        link_extraction "${condition_id}" "${machine}"
        version=$(version_for "${condition_id}")
        python -m asdkit.bin.score \
            "name=${name}" "version=${version}" "dcase=${dcase}" "seed=${seed}" \
            "infer_ver=${infer_ver}" "machine=${machine}" \
            "result_dir=${result_dir}" "experiments=$(backend_for "${condition_id}")"
        python -m asdkit.bin.evaluate \
            "name=${name}" "version=${version}" "dcase=${dcase}" "seed=${seed}" \
            "infer_ver=${infer_ver}" "machine=${machine}" "result_dir=${result_dir}"
    done
done

if [[ "${machine_selector}" == "all" ]]; then
    for condition_id in "${conditions[@]}"; do
        version=$(version_for "${condition_id}")
        python -m asdkit.bin.table \
            "name=${name}" "version=${version}" "dcase=${dcase}" "seed=${seed}" \
            "infer_ver=${infer_ver}" "result_dir=${result_dir}"
    done
    python -m asdkit.bin.summarize_raw_beats_ablation_dcase2024_followup \
        --result-dir "${result_dir}" --name "${name}" --seed "${seed}" \
        --infer-ver "${infer_ver}"
else
    echo "Skipping table and follow-up summary for one-machine smoke run."
fi
