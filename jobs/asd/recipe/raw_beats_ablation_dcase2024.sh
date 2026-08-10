#!/bin/bash

# Controlled DCASE 2024 frozen Original-BEATs ablation (B0--B5).
#
# Usage:
#   bash jobs/asd/recipe/raw_beats_ablation_dcase2024.sh [B0|...|B5|all] [machine|all]
#
# In the all-condition run, B1/B3 share the same AP extraction and B2/B4/B5
# share the same RDP(4) extraction.  Existing outputs are never overwritten.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/../../.." && pwd)
cd "${repo_root}"

condition_selector="${1:-all}"
machine_selector="${2:-all}"

dcase="dcase2024"
seed="${ASDKIT_ABLATION_SEED:-0}"
name="${ASDKIT_ABLATION_NAME:-dcase2024_raw_beats_ablation}"
infer_ver="${ASDKIT_ABLATION_INFER_VER:-last}"
result_dir="${ASDKIT_RESULT_DIR:-./results}"
data_dir="${ASDKIT_DATA_DIR:-../data}"
device="${ASDKIT_DEVICE:-cuda:0}"
reuse_seed="${ASDKIT_ABLATION_REUSE_SEED:-}"
checkpoint="pretrained_models/beats/BEATs_iter3.pt"

all_conditions=(B0 B1 B2 B3 B4 B5)
mapfile -t all_machines < <(
    python - <<'PY'
from asdkit.utils.dcase_utils import MACHINE_DICT
print(*(
    MACHINE_DICT["dcase2024-dev"] + MACHINE_DICT["dcase2024-eval"]
), sep="\n")
PY
)

case "${condition_selector}" in
    all) conditions=("${all_conditions[@]}") ;;
    B0|B1|B2|B3|B4|B5) conditions=("${condition_selector}") ;;
    *)
        echo "condition must be B0, B1, B2, B3, B4, B5, or all" >&2
        exit 2
        ;;
esac

if [[ "${machine_selector}" == "all" ]]; then
    machines=("${all_machines[@]}")
elif printf '%s\n' "${all_machines[@]}" | grep -Fxq -- "${machine_selector}"; then
    machines=("${machine_selector}")
else
    echo "Unknown DCASE 2024 machine: ${machine_selector}" >&2
    exit 2
fi

if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing BEATs checkpoint: ${checkpoint}" >&2
    exit 1
fi
if [[ ! -d "${data_dir}/formatted/${dcase}/raw" ]]; then
    echo "Missing formatted DCASE 2024 data: ${data_dir}/formatted/${dcase}/raw" >&2
    exit 1
fi

version_for() {
    printf 'raw_beats_ablation_%s' "$1"
}

frontend_for() {
    case "$1" in
        B0) printf 'scratch/raw_beats' ;;
        B1|B3) printf 'scratch/raw_beats_freq_ap' ;;
        B2|B4|B5) printf 'scratch/raw_beats_freq_rdp4' ;;
    esac
}

backend_for() {
    case "$1" in
        B0|B1|B2) printf 'knn_raw_beats' ;;
        B3|B4) printf 'beam_raw' ;;
        B5) printf 'beam_varmin4' ;;
    esac
}

output_dir_for() {
    local condition_id=$1
    local machine=$2
    local output_seed=${3:-${seed}}
    printf '%s/%s/%s/%s/%s/output/%s/%s' \
        "${result_dir}" "${name}" "${dcase}" "$(version_for "${condition_id}")" \
        "${output_seed}" "${infer_ver}" "${machine}"
}

run_extract() {
    local condition_id=$1
    local machine=$2
    local version
    version=$(version_for "${condition_id}")
    python -m asdkit.bin.extract \
        "name=${name}" "version=${version}" "dcase=${dcase}" "seed=${seed}" \
        "infer_ver=${infer_ver}" "machine=${machine}" \
        "result_dir=${result_dir}" "data_dir=${data_dir}" "device=${device}" \
        "experiments=$(frontend_for "${condition_id}")" \
        "datamodule.train.collator.sec=dcase2024" \
        "+datamodule.train.collator.pad_mode=tile" \
        "datamodule.train.dataset.audio_channel=first"
}

link_extraction() {
    local source_id=$1
    local target_id=$2
    local machine=$3
    local source_dir target_dir artifact
    source_dir=$(output_dir_for "${source_id}" "${machine}")
    target_dir=$(output_dir_for "${target_id}" "${machine}")
    mkdir -p "${target_dir}"
    for artifact in train_extract.npz test_extract.npz .hydra_extract extract.log; do
        if [[ ! -e "${source_dir}/${artifact}" ]]; then
            echo "Missing extraction artifact: ${source_dir}/${artifact}" >&2
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
            echo "Extraction sharing check failed for ${target_id} ${machine}" >&2
            exit 1
        fi
    done
}

link_seed_extraction() {
    local condition_id=$1
    local machine=$2
    local source_seed=$3
    local source_dir target_dir artifact
    source_dir=$(output_dir_for "${condition_id}" "${machine}" "${source_seed}")
    target_dir=$(output_dir_for "${condition_id}" "${machine}")
    mkdir -p "${target_dir}"
    for artifact in train_extract.npz test_extract.npz .hydra_extract extract.log; do
        if [[ ! -e "${source_dir}/${artifact}" ]]; then
            echo "Missing seed-${source_seed} extraction artifact: ${source_dir}/${artifact}" >&2
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
            echo "Cross-seed extraction sharing check failed for ${condition_id} ${machine}" >&2
            exit 1
        fi
    done
}

link_seed_deterministic_scores() {
    local condition_id=$1
    local machine=$2
    local source_seed=$3
    local source_dir target_dir artifact
    source_dir=$(output_dir_for "${condition_id}" "${machine}" "${source_seed}")
    target_dir=$(output_dir_for "${condition_id}" "${machine}")
    for artifact in \
        train_score.csv test_score.csv test_evaluate.csv \
        .hydra_score score.log .hydra_evaluate evaluate.log; do
        if [[ ! -e "${source_dir}/${artifact}" ]]; then
            echo "Missing deterministic seed-${source_seed} artifact: ${source_dir}/${artifact}" >&2
            exit 1
        fi
        if [[ -e "${target_dir}/${artifact}" || -L "${target_dir}/${artifact}" ]]; then
            echo "Refusing to overwrite ${target_dir}/${artifact}" >&2
            exit 1
        fi
        ln -s "$(realpath "${source_dir}/${artifact}")" "${target_dir}/${artifact}"
    done
    for artifact in train_score.csv test_score.csv test_evaluate.csv; do
        if [[ ! "${source_dir}/${artifact}" -ef "${target_dir}/${artifact}" ]]; then
            echo "Deterministic score sharing check failed for ${condition_id} ${machine}" >&2
            exit 1
        fi
    done
}

run_score_and_evaluate() {
    local condition_id=$1
    local machine=$2
    local version
    version=$(version_for "${condition_id}")
    python -m asdkit.bin.score \
        "name=${name}" "version=${version}" "dcase=${dcase}" "seed=${seed}" \
        "infer_ver=${infer_ver}" "machine=${machine}" \
        "result_dir=${result_dir}" "experiments=$(backend_for "${condition_id}")"
    python -m asdkit.bin.evaluate \
        "name=${name}" "version=${version}" "dcase=${dcase}" "seed=${seed}" \
        "infer_ver=${infer_ver}" "machine=${machine}" "result_dir=${result_dir}"
}

run_table() {
    local condition_id=$1
    local version
    version=$(version_for "${condition_id}")
    python -m asdkit.bin.table \
        "name=${name}" "version=${version}" "dcase=${dcase}" "seed=${seed}" \
        "infer_ver=${infer_ver}" "result_dir=${result_dir}"
}

echo "DCASE=${dcase} conditions=${conditions[*]} machines=${machines[*]}"
echo "controls: seed=${seed} device=${device} sec=12 pad_mode=tile channel=first"
echo "checkpoint_sha256=$(sha256sum "${checkpoint}" | cut -d' ' -f1)"

if [[ -n "${reuse_seed}" && "${reuse_seed}" == "${seed}" ]]; then
    echo "ASDKIT_ABLATION_REUSE_SEED must differ from ASDKIT_ABLATION_SEED" >&2
    exit 2
fi

if [[ "${condition_selector}" == "all" ]]; then
    for machine in "${machines[@]}"; do
        if [[ -n "${reuse_seed}" ]]; then
            for condition_id in "${all_conditions[@]}"; do
                echo "[reuse extraction] ${machine}: ${condition_id} seed ${reuse_seed} -> ${seed}"
                link_seed_extraction "${condition_id}" "${machine}" "${reuse_seed}"
            done
        else
            echo "[extract] ${machine}: B0 global AP"
            run_extract B0 "${machine}"
            echo "[extract] ${machine}: B1/B3 shared frequency AP"
            run_extract B1 "${machine}"
            echo "[extract] ${machine}: B2/B4/B5 shared frequency RDP(4)"
            run_extract B2 "${machine}"
            link_extraction B1 B3 "${machine}"
            link_extraction B2 B4 "${machine}"
            link_extraction B2 B5 "${machine}"
        fi
        for condition_id in "${conditions[@]}"; do
            if [[ -n "${reuse_seed}" && "${condition_id}" =~ ^B[345]$ ]]; then
                echo "[reuse deterministic score/evaluate] ${machine}: ${condition_id} seed ${reuse_seed} -> ${seed}"
                link_seed_deterministic_scores "${condition_id}" "${machine}" "${reuse_seed}"
            else
                echo "[score/evaluate] ${machine}: ${condition_id}"
                run_score_and_evaluate "${condition_id}" "${machine}"
            fi
        done
    done
else
    for machine in "${machines[@]}"; do
        echo "[extract] ${machine}: ${condition_selector}"
        run_extract "${condition_selector}" "${machine}"
        echo "[score/evaluate] ${machine}: ${condition_selector}"
        run_score_and_evaluate "${condition_selector}" "${machine}"
    done
fi

if [[ "${machine_selector}" == "all" ]]; then
    for condition_id in "${conditions[@]}"; do
        echo "[table] ${condition_id}"
        run_table "${condition_id}"
    done
else
    echo "Skipping ASDKit table: a one-machine smoke run is intentionally incomplete."
fi

if [[ "${condition_selector}" == "all" && "${machine_selector}" == "all" ]]; then
    python -m asdkit.bin.summarize_raw_beats_ablation_dcase2024 \
        --result-dir "${result_dir}" --name "${name}" --seed "${seed}" \
        --infer-ver "${infer_ver}" --checkpoint "${checkpoint}"
else
    echo "Skipping six-condition summary until all conditions and machines exist."
fi
