#!/bin/bash

# Frozen BEATs_iter3 reproduction ablations with optional BEAM + VarMin (K=4).

# ---------------------------- #
dcase=$1
ablation="${2:-freq_rdp4_beam}"
beam_mode="${3:-varmin4}"
seed=0
name="recipe"
version="raw_beats_${ablation}"
infer_ver="last"
# ---------------------------- #
extract_overrides=()
case "${ablation}" in
    global_ap)
        experiments_extract="scratch/raw_beats"
        experiments_score="default"
        ;;
    freq_ap)
        experiments_extract="scratch/raw_beats_freq_ap"
        experiments_score="default"
        ;;
    freq_ap_beam)
        experiments_extract="scratch/raw_beats_freq_ap"
        experiments_score="beam_${beam_mode}"
        ;;
    freq_rdp4_beam)
        experiments_extract="scratch/raw_beats_freq_rdp4"
        experiments_score="beam_${beam_mode}"
        ;;
    freq_rdp8_beam)
        experiments_extract="scratch/raw_beats_freq_rdp4"
        experiments_score="beam_${beam_mode}"
        extract_overrides=("scratch_frontend.gamma=8")
        ;;
    *)
        echo "Unknown ablation: ${ablation}" >&2
        exit 1
        ;;
esac

if [[ "${ablation}" == *_beam ]] && [[ "${beam_mode}" != "raw" && "${beam_mode}" != "varmin4" ]]; then
    echo "beam_mode must be raw or varmin4, got: ${beam_mode}" >&2
    exit 1
fi

if [[ "${ablation}" == *_beam ]]; then
    version="${version}_${beam_mode}"
fi

if [ "${dcase}" = "dcase2026" ]; then
    extract_overrides+=(
        "datamodule.train.collator.sec=all"
        "datamodule.train.dataloader.batch_size=1"
    )
fi
# ---------------------------- #
source ../base/base.sh

for machine in $machines; do
    asdkit_extract experiments="${experiments_extract}" "${extract_overrides[@]}"
    asdkit_score experiments="${experiments_score}"
    asdkit_evaluate
    asdkit_visualize
done
asdkit_table
