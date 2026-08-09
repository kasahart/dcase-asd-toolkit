#!/bin/bash
# ---------------------------- #
dcase="${1:-dcase2026}"
ablation="${2:-freq_rdp4_beam}"
beam_mode="${3:-varmin4}"
# ---------------------------- #

source ../../../venv/bin/activate
cd ../recipe
bash "raw_beats_beam_rdp4.sh" "${dcase}" "${ablation}" "${beam_mode}"
