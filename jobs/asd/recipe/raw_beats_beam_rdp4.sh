#!/bin/bash

# Frozen BEATs_iter3 + frequency-wise RDP (gamma=4) + BEAM + VarMin (K=4)

# ---------------------------- #
dcase=$1
seed=0
name="recipe"
version="raw_beats_beam_rdp4"
infer_ver="last"
# ---------------------------- #
experiments_extract="scratch/raw_beats_freq_rdp4"
experiments_score="beam_varmin4"
# ---------------------------- #
source ../base/base.sh

for machine in $machines; do
    asdkit_extract experiments="${experiments_extract}"
    asdkit_score experiments="${experiments_score}"
    asdkit_evaluate
    asdkit_visualize
done
asdkit_table
