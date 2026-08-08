#!/bin/bash
# ---------------------------- #
dcase="dcase2023"
# ---------------------------- #

source ../../../venv/bin/activate
cd ../recipe
bash "raw_beats_beam_rdp4.sh" "${dcase}"
