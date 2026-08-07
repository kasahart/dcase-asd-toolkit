dst_dir=$1

set -euo pipefail

dev_dir="${dst_dir}/dcase2026/dev_data/raw"
eval_dir="${dst_dir}/dcase2026/eval_data/raw"
mkdir -p "${dev_dir}"
mkdir -p "${eval_dir}"

# Download development data.
cd "${dev_dir}"
for machine_type in \
    ToyCar_r2 \
    ToyCarEmu \
    bearingEmu \
    fan \
    gearboxEmu \
    sliderEmu \
    valveEmu \
; do
curl -L -C - -O "https://zenodo.org/records/19336329/files/dev_${machine_type}.zip"
unzip -n "dev_${machine_type}.zip"
done

# Download additional training data for the evaluation machines.
cd "${eval_dir}"
for machine_type in \
    BlowerDustCollector \
    Sander \
    SewingMachine \
    ToothBrush \
    ToyDrone \
; do
curl -L -C - -O "https://zenodo.org/records/20151556/files/eval_data_${machine_type}_train.zip"
unzip -n "eval_data_${machine_type}_train.zip"
done

# Download evaluation test data.
for machine_type in \
    BlowerDustCollector \
    Sander \
    SewingMachine \
    ToothBrush \
    ToyDrone \
; do
curl -L -C - -O "https://zenodo.org/records/20437238/files/eval_data_${machine_type}_test.zip"
unzip -n "eval_data_${machine_type}_test.zip"
done
