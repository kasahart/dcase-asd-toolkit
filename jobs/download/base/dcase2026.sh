dst_dir=$1

set -euo pipefail

dev_dir="${dst_dir}/dcase2026/dev_data/raw"
eval_dir="${dst_dir}/dcase2026/eval_data/raw"
evaluator_dir="${dst_dir}/dcase2026/evaluator"
DCASE2026_EVALUATOR_REV="f6a94a2b5e614a9626c9d1ccff6df0705e6aaa75"
evaluator_url="https://github.com/nttcslab/dcase2026_task2_evaluator.git"
mkdir -p "${dev_dir}"
mkdir -p "${eval_dir}"

download_archive() {
  archive=$1
  url=$2

  if ! unzip -tq "${archive}" >/dev/null 2>&1; then
    curl --fail --location --continue-at - --output "${archive}" "${url}"
  fi
  if ! unzip -tq "${archive}" >/dev/null 2>&1; then
    echo "Incomplete or corrupt archive after download: ${archive}" >&2
    return 1
  fi
  unzip -n "${archive}"
}

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
archive="dev_${machine_type}.zip"
download_archive \
  "${archive}" \
  "https://zenodo.org/records/19336329/files/${archive}"
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
archive="eval_data_${machine_type}_train.zip"
download_archive \
  "${archive}" \
  "https://zenodo.org/records/20151556/files/${archive}"
done

# Download evaluation test data.
for machine_type in \
    BlowerDustCollector \
    Sander \
    SewingMachine \
    ToothBrush \
    ToyDrone \
; do
archive="eval_data_${machine_type}_test.zip"
download_archive \
  "${archive}" \
  "https://zenodo.org/records/20437238/files/${archive}"
done

# Download the official post-challenge evaluation labels and evaluator.
if [ ! -e "${evaluator_dir}" ]; then
  git init "${evaluator_dir}"
  git -C "${evaluator_dir}" remote add origin "${evaluator_url}"
elif [ ! -d "${evaluator_dir}/.git" ]; then
  echo "Incomplete evaluator directory (not a git repository): ${evaluator_dir}" >&2
  exit 1
else
  git -C "${evaluator_dir}" remote set-url origin "${evaluator_url}"
fi

if [ -n "$(git -C "${evaluator_dir}" status --porcelain --untracked-files=all)" ]; then
  echo "Evaluator directory has local changes; refusing to use or overwrite it: ${evaluator_dir}" >&2
  exit 1
fi

git -C "${evaluator_dir}" fetch --depth 1 origin "${DCASE2026_EVALUATOR_REV}"
git -C "${evaluator_dir}" checkout --detach FETCH_HEAD
actual_evaluator_rev=$(git -C "${evaluator_dir}" rev-parse HEAD)
echo "DCASE 2026 evaluator revision: ${actual_evaluator_rev}"
if [ "${actual_evaluator_rev}" != "${DCASE2026_EVALUATOR_REV}" ]; then
  echo "Unexpected evaluator revision: ${actual_evaluator_rev}" >&2
  exit 1
fi
if [ -n "$(git -C "${evaluator_dir}" status --porcelain --untracked-files=all)" ]; then
  echo "Evaluator checkout is not clean: ${evaluator_dir}" >&2
  exit 1
fi

for machine_type in \
    BlowerDustCollector \
    Sander \
    SewingMachine \
    ToothBrush \
    ToyDrone \
; do
  ground_truth_csv="${evaluator_dir}/ground_truth_attributes/ground_truth_${machine_type}_section_00_test.csv"
  if [ ! -s "${ground_truth_csv}" ]; then
    echo "Missing evaluator ground truth: ${ground_truth_csv}" >&2
    exit 1
  fi
done
