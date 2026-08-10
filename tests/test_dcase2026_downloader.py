import os
import re
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "jobs/download/base/dcase2026.sh"
DEV_MACHINES = [
    "ToyCar_r2",
    "ToyCarEmu",
    "bearingEmu",
    "fan",
    "gearboxEmu",
    "sliderEmu",
    "valveEmu",
]
EVAL_MACHINES = [
    "BlowerDustCollector",
    "Sander",
    "SewingMachine",
    "ToothBrush",
    "ToyDrone",
]


def _write_executable(path, contents):
    path.write_text(contents)
    path.chmod(0o755)


def _prepare_download_tree(tmp_path, *, partial_archive=None):
    data_dir = tmp_path / "original"
    dev_dir = data_dir / "dcase2026/dev_data/raw"
    eval_dir = data_dir / "dcase2026/eval_data/raw"
    evaluator_dir = data_dir / "dcase2026/evaluator"
    dev_dir.mkdir(parents=True)
    eval_dir.mkdir(parents=True)
    (evaluator_dir / ".git").mkdir(parents=True)

    for machine in DEV_MACHINES:
        archive = dev_dir / f"dev_{machine}.zip"
        archive.write_text("partial" if archive.name == partial_archive else "complete")
    for machine in EVAL_MACHINES:
        (eval_dir / f"eval_data_{machine}_train.zip").write_text("complete")
        (eval_dir / f"eval_data_{machine}_test.zip").write_text("complete")
        ground_truth = (
            evaluator_dir
            / "ground_truth_attributes"
            / f"ground_truth_{machine}_section_00_test.csv"
        )
        ground_truth.parent.mkdir(exist_ok=True)
        ground_truth.write_text("anonymous.wav,labeled_normal.wav\n")

    return data_dir, evaluator_dir


def _fake_command_environment(
    tmp_path, *, evaluator_dirty=False, resume_produces_valid_archive=True
):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log_path = tmp_path / "commands.log"
    revision = re.search(
        r'DCASE2026_EVALUATOR_REV="([0-9a-f]{40})"', SCRIPT.read_text()
    ).group(1)

    _write_executable(
        fake_bin / "curl",
        """#!/bin/sh
set -eu
printf 'curl %s\n' "$*" >> "${DCASE_TEST_LOG}"
archive=
resume=0
while [ "$#" -gt 0 ]; do
  if [ "$1" = "--output" ]; then
    shift
    archive=$1
  elif [ "$1" = "--continue-at" ]; then
    resume=1
    shift
  fi
  shift
done
if [ "${resume}" = 1 ] && [ "${DCASE_TEST_RESUME_VALID}" = 0 ]; then
  exit 0
fi
printf complete > "${archive}"
""",
    )
    _write_executable(
        fake_bin / "unzip",
        """#!/bin/sh
set -eu
if [ "$1" = "-tq" ]; then
  [ "$(cat "$2")" = complete ]
else
  archive=$2
  destination=$4
  printf 'unzip %s\n' "${archive}" >> "${DCASE_TEST_LOG}"
  [ "$(cat "${archive}")" = complete ]
  payload_dir="${destination}/${archive%.zip}"
  mkdir -p "${payload_dir}"
  printf complete > "${payload_dir}/payload.wav"
fi
""",
    )
    _write_executable(
        fake_bin / "git",
        """#!/bin/sh
set -eu
if [ "$1" = "-C" ]; then
  shift 2
fi
case "$1" in
  status)
    if [ "${DCASE_TEST_DIRTY:-0}" = 1 ]; then
      printf ' M ground_truth_attributes/ground_truth_ToyDrone_section_00_test.csv\n'
    fi
    ;;
  rev-parse)
    printf '%s\n' "${DCASE_TEST_REV}"
    ;;
esac
""",
    )

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "DCASE_TEST_DIRTY": "1" if evaluator_dirty else "0",
            "DCASE_TEST_LOG": str(log_path),
            "DCASE_TEST_REV": revision,
            "DCASE_TEST_RESUME_VALID": (
                "1" if resume_produces_valid_archive else "0"
            ),
        }
    )
    return env, log_path


def test_downloader_resumes_partial_archive_before_extracting(tmp_path):
    archive_name = "dev_ToyCar_r2.zip"
    data_dir, _ = _prepare_download_tree(
        tmp_path, partial_archive=archive_name
    )
    interrupted_output = (
        data_dir
        / "dcase2026/dev_data/raw"
        / archive_name.removesuffix(".zip")
        / "payload.wav"
    )
    interrupted_output.parent.mkdir()
    interrupted_output.write_text("partial")
    env, log_path = _fake_command_environment(tmp_path)

    result = subprocess.run(
        ["bash", str(SCRIPT), str(data_dir)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    log = log_path.read_text().splitlines()
    curl_entries = [entry for entry in log if entry.startswith("curl ")]
    assert len(curl_entries) == 1
    assert "--continue-at -" in curl_entries[0]
    assert f"--output {archive_name}" in curl_entries[0]
    assert log.index(curl_entries[0]) < log.index(f"unzip {archive_name}")
    assert (
        data_dir / "dcase2026/dev_data/raw" / archive_name
    ).read_text() == "complete"
    assert interrupted_output.read_text() == "complete"


def test_downloader_restarts_corrupt_archive_from_zero(tmp_path):
    archive_name = "dev_ToyCar_r2.zip"
    data_dir, _ = _prepare_download_tree(
        tmp_path, partial_archive=archive_name
    )
    archive = data_dir / "dcase2026/dev_data/raw" / archive_name
    archive.write_text("corrupt-full-length")
    env, log_path = _fake_command_environment(
        tmp_path, resume_produces_valid_archive=False
    )

    result = subprocess.run(
        ["bash", str(SCRIPT), str(data_dir)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    curl_entries = [
        entry
        for entry in log_path.read_text().splitlines()
        if entry.startswith("curl ")
    ]
    assert len(curl_entries) == 2
    assert "--continue-at -" in curl_entries[0]
    assert "--continue-at" not in curl_entries[1]
    assert f"--output {archive_name}.fresh" in curl_entries[1]
    assert archive.read_text() == "complete"


def test_downloader_rejects_dirty_pinned_evaluator(tmp_path):
    data_dir, evaluator_dir = _prepare_download_tree(tmp_path)
    changed_csv = (
        evaluator_dir
        / "ground_truth_attributes/ground_truth_ToyDrone_section_00_test.csv"
    )
    changed_csv.write_text("locally modified\n")
    env, _ = _fake_command_environment(tmp_path, evaluator_dirty=True)

    result = subprocess.run(
        ["bash", str(SCRIPT), str(data_dir)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode != 0
    assert "Evaluator directory has local changes" in result.stderr
    assert changed_csv.read_text() == "locally modified\n"
