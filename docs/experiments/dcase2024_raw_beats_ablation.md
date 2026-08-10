# DCASE 2024 frozen raw-BEATs ablation

This experiment isolates representation, temporal pooling, band-wise scoring,
and variance-minimum rescaling around the same frozen Original BEATs_iter3
encoder. It does not include NA-BEATs, LoRA, pseudo labels, ensembles, denoising,
or two-microphone processing.

## Fixed controls

All six conditions use DCASE 2024 Task 2, seed 0, mono/first channel at 16 kHz,
a 12-second waveform crop with the existing tile-padding policy, the official
DCASE 2024 machine order and train/test split, and
`pretrained_models/beats/BEATs_iter3.pt`. The encoder remains frozen. ASDKit's
same `official24` evaluation and table code is used for every condition.

The KNN conditions use the repository's published raw-BEATs KNN-with-SMOTE
entry without changing its behavior:

- cosine mode (L2 normalize, Euclidean neighbor distance, divide by 2)
- one source-domain neighbor and one target-domain neighbor
- take the smaller source/target score
- target SMOTE ratio 0.2 with two SMOTE neighbors and the experiment seed
- `sep_section=false`, so one backend is fitted across section 0
- flattened `embed` input

BEAM uses all normal training references in one aligned memory per frequency,
`0.5 * (1 - cosine)`, an independent nearest reference per band, and a uniform
mean over the eight bands. It does not use Euclidean distance or energy
weighting. VarMin uses K=4, TrainAll leave-one-out calibration, one alpha per
band, rescaling before the band mean, and minimizes over the complete adjusted
distance set without clipping alpha or scores.

## Primary matrix

| ID | Frontend config | Backend config | Backend input | Meaning |
| --- | --- | --- | --- | --- |
| B0 | `scratch/raw_beats` | `knn_raw_beats` | `embed [B,768]` | Global AP + KNN |
| B1 | `scratch/raw_beats_freq_ap` | `knn_raw_beats` | flattened `embed [B,6144]` | Frequency AP + KNN |
| B2 | `scratch/raw_beats_freq_rdp4` | `knn_raw_beats` | flattened `embed [B,6144]` | Frequency RDP(4) + KNN |
| B3 | `scratch/raw_beats_freq_ap` | `beam_raw` | `embed_freq [B,8,768]` | Frequency AP + BEAM |
| B4 | `scratch/raw_beats_freq_rdp4` | `beam_raw` | `embed_freq [B,8,768]` | Frequency RDP(4) + BEAM |
| B5 | `scratch/raw_beats_freq_rdp4` | `beam_varmin4` | `embed_freq [B,8,768]` | Frequency RDP(4) + BEAM + VarMin(K=4) |

B1 and B3 share exactly the same extracted NPZ files. B2, B4, and B5 also
share exactly the same extracted NPZ files. The all-six runner creates symlinks
and the summarizer rejects results that do not resolve to the same artifacts.

The B1-B0 comparison includes both frequency retention and the increase from
768 to 6144 KNN input dimensions. No PCA or additional normalization is used.

## Controlled interpretation

- B1 vs B0: frequency-preserving representation effect
- B2 vs B1: RDP effect under KNN
- B3 vs B1: BEAM effect under AP
- B4 vs B3: RDP effect under BEAM
- B4 vs B2: BEAM effect under RDP
- B5 vs B4: VarMin effect

These labels describe the controlled implementation differences. They do not
establish causality beyond those controls.

## Commands

Run one condition across all official machines:

```bash
bash jobs/asd/recipe/raw_beats_ablation_dcase2024.sh B0 all
bash jobs/asd/recipe/raw_beats_ablation_dcase2024.sh B1 all
bash jobs/asd/recipe/raw_beats_ablation_dcase2024.sh B2 all
bash jobs/asd/recipe/raw_beats_ablation_dcase2024.sh B3 all
bash jobs/asd/recipe/raw_beats_ablation_dcase2024.sh B4 all
bash jobs/asd/recipe/raw_beats_ablation_dcase2024.sh B5 all
```

Run a one-machine smoke test under a distinct result name:

```bash
ASDKIT_ABLATION_NAME=dcase2024_raw_beats_ablation_smoke \
  bash jobs/asd/recipe/raw_beats_ablation_dcase2024.sh all bearing
```

Run the complete study with shared extraction:

```bash
bash jobs/asd/recipe/raw_beats_ablation_dcase2024.sh all all
```

Run seeds 0--4 and aggregate sample standard deviation:

```bash
bash jobs/asd/recipe/raw_beats_ablation_dcase2024_multiseed.sh
```

The multi-seed runner uses the exact seed-0 extraction artifacts for all five
seeds. This is safe because all DCASE 2024 files are at most 12 seconds, the
collator has `shuffle=false`, and frozen BEATs inference is deterministic. The
seed therefore varies the stochastic KNN SMOTE backend without introducing an
input-crop or frontend difference. BEAM and VarMin are deterministic controls
and are expected to have zero seed variance.

## Deterministic follow-up controls

The follow-up runner fills the missing AP + BEAM + VarMin cell and adds a
matched-memory coupled-reference control:

| ID | Pooling | Reference selection | VarMin |
| --- | --- | --- | --- |
| A6 | AP | independent nearest reference per band | K=4 |
| CAP | AP | one reference after uniform band averaging | off |
| CRDP | RDP(4) | one reference after uniform band averaging | off |

CAP/CRDP use the same all-normal memory and exact scaled cosine distance as
BEAM. Their score is `min_i mean_f D(x[f], y[i,f])`; B3/B4 use
`mean_f min_i D(x[f], y[i,f])`. Run all three controls with:

```bash
bash jobs/asd/recipe/raw_beats_ablation_dcase2024_followup.sh all
```

The runner reuses B1 extraction for A6/CAP and B2 extraction for CRDP. It
reports A6-B3, B5-A6, B3-CAP, B4-CRDP, and CRDP-CAP without adding these
controls to the six primary configurations.

The complete runner performs extract, score, evaluate, and table generation,
then writes `summary.csv`, `machine_scores.csv`, `machine_deltas.csv`,
`comparison_diagnostics.csv`, `reproducibility_metadata.csv`, and `REPORT.md`
under the result name. Existing extraction, score, evaluation, table, and
summary files are not overwritten.

Environment overrides are available for isolated runs:

```bash
ASDKIT_ABLATION_NAME=my_run \
ASDKIT_DATA_DIR=/absolute/path/to/data \
ASDKIT_RESULT_DIR=/absolute/path/to/results \
ASDKIT_DEVICE=cuda:0 \
  bash jobs/asd/recipe/raw_beats_ablation_dcase2024.sh all all
```
