# Frequency RDP + BEAM + variance-minimum rescaling

This implementation provides the frozen Original BEATs backend used as the first
step toward the NA-BEATs pipeline described in [Anomalous Sound Detection Meets
Noise-Aware Self-Supervised Learning](https://arxiv.org/html/2608.00447v1). It
does not implement NA-BEATs itself.

The status of each important behavior is explicit:

- **Paper-defined:** frozen Original BEATs embeddings, frequency-wise pooling,
  BEAM matching, and variance-minimum score rescaling.
- **Implementation choice:** RDP runs during extraction so standard NPZ files
  keep `[N,F,D]`, not the full `[N,L,D]` patch sequence.
- **Reproduction assumption:** VarMin uses TrainAll leave-one-out calibration,
  one alpha per band, and is applied before the frequency mean.
- **Engineering optimization:** query/reference and reference/reference distance
  calculations are chunked without changing the exact cosine definition.
- **Non-goal:** this is embedding-space ASD, not waveform separation or physical
  sound-level measurement.

## Pipeline and tensor shapes

The standard pipeline is:

```text
waveform [B, samples]
  -> frozen BEATs_iter3 final Transformer layer [B, L, D]
  -> time-frequency patch grid [B, T_p, F_p, D], L = T_p * F_p
  -> time pooling independently at every frequency [B, F_p, D]
  -> frequency-aligned BEAM memories [R, F_p, D]
  -> band-wise exact cosine distances [B, R, F_p]
  -> optional per-band variance-minimum rescaling
  -> uniform mean over F_p -> sample anomaly score [B]
```

`T_p` and `F_p` come from the actual BEATs patch convolution output; `F_p` is not
hard-coded by the frontend. BEATs flattens the convolution output in row-major
`(time, frequency)` order, so `[B,L,D]` is restored directly as
`[B,T_p,F_p,D]` after checking `L == T_p * F_p`. Duration is a pipeline setting:
the canonical DCASE 2026 recipe explicitly uses `sec=all` and batch size 1 for
both train and test extraction, while historical recipes retain their existing
fixed-duration `${dcase}` behavior. A fixed-duration DCASE 2026 experiment is
still possible only through an explicit Hydra override.

The frequency frontend saves `embed_freq` with shape `[B,F_p,D]` and a flattened
`embed` with shape `[B,F_p*D]`. `patch_sequence` is emitted only when
`emit_patch_sequence: true`; standard configurations keep it disabled to avoid
large NPZ files.

## Relative Deviation Pooling

For every sample and frequency band, let

```text
mu_f       = mean_t x[t,f,:]
d[t,f]     = ||x[t,f,:] - mu_f||_2
d_hat[t,f] = d[t,f] / max_t d[t,f]
a[t,f]     = (1 + d_hat[t,f]) ** gamma
w[t,f]     = a[t,f] / sum_t a[t,f]
pooled[f]  = sum_t w[t,f] * x[t,f,:]
```

When the maximum deviation is at most `eps`, normalized deviations are zero.
The implementation also accepts an optional boolean valid-time mask. `gamma=0`
is ordinary time averaging. The reproduction default is `gamma=4`; changing the
extract setting to `gamma: 8` gives the RDP(8) ablation. `pooling: mean` gives
frequency average pooling.

RDP is mathematically a backend aggregation in the cited work, but is executed
during ASDKit extraction here. This preserves only compact band embeddings in
NPZ instead of retaining all `[N,L,D]` patches. The pooling utility itself is
encoder-independent and can be reused by future NA-BEATs, EAT, or other models
that supply a time-frequency grid.

## BEAM

[BEAM](https://arxiv.org/html/2603.13749) creates one normal memory per aligned
frequency index. For normalized vectors, this implementation uses the exact
distance

```text
D(x,y) = 0.5 * (1 - dot(normalize(x), normalize(y))).
s_f(x) = min_i D(x[f], y_i[f])
S_raw  = mean_f s_f(x).
```

Each band chooses its own reference; references are not tied across frequency.
Bands are averaged uniformly without energy weighting. Source and target normal
samples share one memory, and `is_target` is not used. If `is_normal` is present,
only rows with `is_normal == 1` enter memory. `sep_section` retains the meaning
used by existing ASDKit backends and defaults to `false` in reproduction configs.

## Variance-minimum score rescaling

The implementation follows [Matsumoto et al., DCASE 2025](https://dcase.community/documents/workshop2025/proceedings/DCASE2025Workshop_Matsumoto_12.pdf).
For reference `i` and band `f`, `b[i,f]` is the mean cosine distance to the `K`
nearest *other* references in the same band. The default is `K=4`. With fewer
than `K+1` references, `min(K,R-1)` is used with a warning; fewer than two is an
error.

TrainAll calibration treats each normal training reference as validation and
uses leave-one-out raw matching. With raw nearest-neighbor distance `d_zf` and
the selected neighbor density `b_zf`, each band uses

```text
alpha[f] = Cov(d_zf, b_zf) / Var(b_zf).
```

The covariance and variance use matching centered moments. If the denominator
is at most `eps`, alpha is zero with a warning. Alpha is not clipped. Query
scores implement Eq. 5 over the complete adjusted reference set:

```text
s_rescaled_f(x) = min_i (D(x[f], y_i[f]) - alpha[f] * b[i,f])
S_main          = mean_f s_rescaled_f(x).
```

Thus the adjusted neighbor may differ from the raw nearest neighbor, and final
scores may be negative. The backend returns one-dimensional `main`, `raw`, and
`rescale_delta = raw - main` arrays. Only `main` uses the canonical `AS-` prefix
and enters standard evaluate/table aggregation. `raw` is an ablation and
`rescale_delta` is diagnostic; both use a `diagnostic-` prefix. Distance matrices
are constructed in configurable query chunks. Reference-reference density
computation is also chunked.

Training anomaly-score output excludes every reference having the same `path`
as the query. TrainAll alpha calibration always excludes the reference itself by
index. If paths are unavailable, the backend logs a warning because query-time
self-match exclusion cannot be guaranteed.

## Configuration and execution

The entrypoint defaults to the canonical DCASE 2026 RDP(4) + BEAM + VarMin run:

```bash
cd jobs/asd/call
bash raw_beats_beam_rdp4.sh
```

Its positional arguments are `dcase`, `ablation`, and `beam_mode`. The five
Original BEATs comparisons are:

```bash
bash raw_beats_beam_rdp4.sh dcase2026 global_ap
bash raw_beats_beam_rdp4.sh dcase2026 freq_ap
bash raw_beats_beam_rdp4.sh dcase2026 freq_ap_beam raw
bash raw_beats_beam_rdp4.sh dcase2026 freq_rdp4_beam raw
bash raw_beats_beam_rdp4.sh dcase2026 freq_rdp8_beam raw
```

Replace `raw` with `varmin4` for a BEAM + VarMin comparison. DCASE 2023 remains
available as a historical smoke test:

```bash
bash raw_beats_beam_rdp4.sh dcase2023 freq_rdp4_beam varmin4
```

For DCASE 2026 every variant receives these extraction overrides:

```text
datamodule.train.collator.sec=all
datamodule.train.dataloader.batch_size=1
```

The test loader inherits both settings. The configured audio channel is `first`
for training, validation, and restored extraction (the near channel in the
DCASE 2026 two-channel recordings); `second`, `mean`, and `diff` remain explicit
experiment overrides. Formatting is a separate prerequisite: use `hidden` for
anonymous submission-style inference or `public` for post-challenge metrics.

## Reproduction record

The downloader identifies its dataset inputs by Zenodo records `19336329`
(development, including `dev_ToyCar_r2.zip`), `20151556` (additional evaluation
machine training audio), and `20437238` (evaluation test audio). Public labels
come from evaluator commit
`f6a94a2b5e614a9626c9d1ccff6df0705e6aaa75`. The model checkpoint is the
Original `pretrained_models/beats/BEATs_iter3.pt`; record its local checksum when
reporting an experiment.

The paper reference targets are not unit-test constants:

| configuration | reference score | local score | difference |
|---|---:|---:|---:|
| Global AP | 57.16 | 58.44* | +1.28 |
| Freq AP | 57.94 | 60.26* | +2.32 |
| Freq AP + BEAM + VarMin | 60.28 | 60.283 | +0.003 |
| Freq RDP(4) + BEAM + VarMin | 61.32 | 61.329 | +0.009 |
| Freq RDP(8) + BEAM + VarMin | 62.02 | 62.015 | -0.005 |

These local scores were run on 2026-08-09 over the seven DCASE 2026 development
machines from Zenodo record `19336329`, with `audio_channel=first`, original
duration, batch size 1, evaluator mode not applicable to development labels, and
checkpoint SHA-256
`8d1b234032a9ccff353612dc6c20982346dc2968b205b79d97303eb5e77bfb34`.
BEAM used shared source+target normal memory, exact scaled cosine, VarMin K=4,
TrainAll leave-one-out calibration, and the per-band scope. Results are stored
under `/tmp/dcase-asd-toolkit-reproduction-results` in the run environment.

The starred AP-only rows use the first backend in ASDKit's existing `default`
score configuration: cosine 1-NN with separate source and target memories. That
backend protocol is not established as identical to the paper's AP-only rows,
and is the leading explanation for their material difference. BEAM raw on the
Freq AP embeddings scored 60.139; adding the canonical VarMin assumption produced
the reported 60.283. Differences should be investigated rather than adjusted.
Other likely causes include duration policy, preprocessing, T-F layout, RDP,
self-match exclusion, VarMin calibration/scope, dataset revision, and evaluation
definition. The roughly 70.24 `Fujimura_MERL_task2_3` full-system result is not a
target for this stack.

## Explicit assumptions where the combined paper is underspecified

1. VarMin validation is TrainAll.
2. Alpha is estimated independently for each frequency band.
3. VarMin is applied to band scores before the frequency mean.
4. RDP is applied over time independently per frequency after T-F restoration.
5. RDP runs in extraction for ASDKit NPZ memory efficiency.

## Current non-goals and limits

NA-BEATs and Dis NA-BEATs, discriminative fine-tuning, LoRA, pseudo-labeling,
waveform noise cancellation, clean-waveform generation, anomalous-waveform
separation, RMS `[Pa]`, SPL/calibration, and guaranteed inspection thresholds are
not implemented. Dataset formatting and DCASE metrics exist elsewhere in ASDKit,
but this method remains an embedding-space anomaly detector and does not claim
any source-separation or physical-magnitude result.
