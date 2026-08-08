# Frequency RDP + BEAM + variance-minimum rescaling

This implementation provides the frozen Original BEATs backend used as the first
step toward the NA-BEATs pipeline described in [Anomalous Sound Detection Meets
Noise-Aware Self-Supervised Learning](https://arxiv.org/html/2608.00447v1). It
does not implement NA-BEATs itself.

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

`T_p` and `F_p` come from the actual BEATs patch convolution output; neither the
audio duration nor `F_p` is hard-coded. BEATs flattens the convolution output in
row-major `(time, frequency)` order, so `[B,L,D]` is restored directly as
`[B,T_p,F_p,D]` after checking `L == T_p * F_p`.

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
`rescale_delta = raw - main` arrays. Distance matrices are constructed in
configurable query chunks. Reference-reference density computation is also
chunked.

Training anomaly-score output excludes every reference having the same `path`
as the query. TrainAll alpha calibration always excludes the reference itself by
index. If paths are unavailable, the backend logs a warning because query-time
self-match exclusion cannot be guaranteed.

## Configuration and execution

Frequency AP and RDP extraction use, respectively:

```bash
python -m asdkit.bin.extract dcase=dcase2023 name=recipe \
  version=raw_beats_freq_ap seed=0 infer_ver=last machine=bearing \
  experiments=scratch/raw_beats_freq_ap

python -m asdkit.bin.extract dcase=dcase2023 name=recipe \
  version=raw_beats_freq_rdp4 seed=0 infer_ver=last machine=bearing \
  experiments=scratch/raw_beats_freq_rdp4
```

Select `experiments=beam_raw` or `experiments=beam_varmin4` when running
`asdkit.bin.score`. The complete RDP(4) recipe is:

```bash
cd jobs/asd/call
bash raw_beats_beam_rdp4.sh
```

To compare the Original BEATs stages reported in the NA-BEATs paper, extract the
same data with the existing global `raw_beats`, frequency AP, RDP(4), and an
RDP(8) override, then pair frequency embeddings with raw BEAM or VarMin BEAM.
The published DCASE 2026 development reference scores (with its experimental
setup) are 60.28 for Freq AP + BEAM, 61.32 for RDP(4) + BEAM, and 62.02 for
RDP(8) + BEAM. Exact agreement is not claimed without the paper's complete data
and evaluation setup.

## Explicit assumptions where the combined paper is underspecified

1. VarMin validation is TrainAll.
2. Alpha is estimated independently for each frequency band.
3. VarMin is applied to band scores before the frequency mean.
4. RDP is applied over time independently per frequency after T-F restoration.
5. RDP runs in extraction for ASDKit NPZ memory efficiency.

## Current non-goals and limits

NA-BEATs/Dis NA-BEATs layers and training, two-channel DCASE 2026 input changes,
LoRA changes, pseudo-labeling, dataset formatting, evaluation metrics, waveform
denoising, clean-waveform generation, and calibrated pressure/SPL output are not
implemented. This backend only computes anomaly scores in embedding space.
