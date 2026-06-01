# Threshold Boundary Notes

These notes record the evidence behind `results/colab/threshold_sweep_t4.json`,
the current Table 3 artifact for the `1/255` cutoff transition.

## Mechanism

The renderer applies a hard alpha contribution gate:

```text
if alpha < 1 / 255: skip contribution
```

For an image-space loss, a Gaussian whose effective alpha is below this cutoff
at every contributing pixel receives no image-driven opacity-gradient
contribution from the skipped branch.

## Reset Opacity Sweep

The current threshold-isolation draft uses a synthetic 555K-Gaussian probe. The
reset opacity was swept around `1 / 255 ~= 0.00392`:

| Reset opacity | Relative to `1/255` | Recovery |
|---:|---|---:|
| `0.0035` | below | `0.00 dB` |
| `0.00395` | near boundary | `+0.10 dB` |
| `0.0050` | above | `+0.13 dB` |

The transition from zero to nonzero recovery aligns with the renderer cutoff.

## Artifact

The checked-in artifact is:

```text
results/colab/threshold_sweep_t4.json
```

For a stronger archival release, add a fully rerunnable threshold-sweep path to
`experiments/reproduce_gsplat_opacity_reset.py`. The current script can emit the
archived JSON with:

```bash
python experiments/reproduce_gsplat_opacity_reset.py --only threshold
```
