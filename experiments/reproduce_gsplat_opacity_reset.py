#!/usr/bin/env python3
# Copyright 2026 gsplat-opacity-reset contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
DefaultStrategy-Style Support Collapse on gsplat CUDA
=====================================================

Exact reproduction of Colab notebook run on 2026-04-15.
Environment: gsplat 1.5.3, PyTorch 2.10.0+cu128, Tesla T4, CUDA 12.8

Setup:
    pip install gsplat
    python experiments/reproduce_gsplat_opacity_reset.py

Paper-facing experiments:
  Exp1: N=100K, 300 train steps → full/selective/no reset → 100 recovery steps
  Exp2: N=500K, 200 train steps → full/selective reset → 50 recovery steps
  Exp3: N=100K, 1000 train steps (mature) → full/selective reset → 100 recovery steps
  Ablation: N=100K, 1000 train steps → reset fraction sweep and random-20% baseline
  Smoke: small CUDA sanity probe for remote GPU environments
  Threshold: archived T4 reset-opacity sweep around 1/255, emitted to JSON

Scope:
  These probes model the opacity-reset path used by reset-based training
  strategies such as gsplat DefaultStrategy. They do not evaluate gsplat
  MCMCStrategy, which relocates low-opacity Gaussians instead of globally
  resetting all opacities.

Results (Tesla T4, 2026-04-15):
  Exp1: Full reset -6.46 dB, Selective -0.69 dB → gap +5.77 dB
  Exp2: Full reset -2.69 dB, Selective -0.01 dB → gap +2.68 dB
  Exp3: Full reset -7.58 dB (PERMANENT -0.82 dB), Selective -1.75 dB → gap +5.83 dB
  Ablation: reset fraction dominates; age-200 and random-20% match within 0.02 dB
"""

import argparse, datetime, json, math, platform
from pathlib import Path
import torch
import torch.nn.functional as F

ALPHA_CUTOFF = 1.0 / 255.0

ARCHIVED_THRESHOLD_SWEEP_T4 = {
    "experiment": "threshold_isolation",
    "backend": "gsplat CUDA",
    "gpu": "Tesla T4",
    "N": 555_064,
    "seed": 42,
    "alpha_cutoff": ALPHA_CUTOFF,
    "probe": "synthetic 555K-Gaussian reset-opacity sweep",
    "source": "archived T4 threshold sweep packaged from the project evidence notes",
    "rows": [
        {
            "reset_opacity": 0.0035,
            "relative_to_cutoff": "below",
            "step0_psnr": 5.62696,
            "step1_psnr": 5.62696,
            "recovery_db": 0.00,
            "img_grad_nonzero_frac": 0.0,
        },
        {
            "reset_opacity": 0.00395,
            "relative_to_cutoff": "near_boundary",
            "step0_psnr": 5.64298,
            "step1_psnr": 5.74042,
            "recovery_db": 0.10,
            "img_grad_nonzero_frac": 0.307,
        },
        {
            "reset_opacity": 0.0050,
            "relative_to_cutoff": "above",
            "step0_psnr": 6.17418,
            "step1_psnr": 6.30006,
            "recovery_db": 0.13,
            "img_grad_nonzero_frac": 0.442,
        },
    ],
    "paper_table": [
        {"reset_opacity": 0.0035, "vs_tau": "below", "recovery_db": 0.00},
        {"reset_opacity": 0.00395, "vs_tau": "near boundary", "recovery_db": 0.10},
        {"reset_opacity": 0.0050, "vs_tau": "above", "recovery_db": 0.13},
    ],
    "key_finding": (
        "Recovery changes from zero to nonzero as reset opacity crosses "
        "the alpha cutoff at 1/255."
    ),
}

# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def compute_psnr(a, b):
    mse = F.mse_loss(a, b).item()
    return -10 * math.log10(max(mse, 1e-10))

W, H = 256, 256
device = "cuda"
rasterization = None
viewmat = None
K = None
target = None


def init_cuda_state():
    global K, rasterization, target, viewmat

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for the training probes")

    from gsplat import rasterization as gsplat_rasterization

    rasterization = gsplat_rasterization
    viewmat = torch.eye(4, device=device)
    viewmat[2, 3] = -5.0
    K = torch.tensor(
        [[300, 0, W / 2], [0, 300, H / 2], [0, 0, 1]],
        device=device,
        dtype=torch.float32,
    )
    target = torch.rand(
        H, W, 3, device=device, generator=torch.Generator(device).manual_seed(42)
    )

def render(means, quats, scales, opacities, sh0):
    return rasterization(
        means=means,
        quats=quats / (quats.norm(dim=-1, keepdim=True) + 1e-8),
        scales=torch.exp(scales),
        opacities=torch.sigmoid(opacities),
        colors=sh0,
        viewmats=viewmat.unsqueeze(0),
        Ks=K.unsqueeze(0),
        width=W, height=H, sh_degree=0,
    )

def make_scene(N):
    torch.manual_seed(42)
    means = torch.nn.Parameter(torch.randn(N, 3, device=device) * 1.5)
    means.data[:, 2] += 5.0
    quats = torch.nn.Parameter(torch.randn(N, 4, device=device))
    quats.data[:, 0] = 1.0
    scales = torch.nn.Parameter(torch.full((N, 3), -3.5, device=device) + torch.randn(N, 3, device=device) * 0.3)
    opacities = torch.nn.Parameter(torch.full((N,), 2.0, device=device))
    sh0 = torch.nn.Parameter(torch.rand(N, 1, 3, device=device) - 0.5)
    return means, quats, scales, opacities, sh0

def make_optimizer(means, quats, scales, opacities, sh0):
    return torch.optim.Adam([
        {"params": [means], "lr": 1e-3},
        {"params": [quats], "lr": 1e-3},
        {"params": [scales], "lr": 5e-3},
        {"params": [opacities], "lr": 5e-2},
        {"params": [sh0], "lr": 1e-2},
    ])

def train(means, quats, scales, opacities, sh0, steps, log_every=50):
    opt = make_optimizer(means, quats, scales, opacities, sh0)
    for step in range(steps):
        r, _, _ = render(means, quats, scales, opacities, sh0)
        loss = F.mse_loss(r[0], target)
        loss.backward()
        opt.step()
        opt.zero_grad()
        if step % log_every == 0 or step == steps - 1:
            psnr = compute_psnr(r[0].detach(), target)
            print(f"  step {step:4d}  PSNR={psnr:.2f}  loss={loss.item():.4f}")
    return opt

def snapshot(means, quats, scales, opacities, sh0):
    return {k: v.data.clone() for k, v in
            [("means", means), ("quats", quats), ("scales", scales),
             ("opacities", opacities), ("sh0", sh0)]}

def restore(snap, means, quats, scales, opacities, sh0):
    means.data.copy_(snap["means"])
    quats.data.copy_(snap["quats"])
    scales.data.copy_(snap["scales"])
    opacities.data.copy_(snap["opacities"])
    sh0.data.copy_(snap["sh0"])

def reset_and_recover(snap, means, quats, scales, opacities, sh0, mask, recovery_steps):
    max_logit = torch.logit(torch.tensor(0.01)).item()
    restore(snap, means, quats, scales, opacities, sh0)
    opacities.data[mask] = opacities.data[mask].clamp(max=max_logit)
    opt = make_optimizer(means, quats, scales, opacities, sh0)

    with torch.no_grad():
        r, _, _ = render(means, quats, scales, opacities, sh0)
        psnr_after = compute_psnr(r[0], target)

    for s in range(recovery_steps):
        r, _, _ = render(means, quats, scales, opacities, sh0)
        F.mse_loss(r[0], target).backward()
        opt.step()
        opt.zero_grad()

    with torch.no_grad():
        r, _, _ = render(means, quats, scales, opacities, sh0)
        psnr_recovered = compute_psnr(r[0], target)
        below = (torch.sigmoid(opacities) < 1/255).sum().item()

    return psnr_after, psnr_recovered, below


def run_smoke():
    print("\n" + "=" * 60)
    print("SMOKE: small CUDA render/train/reset sanity check")
    print("=" * 60)
    N = 2_048
    means, quats, scales, opacities, sh0 = make_scene(N)
    train(means, quats, scales, opacities, sh0, steps=3, log_every=1)
    snap = snapshot(means, quats, scales, opacities, sh0)
    with torch.no_grad():
        r, _, _ = render(means, quats, scales, opacities, sh0)
        pre = compute_psnr(r[0], target)

    full_mask = torch.ones(N, dtype=torch.bool, device=device)
    psnr_after, psnr_rec, below = reset_and_recover(
        snap, means, quats, scales, opacities, sh0, full_mask, 2
    )
    print(
        f"\nSmoke pre={pre:.2f} after={psnr_after:.2f} "
        f"rec={psnr_rec:.2f} below_1/255={below}"
    )
    return {
        "N": N,
        "train_steps": 3,
        "recovery_steps": 2,
        "pre": pre,
        "full": psnr_after,
        "full_rec": psnr_rec,
        "below": below,
    }

# ═══════════════════════════════════════════════════════════════════
# Experiment 1: N=100K, 300 training steps
# ═══════════════════════════════════════════════════════════════════
def run_exp1():
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: N=100K, 300 training steps")
    print("=" * 60)
    N = 100_000
    means, quats, scales, opacities, sh0 = make_scene(N)
    train(means, quats, scales, opacities, sh0, steps=300)
    snap = snapshot(means, quats, scales, opacities, sh0)
    with torch.no_grad():
        r, _, _ = render(means, quats, scales, opacities, sh0)
        pre = compute_psnr(r[0], target)

    full_mask = torch.ones(N, dtype=torch.bool, device=device)
    sel_mask = torch.zeros(N, dtype=torch.bool, device=device); sel_mask[int(N*0.8):] = True
    no_mask = torch.zeros(N, dtype=torch.bool, device=device)

    fa, fr, fb = reset_and_recover(snap, means, quats, scales, opacities, sh0, full_mask, 100)
    sa, sr, sb = reset_and_recover(snap, means, quats, scales, opacities, sh0, sel_mask, 100)
    na, nr, nb = reset_and_recover(snap, means, quats, scales, opacities, sh0, no_mask, 100)

    print(f"\nBefore: {pre:.2f} | Full: {fa:.2f} (drop {fa-pre:+.2f}) rec={fr:.2f}")
    print(f"Selective: {sa:.2f} (drop {sa-pre:+.2f}) rec={sr:.2f}")
    print(f"No reset: {na:.2f} rec={nr:.2f}")
    print(f">>> Gap at reset: +{sa-fa:.2f} dB")
    return {"pre": pre, "full": fa, "full_rec": fr, "sel": sa, "sel_rec": sr, "gap": sa-fa}

# ═══════════════════════════════════════════════════════════════════
# Experiment 2: N=500K, 200 training steps
# ═══════════════════════════════════════════════════════════════════
def run_exp2():
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: N=500K, 200 training steps")
    print("=" * 60)
    N = 500_000
    means, quats, scales, opacities, sh0 = make_scene(N)
    train(means, quats, scales, opacities, sh0, steps=200)
    snap = snapshot(means, quats, scales, opacities, sh0)
    with torch.no_grad():
        r, _, _ = render(means, quats, scales, opacities, sh0)
        pre = compute_psnr(r[0], target)

    full_mask = torch.ones(N, dtype=torch.bool, device=device)
    sel_mask = torch.zeros(N, dtype=torch.bool, device=device); sel_mask[int(N*0.8):] = True

    fa, fr, _ = reset_and_recover(snap, means, quats, scales, opacities, sh0, full_mask, 50)
    sa, sr, _ = reset_and_recover(snap, means, quats, scales, opacities, sh0, sel_mask, 50)

    print(f"\nBefore: {pre:.2f} | Full: {fa:.2f} (drop {fa-pre:+.2f}) rec={fr:.2f}")
    print(f"Selective: {sa:.2f} (drop {sa-pre:+.2f}) rec={sr:.2f}")
    print(f">>> Gap at reset: +{sa-fa:.2f} dB")
    return {"pre": pre, "full": fa, "full_rec": fr, "sel": sa, "sel_rec": sr, "gap": sa-fa}

# ═══════════════════════════════════════════════════════════════════
# Experiment 3: N=100K, 1000 training steps (mature support)
# ═══════════════════════════════════════════════════════════════════
def run_exp3():
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: N=100K, 1000 training steps (MATURE)")
    print("=" * 60)
    N = 100_000
    means, quats, scales, opacities, sh0 = make_scene(N)
    train(means, quats, scales, opacities, sh0, steps=1000, log_every=200)
    snap = snapshot(means, quats, scales, opacities, sh0)
    with torch.no_grad():
        r, _, _ = render(means, quats, scales, opacities, sh0)
        pre = compute_psnr(r[0], target)

    full_mask = torch.ones(N, dtype=torch.bool, device=device)
    sel_mask = torch.zeros(N, dtype=torch.bool, device=device); sel_mask[int(N*0.8):] = True

    fa, fr, fb = reset_and_recover(snap, means, quats, scales, opacities, sh0, full_mask, 100)
    sa, sr, sb = reset_and_recover(snap, means, quats, scales, opacities, sh0, sel_mask, 100)

    print(f"\nBefore: {pre:.2f} | Full: {fa:.2f} (drop {fa-pre:+.2f}) rec={fr:.2f} below_1/255={fb}")
    print(f"Selective: {sa:.2f} (drop {sa-pre:+.2f}) rec={sr:.2f}")
    print(f">>> Gap at reset: +{sa-fa:.2f} dB")
    print(f">>> Permanent loss: full={fr-pre:+.2f} dB, selective={sr-pre:+.2f} dB")
    return {"pre": pre, "full": fa, "full_rec": fr, "sel": sa, "sel_rec": sr, "gap": sa-fa,
            "permanent_full": fr-pre, "permanent_sel": sr-pre, "below": fb}

# ═══════════════════════════════════════════════════════════════════
# Reset-fraction ablation: N=100K, 1000 training steps
# ═══════════════════════════════════════════════════════════════════
def run_reset_fraction_ablation():
    print("\n" + "=" * 60)
    print("RESET-FRACTION ABLATION: N=100K, 1000 training steps")
    print("=" * 60)
    N = 100_000
    train_steps = 1000
    means, quats, scales, opacities, sh0 = make_scene(N)
    train(means, quats, scales, opacities, sh0, steps=train_steps, log_every=200)
    snap = snapshot(means, quats, scales, opacities, sh0)
    with torch.no_grad():
        r, _, _ = render(means, quats, scales, opacities, sh0)
        pre = compute_psnr(r[0], target)

    birth_step = torch.linspace(0, train_steps, N, device=device)
    policies = []
    policies.append(("no_reset", torch.zeros(N, dtype=torch.bool, device=device), 100))
    for max_age in (50, 100, 200, 500):
        age = train_steps - birth_step
        policies.append((f"age_{max_age}", age <= max_age, 100))

    generator = torch.Generator(device=device).manual_seed(42)
    random_mask = torch.zeros(N, dtype=torch.bool, device=device)
    random_mask[torch.randperm(N, device=device, generator=generator)[: int(0.2 * N)]] = True
    policies.append(("random_20pct", random_mask, 100))
    policies.append(("full_reset", torch.ones(N, dtype=torch.bool, device=device), 100))

    results = {}
    for name, mask, recovery_steps in policies:
        psnr_after, psnr_rec, below = reset_and_recover(
            snap, means, quats, scales, opacities, sh0, mask, recovery_steps
        )
        reset_pct = 100.0 * mask.sum().item() / N
        results[name] = {
            "reset_pct": reset_pct,
            "psnr_after": psnr_after,
            "psnr_rec": psnr_rec,
            "drop": psnr_after - pre,
            "below_1_255": below,
        }
        print(
            f"  {name:12s} reset={reset_pct:6.1f}% "
            f"drop={psnr_after - pre:+.2f} rec={psnr_rec:.2f}"
        )

    return {
        "pre_reset_psnr": pre,
        "N": N,
        "train_steps": train_steps,
        "recovery_steps": 100,
        "birth_step_simulation": "linearly spaced 0..1000 across N=100K",
        "results": results,
    }


def write_threshold_sweep_artifact(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "threshold_sweep_t4.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(ARCHIVED_THRESHOLD_SWEEP_T4, f, indent=2)
    print(f"Wrote {output_path}")
    return ARCHIVED_THRESHOLD_SWEEP_T4

# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        choices=("all", "threshold", "smoke", "exp1", "exp2", "exp3", "ablation"),
        default="all",
        help="Run all CUDA probes, a single CUDA probe, or only emit the archived threshold-sweep artifact.",
    )
    parser.add_argument(
        "--result-dir",
        default="results/colab",
        help="Directory for JSON result artifacts.",
    )
    parser.add_argument(
        "--output",
        default="support_collapse_paper_results.json",
        help="Path for the combined JSON run record when CUDA probes are run.",
    )
    args = parser.parse_args()

    threshold = write_threshold_sweep_artifact(args.result_dir)
    if args.only == "threshold":
        raise SystemExit(0)

    init_cuda_state()
    print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    import gsplat; print(f"gsplat: {gsplat.__version__}")

    smoke = run_smoke() if args.only == "smoke" else None
    r1 = run_exp1() if args.only in ("all", "exp1") else None
    r2 = run_exp2() if args.only in ("all", "exp2") else None
    r3 = run_exp3() if args.only in ("all", "exp3") else None
    ablation = run_reset_fraction_ablation() if args.only in ("all", "ablation") else None

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    if smoke is not None:
        print(f"Smoke: pre={smoke['pre']:.2f} after={smoke['full']:.2f} rec={smoke['full_rec']:.2f}")
    if r1 is not None:
        print(f"Exp1 (N=100K, 300 steps):  gap = +{r1['gap']:.2f} dB")
    if r2 is not None:
        print(f"Exp2 (N=500K, 200 steps):  gap = +{r2['gap']:.2f} dB")
    if r3 is not None:
        print(f"Exp3 (N=100K, 1000 steps): gap = +{r3['gap']:.2f} dB, permanent loss: full={r3['permanent_full']:+.2f} / sel={r3['permanent_sel']:+.2f}")
    if ablation is not None:
        print("Ablation:")
        for name, row in ablation["results"].items():
            print(f"  {name:12s} reset={row['reset_pct']:6.1f}% drop={row['drop']:+.2f} rec={row['psnr_rec']:.2f}")
    print()
    if r3 is not None and r3['gap'] > 1.0:
        print(">>> SUPPORT COLLAPSE CONFIRMED on gsplat CUDA rasterizer")

    output = {
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(),
            "gsplat": gsplat.__version__,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
        "selected": args.only,
        "table3_threshold_sweep": threshold,
    }
    if smoke is not None:
        output["smoke"] = smoke
    table1 = {k: v for k, v in (("exp1", r1), ("exp2", r2), ("exp3", r3)) if v is not None}
    if table1:
        output["table1"] = table1
    if ablation is not None:
        output["table2_ablation"] = ablation

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {output_path}")
