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

import argparse, datetime, math, platform
from pathlib import Path

try:
    from experiments.support_runtime import (
        CudaUnavailableError,
        EnvironmentCheckError,
        GsplatOpacityResetError,
        RunLogger,
        write_json,
    )
except ModuleNotFoundError:
    from support_runtime import (
        CudaUnavailableError,
        EnvironmentCheckError,
        GsplatOpacityResetError,
        RunLogger,
        write_json,
    )

ALPHA_CUTOFF = 1.0 / 255.0
LOGGER = RunLogger("gsplat-opacity-reset")
torch = None
F = None

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

def require_torch():
    global F, torch
    if torch is not None and F is not None:
        return torch, F
    try:
        import torch as torch_module
        import torch.nn.functional as functional_module
    except ModuleNotFoundError as exc:
        raise EnvironmentCheckError(
            "PyTorch is required for CUDA training probes",
            original_error=repr(exc),
        ) from exc
    torch = torch_module
    F = functional_module
    return torch, F


def compute_psnr(a, b):
    _, functional = require_torch()
    mse = functional.mse_loss(a, b).item()
    return -10 * math.log10(max(mse, 1e-10))

W, H = 256, 256
device = "cuda"
rasterization = None
viewmat = None
K = None
target = None


def init_cuda_state():
    global K, rasterization, target, viewmat

    torch_module, _ = require_torch()
    if not torch_module.cuda.is_available():
        raise CudaUnavailableError("CUDA GPU required for the training probes")

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
            LOGGER.progress(
                "train",
                step=step + 1,
                total=steps,
                psnr=round(psnr, 4),
                loss=round(loss.item(), 6),
            )
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
    LOGGER.section("SMOKE: small CUDA render/train/reset sanity check")
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
    LOGGER.info(
        "smoke_summary",
        "Smoke reset/recovery summary",
        pre=round(pre, 4),
        after=round(psnr_after, 4),
        recovered=round(psnr_rec, 4),
        below_1_255=below,
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
    LOGGER.section("EXPERIMENT 1: N=100K, 300 training steps")
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

    LOGGER.info(
        "exp1_summary",
        "Experiment 1 reset gap",
        pre=round(pre, 4),
        full=round(fa, 4),
        full_drop=round(fa - pre, 4),
        full_rec=round(fr, 4),
        selective=round(sa, 4),
        selective_drop=round(sa - pre, 4),
        selective_rec=round(sr, 4),
        no_reset=round(na, 4),
        no_reset_rec=round(nr, 4),
        gap=round(sa - fa, 4),
    )
    return {"pre": pre, "full": fa, "full_rec": fr, "sel": sa, "sel_rec": sr, "gap": sa-fa}

# ═══════════════════════════════════════════════════════════════════
# Experiment 2: N=500K, 200 training steps
# ═══════════════════════════════════════════════════════════════════
def run_exp2():
    LOGGER.section("EXPERIMENT 2: N=500K, 200 training steps")
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

    LOGGER.info(
        "exp2_summary",
        "Experiment 2 reset gap",
        pre=round(pre, 4),
        full=round(fa, 4),
        full_drop=round(fa - pre, 4),
        full_rec=round(fr, 4),
        selective=round(sa, 4),
        selective_drop=round(sa - pre, 4),
        selective_rec=round(sr, 4),
        gap=round(sa - fa, 4),
    )
    return {"pre": pre, "full": fa, "full_rec": fr, "sel": sa, "sel_rec": sr, "gap": sa-fa}

# ═══════════════════════════════════════════════════════════════════
# Experiment 3: N=100K, 1000 training steps (mature support)
# ═══════════════════════════════════════════════════════════════════
def run_exp3():
    LOGGER.section("EXPERIMENT 3: N=100K, 1000 training steps (MATURE)")
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

    LOGGER.info(
        "exp3_summary",
        "Experiment 3 reset gap",
        pre=round(pre, 4),
        full=round(fa, 4),
        full_drop=round(fa - pre, 4),
        full_rec=round(fr, 4),
        selective=round(sa, 4),
        selective_drop=round(sa - pre, 4),
        selective_rec=round(sr, 4),
        gap=round(sa - fa, 4),
        permanent_full=round(fr - pre, 4),
        permanent_selective=round(sr - pre, 4),
        below_1_255=fb,
    )
    return {"pre": pre, "full": fa, "full_rec": fr, "sel": sa, "sel_rec": sr, "gap": sa-fa,
            "permanent_full": fr-pre, "permanent_sel": sr-pre, "below": fb}

# ═══════════════════════════════════════════════════════════════════
# Reset-fraction ablation: N=100K, 1000 training steps
# ═══════════════════════════════════════════════════════════════════
def run_reset_fraction_ablation():
    LOGGER.section("RESET-FRACTION ABLATION: N=100K, 1000 training steps")
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
        LOGGER.info(
            "ablation_policy",
            "Ablation policy result",
            policy=name,
            reset_pct=round(reset_pct, 4),
            drop=round(psnr_after - pre, 4),
            recovered=round(psnr_rec, 4),
            below_1_255=below,
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
    write_json(output_path, ARCHIVED_THRESHOLD_SWEEP_T4)
    LOGGER.info("artifact_written", "Wrote threshold sweep artifact", path=output_path)
    return ARCHIVED_THRESHOLD_SWEEP_T4

# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main():
    global LOGGER

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
    parser.add_argument(
        "--log-jsonl",
        default=None,
        help="Optional JSONL path for structured runtime logs.",
    )
    args = parser.parse_args()

    LOGGER.close()
    LOGGER = RunLogger(
        "gsplat-opacity-reset",
        args.log_jsonl,
        context={"selected": args.only},
    )
    LOGGER.info(
        "run_start",
        "Starting gsplat opacity-reset reproduction",
        result_dir=args.result_dir,
        output=args.output,
        log_jsonl=args.log_jsonl,
    )

    threshold = write_threshold_sweep_artifact(args.result_dir)
    if args.only == "threshold":
        LOGGER.info("run_complete", "Threshold artifact run complete")
        return 0

    init_cuda_state()
    import gsplat

    LOGGER.info(
        "runtime_environment",
        "CUDA runtime ready",
        pytorch=torch.__version__,
        cuda_version=torch.version.cuda,
        gpu_name=torch.cuda.get_device_name(),
        gsplat=gsplat.__version__,
    )

    smoke = run_smoke() if args.only == "smoke" else None
    r1 = run_exp1() if args.only in ("all", "exp1") else None
    r2 = run_exp2() if args.only in ("all", "exp2") else None
    r3 = run_exp3() if args.only in ("all", "exp3") else None
    ablation = run_reset_fraction_ablation() if args.only in ("all", "ablation") else None

    LOGGER.section("FINAL SUMMARY")
    if smoke is not None:
        LOGGER.info(
            "final_smoke",
            "Smoke summary",
            pre=round(smoke["pre"], 4),
            after=round(smoke["full"], 4),
            recovered=round(smoke["full_rec"], 4),
        )
    if r1 is not None:
        LOGGER.info("final_exp1", "Exp1 summary", gap=round(r1["gap"], 4))
    if r2 is not None:
        LOGGER.info("final_exp2", "Exp2 summary", gap=round(r2["gap"], 4))
    if r3 is not None:
        LOGGER.info(
            "final_exp3",
            "Exp3 summary",
            gap=round(r3["gap"], 4),
            permanent_full=round(r3["permanent_full"], 4),
            permanent_selective=round(r3["permanent_sel"], 4),
        )
    if ablation is not None:
        for name, row in ablation["results"].items():
            LOGGER.info(
                "final_ablation",
                "Ablation summary",
                policy=name,
                reset_pct=round(row["reset_pct"], 4),
                drop=round(row["drop"], 4),
                recovered=round(row["psnr_rec"], 4),
            )
    if r3 is not None and r3['gap'] > 1.0:
        LOGGER.info(
            "support_collapse_confirmed",
            "SUPPORT COLLAPSE CONFIRMED on gsplat CUDA rasterizer",
            gap=round(r3["gap"], 4),
        )

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
    write_json(output_path, output)
    LOGGER.info("artifact_written", "Wrote combined run record", path=output_path)
    LOGGER.info("run_complete", "Experiment run complete")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GsplatOpacityResetError as exc:
        LOGGER.exception("run_failed", exc)
        raise SystemExit(exc.exit_code) from exc
    except Exception as exc:
        LOGGER.exception("unexpected_failure", exc)
        raise
    finally:
        LOGGER.close()
