"""Long-horizon consistency metrics + frame strips for the eval videos.

Per condition (init / released / sgf / sf), per prompt video (60s, 16fps):
- subject consistency: DINO ViT-B/16 features, VBench formula
  mean_i (cos(f0,fi) + cos(f_{i-1},f_i)) / 2 on ~4fps samples
- background consistency: same formula with CLIP ViT-B/32 features
- identity drift curve: cos(f0, f_t) per 5s window
- flicker: mean abs consecutive-frame diff (native rate, /255, lower=better)
Short horizon = first 5s window; long = last 5s window / full clip.
Outputs JSON (also printed to the log) + figures + paired frame strips.
"""
import json
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as TF

OUT = Path(os.environ.get("SGF_SHARED", "/shared")) / "outputs"
EVAL = OUT / "eval"
MET = OUT / "metrics"
WINDOW_S, FPS = 5, 16
FEAT_STRIDE = 4          # 4 fps feature sampling
CONDITIONS = ["init", "released", "sgf", "sf"]


def read_video(path):
    reader = imageio.get_reader(str(path))
    frames = np.stack([f for f in reader], axis=0)  # [T,H,W,3] uint8
    reader.close()
    return frames


class Extractors:
    def __init__(self, device):
        self.device = device
        self.dino = torch.hub.load("facebookresearch/dino:main", "dino_vitb16").to(device).eval()
        import open_clip
        self.clip, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k")
        self.clip = self.clip.to(device).eval()
        self.dino_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.dino_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self.clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
        self.clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def feats(self, frames_u8):
        """frames_u8 [N,H,W,3] -> dict of L2-normalized [N,D] features."""
        x = torch.from_numpy(frames_u8).to(self.device).permute(0, 3, 1, 2).float() / 255.
        out = {}
        for name, size, mean, std, model in (
                ("dino", 224, self.dino_mean, self.dino_std, self.dino),
                ("clip", 224, self.clip_mean, self.clip_std,
                 lambda t: self.clip.encode_image(t))):
            fs = []
            for i in range(0, len(x), 32):
                xb = TF.interpolate(x[i:i + 32], size=(size, size), mode="bilinear", antialias=True)
                xb = (xb - mean) / std
                fs.append(model(xb).float())
            f = torch.cat(fs)
            out[name] = f / f.norm(dim=-1, keepdim=True)
        return out


def vbench_consistency(f):
    """mean_i (cos(f0,fi)+cos(f_{i-1},f_i))/2 for i>=1."""
    c_first = (f[1:] @ f[0]).clamp(0)
    c_prev = (f[1:] * f[:-1]).sum(-1).clamp(0)
    return ((c_first + c_prev) / 2).mean().item()


def windowed(vals_t, n_windows, agg=np.mean):
    idx = np.linspace(0, len(vals_t), n_windows + 1).astype(int)
    return [float(agg(vals_t[a:b])) if b > a else float("nan")
            for a, b in zip(idx[:-1], idx[1:])]


def analyze_video(path, ex):
    frames = read_video(path)
    T = len(frames)
    n_windows = max(1, T // (WINDOW_S * FPS))
    sub = frames[::FEAT_STRIDE]
    f = ex.feats(sub)

    res = {"frames": T, "n_windows": n_windows}
    for name, key in (("dino", "subject_consistency"), ("clip", "background_consistency")):
        feat = f[name]
        res[key] = vbench_consistency(feat)
        drift = (feat[1:] @ feat[0]).cpu().numpy()
        prev = (feat[1:] * feat[:-1]).sum(-1).cpu().numpy()
        res[f"{key}_drift_curve"] = windowed(drift, n_windows)
        res[f"{key}_windowed"] = windowed((drift.clip(0) + prev.clip(0)) / 2, n_windows)

    diffs = np.abs(frames[1:].astype(np.int16) - frames[:-1].astype(np.int16)).mean(axis=(1, 2, 3)) / 255.
    res["flicker_mad"] = float(diffs.mean())
    res["flicker_mad_windowed"] = windowed(diffs, n_windows)
    return res


def frame_strip(cond_paths, out_png, times_s=(0, 10, 20, 30, 40, 50, 59)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = []
    labels = []
    for cond, path in cond_paths.items():
        frames = read_video(path)
        idxs = [min(int(t * FPS), len(frames) - 1) for t in times_s]
        rows.append([frames[i] for i in idxs])
        labels.append(cond)
    fig, axes = plt.subplots(len(rows), len(times_s),
                             figsize=(2.2 * len(times_s), 1.45 * len(rows)))
    axes = np.atleast_2d(axes)
    for r, (row, lab) in enumerate(zip(rows, labels)):
        for c, img in enumerate(row):
            ax = axes[r, c]
            ax.imshow(img)
            ax.set_xticks([]), ax.set_yticks([])
            if r == 0:
                ax.set_title(f"t={times_s[c]}s", fontsize=8)
            if c == 0:
                ax.set_ylabel(lab, fontsize=8)
    fig.tight_layout(pad=0.3)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main():
    device = torch.device("cuda")
    os.environ.setdefault("TORCH_HOME", str(OUT.parent / "torch-cache"))
    MET.mkdir(parents=True, exist_ok=True)
    ex = Extractors(device)

    all_results = {}
    videos_by_prompt = {}
    for cond in CONDITIONS:
        d = EVAL / cond
        if not d.is_dir():
            print(f"[metrics] skip missing condition {cond}")
            continue
        vids = sorted(d.glob("*.mp4"))
        cond_res = []
        for v in vids:
            r = analyze_video(v, ex)
            r["video"] = v.stem
            cond_res.append(r)
            videos_by_prompt.setdefault(v.stem[:60], {})[cond] = v
            print(f"[metrics] {cond}/{v.stem[:50]}: subj={r['subject_consistency']:.4f} "
                  f"bg={r['background_consistency']:.4f} flicker={r['flicker_mad']:.4f}", flush=True)
        all_results[cond] = cond_res

    summary = {}
    for cond, rs in all_results.items():
        if not rs:
            continue
        agg = {}
        for key in ("subject_consistency", "background_consistency", "flicker_mad"):
            agg[key] = float(np.mean([r[key] for r in rs]))
        for key in ("subject_consistency_windowed", "background_consistency_windowed",
                    "subject_consistency_drift_curve", "flicker_mad_windowed"):
            n = min(len(r[key]) for r in rs)
            agg[key] = np.mean([r[key][:n] for r in rs], axis=0).tolist()
            agg[key + "_sem"] = (np.std([r[key][:n] for r in rs], axis=0) /
                                 np.sqrt(len(rs))).tolist()
        agg["short_horizon_subject"] = agg["subject_consistency_windowed"][0]
        agg["long_horizon_subject"] = agg["subject_consistency_windowed"][-1]
        summary[cond] = agg

    (MET / "metrics_full.json").write_text(json.dumps(all_results, indent=1))
    (MET / "metrics_summary.json").write_text(json.dumps(summary, indent=1))
    print("[metrics] SUMMARY " + json.dumps(summary), flush=True)

    # figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figs = {
        "subject_consistency_windowed": ("DINO subject consistency (per 5s window)", "higher = better"),
        "subject_consistency_drift_curve": ("DINO similarity to first frame", "higher = less identity drift"),
        "background_consistency_windowed": ("CLIP background consistency (per 5s window)", "higher = better"),
        "flicker_mad_windowed": ("Mean abs consecutive-frame diff (per 5s window)", "lower = less flicker"),
    }
    for key, (title, sub) in figs.items():
        fig, ax = plt.subplots(figsize=(6, 3.6))
        for cond in CONDITIONS:
            if cond not in summary:
                continue
            y = np.array(summary[cond][key])
            sem = np.array(summary[cond][key + "_sem"])
            t = (np.arange(len(y)) + 0.5) * WINDOW_S
            ax.plot(t, y, marker="o", ms=3, label=cond)
            ax.fill_between(t, y - sem, y + sem, alpha=0.15)
        ax.set_xlabel("time (s)")
        ax.set_title(f"{title}\n({sub})", fontsize=9)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(MET / f"{key}.png", dpi=130)
        plt.close(fig)

    strips = MET / "strips"
    strips.mkdir(exist_ok=True)
    for stem, conds in videos_by_prompt.items():
        if len(conds) >= 2:
            safe = "".join(ch if ch.isalnum() else "_" for ch in stem)[:60]
            frame_strip(conds, strips / f"{safe}.png")
    print(f"[metrics] wrote figures + {len(list(strips.glob('*.png')))} strips to {MET}", flush=True)


if __name__ == "__main__":
    main()
