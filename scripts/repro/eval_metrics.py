"""VBench-style consistency metrics over generated videos.

Layout: <root>/<condition>_f<latents>/<prompt>.mp4
Metrics (VBench formulas, reference models):
  subject_consistency   — DINO ViT-B/16 cosine sim, (first + consecutive)/2
  background_consistency— CLIP ViT-B/32 cosine sim, (first + consecutive)/2
  temporal_flickering   — 1 - mean|f_t - f_{t+1}|/255 (static stability)
  drift curve           — DINO sim(frame_0, frame_t) vs time (identity drift)
Everything needed for analysis is printed to stdout as JSON lines.
"""
import argparse
import json
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

FPS = 16
SAMPLE_EVERY = 4  # analyze at 4 fps


def load_video(path, sample_every=SAMPLE_EVERY):
    reader = imageio.get_reader(path)
    frames = [f for i, f in enumerate(reader) if i % sample_every == 0]
    reader.close()
    return np.stack(frames)  # [T, H, W, 3] uint8


class Embedders:
    def __init__(self, device):
        self.device = device
        self.dino = torch.hub.load("facebookresearch/dino:main", "dino_vitb16").to(device).eval()
        import open_clip
        self.clip, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        self.clip = self.clip.to(device).eval()
        self.im_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.im_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self.clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
        self.clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def embed(self, frames_uint8, which, bs=32):
        feats = []
        for i in range(0, len(frames_uint8), bs):
            x = torch.from_numpy(frames_uint8[i:i + bs]).to(self.device).permute(0, 3, 1, 2).float() / 255.0
            x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False)
            if which == "dino":
                x = (x - self.im_mean) / self.im_std
                f = self.dino(x)
            else:
                x = (x - self.clip_mean) / self.clip_std
                f = self.clip.encode_image(x)
            feats.append(F.normalize(f.float(), dim=-1))
        return torch.cat(feats)


def consistency(feats):
    first = (feats @ feats[0]).clamp(min=0)[1:]
    consec = (feats[1:] * feats[:-1]).sum(-1).clamp(min=0)
    return ((first + consec) / 2).mean().item()


def flicker(frames_uint8):
    a = frames_uint8.astype(np.float32)
    mae = np.abs(a[1:] - a[:-1]).mean()
    return 1.0 - mae / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--strips_dir", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb = Embedders(device)
    root = Path(args.root)
    summary = {}

    for cond_dir in sorted(p for p in root.iterdir() if p.is_dir() and "_f" in p.name):
        vids = sorted(cond_dir.glob("*.mp4"))
        if not vids:
            continue
        agg = {"subject_consistency": [], "background_consistency": [], "temporal_flickering": []}
        drift_curves = []
        for v in vids:
            frames = load_video(str(v))
            d = emb.embed(frames, "dino")
            c = emb.embed(frames, "clip")
            sc, bc, fl = consistency(d), consistency(c), flicker(frames)
            agg["subject_consistency"].append(sc)
            agg["background_consistency"].append(bc)
            agg["temporal_flickering"].append(fl)
            drift = (d @ d[0]).cpu().numpy()
            drift_curves.append(drift)
            print(json.dumps({"video": f"{cond_dir.name}/{v.stem[:40]}",
                              "subject_consistency": round(sc, 4),
                              "background_consistency": round(bc, 4),
                              "temporal_flickering": round(fl, 4),
                              "n_frames_sampled": len(frames)}), flush=True)
        L = min(len(x) for x in drift_curves)
        mean_drift = np.stack([x[:L] for x in drift_curves]).mean(0)
        t_sec = (np.arange(L) * SAMPLE_EVERY / FPS).round(2)
        summary[cond_dir.name] = {
            **{k: round(float(np.mean(vals)), 4) for k, vals in agg.items()},
            "n_videos": len(vids),
            "drift_t_sec": t_sec.tolist(),
            "drift_dino_sim": [round(float(x), 4) for x in mean_drift],
        }

    print("[eval-metrics-summary] " + json.dumps(summary), flush=True)

    if args.strips_dir:
        os.makedirs(args.strips_dir, exist_ok=True)
        conds = sorted(summary.keys())
        prompts = {}
        for cname in conds:
            for v in sorted((root / cname).glob("*.mp4")):
                prompts.setdefault(v.stem, []).append((cname, v))
        for stem, items in prompts.items():
            rows = []
            for cname, v in items:
                frames = load_video(str(v), sample_every=1)
                idxs = np.linspace(0, len(frames) - 1, 6).astype(int)
                rows.append(np.concatenate([frames[i] for i in idxs], axis=1))
            strip = np.concatenate(rows, axis=0)
            out = Path(args.strips_dir) / f"{stem[:60]}.png"
            imageio.imwrite(str(out), strip)
            print(f"[strip] {out} rows={[c for c, _ in items]}", flush=True)


if __name__ == "__main__":
    main()
