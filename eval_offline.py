"""Offline evaluation of a WAM checkpoint on held-out LIBERO episodes.

Reports action L1 (normalized space) and world-model quality vs. the "no change" baseline
(predicting that frame t+F embeds like frame t). A useful world model must beat the baseline.

Usage:  python eval_offline.py --ckpt runs/wam_qwen3vl2b_libero/ckpt_002000.pt --n 500
"""
import argparse, json, os

import numpy as np
import torch
import torch.nn.functional as F

from train import LiberoWMDataset, prepare
from wam import QwenWAM, cos, future_latent, wm_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=500, help="number of random held-out frames")
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="", help="optional json path for the results")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu")
    t = ck["args"]
    ds = LiberoWMDataset(t["data"], t["chunk"], t["future"], "val", t["holdout_every"])
    dev = torch.device(a.device)
    model = QwenWAM(t["base"], chunk=t["chunk"], res=ds.meta["res"], lora_r=t["lora_r"], grad_ckpt=False).to(dev)
    missing = model.load_state_dict(ck["model"], strict=False)
    assert not [k for k in missing.unexpected_keys], missing.unexpected_keys
    model.eval()

    ln = lambda x: F.layer_norm(x, x.shape[-1:])
    sel = np.random.default_rng(0).choice(len(ds), min(a.n, len(ds)), replace=False)
    m = {k: [] for k in ("act_l1", "act_l1_first", "grip_acc", "wm_mse", "wm_mse_copy", "wm_cos", "wm_cos_copy")}
    with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        for s in range(0, len(sel), a.bs):
            batch = [ds[int(j)] for j in sel[s:s + a.bs]]
            enc, state, target, scene, fut = prepare(batch, model, dev)
            z = model.vision_targets(fut + scene)
            z_fut, z_now = z[:len(fut)], z[len(fut):]
            pa, pz = model(enc, state)
            pa, pz = pa.float(), future_latent(pz.float(), z_now, t.get("wm_target", "abs"))
            m["act_l1"].append((pa - target).abs().mean().item())
            m["act_l1_first"].append((pa[:, 0] - target[:, 0]).abs().mean().item())
            m["grip_acc"].append(((pa[..., -1] > 0) == (target[..., -1] > 0)).float().mean().item())
            m["wm_mse"].append(wm_loss(pz, z_fut).item())
            m["wm_mse_copy"].append(wm_loss(ln(z_now), z_fut).item())
            m["wm_cos"].append(cos(pz, ln(z_fut)).item())
            m["wm_cos_copy"].append(cos(ln(z_now), ln(z_fut)).item())
    res = {k: float(np.mean(v)) for k, v in m.items()}
    res.update(ckpt=a.ckpt, step=ck["step"], frames=len(sel), val_frames=len(ds))
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
