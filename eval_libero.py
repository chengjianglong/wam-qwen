"""Closed-loop LIBERO evaluation of a WAM checkpoint.

Reuses the simulator loop of ../vla-qwen/eval_libero.py unchanged (same observation processing,
episode limits, seeds and replanning) and only swaps in a WAM policy, so success rates are directly
comparable with the VLA. The foresight tokens need no future frame at test time; their predictions
are simply ignored when acting.

Run with the LIBERO env:
  ./run_eval.sh --ckpt runs/wam_2k/ckpt_002000.pt --trials 10
"""
import importlib.util, json, os, sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
VLA_DIR = os.path.join(HERE, "..", "vla-qwen")
sys.path.insert(0, HERE)            # our wam.py
sys.path.append(VLA_DIR)            # their eval_libero.py (and vla.py, which it imports)

_spec = importlib.util.spec_from_file_location("vla_eval_libero", os.path.join(VLA_DIR, "eval_libero.py"))
base_eval = importlib.util.module_from_spec(_spec)  # same file name as ours, so load it by path
_spec.loader.exec_module(base_eval)
from wam import Normalizer, QwenWAM  # noqa: E402


class WAMPolicy:
    def __init__(self, ckpt, dev):
        run = os.path.dirname(ckpt)
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        args = ck.get("args") or json.load(open(os.path.join(run, "args.json")))
        stats = json.load(open(os.path.join(run, "norm_stats.json")))
        meta = os.path.join(args["data"], "meta.json")
        self.res = json.load(open(meta))["res"] if os.path.exists(meta) else 224
        self.chunk = args["chunk"]
        self.model = QwenWAM(args["base"], chunk=self.chunk, res=self.res, lora_r=args["lora_r"],
                             grad_ckpt=False)
        sd = ck["model"]
        _, unexpected = self.model.load_state_dict(sd, strict=False)
        assert not unexpected, unexpected[:5]
        assert all(k in sd for k in self.model.trainable_state_dict()), "checkpoint is missing trainable weights"
        self.model.to(dev).eval()
        self.dev = dev
        self.snorm = Normalizer(stats["state"])
        self.anorm = Normalizer(stats["actions"], skip_last=True)

    @torch.no_grad()
    def __call__(self, scene, wrist, state, task):
        enc = self.model.encode([[scene, wrist]], [task]).to(self.dev)
        st = torch.from_numpy(self.snorm.norm(state)[None].astype(np.float32)).to(self.dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            act, _ = self.model(enc, st)
        pred = act[0].float().cpu().numpy()  # [chunk, 7] normalized
        out = self.anorm.unnorm(pred)
        out[:, -1] = np.where(pred[:, -1] > 0, 1.0, -1.0)  # binarize gripper
        return out


base_eval.Policy = WAMPolicy

if __name__ == "__main__":
    base_eval.main()
