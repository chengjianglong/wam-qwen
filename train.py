"""Train the Qwen3-VL World Action Model on LIBERO (same prepared data as ../vla-qwen).

Single GPU:  python train.py
Both GPUs:   torchrun --nproc_per_node 2 train.py
Smoke test:  python train.py --steps 20 --log_every 1 --save_every 0
"""
import argparse, json, math, os, time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from wam import Normalizer, QwenWAM, cos, wm_loss

DEFAULT_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vla-qwen", "data", "libero")


class LiberoWMDataset(Dataset):
    """Frame t -> (scene, wrist) images, state, actions t..t+H-1, and the scene frame at t+F.

    split: "train" / "val" / "all". Every `holdout_every`-th episode is held out for "val".
    """

    def __init__(self, root, chunk, future, split="train", holdout_every=50):
        self.meta = json.load(open(os.path.join(root, "meta.json")))
        a = np.load(os.path.join(root, "arrays.npz"))
        self.state, self.actions, self.task_index = a["state"], a["actions"], a["task_index"]
        R, N = self.meta["res"], self.meta["total"]
        self.img_path, self.shape = os.path.join(root, "images.u8"), (N, 2, R, R, 3)
        self.images = None  # opened lazily per worker
        self.chunk, self.future = chunk, future
        starts = np.array(self.meta["episode_starts"]); lens = np.array(self.meta["episode_lengths"])
        ep = np.searchsorted(starts, np.arange(N), side="right") - 1
        self.ep_end = (starts + lens - 1)[ep]
        held = (ep % holdout_every == holdout_every - 1) if holdout_every else np.zeros(N, bool)
        self.idx = {"train": np.flatnonzero(~held), "val": np.flatnonzero(held), "all": np.arange(N)}[split]
        self.snorm = Normalizer(self.meta["stats"]["state"])
        self.anorm = Normalizer(self.meta["stats"]["actions"], skip_last=True)
        self.tasks = {int(k): v for k, v in self.meta["tasks"].items()}

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, j):
        if self.images is None:
            self.images = np.memmap(self.img_path, dtype=np.uint8, mode="r", shape=self.shape)
        i = int(self.idx[j])
        e = self.ep_end[i]
        act_idx = np.minimum(np.arange(i, i + self.chunk), e)
        return {
            "images": np.array(self.images[i]),                           # [2, R, R, 3]
            "future": np.array(self.images[min(i + self.future, e), 0]),  # scene cam at t+F
            "task": self.tasks[int(self.task_index[i])],
            "state": self.snorm.norm(self.state[i]).astype(np.float32),
            "actions": self.anorm.norm(self.actions[act_idx]).astype(np.float32),
        }


def augment(imgs, rng):
    """Same random-resized-crop (90% area) + brightness/contrast jitter applied to every image in
    `imgs` (uint8 HxWx3). Used so current and future scene frames share one augmentation and the
    world-model target stays predictable."""
    R = imgs[0].shape[0]
    s = int(R * math.sqrt(rng.uniform(0.9, 1.0)))
    y, x = rng.integers(0, R - s + 1), rng.integers(0, R - s + 1)
    c, b = rng.uniform(0.8, 1.2), rng.uniform(0.8, 1.2)
    out = []
    for img in imgs:
        t = torch.from_numpy(img[y:y + s, x:x + s]).permute(2, 0, 1)[None].float()
        t = torch.nn.functional.interpolate(t, size=(R, R), mode="bilinear", align_corners=False)[0]
        t = (t - t.mean()) * c + t.mean() * b
        out.append(t.clamp(0, 255).byte().permute(1, 2, 0).numpy())
    return out


def prepare(batch, model, dev, rng=None):
    scene, wrist, fut = [], [], []
    for b in batch:
        if rng is not None:
            s, f = augment([b["images"][0], b["future"]], rng)
            (w,) = augment([b["images"][1]], rng)
        else:
            s, w, f = b["images"][0], b["images"][1], b["future"]
        scene.append(s); wrist.append(w); fut.append(f)
    enc = model.encode(list(zip(scene, wrist)), [b["task"] for b in batch]).to(dev)
    state = torch.from_numpy(np.stack([b["state"] for b in batch])).to(dev)
    target = torch.from_numpy(np.stack([b["actions"] for b in batch])).to(dev)
    return enc, state, target, scene, fut


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--base", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--out", default="runs/wam_qwen3vl2b_libero")
    ap.add_argument("--chunk", type=int, default=8, help="action chunk length H")
    ap.add_argument("--future", type=int, default=8, help="world-model horizon F (frames ahead)")
    ap.add_argument("--wm_weight", type=float, default=0.5)
    ap.add_argument("--holdout_every", type=int, default=50, help="hold out every N-th episode (0 = none)")
    ap.add_argument("--bs", type=int, default=1, help="per-GPU micro batch")
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--steps", type=int, default=50000, help="optimizer steps")
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--save_every", type=int, default=2500)
    ap.add_argument("--no_aug", action="store_true")
    ap.add_argument("--resume", default="")
    a = ap.parse_args()

    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        rank, world, local = dist.get_rank(), dist.get_world_size(), int(os.environ["LOCAL_RANK"])
    else:
        rank, world, local = 0, 1, 0
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    rng = np.random.default_rng(rank)

    ds = LiberoWMDataset(a.data, a.chunk, a.future, "train", a.holdout_every)
    torch.manual_seed(0)  # identical init on every rank, so DDP needs no initial broadcast
    model = QwenWAM(a.base, chunk=a.chunk, res=ds.meta["res"], lora_r=a.lora_r).to(dev)
    torch.manual_seed(rank)
    step0 = 0
    if a.resume:
        ck = torch.load(a.resume, map_location="cpu")
        model.load_state_dict(ck["model"], strict=False)
        step0 = ck["step"]
    params = [p for p in model.parameters() if p.requires_grad]
    if rank == 0:
        print(f"trainable params: {sum(p.numel() for p in params) / 1e6:.1f}M, "
              f"foresight tokens: {model.n_fut}, train frames: {len(ds)}")
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.95))
    if a.resume and "opt" in ck:
        opt.load_state_dict(ck["opt"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / a.warmup) *
                                              0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
    for _ in range(step0):
        sched.step()

    dmodel = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local], find_unused_parameters=False,
                                                       init_sync=False) if ddp else model
    sampler = DistributedSampler(ds, world, rank, shuffle=True, seed=0) if ddp else None
    dl = DataLoader(ds, batch_size=a.bs, sampler=sampler, shuffle=sampler is None, num_workers=a.workers,
                    collate_fn=lambda b: b, drop_last=True, persistent_workers=a.workers > 0,
                    prefetch_factor=4 if a.workers else None)

    if rank == 0:
        os.makedirs(a.out, exist_ok=True)
        json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)
        json.dump(ds.meta["stats"], open(os.path.join(a.out, "norm_stats.json"), "w"))
        tb = SummaryWriter(a.out)

    step, micro, epoch, t0 = step0, 0, 0, time.time()
    acc = {"loss": 0.0, "act": 0.0, "wm": 0.0}
    model.train()
    while step < a.steps:
        if sampler:
            sampler.set_epoch(epoch)
        for batch in dl:
            enc, state, target, scene, fut = prepare(batch, model, dev, None if a.no_aug else rng)
            sync = (micro + 1) % a.accum == 0
            log = rank == 0 and sync and (step + 1) % a.log_every == 0
            # frozen-encoder targets; on log steps also embed the current frame for a "no change" baseline
            z = model.vision_targets(fut + scene if log else fut)
            z_fut = z[:len(fut)]

            ctx = dmodel.no_sync() if (ddp and not sync) else torch.enable_grad()
            with ctx, torch.autocast("cuda", dtype=torch.bfloat16):
                pred_a, pred_z = dmodel(enc, state)
                l_act = torch.nn.functional.l1_loss(pred_a, target)
                l_wm = wm_loss(pred_z, z_fut)
                loss = l_act + a.wm_weight * l_wm
                (loss / a.accum).backward()
            for k, v in (("loss", loss), ("act", l_act), ("wm", l_wm)):
                acc[k] += v.item() / a.accum
            micro += 1
            if not sync:
                continue

            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if log:
                dt = (time.time() - t0) / a.log_every
                mem = torch.cuda.max_memory_allocated(dev) / 2**30
                # compare against the "no change" baseline: predict that frame t+F looks like frame t
                ln = lambda x: torch.nn.functional.layer_norm(x, x.shape[-1:])
                c_pred = cos(pred_z.detach().float(), ln(z_fut)).item()
                c_copy = cos(ln(z[len(fut):]), ln(z_fut)).item()
                m_copy = wm_loss(ln(z[len(fut):]), z_fut).item()
                print(f"step {step} loss {acc['loss']:.4f} act {acc['act']:.4f} wm {acc['wm']:.4f} "
                      f"(copy {m_copy:.4f}) cos {c_pred:.3f} (copy {c_copy:.3f}) "
                      f"gn {gn:.2f} lr {sched.get_last_lr()[0]:.2e} "
                      f"{dt:.2f}s/it mem {mem:.2f}G", flush=True)
                for k, v in acc.items():
                    tb.add_scalar(f"loss/{k}", v, step)
                tb.add_scalar("wm/cos_pred", c_pred, step); tb.add_scalar("wm/cos_copy", c_copy, step)
                tb.add_scalar("wm/mse_copy", m_copy, step)
                tb.add_scalar("lr", sched.get_last_lr()[0], step)
                t0 = time.time()
            acc = dict.fromkeys(acc, 0.0)
            if rank == 0 and a.save_every and (step % a.save_every == 0 or step == a.steps):
                torch.save({"model": model.trainable_state_dict(), "opt": opt.state_dict(), "step": step,
                            "args": vars(a)}, os.path.join(a.out, f"ckpt_{step:06d}.pt"))
            if step >= a.steps:
                break
        epoch += 1

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
