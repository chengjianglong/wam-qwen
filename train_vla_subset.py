"""Train the ../vla-qwen VLA, unchanged, on the same episode subset as `train.py --train_every N`.

Loads ../vla-qwen/train.py by path and only narrows its dataset to every N-th episode (excluding the episodes
train.py holds out), so low-data VLA and WAM runs see identical demonstrations. All other flags pass through:
  python train_vla_subset.py --train_every 10 --data ../vla-qwen/data/libero --bs 1 --accum 16 --steps 5000 ...
"""
import importlib.util, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VLA_DIR = os.path.join(HERE, "..", "vla-qwen")
sys.path.insert(0, VLA_DIR)  # for `from vla import ...` inside their train.py


def pop_flag(name, default, cast):
    if name in sys.argv:
        i = sys.argv.index(name)
        v = cast(sys.argv[i + 1]); del sys.argv[i:i + 2]
        return v
    return default


TRAIN_EVERY = pop_flag("--train_every", 10, int)
HOLDOUT_EVERY = pop_flag("--holdout_every", 50, int)

spec = importlib.util.spec_from_file_location("vla_train", os.path.join(VLA_DIR, "train.py"))
vla_train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vla_train)


class SubsetDataset(vla_train.LiberoDataset):
    def __init__(self, root, chunk):
        super().__init__(root, chunk)
        starts = np.array(self.meta["episode_starts"])
        ep = np.searchsorted(starts, np.arange(self.shape[0]), side="right") - 1
        held = (ep % HOLDOUT_EVERY == HOLDOUT_EVERY - 1) if HOLDOUT_EVERY else np.zeros(len(ep), bool)
        self.idx = np.flatnonzero(~held & (ep % TRAIN_EVERY == 0))
        print(f"subset: every {TRAIN_EVERY}th episode -> {len(np.unique(ep[self.idx]))} episodes, "
              f"{len(self.idx)} frames", flush=True)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, j):
        return super().__getitem__(int(self.idx[j]))


vla_train.LiberoDataset = SubsetDataset

if __name__ == "__main__":
    vla_train.main()
