"""Generate the README figures in media/:

  architecture.svg   WAM diagram (real camera frames + the real latent target of frame t+8 embedded)
  wm_task.gif        the training task over one demo: frame t, frame t+F to imagine, actions with the chunk window
  latent_space.png   the frozen Qwen3-VL vision latent space the world model predicts in:
                     token-PCA maps over an episode, frame-to-frame similarity, and how fast the
                     "no change" baseline decays with the prediction horizon
  demos.gif          2x2 grid of demonstrations from four LIBERO task suites
  train_curves.png   action / world-model losses (+ no-change baseline) parsed from a train.py log

Usage:  python make_media.py [--log runs_2k.log] [--only arch,task,latent,demos,curves]
Vision embeddings are computed on CPU (to leave the GPUs to training) and cached in runs/media_cache.npz.
"""
import argparse, base64, io, json, os, re, textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from train import DEFAULT_DATA

SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                                                      "#e87ba4", "#008300", "#6250d6")
OUT, CACHE = "media", "runs/media_cache.npz"
BASE = "Qwen/Qwen3-VL-2B-Instruct"
ARCH_TASK, ARCH_FRAME, FUT = 10, 44, 8
CURVE_TASKS = (0, 3, 7, 10, 14, 17, 21, 24, 28, 31, 36, 39)  # one demo each, all four suites


def font(size, bold=False):
    p = f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf"
    return ImageFont.truetype(p, size) if os.path.exists(p) else ImageFont.load_default()


def style(ax, title=None):
    ax.set_facecolor(SURF)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    if title:
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")


class Data:
    def __init__(self, root):
        self.m = json.load(open(os.path.join(root, "meta.json")))
        self.a = np.load(os.path.join(root, "arrays.npz"))
        R, N = self.m["res"], self.m["total"]
        self.img = np.memmap(os.path.join(root, "images.u8"), np.uint8, "r", shape=(N, 2, R, R, 3))
        self.starts = np.array(self.m["episode_starts"]); self.lens = np.array(self.m["episode_lengths"])
        self.ep_task = self.a["task_index"][self.starts]

    def episode(self, task, k=0):
        e = np.flatnonzero(self.ep_task == task)[k]
        s, L = self.starts[e], self.lens[e]
        return e, self.img[s:s + L], self.a["actions"][s:s + L], self.m["tasks"][str(task)]


# ----------------------------------------------------------------------------------------------
class Encoder:
    """Frozen Qwen3-VL vision tower (post-merger tokens = the world-model target space), on CPU in fp32."""

    def __init__(self):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        self.proc = AutoProcessor.from_pretrained(BASE).image_processor
        vlm = Qwen3VLForConditionalGeneration.from_pretrained(BASE, dtype=torch.bfloat16)
        self.visual = vlm.model.visual.float().eval()
        self.merge = vlm.config.vision_config.spatial_merge_size
        del vlm

    @torch.no_grad()
    def __call__(self, images, bs=16):
        out = []
        for i in range(0, len(images), bs):
            px = self.proc(images=[np.asarray(x) for x in images[i:i + bs]], return_tensors="pt")
            z = self.visual(px["pixel_values"], grid_thw=px["image_grid_thw"], return_dict=True).pooler_output
            out.append(z.view(len(images[i:i + bs]), -1, z.shape[-1]).numpy().astype(np.float16))
        return np.concatenate(out)


def embeddings(d):
    """Scene-camera latents for every frame of the CURVE_TASKS demos (cached). -> {task: [L, 64, D]}"""
    if os.path.exists(CACHE):
        c = np.load(CACHE)
        if all(f"t{t}" in c for t in CURVE_TASKS):
            return {t: c[f"t{t}"] for t in CURVE_TASKS}
    torch.set_num_threads(6)
    enc, res = Encoder(), {}
    for t in CURVE_TASKS:
        _, ims, _, _ = d.episode(t)
        res[t] = enc([im[0] for im in ims]); print(f"  embedded task {t}: {len(ims)} frames", flush=True)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez(CACHE, **{f"t{t}": v for t, v in res.items()})
    return res


def ln(z):
    z = z.astype(np.float32)
    return (z - z.mean(-1, keepdims=True)) / (z.std(-1, keepdims=True) + 1e-5)


def token_cos(a, b):
    """Mean per-token cosine of layer-normalized latents (the metric logged in training)."""
    a, b = ln(a), ln(b)
    return ((a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))).mean(-1)


def pca_rgb(z_all, z):
    """Project tokens onto the first 3 principal components of z_all -> 8x8 RGB grids in [0, 1]."""
    X = ln(z_all).reshape(-1, z_all.shape[-1]); mu = X.mean(0)
    _, _, Vt = np.linalg.svd(X[::4] - mu, full_matrices=False)
    P = lambda q: (ln(q).reshape(-1, q.shape[-1]) - mu) @ Vt[:3].T
    ref = P(z_all)
    lo, hi = np.percentile(ref, 2, 0), np.percentile(ref, 98, 0)
    g = int(np.sqrt(z.shape[-2]))
    return np.clip((P(z) - lo) / (hi - lo), 0, 1).reshape(*z.shape[:-2], g, g, 3)


# ----------------------------------------------------------------------------------------------
def architecture_svg(d, emb, path):
    _, ims, act, task = d.episode(ARCH_TASK)
    f = ARCH_FRAME

    def b64(im, size=112, nearest=False):
        buf = io.BytesIO()
        Image.fromarray(np.asarray(im)).resize((size, size), Image.NEAREST if nearest else Image.BICUBIC).save(buf, "PNG")
        return base64.b64encode(buf.getvalue()).decode()

    W, H = 1000, 760
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
         f'viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Helvetica, Arial, sans-serif">',
         f'<rect width="{W}" height="{H}" rx="14" fill="{SURF}"/>',
         '<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
         f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{MUTED}"/></marker></defs>']

    def box(x, y, w, h, fill, stroke, lab, sub=None, fs=13, dash=False):
        da = ' stroke-dasharray="5 4"' if dash else ""
        s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="{stroke}" stroke-width="1.5"{da}/>')
        ty = y + h / 2 + (fs * 0.35 if sub is None else -2)
        s.append(f'<text x="{x + w / 2}" y="{ty}" text-anchor="middle" font-size="{fs}" font-weight="600" fill="{INK}">{lab}</text>')
        if sub:
            s.append(f'<text x="{x + w / 2}" y="{ty + 17}" text-anchor="middle" font-size="11" fill="{INK2}">{sub}</text>')

    def arrow(x1, y1, x2, y2, dash=False):
        da = ' stroke-dasharray="4 3"' if dash else ""
        s.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{MUTED}" stroke-width="1.6"{da} marker-end="url(#ar)"/>')

    def path_(d_, dash=False):
        da = ' stroke-dasharray="4 3"' if dash else ""
        s.append(f'<path d="{d_}" stroke="{MUTED}" stroke-width="1.6" fill="none"{da} marker-end="url(#ar)"/>')

    def label(x, y, t, fs=12, c=INK2, anchor="middle", w="400"):
        s.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{fs}" font-weight="{w}" fill="{c}">{t}</text>')

    def image(x, y, im, size=112, nearest=False):
        s.append(f'<image x="{x}" y="{y}" width="{size}" height="{size}" xlink:href="data:image/png;base64,{b64(im, size, nearest)}"/>')
        s.append(f'<rect x="{x}" y="{y}" width="{size}" height="{size}" fill="none" stroke="{GRID}"/>')

    # --- inputs
    label(40, 22, "INPUTS (time t)", 11, MUTED, "start", "700")
    for i, (nm, im) in enumerate([("scene camera", ims[f, 0]), ("wrist camera", ims[f, 1])]):
        image(40 + i * 128, 36, im); label(96 + i * 128, 166, nm, 11.5)
    box(300, 62, 260, 60, "#f0efec", GRID, "language instruction", textwrap.shorten(f'"{task}"', 44), fs=13)
    box(580, 62, 120, 60, "#fdf0ea", ORANGE, "robot state", "8-D", fs=13)
    box(716, 62, 128, 60, "#fdf0f5", MAGENTA, "64 foresight", "queries (learned)", fs=12.5)
    box(858, 62, 112, 60, "#e8f7f1", AQUA, "8 action", "queries (learned)", fs=12.5)

    # --- encoders
    box(40, 196, 240, 50, "#eaf2fc", BLUE, "Qwen3-VL vision encoder", "frozen · 2×64 visual tokens", fs=13)
    box(300, 196, 260, 50, "#f0efec", GRID, "tokenizer + embedding", "frozen (kept on CPU)", fs=13)
    box(580, 196, 120, 50, "#fdf0ea", ORANGE, "state MLP", "8 → 2048", fs=13)
    arrow(160, 176, 160, 194); arrow(430, 124, 430, 194); arrow(640, 124, 640, 194)

    # --- token sequence
    y, x, tw, step = 272, 40, 10, 13
    seq = [("scene", BLUE, 14), ("s1", MUTED, 2), ("wrist", BLUE, 14), ("text", MUTED, 10), ("state", ORANGE, 1),
           ("s2", MUTED, 2), ("fut", MAGENTA, 16), ("s3", MUTED, 2), ("act", AQUA, 8)]
    cen, xs = {}, {"fut": [], "act": []}
    for name, col, n in seq:
        x0 = x
        for _ in range(n):
            s.append(f'<rect x="{x}" y="{y}" width="{tw}" height="22" rx="3" fill="{col}"/>')
            if name in xs:
                xs[name].append(x + tw / 2)
            x += step
        cen[name] = (x0 + x - step + tw) / 2
    arrow(100, 248, cen["scene"], 270); arrow(220, 248, cen["wrist"], 270)
    arrow(430, 248, cen["text"], 270); arrow(640, 248, cen["state"], 270)
    path_(f'M780,124 V256 H{cen["fut"]} V270'); path_(f'M914,124 V256 H{cen["act"]} V270')
    for k, c, t in (("scene", BLUE, "scene"), ("wrist", BLUE, "wrist"), ("text", MUTED, "task text"),
                    ("fut", MAGENTA, "foresight tokens (×64)"), ("act", AQUA, "actions")):
        label(cen[k], 312, t, 11, c)
    label(cen["state"], 326, "state", 11, ORANGE)

    # --- LLM
    box(40, 336, 930, 60, "#f1effc", VIOLET, "Qwen3-VL-2B language model  (28 layers, frozen + LoRA r=32 on all linear layers)",
        "one causal forward pass · foresight tokens see images, task and state · action queries additionally see the foresight tokens",
        fs=14)
    for xx in xs["fut"][::3] + xs["act"]:
        arrow(xx, 296, xx, 334)
    for xx in xs["fut"][::3] + xs["act"][::2]:
        arrow(xx, 398, xx, 426)

    # --- heads
    fx, ax_ = cen["fut"], cen["act"]
    box(fx - 95, 428, 190, 50, "#fdf0f5", MAGENTA, "world-model head", "LN → MLP → 64×2048", fs=13)
    box(ax_ - 76, 428, 152, 50, "#e8f7f1", AQUA, "action head", "LN → MLP → 7", fs=13)

    # world-model branch: predicted latent vs. frozen-encoder target of frame t+F
    z = emb[ARCH_TASK]
    rgb = pca_rgb(z, z[[min(f + FUT, len(z) - 1)]])[0]
    gy = 516
    arrow(fx, 480, fx, gy - 2)
    s.append(f'<rect x="{fx - 48}" y="{gy}" width="96" height="96" rx="4" fill="#f6f5f2" stroke="{MAGENTA}" '
             f'stroke-width="1.5" stroke-dasharray="5 4"/>')
    for i in range(1, 8):
        s.append(f'<line x1="{fx - 48 + i * 12}" y1="{gy}" x2="{fx - 48 + i * 12}" y2="{gy + 96}" stroke="{GRID}"/>')
        s.append(f'<line x1="{fx - 48}" y1="{gy + i * 12}" x2="{fx + 48}" y2="{gy + i * 12}" stroke="{GRID}"/>')
    label(fx, gy + 112, "imagined latent of t+8", 11.5, INK, "middle", "600")
    label(fx, gy + 127, "8×8 tokens × 2048", 10.5)
    # target, to the left
    tx = fx - 185
    image(tx - 48, gy, (rgb * 255).astype(np.uint8), 96, nearest=True)
    label(tx, gy + 112, "target latent of t+8", 11.5, INK, "middle", "600")
    label(tx, gy + 127, "(first 3 PCA comps as RGB)", 10.5)
    s.append(f'<line x1="{tx + 54}" y1="{gy + 48}" x2="{fx - 54}" y2="{gy + 48}" stroke="{MAGENTA}" stroke-width="1.6" stroke-dasharray="4 3"/>')
    label((tx + fx) / 2, gy + 42, "MSE", 12, MAGENTA, "middle", "700")
    vx = tx - 205
    image(vx - 48, gy, ims[min(f + FUT, len(ims) - 1), 0], 96)
    label(vx, gy + 112, "real scene frame t+8", 11.5, INK, "middle", "600")
    label(vx, gy + 127, "training only", 10.5)
    box(vx + 62, gy + 24, 100, 48, "#eaf2fc", BLUE, "vision enc.", "frozen, no grad", fs=12)
    arrow(vx + 50, gy + 48, vx + 60, gy + 48); arrow(vx + 164, gy + 48, tx - 52, gy + 48)

    # action branch: chunk heat strip
    arrow(ax_, 480, ax_, gy - 2)
    lo = np.array(d.m["stats"]["actions"]["q01"]); hi = np.array(d.m["stats"]["actions"]["q99"])
    chunk = np.clip(2 * (act[f:f + 8] - lo) / (hi - lo) - 1, -1, 1)
    cw, ch = 11, 13
    x0 = ax_ - 4 * cw
    for t in range(8):
        for k in range(7):
            v = chunk[t, k]
            c = np.array([0x2a, 0x78, 0xd6]) if v < 0 else np.array([0xeb, 0x68, 0x34])
            c = (np.array([0xf0, 0xef, 0xec]) * (1 - abs(v)) + c * abs(v)).astype(int)
            s.append(f'<rect x="{x0 + t * cw + 0.5}" y="{gy + 4 + k * ch}" width="{cw - 1}" height="{ch - 1}" '
                     f'fill="#{c[0]:02x}{c[1]:02x}{c[2]:02x}"/>')
    label(ax_, gy + 112, "8 × 7 action chunk", 11.5, INK, "middle", "600")
    label(ax_, gy + 127, "L1 vs. demonstration", 10.5)

    # --- notes
    label(40, 448, "Trainable (49.7M params):", 12.5, INK, "start", "700")
    for i, t in enumerate(["LoRA adapters in the LLM", "foresight + action queries", "state MLP",
                           "world-model + action heads"]):
        label(52, 468 + i * 17, f"• {t}", 11.5, INK2, "start")
    label(40, 692, "Loss = L1(actions) + 0.5 · MSE(imagined, LN(target)).  At test time no future frame is needed: the model "
                   "imagines t+8 and acts on it in the same pass.", 12, INK, "start", "600")
    label(40, 712, "The target lives in the token space the LLM already reads images in, so no pixel decoder is "
                   "needed and the world model costs one extra no-grad vision pass.", 11.5, INK2, "start")
    s.append("</svg>")
    open(path, "w").write("\n".join(s))


# ----------------------------------------------------------------------------------------------
def wm_task_gif(d, path, task=ARCH_TASK, stride=2, chunk=8, fut=FUT):
    _, ims, act, text = d.episode(task)
    L, R = len(act), 224
    fig, axs = plt.subplots(2, 1, figsize=(4.48, 4.48), dpi=100, sharex=True, facecolor=SURF)
    for ax, (lo, labels, title) in zip(axs, [(0, "xyz", "translation Δ"), (3, ["roll", "pitch", "yaw"], "rotation Δ")]):
        ylo, yhi = act[:, lo:lo + 3].min() * 1.15, act[:, lo:lo + 3].max() * 1.15
        ax.fill_between(np.arange(L), ylo, yhi, where=act[:, 6] > 0, color="#efeeea", step="mid", lw=0)
        for k, c in enumerate([BLUE, ORANGE, AQUA]):
            ax.plot(act[:, lo + k], color=c, lw=1.5, label=labels[k])
        ax.set_ylim(ylo, yhi); ax.set_xlim(0, L - 1)
        style(ax); ax.tick_params(labelsize=7)
        ax.set_title(title, fontsize=8, color=INK2, loc="left")
        ax.legend(loc="upper right", ncol=3, fontsize=6.5, frameon=False, handlelength=1)
    axs[1].set_xlabel("timestep", fontsize=7.5, color=INK2)
    fig.tight_layout(pad=0.4); fig.canvas.draw()
    bg = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    fh = bg.shape[0]
    to_px = lambda ax, t: ax.transData.transform((t, 0))[0]
    boxes = [(ax.get_window_extent().y0, ax.get_window_extent().y1) for ax in axs]
    plt.close(fig)

    cap = 46
    Wt, Ht = 2 * R + bg.shape[1], max(2 * R, fh) + cap
    f_cap, f_small, f_b = font(14, True), font(11), font(12, True)
    pink = tuple(int(MAGENTA[i:i + 2], 16) for i in (1, 3, 5))
    frames = []
    for t in range(0, L, stride):
        tf = min(t + fut, L - 1)
        fr = Image.new("RGB", (Wt, Ht), SURF)
        dr = ImageDraw.Draw(fr)
        dr.text((10, 8), f'"{text}"', fill=INK, font=f_cap)
        dr.text((10, 28), f"input: cameras at t (left)   →   imagine: scene at t+{fut} (pink)   +   act: next {chunk} "
                          f"actions (blue band)", fill=INK2, font=f_small)
        fr.paste(Image.fromarray(np.asarray(ims[t, 0])), (0, cap))
        fr.paste(Image.fromarray(np.asarray(ims[t, 1])), (0, cap + R))
        fr.paste(Image.fromarray(np.asarray(ims[tf, 0])), (R, cap))
        dr.rectangle([R, cap, 2 * R - 1, cap + R - 1], outline=pink, width=3)
        # the future frame is a target only; below it, state what the model gets at test time
        dr.text((R + 12, cap + R + 30), "World-model target", fill=INK, font=f_b)
        for j, line in enumerate(["The frozen vision encoder embeds", f"frame t+{fut}; the 64 foresight tokens",
                                  "regress that latent (training only).", "",
                                  "The action queries read the", "imagined latent before acting."]):
            dr.text((R + 12, cap + R + 52 + j * 17), line, fill=INK2, font=f_small)
        plot = Image.fromarray(bg)
        pd = ImageDraw.Draw(plot, "RGBA")
        for ax_i, (y0, y1) in enumerate(boxes):
            xa, xb = to_px(axs[ax_i], t), to_px(axs[ax_i], min(t + chunk, L - 1))
            pd.rectangle([xa, fh - y1, xb, fh - y0], fill=(42, 120, 214, 50))
            pd.line([xa, fh - y1, xa, fh - y0], fill=(11, 11, 11, 200), width=1)
            xf = to_px(axs[ax_i], tf)
            pd.line([xf, fh - y1, xf, fh - y0], fill=pink + (255,), width=2)
        fr.paste(plot, (2 * R, cap))
        for (xx, yy, txt) in [(6, cap + 6, f"scene  t = {t}"), (6, cap + R + 6, "wrist  t"),
                              (R + 6, cap + 6, f"scene  t+{fut} = {tf}")]:
            dr.text((xx, yy), txt, fill="white", font=f_small, stroke_width=2, stroke_fill="black")
        frames.append(fr.quantize(128, method=Image.Quantize.MEDIANCUT))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=80, loop=0, optimize=True)


# ----------------------------------------------------------------------------------------------
def latent_png(d, emb, path, task=ARCH_TASK, n_strip=6, kmax=40):
    _, ims, act, text = d.episode(task)
    z = emb[task]; L = len(z)
    fr_idx = np.linspace(0, L - 1 - FUT, n_strip).round().astype(int)
    g = int(np.sqrt(z.shape[1]))

    fig = plt.figure(figsize=(13, 9.2), dpi=110, facecolor=SURF)
    gs = fig.add_gridspec(3, n_strip, height_ratios=[1, 1, 2.2], hspace=0.28, wspace=0.06)
    for i, fi in enumerate(fr_idx):
        a1 = fig.add_subplot(gs[0, i]); a1.imshow(ims[fi, 0]); a1.set_xticks([]); a1.set_yticks([])
        a1.set_title(f"t = {fi}", fontsize=9, color=INK2)
        # per-token change between t and t+F, shown over frame t+F
        chg = 1 - token_cos(z[fi][:, None], z[fi + FUT][:, None])  # [64]
        a2 = fig.add_subplot(gs[1, i]); a2.imshow(ims[fi + FUT, 0])
        a2.imshow(chg.reshape(g, g), cmap="magma", alpha=0.6, vmin=0, vmax=0.35,
                  extent=(-0.5, ims.shape[2] - 0.5, ims.shape[2] - 0.5, -0.5), interpolation="bilinear")
        a2.set_xticks([]); a2.set_yticks([])
        a2.set_title(f"t+{FUT} = {fi + FUT}", fontsize=9, color=INK2)
        if i == 0:
            a1.set_ylabel("frame t", fontsize=9, color=INK2); a2.set_ylabel(f"t+{FUT}, change", fontsize=9, color=INK2)
    fig.text(0.125, 0.915, f"A  What the world model must predict: per-token latent change from t to t+{FUT} "
                           f"(bright = changed) — \"{textwrap.shorten(text, 50)}\"",
             fontsize=10.5, color=INK, weight="bold")

    sub = gs[2, :].subgridspec(1, 2, width_ratios=[1, 1.45], wspace=0.32)
    u = ln(z); u /= np.linalg.norm(u, axis=-1, keepdims=True)   # unit tokens: mean per-token cosine
    u = u.reshape(L, -1)                                         # == (u_i . u_j) / n_tokens
    S = u @ u.T / z.shape[1]
    ax = fig.add_subplot(sub[0])
    im = ax.imshow(S, cmap="magma", vmin=np.percentile(S, 2), vmax=1)
    ax.set_title("B  Latent similarity, every pair of frames", fontsize=10.5, color=INK, loc="left", weight="bold")
    ax.set_xlabel("frame", fontsize=9, color=INK2); ax.set_ylabel("frame", fontsize=9, color=INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.ax.tick_params(labelsize=7, colors=INK2)

    ax = fig.add_subplot(sub[1])
    ks = np.arange(1, kmax + 1)
    curves = np.array([[token_cos(emb[t][:-k], emb[t][k:]).mean() for k in ks] for t in CURVE_TASKS])
    for c in curves:
        ax.plot(ks, c, color=BLUE, lw=0.8, alpha=0.25)
    med = np.median(curves, 0)
    ax.plot(ks, med, color=BLUE, lw=2.2, label=f"median over {len(CURVE_TASKS)} demos (one per task)")
    ax.axvline(FUT, color=MAGENTA, lw=1.5, ls="--")
    ax.annotate(f"world-model horizon F = {FUT}\nno-change cos = {med[FUT - 1]:.3f}", (FUT, med[FUT - 1]),
                xytext=(14, 18), textcoords="offset points", fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    style(ax)
    ax.set_title('C  "No change" baseline: similarity of frame t+k to frame t', fontsize=10.5, color=INK,
                 loc="left", weight="bold")
    ax.set_xlabel("horizon k (frames, 10 Hz)", fontsize=9, color=INK2)
    ax.set_ylabel("per-token cosine", fontsize=9, color=INK2)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.set_xlim(1, kmax)
    fig.savefig(path, facecolor=SURF, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------------------------
def demos_gif(d, path, tasks=(3, 10, 24, 36), stride=3, tile=224):
    eps = [d.episode(t) for t in tasks]
    T = max(len(a) for _, _, a, _ in eps)
    cap, f_cap = 36, font(10)
    frames = []
    for t in range(0, T, stride):
        fr = Image.new("RGB", (2 * tile, 2 * (tile + cap)), SURF)
        dr = ImageDraw.Draw(fr)
        for i, (_, ims, act, text) in enumerate(eps):
            x, y = (i % 2) * tile, (i // 2) * (tile + cap)
            fr.paste(Image.fromarray(np.asarray(ims[min(t, len(act) - 1), 0])), (x, y + cap))
            for j, line in enumerate(textwrap.wrap(text, 36)[:2]):
                dr.text((x + 6, y + 3 + j * 15), line, fill=INK, font=f_cap)
            if t >= len(act):
                dr.text((x + tile - 40, y + cap + 6), "done", fill="white", font=f_cap, stroke_width=2, stroke_fill="black")
        frames.append(fr.quantize(128, method=Image.Quantize.MEDIANCUT))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=100, loop=0, optimize=True)


# ----------------------------------------------------------------------------------------------
def curves_png(log, path):
    pat = (r"^step (\d+) loss ([\d.]+) act ([\d.]+) wm ([\d.]+) \(copy ([\d.]+)\) cos ([\d.]+) \(copy ([\d.]+)\)")
    pts = [tuple(map(float, m.groups())) for m in re.finditer(pat, open(log).read(), re.M)] if os.path.exists(log) else []
    if not pts:
        print("no WAM loss lines in", log, "- skipping curves"); return False
    st, _, la, lw, lc, cp, cc = np.array(pts).T

    def sm(x, k=9):
        k = min(k, len(x) // 2 * 2 + 1)
        return np.convolve(np.pad(x, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")

    fig, axs = plt.subplots(1, 2, figsize=(11, 3.4), dpi=120, facecolor=SURF)
    ax = axs[0]
    ax.plot(st, la, color=AQUA, lw=1, alpha=0.3); ax.plot(st, sm(la), color=AQUA, lw=2)
    ax.annotate(f"{sm(la)[-1]:.3f}", (st[-1], sm(la)[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9)
    style(ax, "Action L1 (normalized)"); ax.set_ylim(0, None); ax.set_xlabel("optimizer step", fontsize=9, color=INK2)
    ax = axs[1]
    ax.plot(st, lw, color=MAGENTA, lw=1, alpha=0.3); ax.plot(st, sm(lw), color=MAGENTA, lw=2, label="world model")
    ax.plot(st, sm(lc), color=MUTED, lw=1.6, ls="--", label='"no change" baseline')
    for y in (sm(lw)[-1], sm(lc)[-1]):
        ax.annotate(f"{y:.3f}", (st[-1], y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9)
    style(ax, f"Future-latent MSE (frame t+{FUT})"); ax.set_ylim(0, None)
    ax.set_xlabel("optimizer step", fontsize=9, color=INK2); ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout(); fig.savefig(path, facecolor=SURF); plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--log", default="runs_2k.log")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    d = Data(a.data)
    want = lambda k: not a.only or k in a.only.split(",")
    emb = embeddings(d) if want("arch") or want("latent") else None
    jobs = {"arch": lambda: architecture_svg(d, emb, f"{OUT}/architecture.svg"),
            "task": lambda: wm_task_gif(d, f"{OUT}/wm_task.gif"),
            "latent": lambda: latent_png(d, emb, f"{OUT}/latent_space.png"),
            "demos": lambda: demos_gif(d, f"{OUT}/demos.gif"),
            "curves": lambda: curves_png(a.log, f"{OUT}/train_curves.png")}
    for k, fn in jobs.items():
        if want(k):
            fn(); print("done", k, flush=True)


if __name__ == "__main__":
    main()
