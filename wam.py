"""Qwen3-VL based World Action Model (WAM): imagine the future, then act.

  [scene img][wrist img] "Task: ... State: <s> Future: <f_1..f_K> Actions: <a_1..a_H>"
                              Qwen3-VL-2B (LoRA), causal
  hidden(f_k) -> MLP -> z_hat_k    ~ frozen Qwen3-VL vision embedding of the scene frame t+F
  hidden(a_i) -> MLP -> action_t+i (7-D, normalized)

The K foresight tokens come *before* the action queries, so under causal attention every
action query can read the model's imagined future (but not vice versa). The world-model
target is the (frozen, merged) Qwen3-VL vision-encoder output for the future scene frame,
i.e. the same token space the LLM already consumes, so the model learns forward dynamics in
its own visual latent space; no pixel decoder is needed.

Placeholder tokens are swapped for learned embeddings by a forward hook on the input
embedding layer, so Qwen3-VL's image-token handling / M-RoPE is untouched.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import LoraConfig, get_peft_model

ACT_TOKEN = "<|quad_start|>"
STATE_TOKEN = "<|quad_end|>"
FUT_TOKEN = "<|box_start|>"


class Normalizer:
    """Map raw values to [-1, 1] via q01/q99 (gripper dims already in [-1, 1])."""

    def __init__(self, stats, skip_last=False):
        self.lo = np.asarray(stats["q01"], np.float32)
        self.hi = np.asarray(stats["q99"], np.float32)
        if skip_last:
            self.lo[-1], self.hi[-1] = -1.0, 1.0

    def norm(self, x):
        return np.clip(2 * (x - self.lo) / (self.hi - self.lo + 1e-8) - 1, -1.5, 1.5)

    def unnorm(self, y):
        return (np.asarray(y) + 1) / 2 * (self.hi - self.lo + 1e-8) + self.lo


class CPUEmbedding(nn.Module):
    """Frozen embedding table kept in CPU RAM (saves ~620MB of VRAM for Qwen3-VL-2B).
    The weight is a plain attribute, not a Parameter, so model.to(cuda) leaves it alone."""

    def __init__(self, emb: nn.Embedding):
        super().__init__()
        self.cpu_weight = emb.weight.detach().cpu()
        self.anchor = nn.Parameter(torch.empty(0, dtype=emb.weight.dtype), requires_grad=False)  # tracks device

    def forward(self, ids):
        out = nn.functional.embedding(ids.cpu(), self.cpu_weight)
        return out.to(self.anchor.device, non_blocking=True)


def mlp_head(H, hidden, out):
    return nn.Sequential(nn.LayerNorm(H), nn.Linear(H, hidden), nn.GELU(), nn.Linear(hidden, out))


class QwenWAM(nn.Module):
    def __init__(self, base="Qwen/Qwen3-VL-2B-Instruct", chunk=8, action_dim=7, state_dim=8,
                 res=224, lora_r=32, grad_ckpt=True, cpu_embed=True):
        super().__init__()
        self.chunk, self.action_dim = chunk, action_dim
        self.processor = AutoProcessor.from_pretrained(base)
        tok = self.processor.tokenizer
        self.act_id, self.state_id, self.fut_id = (tok.convert_tokens_to_ids(t)
                                                   for t in (ACT_TOKEN, STATE_TOKEN, FUT_TOKEN))
        assert tok.unk_token_id not in (self.act_id, self.state_id, self.fut_id)

        vlm = Qwen3VLForConditionalGeneration.from_pretrained(
            base, dtype=torch.bfloat16, attn_implementation="sdpa")
        vlm.lm_head = nn.Identity()  # we never need vocab logits (saves ~600MB of activations)
        if cpu_embed:
            vlm.model.language_model.embed_tokens = CPUEmbedding(vlm.model.language_model.embed_tokens)
        H = vlm.config.text_config.hidden_size
        D = vlm.config.vision_config.out_hidden_size  # merged vision token dim (== H for Qwen3-VL)
        for p in vlm.parameters():
            p.requires_grad_(False)
        if grad_ckpt:
            vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            vlm.enable_input_require_grads()
        lcfg = LoraConfig(r=lora_r, lora_alpha=lora_r, lora_dropout=0.0,
                          target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
        self.vlm = get_peft_model(vlm, lcfg)

        # number of merged vision tokens per image at this resolution (224 -> 256px -> 8x8 = 64)
        g = self.processor.image_processor(images=[np.zeros((res, res, 3), np.uint8)],
                                           return_tensors="pt")["image_grid_thw"][0]
        self.n_fut = int(g.prod()) // vlm.config.vision_config.spatial_merge_size ** 2

        self.act_queries = nn.Parameter(torch.randn(chunk, H) * 0.02)
        self.fut_queries = nn.Parameter(torch.randn(self.n_fut, H) * 0.02)
        self.state_proj = nn.Sequential(nn.Linear(state_dim, H), nn.GELU(), nn.Linear(H, H))
        self.head = mlp_head(H, 1024, action_dim)
        self.wm_head = mlp_head(H, H, D)

        self._state = None
        self.vlm.get_input_embeddings().register_forward_hook(self._embed_hook)

    # -- swap placeholder embeddings for learned ones -------------------------------
    def _embed_hook(self, module, inputs, out):
        if self._state is None:
            return out
        ids = inputs[0].to(out.device)
        B = ids.shape[0]
        out = out.clone()
        out[ids == self.act_id] = self.act_queries.to(out.dtype).repeat(B, 1)
        out[ids == self.fut_id] = self.fut_queries.to(out.dtype).repeat(B, 1)
        out[ids == self.state_id] = self.state_proj(self._state.float()).to(out.dtype)
        return out

    def build_prompt(self, task):
        return (f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                f"<|vision_start|><|image_pad|><|vision_end|>"
                f"Task: {task.strip().lower()}\nState: {STATE_TOKEN}\n"
                f"Future: {FUT_TOKEN * self.n_fut}\nActions: {ACT_TOKEN * self.chunk}")

    def encode(self, images, tasks):
        """images: list of [scene, wrist] uint8 HxWx3; tasks: list[str]."""
        flat = [im for pair in images for im in pair]
        texts = [self.build_prompt(t) for t in tasks]
        self.processor.tokenizer.padding_side = "left"  # action tokens are always the last `chunk`
        return self.processor(text=texts, images=flat, return_tensors="pt", padding=True)

    @torch.no_grad()
    def vision_targets(self, images):
        """Frozen Qwen3-VL vision embeddings (post-merger) of a list of uint8 images -> [N, K, D]."""
        dev = self.act_queries.device
        px = self.processor.image_processor(images=list(images), return_tensors="pt")
        out = self.vlm.base_model.model.model.get_image_features(
            px["pixel_values"].to(dev), px["image_grid_thw"].to(dev))
        return torch.stack(out.pooler_output).float()

    def forward(self, batch, state):
        """Returns (actions [B, chunk, A], future latents [B, K, D])."""
        self._state = state
        try:
            out = self.vlm(**batch, output_hidden_states=False, use_cache=False)
        finally:
            self._state = None
        h = out.logits  # lm_head is Identity -> last hidden states [B, T, H]
        B = h.shape[0]
        act = self.head(h[:, -self.chunk:].float())
        fut_mask = (batch["input_ids"] == self.fut_id).to(h.device)
        z = self.wm_head(h[fut_mask].view(B, self.n_fut, -1).float())
        return act, z

    def trainable_state_dict(self):
        return {k: v for k, v in self.state_dict().items()
                if "lora_" in k or k.startswith(("act_queries", "fut_queries", "state_proj", "head", "wm_head"))}


def wm_loss(pred, target):
    """MSE against per-token layer-normalized targets (scale-free, like MAE's norm-pix loss)."""
    return F.mse_loss(pred, F.layer_norm(target, target.shape[-1:]))


def cos(a, b):
    return F.cosine_similarity(a, b, dim=-1).mean()
