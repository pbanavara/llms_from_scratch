"""
Qwen3.5 From Scratch
Supplementary code for "Build a Large Language Model From Scratch" by Sebastian Raschka
https://github.com/rasbt/LLMs-from-scratch

Minimal, readable re-implementation of the Qwen3.5 text stack for the
Qwen/Qwen3.5-0.8B checkpoint.

Notes:
- Alternates `linear_attention` and `full_attention` layers
- Reuses `Qwen3_5GatedDeltaNet` from HuggingFace transformers (qwen3_5_transformers.py)
"""

# pip install -r https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/ch05/07_gpt_to_llama/requirements-extra.txt

import json
import os
import re
import time
from pathlib import Path

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file
from tokenizers import Tokenizer

from qwen3_5_transformers import Qwen3_5GatedDeltaNet


# ---------------------------------------------------------------------------
# 1. Architecture
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.silu(x_fc1) * x_fc2
        return self.fc3(x)


class RMSNorm(nn.Module):
    """RMSNorm with (1 + weight) scaling and zero init, as used in Qwen3.5."""

    def __init__(self, emb_dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(emb_dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        x_norm = self._norm(x.float())
        x_norm = x_norm * (1.0 + self.weight.float())
        return x_norm.to(dtype=x.dtype)


def compute_rope_params(
    head_dim,
    theta_base=10_000,
    context_length=4096,
    partial_rotary_factor=1.0,
    dtype=torch.float32,
):
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    rotary_dim = int(head_dim * partial_rotary_factor)
    rotary_dim = max(2, rotary_dim - (rotary_dim % 2))

    inv_freq = 1.0 / (
        theta_base ** (
            torch.arange(0, rotary_dim, 2, dtype=dtype)[: (rotary_dim // 2)].float() / rotary_dim
        )
    )

    positions = torch.arange(context_length, dtype=dtype)
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
    angles = torch.cat([angles, angles], dim=1)

    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return cos, sin


def apply_rope(x, cos, sin):
    _, _, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    rot_dim = cos.shape[-1]
    if rot_dim > head_dim:
        raise ValueError(f"RoPE dim {rot_dim} cannot exceed head_dim {head_dim}.")

    x_rot = x[..., :rot_dim]
    x_pass = x[..., rot_dim:]

    x1 = x_rot[..., : rot_dim // 2]
    x2 = x_rot[..., rot_dim // 2 :]

    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x_rot * cos) + (rotated * sin)

    x_out = torch.cat([x_rotated, x_pass], dim=-1)
    return x_out.to(dtype=x.dtype)


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False, dtype=None):
        super().__init__()
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        if head_dim is None:
            assert d_in % num_heads == 0, "`d_in` must be divisible by `num_heads` if `head_dim` is not set"
            head_dim = d_in // num_heads

        self.head_dim = head_dim
        self.d_out = num_heads * head_dim

        # Qwen3.5 full-attention uses a gated Q projection (2x output dim)
        self.W_query = nn.Linear(d_in, self.d_out * 2, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

    def forward(self, x, mask, cos, sin):
        b, num_tokens, _ = x.shape

        q_and_gate = self.W_query(x)
        q_and_gate = q_and_gate.view(b, num_tokens, self.num_heads, self.head_dim * 2)
        queries, gate = torch.chunk(q_and_gate, 2, dim=-1)
        gate = gate.reshape(b, num_tokens, self.d_out)

        keys = self.W_key(x)
        values = self.W_value(x)

        queries = queries.transpose(1, 2)
        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys = self.k_norm(keys)

        queries = apply_rope(queries, cos, sin)
        keys = apply_rope(keys, cos, sin)

        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(
            attn_scores * (self.head_dim ** -0.5),
            dim=-1,
            dtype=torch.float32,
        ).to(queries.dtype)

        context = (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)

        # Gated Q projection
        context = context * torch.sigmoid(gate)
        return self.out_proj(context)


class _Qwen3_5ConfigAdapter:
    """Maps our config dict to the attribute names expected by HuggingFace's Qwen3_5GatedDeltaNet."""

    def __init__(self, cfg):
        self.hidden_size = cfg["emb_dim"]
        self.linear_num_value_heads = cfg["linear_num_value_heads"]
        self.linear_num_key_heads = cfg["linear_num_key_heads"]
        self.linear_key_head_dim = cfg["linear_key_head_dim"]
        self.linear_value_head_dim = cfg["linear_value_head_dim"]
        self.linear_conv_kernel_dim = cfg["linear_conv_kernel_dim"]
        self.hidden_act = "silu"
        self.rms_norm_eps = cfg.get("rms_norm_eps", 1e-6)
        self.dtype = cfg.get("dtype", None)


class TransformerBlock(nn.Module):
    def __init__(self, cfg, layer_type, layer_idx):
        super().__init__()
        self.layer_type = layer_type

        if layer_type == "full_attention":
            self.token_mixer = GroupedQueryAttention(
                d_in=cfg["emb_dim"],
                num_heads=cfg["n_heads"],
                head_dim=cfg["head_dim"],
                num_kv_groups=cfg["n_kv_groups"],
                qk_norm=cfg["qk_norm"],
                dtype=cfg["dtype"],
            )
        elif layer_type == "linear_attention":
            self.token_mixer = Qwen3_5GatedDeltaNet(_Qwen3_5ConfigAdapter(cfg), layer_idx)
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")

        self.ff = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg["emb_dim"], eps=cfg.get("rms_norm_eps", 1e-6))
        self.norm2 = RMSNorm(cfg["emb_dim"], eps=cfg.get("rms_norm_eps", 1e-6))

    def forward(self, x, mask, cos, sin):
        shortcut = x
        x = self.norm1(x)
        if self.layer_type == "full_attention":
            x = self.token_mixer(x, mask, cos, sin)
        else:
            x = self.token_mixer(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut
        return x


class Qwen3_5Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        layer_types = cfg.get("layer_types", ["full_attention"] * cfg["n_layers"])
        if len(layer_types) != cfg["n_layers"]:
            raise ValueError("len(layer_types) must equal n_layers")

        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg, layer_type, idx) for idx, layer_type in enumerate(layer_types)]
        )

        self.final_norm = RMSNorm(cfg["emb_dim"], eps=cfg.get("rms_norm_eps", 1e-6))
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        head_dim = cfg["emb_dim"] // cfg["n_heads"] if cfg["head_dim"] is None else cfg["head_dim"]
        cos, sin = compute_rope_params(
            head_dim=head_dim,
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"],
            partial_rotary_factor=cfg.get("partial_rotary_factor", 1.0),
            dtype=torch.float32,
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cfg = cfg

    def forward(self, in_idx):
        x = self.tok_emb(in_idx)

        num_tokens = x.shape[1]
        mask = torch.triu(
            torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool),
            diagonal=1,
        )

        for block in self.trf_blocks:
            x = block(x, mask, self.cos, self.sin)

        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg["dtype"]))
        return logits


# ---------------------------------------------------------------------------
# 2. Model configuration
# ---------------------------------------------------------------------------

QWEN3_5_CONFIG = {
    "vocab_size": 248_320,
    "context_length": 262_144,
    "emb_dim": 1_024,
    "n_heads": 8,
    "n_layers": 24,
    "hidden_dim": 3_584,
    "head_dim": 256,
    "qk_norm": True,
    "n_kv_groups": 2,
    "rope_base": 10_000_000.0,
    "partial_rotary_factor": 0.25,
    "rms_norm_eps": 1e-6,
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 16,
    "dtype": torch.bfloat16,
    "layer_types": [
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
    ],
}


# ---------------------------------------------------------------------------
# 3. Weight loading
# ---------------------------------------------------------------------------

def load_weights_into_qwen3_5(model, param_config, params):
    def assign(left, right, tensor_name="unknown"):
        if left.shape != right.shape:
            raise ValueError(
                f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}"
            )
        with torch.no_grad():
            if isinstance(right, torch.Tensor):
                left.copy_(right)
            else:
                left.copy_(torch.as_tensor(right, dtype=left.dtype, device=left.device))
        return left

    if "model.embed_tokens.weight" in params:
        model_prefix = "model"
    elif "model.language_model.embed_tokens.weight" in params:
        model_prefix = "model.language_model"
    else:
        raise KeyError("Could not find embed token weights in checkpoint.")

    def pkey(suffix):
        return f"{model_prefix}.{suffix}"

    model.tok_emb.weight = assign(
        model.tok_emb.weight, params[pkey("embed_tokens.weight")], pkey("embed_tokens.weight")
    )

    n_layers = param_config["n_layers"]
    layer_types = param_config.get("layer_types", ["full_attention"] * n_layers)

    for l in range(n_layers):
        block = model.trf_blocks[l]
        layer_type = layer_types[l]

        if layer_type == "full_attention":
            att = block.token_mixer
            att.W_query.weight = assign(att.W_query.weight, params[pkey(f"layers.{l}.self_attn.q_proj.weight")], pkey(f"layers.{l}.self_attn.q_proj.weight"))
            att.W_key.weight = assign(att.W_key.weight, params[pkey(f"layers.{l}.self_attn.k_proj.weight")], pkey(f"layers.{l}.self_attn.k_proj.weight"))
            att.W_value.weight = assign(att.W_value.weight, params[pkey(f"layers.{l}.self_attn.v_proj.weight")], pkey(f"layers.{l}.self_attn.v_proj.weight"))
            att.out_proj.weight = assign(att.out_proj.weight, params[pkey(f"layers.{l}.self_attn.o_proj.weight")], pkey(f"layers.{l}.self_attn.o_proj.weight"))
            if hasattr(att, "q_norm") and att.q_norm is not None:
                att.q_norm.weight = assign(att.q_norm.weight, params[pkey(f"layers.{l}.self_attn.q_norm.weight")], pkey(f"layers.{l}.self_attn.q_norm.weight"))
            if hasattr(att, "k_norm") and att.k_norm is not None:
                att.k_norm.weight = assign(att.k_norm.weight, params[pkey(f"layers.{l}.self_attn.k_norm.weight")], pkey(f"layers.{l}.self_attn.k_norm.weight"))

        elif layer_type == "linear_attention":
            lat = block.token_mixer
            lat.dt_bias = assign(lat.dt_bias, params[pkey(f"layers.{l}.linear_attn.dt_bias")], pkey(f"layers.{l}.linear_attn.dt_bias"))
            lat.A_log = assign(lat.A_log, params[pkey(f"layers.{l}.linear_attn.A_log")], pkey(f"layers.{l}.linear_attn.A_log"))
            lat.conv1d.weight = assign(lat.conv1d.weight, params[pkey(f"layers.{l}.linear_attn.conv1d.weight")], pkey(f"layers.{l}.linear_attn.conv1d.weight"))
            lat.norm.weight = assign(lat.norm.weight, params[pkey(f"layers.{l}.linear_attn.norm.weight")], pkey(f"layers.{l}.linear_attn.norm.weight"))
            lat.out_proj.weight = assign(lat.out_proj.weight, params[pkey(f"layers.{l}.linear_attn.out_proj.weight")], pkey(f"layers.{l}.linear_attn.out_proj.weight"))
            lat.in_proj_qkv.weight = assign(lat.in_proj_qkv.weight, params[pkey(f"layers.{l}.linear_attn.in_proj_qkv.weight")], pkey(f"layers.{l}.linear_attn.in_proj_qkv.weight"))
            lat.in_proj_z.weight = assign(lat.in_proj_z.weight, params[pkey(f"layers.{l}.linear_attn.in_proj_z.weight")], pkey(f"layers.{l}.linear_attn.in_proj_z.weight"))
            lat.in_proj_b.weight = assign(lat.in_proj_b.weight, params[pkey(f"layers.{l}.linear_attn.in_proj_b.weight")], pkey(f"layers.{l}.linear_attn.in_proj_b.weight"))
            lat.in_proj_a.weight = assign(lat.in_proj_a.weight, params[pkey(f"layers.{l}.linear_attn.in_proj_a.weight")], pkey(f"layers.{l}.linear_attn.in_proj_a.weight"))
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")

        block.norm1.weight = assign(block.norm1.weight, params[pkey(f"layers.{l}.input_layernorm.weight")], pkey(f"layers.{l}.input_layernorm.weight"))
        block.ff.fc1.weight = assign(block.ff.fc1.weight, params[pkey(f"layers.{l}.mlp.gate_proj.weight")], pkey(f"layers.{l}.mlp.gate_proj.weight"))
        block.ff.fc2.weight = assign(block.ff.fc2.weight, params[pkey(f"layers.{l}.mlp.up_proj.weight")], pkey(f"layers.{l}.mlp.up_proj.weight"))
        block.ff.fc3.weight = assign(block.ff.fc3.weight, params[pkey(f"layers.{l}.mlp.down_proj.weight")], pkey(f"layers.{l}.mlp.down_proj.weight"))
        block.norm2.weight = assign(block.norm2.weight, params[pkey(f"layers.{l}.post_attention_layernorm.weight")], pkey(f"layers.{l}.post_attention_layernorm.weight"))

    model.final_norm.weight = assign(model.final_norm.weight, params[pkey("norm.weight")], pkey("norm.weight"))

    if "lm_head.weight" in params:
        model.out_head.weight = assign(model.out_head.weight, params["lm_head.weight"], "lm_head.weight")
    elif pkey("lm_head.weight") in params:
        model.out_head.weight = assign(model.out_head.weight, params[pkey("lm_head.weight")], pkey("lm_head.weight"))
    else:
        model.out_head.weight = model.tok_emb.weight
        print("Model uses weight tying.")


# ---------------------------------------------------------------------------
# 4. Tokenizer
# ---------------------------------------------------------------------------

class Qwen3_5Tokenizer:
    _SPECIALS = [
        "<|endoftext|>",
        "<|im_start|>", "<|im_end|>",
        "<|object_ref_start|>", "<|object_ref_end|>",
        "<|box_start|>", "<|box_end|>",
        "<|quad_start|>", "<|quad_end|>",
        "<|vision_start|>", "<|vision_end|>",
        "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>",
        "<think>", "</think>",
    ]
    _SPLIT_RE = re.compile(r"(<\|[^>]+?\|>|<think>|</think>)")

    def __init__(
        self,
        tokenizer_file_path="tokenizer.json",
        repo_id=None,
        apply_chat_template=True,
        add_generation_prompt=False,
        add_thinking=False,
    ):
        self.apply_chat_template = apply_chat_template
        self.add_generation_prompt = add_generation_prompt
        self.add_thinking = add_thinking

        self._tok = Tokenizer.from_file(str(Path(tokenizer_file_path)))
        self._special_to_id = {}
        for t in self._SPECIALS:
            tid = self._tok.token_to_id(t)
            if tid is not None:
                self._special_to_id[t] = tid

        self.pad_token_id = self._special_to_id["<|endoftext|>"]
        self.eos_token_id = self.pad_token_id

        if repo_id and "Base" not in repo_id:
            eos_token = "<|im_end|>"
        else:
            eos_token = "<|endoftext|>"
        if eos_token in self._special_to_id:
            self.eos_token_id = self._special_to_id[eos_token]

    def encode(self, text, chat_wrapped=None):
        if chat_wrapped is None:
            chat_wrapped = self.apply_chat_template

        stripped = text.strip()
        if stripped in self._special_to_id and "\n" not in stripped:
            return [self._special_to_id[stripped]]

        if chat_wrapped:
            text = self._wrap_chat(text)

        ids = []
        for part in filter(None, self._SPLIT_RE.split(text)):
            if part in self._special_to_id:
                ids.append(self._special_to_id[part])
            else:
                ids.extend(self._tok.encode(part).ids)
        return ids

    def decode(self, ids):
        return self._tok.decode(ids, skip_special_tokens=False)

    def _wrap_chat(self, user_msg):
        s = f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        if self.add_generation_prompt:
            s += "<|im_start|>assistant\n"
            if self.add_thinking:
                s += "<think>\n"
            else:
                s += "<think>\n\n</think>\n\n"
        return s


# ---------------------------------------------------------------------------
# 5. Text generation
# ---------------------------------------------------------------------------

def generate_text_basic_stream(model, token_ids, max_new_tokens, eos_token_id=None):
    """Greedy decoding with streaming — yields one token tensor at a time."""
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(token_ids)[:, -1]
            next_token = torch.argmax(out, dim=-1, keepdim=True)

            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break

            yield next_token
            token_ids = torch.cat([token_ids, next_token], dim=1)


def calc_model_memory_size(model, input_dtype=torch.float32):
    total_params = sum(p.numel() for p in model.parameters())
    total_grads = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_buffers = sum(buf.numel() for buf in model.buffers())
    element_size = torch.tensor(0, dtype=input_dtype).element_size()
    total_memory_bytes = (total_params + total_grads + total_buffers) * element_size
    return total_memory_bytes / (1024 ** 3)


def run_generation(model, tokenizer, device, prompt, max_new_tokens=500):
    input_token_ids = tokenizer.encode(prompt)
    input_token_ids_tensor = torch.tensor(input_token_ids, device=device).unsqueeze(0)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start_time = time.perf_counter()
    generated_tokens = 0

    for token in generate_text_basic_stream(
        model=model,
        token_ids=input_token_ids_tensor,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
    ):
        generated_tokens += 1
        token_id = token.squeeze(0).tolist()
        print(tokenizer.decode(token_id), end="", flush=True)

    elapsed = time.perf_counter() - start_time
    tokens_per_sec = generated_tokens / elapsed if elapsed > 0 else 0.0
    print(f"\n\nGeneration speed: {tokens_per_sec:.2f} tokens/sec")

    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
        print(f"GPU memory used: {peak_gb:.2f} GB")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    repo_id = "Qwen/Qwen3.5-0.8B"
    local_dir = Path(repo_id).parts[-1]

    # --- Build model ---
    torch.manual_seed(123)
    model = Qwen3_5Model(QWEN3_5_CONFIG)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Unique parameters (weight tying): {total_params - model.tok_emb.weight.numel():,}")
    print(f"Memory (float32): {calc_model_memory_size(model, torch.float32):.2f} GB")
    print(f"Memory (bfloat16): {calc_model_memory_size(model, torch.bfloat16):.2f} GB")

    # --- Device ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    model.to(device)

    # --- Load pretrained weights ---
    repo_dir = snapshot_download(repo_id=repo_id, local_dir=local_dir)
    index_path = os.path.join(repo_dir, "model.safetensors.index.json")
    with open(index_path, "r") as f:
        index = json.load(f)

    weights_dict = {}
    for filename in sorted(set(index["weight_map"].values())):
        shard_path = os.path.join(repo_dir, filename)
        weights_dict.update(load_file(shard_path))

    load_weights_into_qwen3_5(model, QWEN3_5_CONFIG, weights_dict)
    model.to(device)
    del weights_dict

    # --- Load tokenizer ---
    hf_hub_download(repo_id=repo_id, filename="tokenizer.json", local_dir=local_dir)
    tokenizer = Qwen3_5Tokenizer(
        tokenizer_file_path=f"{local_dir}/tokenizer.json",
        repo_id=repo_id,
        apply_chat_template=True,
        add_generation_prompt=True,
        add_thinking=True,
    )

    # --- Generate ---
    print("\n--- Prompt 1 ---")
    run_generation(model, tokenizer, device, "Give me a short introduction to large language models.")

    print("\n--- Prompt 2 ---")
    run_generation(
        model, tokenizer, device,
        "A shop gives a 20% discount, then adds 10% tax. Is the final price higher or lower than the original? By how much?"
    )


if __name__ == "__main__":
    main()
