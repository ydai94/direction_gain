"""Encoder forward-pass orchestration for activation patching.

Provides:
  - clean_forward: run text encoder, optionally record per-layer activations
    via record-mode patcher hooks. Returns input_ids, attention_mask,
    last_hidden_state, and (if requested) a dict[layer_idx_1based -> tensor]
    of recorded layer outputs.
  - patched_forward: install a single apply-mode hook at layers[L-1], run
    the encoder, return last_hidden_state.
  - pipeline_encode_prompt_native: thin wrapper around
    `pipe._get_qwen_prompt_embeds` for parity verification.
  - post_process_for_dit: replicate the diffusers QwenImagePipeline pipeline
    post-encoder steps (gather non-pad, drop first 34, repad with zeros,
    return new prompt_embeds + all-ones mask). Single-sample version.
  - build_patch_mask: per-position bool mask over CONTEXT (or
    CONTEXT ∪ TEMPLATE_TRAIL) positions present in BOTH recipient and donor.

All shapes are batch=1 throughout. The encoder is the OUTER
`Qwen2_5_VLForConditionalGeneration` (matching what the diffusers pipeline
calls). The inner block list is at `text_encoder.model.layers`.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Set, Tuple

import torch

from experiments.causal_patching.patcher import Patcher
from experiments.layer_probing import config_local as C


# ---------------------------------------------------------------------------
# Module routing
# ---------------------------------------------------------------------------

def _text_decoder(text_encoder):
    """Return the actual text decoder (Qwen2_5_VLTextModel) regardless of
    whether `text_encoder` is the outer Qwen2_5_VLForConditionalGeneration
    (with `.model.language_model`) or the inner VLModel."""
    m = text_encoder.model if hasattr(text_encoder, "model") else text_encoder
    if hasattr(m, "language_model"):
        m = m.language_model
    if not hasattr(m, "layers"):
        raise RuntimeError(
            f"Could not locate text decoder layers from {type(text_encoder).__name__}"
        )
    return m


# ---------------------------------------------------------------------------
# Tokenization (mirrors diffusers _get_qwen_prompt_embeds for batch=1)
# ---------------------------------------------------------------------------

def tokenize_wrapped(tokenizer, prompt: str, device: str = "cuda"):
    """Wrap with PROMPT_TEMPLATE and tokenize. Returns dict with keys
    input_ids (1, T), attention_mask (1, T) on device."""
    txt = C.PROMPT_TEMPLATE.format(prompt)
    enc = tokenizer(
        [txt],
        max_length=C.TOKENIZER_MAX_LENGTH + C.PROMPT_TEMPLATE_DROP_IDX,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in enc.items()}


# ---------------------------------------------------------------------------
# Forward passes
# ---------------------------------------------------------------------------

def clean_forward(
    text_encoder,
    tokenizer,
    prompt: str,
    record_layers: Iterable[int] = (),
    device: str = "cuda",
) -> Dict:
    """Run the encoder cleanly (no apply patches). If `record_layers` is
    given, install record-mode hooks on each requested layer (1-indexed),
    capture their outputs, and return them.

    Calls the INNER `text_encoder.model` (Qwen2_5_VLTextModel) directly so
    we can read `out.last_hidden_state`. For text-only input (no pixel_values)
    this is what the outer Qwen2_5_VLForConditionalGeneration delegates to,
    and the resulting last_hidden_state equals the outer's
    `output.hidden_states[-1]` (post-final-RMSNorm) which is what the
    diffusers QwenImagePipeline uses.

    Returns dict with keys:
      input_ids: (1, T) long
      attention_mask: (1, T) long
      last_hidden_state: (1, T, D) bf16 on device  (post-final-RMSNorm)
      recorded: dict[L -> (1, T, D) bf16]   (block output BEFORE final norm
                                             — when L=28 this is the
                                             pre-final-norm output of
                                             layers[27], which is what the
                                             apply hook also sees)
    """
    record_layers = tuple(record_layers)
    enc = tokenize_wrapped(tokenizer, prompt, device=device)
    inner = _text_decoder(text_encoder)

    patchers = {}
    try:
        for L in record_layers:
            assert 1 <= L <= C.NUM_HIDDEN_LAYERS, f"bad layer {L}"
            p = Patcher()
            p.mode = "record"
            p.install(inner.layers[L - 1])
            patchers[L] = p

        with torch.no_grad():
            out = inner(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                output_hidden_states=False,
                use_cache=False,
            )
        last_hs = out.last_hidden_state
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "last_hidden_state": last_hs,
            "recorded": {L: patchers[L].recorded for L in record_layers},
        }
    finally:
        for p in patchers.values():
            p.remove()


def patched_forward(
    text_encoder,
    tokenizer,
    prompt: str,
    layer_1based: int,
    donor_act: torch.Tensor,
    patch_mask: torch.Tensor,
    device: str = "cuda",
) -> Dict:
    """Forward the encoder on `prompt` with a single apply-mode hook at
    layers[layer_1based - 1] that overwrites positions where patch_mask is
    True with donor_act values.

    donor_act: (1, T_donor, D) any device/dtype (will be moved to layer dtype/device)
    patch_mask: (1, T_recip) bool any device

    Returns dict with input_ids, attention_mask, last_hidden_state.
    """
    assert 1 <= layer_1based <= C.NUM_HIDDEN_LAYERS
    enc = tokenize_wrapped(tokenizer, prompt, device=device)
    inner = _text_decoder(text_encoder)

    with Patcher() as p:
        p.donor_act = donor_act
        p.patch_mask = patch_mask
        p.mode = "apply"
        p.install(inner.layers[layer_1based - 1])
        with torch.no_grad():
            out = inner(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                output_hidden_states=False,
                use_cache=False,
            )
    last_hs = out.last_hidden_state
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "last_hidden_state": last_hs,
    }


def pipeline_encode_prompt_native(pipe, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Call the diffusers pipeline's _get_qwen_prompt_embeds verbatim.
    Returns (prompt_embeds, prompt_embeds_mask) — already drop-applied and
    padded — exactly what the DiT consumes.
    """
    return pipe._get_qwen_prompt_embeds(prompt=prompt, device=pipe._execution_device)


# ---------------------------------------------------------------------------
# Diffusers post-encoder processing (single-sample reproduction)
# ---------------------------------------------------------------------------

def post_process_for_dit(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    drop_idx: int = C.PROMPT_TEMPLATE_DROP_IDX,
    target_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Replicate steps 2-6 of QwenImagePipeline._get_qwen_prompt_embeds for
    a single-sample batch:
      2. _extract_masked_hidden: gather non-pad positions
      3. drop the first `drop_idx` (34) tokens
      4. build new all-ones mask of length kept
      5. pad to max_seq_len (== kept_len for batch=1) with zeros
    Returns (prompt_embeds[1, T', D], prompt_embeds_mask[1, T']) on the same
    device. dtype follows last_hidden_state unless target_dtype given.
    """
    assert last_hidden_state.shape[0] == 1, "single-sample only"
    device = last_hidden_state.device
    bool_mask = attention_mask.bool()                    # (1, T)
    valid = last_hidden_state[0][bool_mask[0]]           # (T_valid, D)
    kept = valid[drop_idx:]                              # (T_valid - 34, D)
    if kept.shape[0] == 0:
        raise RuntimeError(
            "post_process_for_dit: prompt has no content tokens after drop_idx; "
            f"T_valid={valid.shape[0]}, drop_idx={drop_idx}"
        )
    new_mask = torch.ones(kept.shape[0], dtype=torch.long, device=device)
    prompt_embeds = kept.unsqueeze(0)                    # (1, T', D)
    prompt_embeds_mask = new_mask.unsqueeze(0)           # (1, T')
    if target_dtype is not None:
        prompt_embeds = prompt_embeds.to(dtype=target_dtype)
    return prompt_embeds, prompt_embeds_mask


# ---------------------------------------------------------------------------
# Patch-mask construction from token_labels.parquet
# ---------------------------------------------------------------------------

def build_patch_mask(
    labels_recipient,           # iterable of label strings, length T_recip
    labels_donor,               # iterable of label strings, length T_donor
    scope_set: Set[str],
    T_recip: int,
    T_donor: int,
) -> torch.Tensor:
    """Position-aligned mask: mask[p] = True iff
       labels_recipient[p] in scope AND p < T_donor AND labels_donor[p] in scope.
    Returns bool tensor of shape (1, T_recip).
    """
    rec = list(labels_recipient)
    don = list(labels_donor)
    assert len(rec) == T_recip, f"recipient labels {len(rec)} != T_recip {T_recip}"
    assert len(don) == T_donor, f"donor labels {len(don)} != T_donor {T_donor}"

    mask = torch.zeros((1, T_recip), dtype=torch.bool)
    T = min(T_recip, T_donor)
    for p in range(T):
        if rec[p] in scope_set and don[p] in scope_set:
            mask[0, p] = True
    return mask


def labels_for_row(token_labels_df, row_idx: int):
    """Return the per-position label list for a given row_idx, sorted by tok_pos."""
    sub = token_labels_df[token_labels_df["row_idx"] == row_idx].sort_values("tok_pos")
    return sub["label"].tolist()
