"""Frozen output operators, measured BF16 budget matching, and Qwen adapter."""
import sys
import math
from pathlib import Path
from common import CONFIG, ROOT

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def tensor_hash(tensor):
    from exp264b_common import sha256_tensor
    return sha256_tensor(tensor)


def norm(tensor):
    return float(tensor.float().norm())


def positional(neutral, stereo, anti):
    import torch.nn.functional as functional
    length = max(stereo.shape[1], anti.shape[1], neutral.shape[1])
    s = functional.pad(stereo.float(), (0, 0, 0, length - stereo.shape[1]))
    a = functional.pad(anti.float(), (0, 0, 0, length - anti.shape[1]))
    return (a - s)[:, :neutral.shape[1]].to(neutral.dtype)


def match_budget(neutral, raw, budget):
    """Find a multiplier using executed native-dtype norms, without model scores."""
    import torch
    if budget == 0:
        return neutral.clone(), 0.0
    raw_norm = norm(raw)
    if raw_norm <= 1e-8 or not math.isfinite(raw_norm):
        raise AssertionError("zero or invalid raw direction")
    def evaluate(scale):
        edited = neutral + (raw.float() * scale).to(neutral.dtype)
        return edited, norm(edited - neutral)
    lo, hi = 0.0, budget / raw_norm * 2.0
    for _ in range(20):
        _, actual = evaluate(hi)
        if actual >= budget:
            break
        hi *= 2.0
    else:
        raise AssertionError("cannot bracket executed norm")
    best = None
    for _ in range(40):
        scale = (lo + hi) / 2
        edited, actual = evaluate(scale)
        error = abs(actual - budget) / budget
        if best is None or error < best[0]:
            best = (error, edited, scale)
        if actual < budget:
            lo = scale
        else:
            hi = scale
    if best[0] > CONFIG["norm_tolerance"] or not torch.isfinite(best[1]).all():
        raise AssertionError(f"executed norm mismatch: {best[0]}")
    return best[1], best[2]


def build_conditionings(neutral, stereo, anti, pooled_unit, pooled_raw_norm, rows):
    import torch
    mask = torch.zeros_like(neutral, dtype=torch.bool)
    mask[:, rows, :] = True
    d = positional(neutral, stereo, anti)
    pooled = pooled_unit.to(neutral.device).float().view(1, 1, -1)
    raw = {"F": d, "M": d * mask, "A": pooled.expand_as(neutral) * mask}
    reference = float(pooled_raw_norm) * math.sqrt(len(rows))
    edits = {"C": neutral.clone(), "F2": neutral + 2.0*d,
             "A2": neutral + 2.0 * pooled.to(neutral.dtype) * mask.to(neutral.dtype)}
    metadata = {}
    for family, shape in raw.items():
        for i, dose in enumerate(CONFIG["doses"]):
            arm = f"{family}_d{i}"
            structural_zero = norm(shape) <= 1e-8
            if structural_zero and family != "M":
                raise AssertionError("full/pooled direction is unexpectedly zero")
            if structural_zero:
                edits[arm], multiplier = neutral.clone(), 0.0
            else:
                edits[arm], multiplier = match_budget(neutral, shape, dose * reference)
            metadata[arm] = {"dose": dose, "requested_norm": dose * reference,
                             "multiplier": multiplier, "structural_zero": structural_zero,
                             "budget_feasible": not structural_zero}
    for arm, edited in edits.items():
        delta = edited - neutral
        active = delta.float().norm(dim=-1)[0] > 0
        if arm.startswith(("M", "A")) and torch.count_nonzero(delta[~mask]):
            raise AssertionError("localized edit spills outside anchor")
        info = metadata.setdefault(arm, {"dose": None, "requested_norm": None, "multiplier": None})
        info.update({"actual_norm": norm(delta), "reference_norm": reference,
                     "relative_full_norm": norm(delta) / max(norm(neutral), 1e-8),
                     "relative_region_norm": norm(delta) / max(norm(neutral[:, active]), 1e-8),
                     "active_rows": torch.where(active)[0].cpu().tolist(),
                     "conditioning_sha256": tensor_hash(edited)})
    return edits, metadata


def load_pipe():
    from experiments.causal_patching.run_three_methods import load_pipe as load
    import torch
    if torch.cuda.get_device_properties(0).total_memory < 75 * 1024**3:
        raise RuntimeError("generation/probe requires an >=75 GiB device")
    pipe = load()
    for module in (pipe.text_encoder, pipe.transformer, pipe.vae):
        module.eval()
        module.requires_grad_(False)
    return pipe


def encode_case(pipe, case):
    import exp256_gradient_screen_qwen as core
    from exp264b_common import output_rows_for_pole, unit_direction
    neutral, mask = core.encode(pipe, case["prompt_neutral"])
    stereo, stereo_mask = core.encode(pipe, case["prompt_stereotype"])
    anti, anti_mask = core.encode(pipe, case["prompt_anti_stereotype"])
    sr, ar = output_rows_for_pole(pipe.tokenizer, case["prompt_stereotype"], case["prompt_anti_stereotype"])
    unit, raw_norm = unit_direction(stereo, sr, anti, ar)
    edits, metadata = build_conditionings(neutral, stereo, anti, unit, raw_norm, case["output_rows"])
    return edits, metadata, (mask, stereo, stereo_mask, anti, anti_mask)


def render(pipe, embedding, mask, seed):
    import torch
    return pipe(prompt_embeds=embedding, prompt_embeds_mask=mask,
                num_inference_steps=CONFIG["steps"], true_cfg_scale=CONFIG["cfg"],
                negative_prompt=" ", generator=torch.Generator("cuda").manual_seed(seed)).images[0]
