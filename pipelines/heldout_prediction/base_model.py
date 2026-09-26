"""Exact Exp243 output write with separate same-latent reference controls.

No internal-layer selection and no behaviour-derived sign correction.
"""
from __future__ import annotations

import re
import sys

import torch

from common import CONFIG, ROOT, stable_seed

GENERIC = ("person", "man", "woman", "people", "girl", "boy", "child", "worker")
SPECS = {"sd3": {"dose": 2., "steps": 28, "timesteps": (6, 10)},
         "flux": {"dose": 1., "steps": 8, "timesteps": (2, 3)},
         "qwen": {"dose": 2., "steps": 50, "timesteps": (10, 16)}}


class NotWritable(ValueError):
    """An explicit data/operator coverage status, not a runtime exception."""


def diff_rows(a, b):
    length = min(a.shape[0], b.shape[0])
    if length == 0:
        return []
    norms = (a[:length] - b[:length]).norm(dim=-1).float()
    return torch.where(norms > max(1e-3, .05 * float(norms.max())))[0].tolist()


def cosine(a, b):
    x, y = a.float().flatten(), b.float().flatten()
    denom = float(x.norm()) * float(y.norm())
    if denom == 0:
        raise NotWritable("zero_response_or_reference")
    return float((x @ y) / denom)


def norm_matched(direction, label):
    generator = torch.Generator(device="cpu").manual_seed(stable_seed(label))
    vector = torch.randn(direction.shape, generator=generator, dtype=torch.float32)
    return vector * (float(direction.float().norm()) / float(vector.norm()))


class Adapter:
    def __init__(self, model):
        sys.path.insert(0, str(ROOT))
        from experiments.interp_program.exp243_gen import build
        self.model, self.spec = model, SPECS[model]
        self.pipe, self._encode, self._generate = build(model)
        self.pipe.set_progress_bar_config(disable=True)
        torch.set_grad_enabled(False)
        self._capture = None
        if model == "flux":
            self.pipe.transformer.register_forward_pre_hook(self._flux_hook, with_kwargs=True)
        if model == "qwen":
            self.unconditional = self.encode(" ")

    def encode(self, text):
        raw, mask = self._encode(text)
        return {"raw": raw, "pe": raw[0] if isinstance(raw, tuple) else raw,
                "mask": mask, "text": text}

    def replace(self, pack, pe):
        result = dict(pack)
        result["pe"] = pe
        result["raw"] = (pe,) + pack["raw"][1:] if isinstance(pack["raw"], tuple) else pe
        result.pop("flux_text_kwargs", None)
        return result

    def write(self, pack, direction, anchors):
        pe = pack["pe"].clone()
        for k in anchors[:2]:
            if not 0 <= k < pe.shape[1]:
                raise AssertionError("anchor outside conditioning tensor")
            pe[0, k] = pe[0, k] + self.spec["dose"] * direction.to(pe.device, pe.dtype)
        if torch.equal(pe, pack["pe"]):
            raise NotWritable("zero_executed_write")
        return self.replace(pack, pe)

    def prepare(self, row):
        stereo, anti = self.encode(row["prompt_stereotype"]), self.encode(row["prompt_anti_stereotype"])
        a, b = stereo["pe"][0], anti["pe"][0]
        changed = diff_rows(a, b)
        if not changed:
            raise NotWritable("identical_or_zero_contrast")
        # Keep native subtraction before float conversion, as in the frozen driver.
        direction = torch.stack([(b[k] - a[k]).float().cpu() for k in changed[:2]]).mean(0)
        if float(direction.norm()) == 0:
            raise NotWritable("zero_pooled_direction")
        neutral = self.encode(row["prompt_neutral"])
        candidates = []
        for key in ("head", "target"):
            value = row.get(key)
            if isinstance(value, str):
                candidates.extend([w for w in re.findall(r"[A-Za-z]+", value) if len(w) > 2][:2])
        anchors = []
        for word in [*candidates, *GENERIC]:
            if not re.search(rf"\b{word}\b", row["prompt_neutral"], flags=re.I):
                continue
            other = self.encode(re.sub(rf"\b{word}\b", "thing", row["prompt_neutral"], count=1, flags=re.I))
            anchors = diff_rows(neutral["pe"][0], other["pe"][0])
            del other
            if anchors:
                break
        if not anchors:
            raise NotWritable("no_anchor")
        edited = self.write(neutral, direction, anchors)
        nulls = [norm_matched(direction, f"write:{self.model}:{row['id']}:{k}")
                 for k in range(CONFIG["null_directions"])]
        delta = edited["pe"].float() - neutral["pe"].float()
        return dict(neutral=neutral, stereo=stereo, anti=anti, edited=edited,
                    direction=direction, anchors=anchors[:2], nulls=nulls,
                    input_direction_norm=float(direction.norm()), executed_write_norm=float(delta.norm()),
                    T_rel=float(delta.norm()) / max(float(neutral["pe"].float().norm()), 1e-12),
                    pair_separability=float((b[:min(len(a), len(b))].float() - a[:min(len(a), len(b))].float()).norm()))

    def generate(self, pack, seed):
        return self._generate(pack["raw"], pack["mask"], seed)

    def _flux_hook(self, module, args, kwargs):
        capture = self._capture
        if capture is None:
            return
        capture["index"] += 1
        index = capture["index"]
        if index in capture["want"]:
            capture["got"][index] = {k: v.detach().clone() if torch.is_tensor(v) else v for k, v in kwargs.items()}
            capture["got"][index]["__args__"] = [v.detach().clone() if torch.is_tensor(v) else v for v in args]
            if capture["abort"]:
                raise _CaptureDone()

    def _flux_capture(self, pack, steps, want, seed, abort=False):
        self._capture = dict(index=-1, want=set(want), got={}, abort=abort)
        try:
            self.pipe(prompt_embeds=pack["pe"].to(torch.bfloat16), num_inference_steps=steps,
                      generator=torch.Generator("cuda").manual_seed(seed))
        except _CaptureDone:
            if not abort:
                raise
        finally:
            got = self._capture["got"]
            self._capture = None
        if set(got) != set(want):
            raise AssertionError("incomplete FLUX latent capture")
        return got

    def trajectory(self, neutral, seed):
        if self.model == "flux":
            return self._flux_capture(neutral, 8, (2, 3), seed)
        if self.model == "sd3":
            got = {}
            def capture(pipe, index, timestep, kwargs):
                if index in (6, 10):
                    got[index] = (kwargs["latents"].detach().clone(), timestep)
                return {}
            self.pipe(prompt=neutral["text"], num_inference_steps=28, guidance_scale=7.,
                      generator=torch.Generator("cuda").manual_seed(seed), callback_on_step_end=capture,
                      callback_on_step_end_tensor_inputs=["latents"])
            if set(got) != {6, 10}:
                raise AssertionError("incomplete SD3 latent capture")
            return got
        from experiments.interp_program import dit_block_ops as dbo
        from experiments.contrastive_denoising_guidance_pilot import cdg_loop
        z, timesteps, shapes, _, _ = dbo._prep(self.pipe, seed, 50)
        self.pipe.scheduler.set_begin_index(0)
        got = {}
        for i, t in enumerate(timesteps):
            if i in (10, 16):
                got[i] = (z.clone(), t, shapes)
            if i > 16:
                break
            cond = self._qwen(z, t, shapes, neutral, "cond")
            unc = self._qwen(z, t, shapes, self.unconditional, "uncond")
            combined = unc + 4. * (cond - unc)
            prediction = combined * (cdg_loop._per_token_norm(cond) / cdg_loop._per_token_norm(combined))
            z = self.pipe.scheduler.step(prediction, t, z, return_dict=False)[0]
        return got

    def _qwen(self, z, t, shapes, pack, context):
        with self.pipe.transformer.cache_context(context):
            return self.pipe.transformer(hidden_states=z, timestep=t.expand(z.shape[0]).to(z.dtype) / 1000,
                guidance=None, encoder_hidden_states_mask=pack["mask"], encoder_hidden_states=pack["pe"],
                img_shapes=shapes, attention_kwargs={}, return_dict=False)[0]

    def velocity(self, snapshot, pack, neutral, is_reference=False):
        if self.model == "qwen":
            z, t, shapes = snapshot
            return self._qwen(z, t, shapes, pack, "probe")[0].float()
        if self.model == "sd3":
            z, t = snapshot
            # Frozen reference keeps the neutral pooled projection for both poles.
            return self.pipe.transformer(hidden_states=z, timestep=t.reshape(1).to(z.dtype),
                encoder_hidden_states=pack["pe"], pooled_projections=neutral["raw"][2],
                return_dict=False)[0][0].float()
        kwargs = {k: v for k, v in snapshot.items() if k != "__args__"}
        kwargs["encoder_hidden_states"] = pack["pe"].to(snapshot["encoder_hidden_states"].dtype)
        if is_reference:
            if "flux_text_kwargs" not in pack:
                pack["flux_text_kwargs"] = self._flux_capture(pack, 1, (0,), 0, abort=True)[0]
            for key, value in pack["flux_text_kwargs"].items():
                current = kwargs.get(key)
                if key not in ("__args__", "encoder_hidden_states") and torch.is_tensor(value) and torch.is_tensor(current):
                    if current.shape != value.shape and snapshot["encoder_hidden_states"].shape[1] in current.shape:
                        kwargs[key] = value
            kwargs["encoder_hidden_states"] = pack["flux_text_kwargs"]["encoder_hidden_states"]
        out = self.pipe.transformer(*snapshot.get("__args__", []), **kwargs)
        out = out[0] if isinstance(out, (tuple, list)) else out.sample
        return out[0].float()

    def probe(self, row, prepared, references):
        p = prepared
        donor_packs = [(self.encode(r["prompt_stereotype"]), self.encode(r["prompt_anti_stereotype"])) for r in references]
        cells = []
        for seed in CONFIG["probe_seeds"]:
            snapshots = self.trajectory(p["neutral"], seed)
            for t in self.spec["timesteps"]:
                snapshot = snapshots[t]
                def response(pack, reference=False):
                    return self.velocity(snapshot, pack, p["neutral"], reference)
                v0 = response(p["neutral"])
                ref = response(p["anti"], True) - response(p["stereo"], True)
                delta = response(p["edited"]) - v0
                record = dict(seed=seed, timestep=t, cosine=cosine(delta, ref), magnitude=float(delta.norm()),
                              reference_norm=float(ref.norm()))
                record["random_write_cosines"] = [cosine(response(self.write(p["neutral"], v, p["anchors"])) - v0, ref) for v in p["nulls"]]
                record["unrelated_reference_cosines"] = [cosine(delta, response(b, True) - response(a, True)) for a, b in donor_packs]
                record["random_reference_cosines"] = []
                for k in range(CONFIG["random_reference_controls"]):
                    random_ref = norm_matched(ref.detach().cpu(), f"reference:{self.model}:{row['id']}:{seed}:{t}:{k}").to(delta.device)
                    record["random_reference_cosines"].append(cosine(delta, random_ref))
                    del random_ref
                cells.append(record)
                del delta, ref, v0
            del snapshots
        return cells


class _CaptureDone(Exception):
    """Intentional pre-forward capture stop, never a failed model operation."""
