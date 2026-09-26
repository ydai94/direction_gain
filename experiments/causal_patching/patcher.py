"""Forward-hook utility for activation patching on Qwen2_5_VLDecoderLayer.

The Qwen2.5-VL decoder layer's `forward` returns a bare Tensor (not a tuple),
so the hook return value must also be a Tensor (or None to leave unchanged).

Usage:
    with Patcher() as p:
        p.donor_act = donor_H_L           # (1, T_donor, 3584) on cuda
        p.patch_mask = mask               # (1, T_recip) bool on cuda
        p.mode = "apply"
        p.install(text_encoder.model.layers[L - 1])
        out = text_encoder(input_ids=..., attention_mask=...,
                           output_hidden_states=False, use_cache=False)
"""

import torch


class Patcher:
    """Hook holder. Modes: off / record / apply.

    apply mode replaces, position-by-position, recipient[p] with donor[p]
    where patch_mask[p] is True; positions outside [0, min(T_recip, T_donor))
    are left untouched.
    """

    def __init__(self):
        self.mode = "off"
        self.donor_act = None        # (1, T_donor, D) on the layer's device
        self.patch_mask = None       # (1, T_recip) bool, on the layer's device
        self.recorded = None         # set when mode == "record"
        self._handle = None

    def _hook(self, module, args, output):
        # Qwen2_5_VLDecoderLayer.forward returns a Tensor; rarely a tuple.
        # Defensively support both — modify [0] if tuple.
        if isinstance(output, tuple):
            hs = output[0]
            modified = self._modify(hs)
            if modified is None:
                return None
            return (modified,) + output[1:]
        modified = self._modify(output)
        return modified  # may be None to leave unchanged

    def _modify(self, hs):
        if self.mode == "record":
            self.recorded = hs.detach().clone()
            return None
        if self.mode == "apply":
            assert self.donor_act is not None and self.patch_mask is not None
            T_recip = hs.shape[1]
            T_donor = self.donor_act.shape[1]
            T = min(T_recip, T_donor)
            if T == 0:
                return None
            mask3 = self.patch_mask[:, :T].to(hs.device).unsqueeze(-1).to(hs.dtype)
            donor = self.donor_act[:, :T, :].to(device=hs.device, dtype=hs.dtype)
            patched = hs.clone()
            patched[:, :T, :] = mask3 * donor + (1 - mask3) * hs[:, :T, :]
            return patched
        return None  # off: no modification

    def install(self, layer_module):
        assert self._handle is None, "Patcher already installed"
        self._handle = layer_module.register_forward_hook(self._hook)

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self.mode = "off"
        self.donor_act = None
        self.patch_mask = None
        self.recorded = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.remove()
        return False
