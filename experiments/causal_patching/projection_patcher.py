"""Projection / additive / k-subspace intervention hooks.

Extends `Patcher` with three new modes.

mode="project" (existing, byte-for-byte preserved):

    Raw:      h'[mask] = h[mask] - alpha * (h[mask] . d_hat) * d_hat
    Centered: h'[mask] = h[mask] - alpha * ((h[mask] - mu) . d_hat) * d_hat

mode="project_k" (new): orthogonal projection onto orthogonal complement of
                       span(B), for B = (k, D) orthonormal rows.

    Centered: h'[mask] = h[mask] - alpha * ((h[mask] - mu) @ B.T @ B)

    Equivalent to k sequential 1-D projections with d_j = B[j] when
    rows of B are orthonormal. Setting alpha=1.0 reproduces the
    parameter-free projection onto the orthogonal complement.

mode="add" (new): pure additive steering. No scalar from h.

    h'[mask] = h[mask] + alpha * d

    Identical pattern used by paper-1 SV-Tail-GT (additive, broadcast).
    Set patch_mask to all-True for "broadcast to all positions".

For all modes, positions outside `patch_mask` are left untouched.

`d` is any D-dim vector (unit norm is conventional but not required for
`mode="add"`; required for `mode="project"` and `mode="project_k"`).
`B` is (k, D) — rows are orthonormal.

Usage:

    with ProjectionPatcher() as p:
        # project (existing)
        p.direction = d_hat                # (D,) unit, on cpu or device
        p.alpha = 1.0
        p.mu = mu_L_scope                  # (D,) OR None (no centering)
        p.patch_mask = mask                # (1, T) bool
        p.mode = "project"

        # project_k (new)
        p.basis = B                        # (k, D) orthonormal rows
        p.alpha = 1.0
        p.mu = mu_L_scope                  # (D,) OR None
        p.patch_mask = mask
        p.mode = "project_k"

        # add (new)
        p.direction = sv                   # (D,) any norm
        p.alpha = 1.0
        p.patch_mask = mask                # all-True for broadcast
        p.mode = "add"

        p.install(text_encoder.model.language_model.layers[L - 1])
        out = text_encoder(input_ids=..., attention_mask=...,
                           output_hidden_states=False, use_cache=False)
"""

import torch

from experiments.causal_patching.patcher import Patcher


class ProjectionPatcher(Patcher):
    """Hook with three intervention modes: project, project_k, add."""

    def __init__(self):
        super().__init__()
        self.direction = None        # (D,) — used by project and add
        self.basis = None            # (k, D) orthonormal rows — used by project_k
        self.alpha = 1.0
        self.mu = None               # (D,) OR None — used by project and project_k

    def _modify(self, hs):
        if self.mode == "project":
            assert self.direction is not None
            assert self.patch_mask is not None
            T = hs.shape[1]
            mask = self.patch_mask[:, :T].to(hs.device)               # (1, T)
            d = self.direction.to(device=hs.device, dtype=hs.dtype)   # (D,)
            d_row = d.view(1, 1, -1)                                  # (1, 1, D)
            if self.mu is not None:
                mu = self.mu.to(device=hs.device, dtype=hs.dtype)
                hs_for_scalar = hs - mu.view(1, 1, -1)
            else:
                hs_for_scalar = hs
            scalar = (hs_for_scalar * d_row).sum(dim=-1, keepdim=True)  # (1, T, 1)
            delta = self.alpha * scalar * d_row                         # (1, T, D)
            mask3 = mask.unsqueeze(-1).to(hs.dtype)                     # (1, T, 1)
            return hs - mask3 * delta

        if self.mode == "project_k":
            assert self.basis is not None
            assert self.patch_mask is not None
            T = hs.shape[1]
            mask = self.patch_mask[:, :T].to(hs.device)                 # (1, T)
            B = self.basis.to(device=hs.device, dtype=hs.dtype)         # (k, D)
            if self.mu is not None:
                mu = self.mu.to(device=hs.device, dtype=hs.dtype)
                hs_for_scalar = hs - mu.view(1, 1, -1)                  # (1, T, D)
            else:
                hs_for_scalar = hs
            coeffs = torch.matmul(hs_for_scalar, B.t())                 # (1, T, k)
            delta = self.alpha * torch.matmul(coeffs, B)                # (1, T, D)
            mask3 = mask.unsqueeze(-1).to(hs.dtype)                     # (1, T, 1)
            return hs - mask3 * delta

        if self.mode == "add":
            assert self.direction is not None
            assert self.patch_mask is not None
            T = hs.shape[1]
            mask = self.patch_mask[:, :T].to(hs.device)                 # (1, T)
            d = self.direction.to(device=hs.device, dtype=hs.dtype)     # (D,)
            d_row = d.view(1, 1, -1)                                    # (1, 1, D)
            delta = self.alpha * d_row.expand(hs.shape[0], T, -1)       # (1, T, D)
            mask3 = mask.unsqueeze(-1).to(hs.dtype)                     # (1, T, 1)
            return hs + mask3 * delta

        return super()._modify(hs)
