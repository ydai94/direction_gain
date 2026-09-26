"""Fixed output write; explicit generation recipe and actual pre-forward snapshots."""
from pathlib import Path
import torch
from common import CODE, read_json
from base_model import Adapter, _CaptureDone

PROTOCOL = read_json(CODE / 'protocol.json')


def tree(value, device):
    if torch.is_tensor(value):
        return value.detach().to(device)
    if isinstance(value, tuple):
        return tuple(tree(v, device) for v in value)
    return value


class FixedAdapter(Adapter):
    def __init__(self, model):
        self.encoding_cache = {}
        self.audit_enabled = False
        self.forward_batches = []
        self.snapshot_capture = None
        if model == 'qwen':
            # Import lazy numerical helpers before capturing source provenance.
            from experiments.interp_program import dit_block_ops
            from experiments.contrastive_denoising_guidance_pilot import cdg_loop
        super().__init__(model)
        self.recipe = PROTOCOL['settings'][model]
        self.pipe.transformer.register_forward_pre_hook(self.audit_hook, with_kwargs=True)

    def encode(self, text):
        if text not in self.encoding_cache:
            pack = super().encode(text)
            self.encoding_cache[text] = (tree(pack['raw'], 'cpu'), tree(pack['mask'], 'cpu'))
        raw, mask = self.encoding_cache[text]
        raw, mask = tree(raw, 'cuda'), tree(mask, 'cuda')
        return dict(raw=raw, pe=raw[0] if isinstance(raw, tuple) else raw, mask=mask, text=text)

    def reset_cache(self):
        self.encoding_cache.clear()

    def audit_hook(self, module, args, kwargs):
        if not self.audit_enabled:
            return
        index = len(self.forward_batches)
        self.forward_batches.append(int(kwargs['hidden_states'].shape[0]))
        if self.model == 'qwen' and self.snapshot_capture is not None:
            if index in (20, 32):
                self.snapshot_capture[index//2] = (kwargs['hidden_states'].detach().clone(), kwargs['timestep'].detach().clone())
            return
        if self.snapshot_capture is not None and index in self.recipe['indices']:
            z, t = kwargs['hidden_states'], kwargs['timestep']
            if z.shape[0] != 2 or not torch.equal(z[:1], z[1:]):
                raise AssertionError('SD3 CFG trajectory must duplicate identical latents')
            if t.numel() != 2 or not torch.equal(t[:1], t[1:]):
                raise AssertionError('SD3 CFG timestep pair differs')
            self.snapshot_capture[index] = (z[:1].detach().clone(), t[:1].detach().clone())

    def audited(self, fn, kind):
        self.forward_batches = []
        self.audit_enabled = True
        try:
            out = fn()
        finally:
            self.audit_enabled = False
        n = self.recipe['steps']
        if self.model == 'qwen':
            n = 100 if kind == 'generate' else 34
        batch = 2 if self.model == 'sd3' else 1
        if self.forward_batches != [batch] * n:
            raise AssertionError(f'Unexpected {self.model} {kind} forwards: {self.forward_batches}')
        self.last_audit = dict(kind=kind, forward_calls=n, batch_size=batch,
            unconditional_executed=self.model != 'flux', guidance_mode=self.recipe['guidance_mode'],
            guidance_scale=self.recipe['guidance_scale'], snapshot_convention='actual_pre_forward',
            indices=self.recipe['indices'])
        return out

    def call_pipe(self, pack, seed, latent=False):
        kw = dict(num_inference_steps=self.recipe['steps'], height=1024, width=1024,
                  generator=torch.Generator('cuda').manual_seed(seed),
                  output_type='latent' if latent else 'pil')
        if self.model == 'sd3':
            pe, ne, pp, np_ = pack['raw']
            return self.pipe(prompt_embeds=pe, negative_prompt_embeds=ne,
                pooled_prompt_embeds=pp, negative_pooled_prompt_embeds=np_,
                guidance_scale=self.recipe['guidance_scale'], **kw)
        # Distilled Klein ignores CFG; 1 avoids an ignored-guidance warning.
        return self.pipe(prompt_embeds=pack['pe'].to(torch.bfloat16), guidance_scale=1., **kw)

    def generate(self, pack, seed):
        if self.model == 'qwen':
            return self.audited(lambda: Adapter.generate(self, pack, seed), 'generate')
        return self.audited(lambda: self.call_pipe(pack, seed).images[0], 'generate')

    def _flux_capture(self, pack, steps, want, seed, abort=False):
        self._capture = dict(index=-1, want=set(want), got={}, abort=abort)
        try:
            self.pipe(prompt_embeds=pack['pe'].to(torch.bfloat16), num_inference_steps=steps,
                guidance_scale=1., height=1024, width=1024, output_type='latent',
                generator=torch.Generator('cuda').manual_seed(seed))
        except _CaptureDone:
            if not abort:
                raise
        finally:
            got = self._capture['got']
            self._capture = None
        if set(got) != set(want):
            raise AssertionError('incomplete FLUX pre-forward capture')
        return got

    def trajectory(self, neutral, seed):
        if self.model == 'qwen':
            self.snapshot_capture = {}
            try:
                snapshots = self.audited(lambda: Adapter.trajectory(self, neutral, seed), 'trajectory')
                actual = self.snapshot_capture
            finally:
                self.snapshot_capture = None
            if set(actual) != {10, 16} or set(snapshots) != {10, 16}:
                raise AssertionError('missing actual Qwen pre-forward states')
            for index, (z, t, shapes) in snapshots.items():
                az, at = actual[index]
                if not torch.equal(z, az) or not torch.equal(t.expand(z.shape[0]).to(z.dtype)/1000, at):
                    raise AssertionError('Qwen returned state differs from actual forward input')
        elif self.model == 'flux':
            snapshots = self.audited(lambda: self._flux_capture(neutral, 8, (2, 3), seed), 'trajectory')
        else:
            self.snapshot_capture = {}
            try:
                self.audited(lambda: self.call_pipe(neutral, seed, latent=True), 'trajectory')
                snapshots = self.snapshot_capture
            finally:
                self.snapshot_capture = None
            if set(snapshots) != {6, 10}:
                raise AssertionError('missing SD3 pre-forward snapshots')
        self.trajectory_audits.append(dict(self.last_audit, seed=seed))
        return snapshots
