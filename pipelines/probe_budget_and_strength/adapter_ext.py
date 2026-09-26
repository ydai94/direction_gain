"""Exp280 adapter: the frozen Exp271 fixed-output adapter with a configurable set of
probe timesteps. Nothing else changes -- the generation recipe, the write operator, the
CFG convention and the forward audit are inherited unmodified.

SD3 and FLUX run their full trajectory regardless of which states are captured, so
capturing more states costs no extra forwards and the inherited audit (which expects
exactly `steps` forwards) still holds. Qwen's trajectory stops at its last probe step
and its audit expects that length, so its probe timesteps stay at the deployed (10, 16)
and the inherited code path is used verbatim.
"""
from adapter import FixedAdapter

QWEN_FIXED = (10, 16)


class ExtAdapter(FixedAdapter):
    def __init__(self, model, indices):
        super().__init__(model)
        idx = tuple(int(i) for i in indices)
        if len(set(idx)) != len(idx):
            raise AssertionError('duplicate probe timestep')
        if model == 'qwen' and set(idx) != set(QWEN_FIXED):
            raise AssertionError('Qwen probe timesteps are fixed at (10, 16); '
                                 'extending them would change the audited trajectory length')
        if not all(0 <= i < self.recipe['steps'] for i in idx):
            raise AssertionError(f'probe timestep outside 0..{self.recipe["steps"] - 1}')
        self.ext_indices = idx
        self.recipe = dict(self.recipe, indices=list(idx))
        self.spec = dict(self.spec, timesteps=idx)
        self.trajectory_audits = []

    def trajectory(self, neutral, seed):
        if self.model == 'qwen':
            return super().trajectory(neutral, seed)
        if self.model == 'flux':
            snapshots = self.audited(
                lambda: self._flux_capture(neutral, self.recipe['steps'], self.ext_indices, seed),
                'trajectory')
        else:
            self.snapshot_capture = {}
            try:
                self.audited(lambda: self.call_pipe(neutral, seed, latent=True), 'trajectory')
                snapshots = self.snapshot_capture
            finally:
                self.snapshot_capture = None
            if set(snapshots) != set(self.ext_indices):
                raise AssertionError(f'missing SD3 pre-forward snapshots: got {sorted(snapshots)}')
        self.trajectory_audits.append(dict(self.last_audit, seed=seed))
        return snapshots
