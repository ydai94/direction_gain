"""Recovery contract tests; synthetic PNGs and fake adapter, no scientific inference."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from PIL import Image
import recovery as r
import control as c


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name); code = root/'code'; code.mkdir()
        original = r.CODE
        for name in ('config.json', 'protocol.json', 'cohort.json', 'cohort300.json'):
            shutil.copy2(original/name, code/name)
        self.patches = [patch.object(r, 'ROOT', root), patch.object(r, 'CODE', code),
                        patch.object(r, 'BASE', root/'out'), patch.object(r, 'release_binding', return_value='test')]
        for p in self.patches:
            p.start(); self.addCleanup(p.stop)
        self.folder = r.folder_for('sd3', 'smoke', 0); self.folder.mkdir(parents=True)
        self.key = r.selected('smoke', 0)[0][0]
        self.audit = dict(kind='generate', forward_calls=28, batch_size=2, unconditional_executed=True,
                          guidance_mode='linear_cfg', guidance_scale=4., snapshot_convention='actual_pre_forward', indices=[6,10])

    def record(self):
        return r.load_record(self.folder, 'sd3', 'smoke', 0, 'test')

    def item(self, arm='clean'):
        path = self.folder/f'{self.key}_{arm}_10.png'
        Image.new('RGB', (1024,1024)).save(path)
        return dict(triplet_key=self.key, arm=arm, seed=10, path=str(path.relative_to(r.ROOT)),
                    audit=self.audit, sha256=r.digest(path))

    def test_static_and_partition(self):
        self.assertEqual(r.preflight()['n_cases'],300)
        all_keys=[k for i in range(25) for k in r.selected('formal',i)[0]]
        self.assertEqual(len(set(all_keys)),300)
        self.assertEqual(sum(len(r.expected('formal',i)) for i in range(25)),3600)

    def test_missing_protocol_fails(self):
        (r.CODE/'protocol.json').unlink()
        with self.assertRaises(FileNotFoundError): r.preflight()

    def test_duplicate_subset_fails(self):
        p=r.CODE/'cohort300.json'; s=r.read_json(p);s['triplet_keys'][1]=s['triplet_keys'][0];r.write_json(p,s)
        with self.assertRaises(ValueError):r.preflight()

    def test_duplicate_manifest_fails(self):
        rec=self.record(); item=self.item();rec['images']=[item,item];r.write_json(self.folder/'manifest.json',rec)
        with self.assertRaises(ValueError):self.record()

    def test_repair_missing_corrupt_orphan(self):
        rec=self.record(); a=self.item();b=self.item('steer');rec['images']=[a,b]
        (r.ROOT/a['path']).unlink();(r.ROOT/b['path']).write_bytes(b'bad')
        (self.folder/'orphan.png.partial').write_bytes(b'partial')
        r.repair_record(rec,self.folder,'sd3')
        self.assertEqual(rec['images'],[]);self.assertEqual(len(rec['recovery_events']),3)

    def test_bad_audit_not_silently_repaired(self):
        rec=self.record();a=self.item();a['audit']={**self.audit,'guidance_scale':7.};rec['images']=[a]
        with self.assertRaises(ValueError):r.repair_record(rec,self.folder,'sd3')

    def test_incomplete_not_success(self):
        rec=self.record();rec['images']=[self.item()];r.write_json(self.folder/'manifest.json',rec)
        with self.assertRaises(ValueError):r.verify_shard('sd3','smoke',0)

    def test_atomic_duplicate_lock(self):
        lock=r.acquire(self.folder)
        with self.assertRaises(FileExistsError):r.acquire(self.folder)
        shutil.rmtree(lock)

    def test_interruption_resume_skips_verified_images(self):
        rec=self.record();rec['images']=[self.item()];r.write_json(self.folder/'manifest.json',rec)
        calls=[];audit=self.audit
        class Adapter:
            def __init__(self,model):self.last_audit=audit
            def reset_cache(self):pass
            def prepare(self,row):
                return dict(neutral='clean',edited='steer',input_direction_norm=1.,pair_separability=1.,
                            T_rel=.1,anchors=[1,2],executed_write_norm=1.)
            def generate(self,pack,seed):calls.append((pack,seed));return Image.new('RGB',(1024,1024))
        torch=types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda:True,
             get_device_properties=lambda n:types.SimpleNamespace(total_memory=80*1024**3)))
        with patch.dict(sys.modules,{'torch':torch,'adapter':types.SimpleNamespace(FixedAdapter=Adapter)}), \
             patch.object(r,'provenance',return_value={'gpu':'fake'}):
            self.assertEqual(r.generate('sd3','smoke',0,0),75)
            self.assertFalse((self.folder/'complete.json').exists())
            self.assertEqual(r.generate('sd3','smoke',0,3600),0)
            self.assertEqual(calls,[('steer',10)])
            self.assertEqual(r.generate('sd3','smoke',0,3600),0)
            self.assertEqual(len(calls),1)
        self.assertEqual(r.verify_shard('sd3','smoke',0)['n_images'],2)

    def test_controller_generation_error_stops(self):
        self.controller_scenario('FAILED','1:0',expected_stop=True)

    def test_controller_budget_resubmits(self):
        self.controller_scenario('FAILED','75:0',expected_stop=False)

    def controller_scenario(self,state,exit_code,expected_stop):
        ctrl=r.BASE/'control';ctrl.mkdir()
        obj=dict(model='sd3',status='ACTIVE',phase='smoke',release_sha256='test',
                 active=dict(job='123',shards=[0],stage='smoke',minutes=45),
                 retries={},progress={},no_progress={},jobs=[])
        r.write_json(ctrl/'sd3.json',obj)
        with patch.object(c,'BASE',r.BASE),patch.object(c,'preflight'),patch.object(c,'release_binding',return_value='test'), \
             patch.object(c,'accounting',return_value={'123_0':dict(state=state,exit_code=exit_code,elapsed='9')}), \
             patch.object(c,'verify_shard',side_effect=ValueError('incomplete')),patch.object(c,'note'), \
             patch.object(c,'folder_for',side_effect=r.folder_for),patch.object(c,'schedule') as schedule:
            c.advance('sd3')
            if expected_stop:
                self.assertEqual(r.read_json(ctrl/'sd3.json')['status'],'STOPPED');schedule.assert_not_called()
            else:
                schedule.assert_called_once();self.assertEqual(schedule.call_args.args[1],[0])


if __name__=='__main__':unittest.main()
