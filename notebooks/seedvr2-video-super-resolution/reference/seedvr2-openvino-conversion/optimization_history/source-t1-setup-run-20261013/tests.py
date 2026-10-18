"""CPU-only source-contract tests; no backend, model, payload, or device access."""
from __future__ import annotations

import ast
import importlib.util
import pathlib
import sys
import types
import unittest
from unittest.mock import patch

HERE=pathlib.Path(__file__).resolve().parent
MODULE_PATH=HERE/'source-only.py'
SPEC=importlib.util.spec_from_file_location('t1_setup_source',MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE=importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name]=MODULE
SPEC.loader.exec_module(MODULE)

class SourceOnlyTests(unittest.TestCase):
    def setUp(self):
        self.text=MODULE_PATH.read_text(encoding='utf-8')

    def test_source_is_small_and_stdlib_only_at_import(self):
        self.assertLessEqual(len(self.text.splitlines()),150)
        tree=ast.parse(self.text)
        allowed={'__future__','argparse','gc','hashlib','importlib','json','os','sys','pathlib'}
        imported=[]
        # Runtime backend imports are intentionally nested under run(); module
        # import/metadata mode must remain stdlib-only.
        for node in tree.body:
            if isinstance(node,ast.Import):imported.extend(a.name.split('.',1)[0] for a in node.names)
            elif isinstance(node,ast.ImportFrom) and node.module:imported.append(node.module.split('.',1)[0])
        self.assertTrue(set(imported)<=allowed,sorted(set(imported)-allowed))

    def test_metadata_pins_current_recovery_epoch(self):
        result=MODULE.source_metadata()
        self.assertEqual(result['status'],'PASS_SOURCE_ONLY')
        self.assertEqual(set(result['source_files']),{str(MODULE.DRIVER),str(MODULE.ROOT_CHECK),
            str(MODULE.RECOVERY),str(MODULE.RECOVERY_JSON),str(MODULE.ROOT_RECOVERY),
            str(MODULE.HEALTH),str(MODULE.OBSERVER)})
        self.assertEqual(result['source_files'][str(MODULE.RECOVERY)]['sha256'],MODULE.RECOVERY_SHA)
        self.assertEqual(result['source_files'][str(MODULE.RECOVERY_JSON)]['sha256'],MODULE.RECOVERY_JSON_SHA)
        self.assertEqual(result['source_files'][str(MODULE.ROOT_RECOVERY)]['sha256'],MODULE.ROOT_RECOVERY_SHA)
        for key in ('torch_imports','core_calls','device_queries','model_payload_reads','dit_forward_calls'):
            self.assertEqual(result[key],0)

    def test_reuses_frozen_seams_and_exact_encode(self):
        for required in ('driver.parent()','old.inputs()','openvino-vae-probe.py','uhd.vae_runtime',
                         'uhd.streamed_dit(torch,device)','tiled_encode','tile_size=(512,512)',
                         'tile_overlap=(64,64)','MEAN_SHA','mean=artifacts.mean_shape(mean)'):
            self.assertIn(required,self.text)
        self.assertNotIn('old.physical_admission',self.text)
        self.assertNotIn('model(',self.text)

    def test_real_artifact_mean_shape_contract_without_torch(self):
        class FakeArray:
            def __init__(self,shape):self.shape=shape;self.ndim=len(shape)
            def __getitem__(self,key):
                if key==(slice(None),slice(None),None):return FakeArray(self.shape[:2]+(1,)+self.shape[2:])
                raise AssertionError(key)
        path=MODULE.E/'source-whole-rgb-minimal-20261012/artifacts.py'
        spec=importlib.util.spec_from_file_location('t1_real_artifacts_contract',path)
        artifact=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,{'numpy':types.SimpleNamespace(ascontiguousarray=lambda value:value)}):
            spec.loader.exec_module(artifact)
            encoded=FakeArray((1,16,270,480))
            self.assertEqual(artifact.mean_shape(encoded).shape,(1,16,1,270,480))
        self.assertNotIn('torch',sys.modules)

    def test_actual_mean_guard_rejects_wrong_spatial_shape(self):
        calls=[node for node in ast.walk(ast.parse(self.text))
               if isinstance(node,ast.Call) and isinstance(node.func,ast.Name)
               and node.func.id=='require' and len(node.args)==2
               and isinstance(node.args[1],ast.Constant)
               and node.args[1].value=='T1 tiled VAE mean shape differs from frozen direct output']
        self.assertEqual(len(calls),1)
        guard=compile(ast.Expression(calls[0].args[0]),str(MODULE_PATH),'eval')
        for shape in ((1,16,270,480),(1,16,480,270),(1,16,269,480),(1,16,1,270,480)):
            self.assertEqual(eval(guard,{'__builtins__':{}},{'mean':types.SimpleNamespace(shape=shape)}),
                             shape==(1,16,270,480))

    def test_measures_public_shallow_clear_without_forward(self):
        before=self.text.index("'dit-setup-before-clear'")
        clear=self.text.index('clear_memory(deep=False,force=True)')
        after=self.text.index("'dit-setup-after-shallow-clear'")
        release=self.text.index("'after-dit-release'")
        self.assertLess(before,clear);self.assertLess(clear,after);self.assertLess(after,release)
        self.assertIn("'dit_forward_calls':0",self.text)
        self.assertIn("require(receipt['dit_forward_calls']==0",self.text)
        self.assertIn("'memory_allocated','memory_reserved'",self.text)
        self.assertIn("'max_memory_allocated','max_memory_reserved'",self.text)

    def test_cpu_vae_is_retained_until_after_dit_setup(self):
        self.assertLess(self.text.index("'after-vae-release'"),self.text.index('uhd.streamed_dit'))
        self.assertLess(self.text.index("'after-dit-release'"),self.text.index('del dit,vae'))
        self.assertIn("_standard_model_movement(vae,device,cpu,'cpu','VAE')",self.text)
        self.assertIn("_standard_model_movement(dit,device,cpu,'cpu','DiT')",self.text)

    def test_no_t5_or_old_124gb_gate_is_embedded(self):
        for forbidden in ('T5GateAuthority','T5_CONTIGUOUS','10_972_581_888','10972581888',
                          '12_400_000_000','12400000000','physical_admission'):
            self.assertNotIn(forbidden,self.text)
        self.assertIn("'old_body_gate_invoked':False",self.text)

    def test_runtime_has_no_retry_or_generic_backend_path(self):
        self.assertNotIn('subprocess',self.text)
        self.assertNotIn('openvino.Core',self.text)
        self.assertIn("'no_retry':True",self.text)
        self.assertIn('output.parent==HERE',self.text)
        self.assertIn('not output.exists()',self.text)

    def test_pre_fix_source_snapshot_is_preserved(self):
        old=(HERE/'source-v1.py').read_text(encoding='utf-8')
        self.assertIn("uhd-sdpa-unprofiled-run-20261012/authority.py",old)
        self.assertIn('mean=artifacts.mean_shape(mean)',old)
        self.assertNotEqual(old,self.text)

if __name__=='__main__':unittest.main()
