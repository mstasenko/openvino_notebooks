"""AST/metadata/tiny arithmetic only; no large fixture reads or Torch import."""
import ast
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import spec

class SourceContracts(unittest.TestCase):
    def test_actual_metadata_and_large_shapes_without_materialization(self):
        receipt,checked=spec.metadata()
        self.assertEqual(receipt['geometry']['total_tokens'],64800)
        self.assertEqual(checked['files']['normalized.bcthw.f32']['bytes'],497664000)
        self.assertEqual(np.prod(spec.SHAPES['mean'])*4,16588800)
        self.assertEqual(np.prod(spec.SHAPES['rgb']),124416000)
        self.assertNotIn('torch',sys.modules)
    def test_exact_original_temporal_vae_and_forward_call_arguments(self):
        source=ast.parse((spec.HERE/'child.py').read_text())
        original=ast.parse((spec.E/'temporal-native-prototype/xpu_t5_reference.py').read_text())
        for method in ('tiled_encode','tiled_decode'):
            def call(tree):return next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr==method)
            a,b=call(source),call(original)
            self.assertEqual([ast.dump(k) for k in a.keywords],[ast.dump(k) for k in b.keywords])
            self.assertEqual(len(a.args),len(b.args))
        text=(spec.HERE/'child.py').read_text()
        self.assertIn('disable_cache=False',text);self.assertIn('adapter.configure_prefix(model,20,',text)
        self.assertIn('ref.vae_runtime(model)',text);self.assertIn('ref.streamed_dit(torch,device)',text)
        self.assertNotIn('blocks_to_swap=0',text);self.assertNotIn('frames=1',text)
    def test_temporal_restore_and_quantizer_tiny_exact_formula(self):
        shape=(2,2,3,16);noise=np.arange(np.prod(shape),dtype=np.float32).reshape(shape)/np.float32(7)
        prediction=noise*np.float32(.25)
        with patch.dict(spec.SHAPES,{'prediction':shape}):
            actual=spec.restore(noise,prediction)
            expected=((noise-prediction)/np.float32(.9152)).transpose(3,0,1,2)[None]
            np.testing.assert_array_equal(actual,expected)
            with self.assertRaises(ValueError):spec.restore(noise[:1],prediction)
        decoded=np.array([-2,-1,-.5,0,.5,1,2,np.nan],np.float32).reshape(1,1,2,2,2)
        with patch.dict(spec.SHAPES,{'decoded':decoded.shape}):
            expected=np.floor((np.clip(decoded[0].transpose(1,2,3,0),-1,1)+np.float32(1))*np.float32(127.5)+np.float32(.5)).astype(np.uint8)
            with np.errstate(invalid='ignore'):np.testing.assert_array_equal(spec.rgb(decoded),expected)
    def test_raw_before_late_gates_and_no_fault_cleanup(self):
        source=(spec.HERE/'child.py').read_text();catch=source.split('    except BaseException as error:',1)[1]
        for bad in ('torch.','clear_memory','model.to','synchronize','empty_cache'):
            self.assertNotIn(bad,catch)
        self.assertLess(source.index('save(call,raw)'),source.index("raw=raw.reshape(spec.SHAPES['prediction'])"))
        self.assertLess(source.index('save(call,raw)'),source.index("'Output finite/shape guard failed'"))
        self.assertLess(source.index("'Output finite/shape guard failed'"),source.index('if args.phase==\'dit\':clear_rope_lru_caches(model)'))
        self.assertIn('torch.set_num_threads(4);torch.set_num_interop_threads(1)',source)
        self.assertIn('admission.physical_admission(',source)
        self.assertLess(source.index('admission.physical_admission('),source.index('model=ref.streamed_dit'))
        supervisor=(spec.HERE/'benchmark.py').read_text()
        self.assertLess(supervisor.index("spec.require(status==0"),supervisor.index('health=guard.health'))
        self.assertLess(supervisor.index('health=guard.health'),supervisor.index('phases.append'))
        self.assertNotIn('import torch',supervisor)
    def test_normalization_tiny_matches_original_formula(self):
        rgb=np.arange(5*2*3*3,dtype=np.uint8).reshape(5,2,3,3)
        with patch.dict(spec.SHAPES,{'rgb':rgb.shape}):
            expected=(rgb.astype(np.float32)/np.float32(127.5)-np.float32(1)).transpose(3,0,1,2)[None]
            np.testing.assert_array_equal(spec.normalize(rgb),expected)
    def test_check_mode_before_large_inputs_or_model(self):
        source=(spec.HERE/'child.py').read_text()
        self.assertLess(source.index('if args.check_metadata:'),source.index('values=spec.values('))
        self.assertLess(source.index('if args.check_metadata:'),source.index('import torch'))
    def test_actual_metadata_authority_skips_exact_payload_digests(self):
        import child
        calls=[];original=spec.digest
        def observe(path):calls.append(Path(path));return original(path)
        with patch.object(child,'load',return_value=SimpleNamespace(source_check=lambda:None)),patch.object(spec,'digest',side_effect=observe):
            child.authority(metadata_only=True)
        payloads={spec.FIXTURE/name for name in spec.FILES.values()}
        self.assertTrue(payloads.isdisjoint(calls))
        self.assertIn(child.PARENT/'xpu-pins.json',calls)
        self.assertNotIn('torch',sys.modules)

    def test_combined_clock_and_healthy_only_allocator(self):
        text=(spec.HERE/'child.py').read_text()
        self.assertIn("'owned_call_plus_healthy_cleanup_seconds':call_seconds+cleanup_seconds",text)
        self.assertLess(text.index("'Output finite/shape guard failed'"),text.index('call_allocator=allocator()'))
        catch=text.split('    except BaseException as error:',1)[1]
        self.assertNotIn('allocator()',catch)
        self.assertIn("('memory_allocated','memory_reserved','max_memory_allocated','max_memory_reserved')",text)
        self.assertIn("receipt['final_allocator']=allocator()",text)

if __name__=='__main__':unittest.main(verbosity=2)
