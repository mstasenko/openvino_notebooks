"""Small CPU contract fixtures only; no Torch/models/device discovery."""
import ast
from pathlib import Path
import unittest
from unittest.mock import patch
import subprocess
import numpy as np
import benchmark as b
import seal
from admission import physical_admission, MIN_FREE, MAX_ALLOCATED

class Contracts(unittest.TestCase):
    def test_uhd_physical_guard(self):
        class Observation:
            def __init__(self, text):
                self.text = text
            def read_text(self):
                if isinstance(self.text, OSError):
                    raise self.text
                return self.text
        for raw, allocated, accepted in (
            (f"size: {MIN_FREE}\nusage: 0\n", MAX_ALLOCATED, True),
            (f"size: {MIN_FREE}\nusage: 1\n", 0, False),
            (f"size: {MIN_FREE}\nusage: 0\n", MAX_ALLOCATED+1, False),
            (f"size: {MIN_FREE}\nusage: 0\n", True, False),
            ("size: 1\nusage: 2\n", 0, False),
            ("size: 9\nsize: 10\nusage: 0\n", 0, False),
            ("size: bogus\nusage: 0\n", 0, False),
            ("size: 9\n", 0, False),
            (OSError("read failed"), 0, False),
        ):
            saved = []
            if accepted:
                physical_admission(Observation(raw), allocated, saved.append)
            else:
                with self.assertRaises((ValueError,OSError)):
                    physical_admission(Observation(raw), allocated, saved.append)
            self.assertEqual(len(saved),1)
            self.assertEqual(saved[0]["torch_allocated_bytes"], allocated)
            self.assertEqual(saved[0]["status"], "ACCEPTED" if accepted else "REJECTED")
            if isinstance(raw,str):
                self.assertEqual(saved[0]["raw"],raw)
        with self.assertRaises(OSError):
            physical_admission(Observation(f"size: {MIN_FREE}\nusage: 0\n"),0,
                               lambda _: (_ for _ in ()).throw(OSError("persist failed")))

    def test_actual_busy_guard_and_shell_syntax(self):
        root = Path(b.__file__).parent
        for name in ("run-root.sh","run-inside.sh"):
            subprocess.run(["bash","-n",str(root/name)],check=True)
        source = (root/"run-root.sh").read_text()
        guard = source[source.index("if [[ -n $(printf"):source.index("journalctl -k")]
        for clients, status, diagnostics, expected in (("client",1,"",75),
            ("",0,"",75),("",2,"error",75),("",1,"",0)):
            prefix = f"otc_clients={clients!r}; fuser() {{ printf '%s' {diagnostics!r}; return {status}; }}; "
            result = subprocess.run(["bash","-c","set -euo pipefail; "+prefix+guard+
                                     "echo command_after_guard"],capture_output=True,text=True)
            self.assertEqual(result.returncode,expected)
            self.assertEqual("command_after_guard" in result.stdout,expected==0)

    def test_fixed_fixture_authority(self):
        for changed in (seal.PRODUCER, seal.FIXTURE_RECEIPT):
            def digest(path):
                if path == changed:
                    return "invalid"
                return (seal.PRODUCER_SHA if path == seal.PRODUCER
                        else seal.FIXTURE_RECEIPT_SHA)
            with patch.object(seal, "sha", digest), patch.object(seal, "load") as loader:
                with self.assertRaises(ValueError):
                    seal.load_fixture_producer()
                loader.assert_not_called()

    def test_geometry_and_schedule(self):
        self.assertEqual(b.CALLS, ("warmup02", "01", "02", "03"))
        self.assertEqual(b.SHAPES, {"encode": (1,32,270,480), "dit": (1,270,480,16),
                                  "decode": (1,3,2160,3840)})
        self.assertEqual(b.HEIGHT * b.WIDTH * 3, 24883200)
        self.assertNotEqual(b.SHAPES["dit"], (1,96,120,16))
        values = np.arange(32, dtype=np.float32)
        with patch.dict(b.SHAPES, {"dit": (1,1,2,16)}):
            result = b.prediction_shape(values)
            np.testing.assert_array_equal(result.reshape(-1), values)
            self.assertEqual(result.shape, (1,1,2,16))
            with self.assertRaises(ValueError):
                b.prediction_shape(values[:-1])

    def test_whole_wall_and_failure(self):
        order = []
        ticks = iter(range(20))
        result = b.sequence(Path("unused"), lambda p: order.append(p) or 0,
                            lambda root, p: order.append("validate:" + p), lambda: next(ticks))
        self.assertEqual(order, ["encode","validate:encode","dit","validate:dit","decode","validate:decode"])
        self.assertEqual(result["whole_video_seconds"], 7)
        self.assertFalse(result["quality_pass"])
        self.assertFalse(result["steady_stage_sum_is_whole_video"])
        for failure in b.PHASES:
            order = []
            def launch(phase):
                order.append(phase)
                return 1 if phase == failure else 0
            with self.assertRaises(RuntimeError):
                b.sequence(Path("unused"), launch, lambda *_: None)
            self.assertEqual(order, list(b.PHASES[:b.PHASES.index(failure)+1]))
        order = []
        with self.assertRaises(ValueError):
            b.sequence(Path("unused"), lambda phase: order.append(phase) or 0,
                       lambda *_: (_ for _ in ()).throw(ValueError("bad handoff")))
        self.assertEqual(order, ["encode"])

    def test_source_math_and_fault_contract(self):
        tree = ast.parse(Path(b.__file__).read_text())
        functions = {node.name: node for node in tree.body if isinstance(node,ast.FunctionDef)}
        for name in ("main","sequence","inputs"):
            self.assertFalse(any(isinstance(node,ast.Import) and any(item.name == "torch" for item in node.names)
                                 for node in ast.walk(functions[name])))
        child = functions["child"]
        handlers = [node for node in ast.walk(child) if isinstance(node,ast.ExceptHandler)]
        self.assertEqual(len(handlers),1)
        self.assertEqual(ast.unparse(handlers[0].body[0]),"diag.fault(error)")
        self.assertIsInstance(handlers[0].body[1],ast.Raise)
        text = ast.unparse(child)
        self.assertIn("tile_size=(512, 512), tile_overlap=(64, 64)",text)
        self.assertIn("dtype=torch.bfloat16",text)
        self.assertIn("disable_cache=False",text)
        self.assertIn("clear_rope_lru_caches(model)",text)
        self.assertNotIn("set_per_process_memory_fraction",text)
        self.assertNotIn("probe.physical_admission",text)
        self.assertIn("physical_admission",text)
        self.assertLess(text.index("save(directory"),text.index("validate_raws(root"))
        self.assertIn("values['frames']['02' if call == 'warmup02' else call]",text)

    def test_packing_small_sentinels(self):
        noise = np.arange(32,dtype=np.float32).reshape(1,1,2,16)
        moments = np.arange(64,dtype=np.float32).reshape(1,32,1,2)
        condition = moments[0,:16].transpose(1,2,0)[None] * np.float32(.9152)
        video = np.concatenate((noise,condition,np.ones((1,1,2,1),np.float32)),axis=-1).reshape(-1,33)
        np.testing.assert_array_equal(video[:,:16],noise.reshape(-1,16))
        np.testing.assert_array_equal(video[:,16:32],condition.reshape(-1,16))
        np.testing.assert_array_equal(video[:,32],1)
        restored = ((noise-noise)/np.float32(.9152)).transpose(3,0,1,2)[None]
        self.assertEqual(restored.shape,(1,16,1,1,2))
        self.assertTrue(np.isfinite(restored).all())

if __name__ == "__main__":
    unittest.main()
