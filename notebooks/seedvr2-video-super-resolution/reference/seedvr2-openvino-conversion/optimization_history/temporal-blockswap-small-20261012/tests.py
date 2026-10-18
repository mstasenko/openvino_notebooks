"""Exercise actual upstream placement/wrapper function bodies with fake tensors."""
import ast
from pathlib import Path
import types
import unittest
import weakref
from adapter import configure_prefix

SOURCE = Path(__file__).resolve().parents[3] / "source/src/optimization/blockswap.py"


def official(events):
    tree = ast.parse(SOURCE.read_text())
    names = {"_configure_blocks", "_wrap_block_forward"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != names:
        raise ValueError("Official primitive definitions absent")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    namespace = {"torch": types.SimpleNamespace(device=lambda device: device), "types": types,
                 "weakref": weakref, "get_module_memory_mb": lambda module: 1.,
                 "_get_swap_start_time": lambda *args: None, "_log_swap_timing": lambda *args: None,
                 "clear_memory": lambda **kwargs: events.append(("clear", kwargs))}
    code = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(code, str(SOURCE), "exec"), namespace)
    return namespace["_configure_blocks"], namespace["_wrap_block_forward"]


class Debug:
    enabled = False


class Tensor:
    def __init__(self, device):
        self.device = device


class Block:
    def __init__(self, index, events):
        self.index, self.events = index, events
        self.parameter = Tensor("xpu:0")
        self.fail = False

    def parameters(self):
        yield self.parameter

    def named_buffers(self):
        return ()  # Placement already moves containing-module buffers in real .to.

    def to(self, device, **kwargs):
        self.events.append(("move", self.index, device, kwargs))
        self.parameter.device = device
        return self

    def forward(self, value):
        self.events.append(("forward", self.index, self.parameter.device))
        if self.fail:
            raise RuntimeError("injected original forward failure")
        return value + self.index


class Model:
    def __init__(self, events):
        self.blocks = [Block(index, events) for index in range(32)]


class CPU(unittest.TestCase):
    def test_N16_placement_chronology_and_replay(self):
        events = []
        configure, wrap = official(events)
        model, debug = Model(events), Debug()
        result = configure_prefix(model, 16, "xpu:0", "cpu", debug, configure, wrap)
        self.assertEqual(model.blocks_to_swap, 15)
        self.assertEqual(result["selected_blocks"], list(range(16)))
        self.assertEqual([block.parameter.device for block in model.blocks], ["cpu"] * 16 + ["xpu:0"] * 16)
        for _ in range(2):
            events.clear()
            value = 0
            for block in model.blocks:
                value = block.forward(value)
            self.assertEqual(value, sum(range(32)))
            self.assertEqual([row[1] for row in events if row[0] == "forward"], list(range(32)))
            self.assertTrue(all(row[2] == "xpu:0" for row in events if row[0] == "forward"))
            for index in range(16):
                move = [row for row in events if row[0] == "move" and row[1] == index]
                self.assertEqual([row[2] for row in move], ["xpu:0", "cpu"])
                self.assertTrue(all(row[3] == {"non_blocking": False} for row in move))
            clears = [row for row in events if row[0] == "clear"]
            self.assertEqual(len(clears), 16)
            self.assertTrue(all(row[1]["force"] is False and row[1]["deep"] is False for row in clears))

    def test_N0_unchanged(self):
        events = []
        configure, wrap = official(events)
        model = Model(events)
        configure_prefix(model, 0, "xpu:0", "cpu", Debug(), configure, wrap)
        self.assertFalse(events)
        self.assertFalse(hasattr(model, "blocks_to_swap"))
        self.assertFalse(any(hasattr(block, "_original_forward") for block in model.blocks))
        for block in model.blocks:
            block.forward(0)
        self.assertFalse(any(row[0] in ("clear", "move") for row in events))

    def test_exception_no_move_cleanup_retry_or_future_block(self):
        events = []
        configure, wrap = official(events)
        model, debug = Model(events), Debug()
        configure_prefix(model, 16, "xpu:0", "cpu", debug, configure, wrap)
        model.blocks[2].fail = True
        events.clear()
        with self.assertRaisesRegex(RuntimeError, "original forward"):
            for block in model.blocks:
                block.forward(0)
        fault = events.index(("forward", 2, "xpu:0"))
        self.assertEqual(events[fault + 1:], [])
        self.assertEqual([row[1] for row in events if row[0] == "forward"], [0, 1, 2])

    def test_invalid_and_preconfigured_fail_before_mutation(self):
        for count in (-1, 1, 15, 17, 32, True):
            model = Model([])
            with self.assertRaises(ValueError):
                configure_prefix(model, count, "xpu:0", "cpu", Debug())
            self.assertFalse(hasattr(model, "blocks_to_swap"))
        model = Model([])
        model.blocks[4]._original_forward = model.blocks[4].forward
        with self.assertRaises(ValueError):
            configure_prefix(model, 16, "xpu:0", "cpu", Debug())
        self.assertFalse(hasattr(model, "blocks_to_swap"))

    def test_no_retry_patch_in_adapter_or_error_handler(self):
        source = (Path(__file__).parent / "adapter.py").read_text()
        self.assertNotIn("_patch_rope_for_blockswap", source)
        self.assertNotIn("_protect_model_from_move", source)
        tree = ast.parse((Path(__file__).parent / "control.py").read_text())
        handlers = [node for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler)]
        self.assertEqual(len(handlers), 1)
        calls = [ast.unparse(node.func) for node in ast.walk(handlers[0]) if isinstance(node, ast.Call)]
        self.assertEqual(sorted(calls), ["receipt.update", "record", "str"])


if __name__ == "__main__":
    unittest.main()
