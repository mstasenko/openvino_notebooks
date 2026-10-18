"""Portable launcher for the extracted FlashVSR+ research runner.

The underlying runner remains deliberately explicit and strict.  This thin
launcher only supplies import paths and external-resource locations; all
research options (``--run``, ``--input``, ``--output``, ``--mode`` and the
reviewed memory/decoder switches) are forwarded unchanged.

No default points into the original workstation.  Model/source/scratch paths
can be supplied as launcher options or through the corresponding
``FLASHVSR_PLUS_*`` environment variables documented in ``README.md``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import runpy
import sys


REFERENCE = Path(__file__).resolve().parent.parent / "reference"
FULL_RUNTIME = REFERENCE / "openvino" / "full_runtime"
OPENVINO_TREE = REFERENCE / "openvino"
CHUNKED_ATTENTION = OPENVINO_TREE / "chunked_attention"
SOURCE_TREE = REFERENCE / "source"
FLASHVSR_ATTN = SOURCE_TREE / "operators" / "flashvsr_attn"


def _path_option(parser: argparse.ArgumentParser, name: str, env_name: str) -> None:
    parser.add_argument(
        name,
        type=Path,
        help=f"external path for {env_name} (also accepted as an environment variable)",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch the extracted FlashVSR+ research runner with explicit resources."
    )
    _path_option(parser, "--source-root", "FLASHVSR_PLUS_SOURCE")
    _path_option(parser, "--models-root", "FLASHVSR_PLUS_MODELS")
    _path_option(parser, "--scratch-root", "FLASHVSR_PLUS_SCRATCH")
    _path_option(parser, "--evidence-root", "FLASHVSR_PLUS_EVIDENCE")
    _path_option(parser, "--prompt", "FLASHVSR_PLUS_PROMPT")
    _path_option(parser, "--tcdecoder", "FLASHVSR_PLUS_TCDECODER")
    _path_option(parser, "--dit", "FLASHVSR_PLUS_DIT")
    _path_option(parser, "--lq", "FLASHVSR_PLUS_LQ")
    parser.add_argument(
        "--openvino-runtime",
        type=Path,
        help="optional Python runtime directory prepended to imports",
    )
    parser.add_argument(
        "--rgb-reference",
        type=Path,
        help="external authenticated RGB-reference directory",
    )
    return parser


def _set_paths(args: argparse.Namespace) -> None:
    mapping = {
        "source_root": "FLASHVSR_PLUS_SOURCE",
        "models_root": "FLASHVSR_PLUS_MODELS",
        "scratch_root": "FLASHVSR_PLUS_SCRATCH",
        "evidence_root": "FLASHVSR_PLUS_EVIDENCE",
        "prompt": "FLASHVSR_PLUS_PROMPT",
        "tcdecoder": "FLASHVSR_PLUS_TCDECODER",
        "dit": "FLASHVSR_PLUS_DIT",
        "lq": "FLASHVSR_PLUS_LQ",
    }
    for attribute, variable in mapping.items():
        value = getattr(args, attribute)
        if value is not None:
            os.environ[variable] = str(value.expanduser().resolve())
    if args.rgb_reference is not None:
        os.environ["FLASHVSR_PLUS_RGB_REFERENCE"] = str(
            args.rgb_reference.expanduser().resolve()
        )


def _prepend_imports(runtime: Path | None) -> None:
    entries = []
    if runtime is not None:
        entries.append(runtime.expanduser().resolve())
    entries.extend((FULL_RUNTIME, CHUNKED_ATTENTION, OPENVINO_TREE, REFERENCE,
                    SOURCE_TREE, FLASHVSR_ATTN))
    for entry in reversed(entries):
        text = str(entry)
        if text not in sys.path:
            sys.path.insert(0, text)


def main(argv: list[str] | None = None) -> None:
    args, forwarded = _parser().parse_known_args(argv)
    _set_paths(args)
    _prepend_imports(args.openvino_runtime)
    runner = FULL_RUNTIME / "run_plus_resident.py"
    if not runner.is_file():
        raise FileNotFoundError(runner)
    sys.argv = [str(runner), *forwarded]
    runpy.run_path(str(runner), run_name="__main__")


if __name__ == "__main__":
    main()
