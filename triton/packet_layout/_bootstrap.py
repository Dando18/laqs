"""Import the existing solver/harness without changing the active v2 sources."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / 'triton/experiments'),
                str(ROOT / 'triton'), str(ROOT / 'triton/tritonbench')]


def activate(platform):
    import importlib.util
    spec = importlib.util.spec_from_file_location('packet_legacy_runner', ROOT / 'triton/experiments/run-search.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._activate_triton_source(platform)
