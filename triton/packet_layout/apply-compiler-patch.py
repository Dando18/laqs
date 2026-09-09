#!/usr/bin/env python3
"""Apply the packet identity lowering to the source used by a platform build."""
import argparse
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--build-dir', type=Path, required=True)
args = parser.parse_args()
cache = (args.build_dir / 'CMakeCache.txt').read_text().splitlines()
source = Path(next(line.split('=', 1)[1] for line in cache if line.startswith('CMAKE_HOME_DIRECTORY:')))
patch = Path(__file__).resolve().with_name('transparent-carrier.patch')
check = ['git', '-C', str(source), 'apply', '--check']
if subprocess.run([*check, '--reverse', str(patch)], capture_output=True).returncode:
    subprocess.run([*check, str(patch)], check=True)
    subprocess.run(['git', '-C', str(source), 'apply', str(patch)], check=True)
print(f'Packet identity lowering ready in {source}; rebuild libtriton as well as the plugin.')
