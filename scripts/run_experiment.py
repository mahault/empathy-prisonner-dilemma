#!/usr/bin/env python
"""
Main experiment launcher.

Usage:
------
# Smoke diagnostic
python scripts/run_experiment.py --experiment smoke --n_workers 1

# Core experiment
python scripts/run_experiment.py --experiment killer_test --n_workers 8

# Full suite
python scripts/run_experiment.py --experiment all --n_workers 8
"""

import sys
import os
from pathlib import Path
import argparse

# Add project root to path (must be before imports that use project modules)
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Parse CLI args early to check for numba_threads and n_workers (before Numba imports)
# Use a minimal parser to avoid importing full argument parsing module
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--numba-threads", type=int, default=None, dest="numba_threads")
_parser.add_argument("--n_workers", type=int, default=8, dest="n_workers")  # Default matches CLI default
_ns, _ = _parser.parse_known_args()
_numba_threads = _ns.numba_threads
_n_workers = _ns.n_workers

# Configure Numba parallelization in parent process (before any imports)
# On macOS (spawn multiprocessing), Numba reads NUMBA_NUM_THREADS during
# module import. Setting it in worker functions is too late - must set here.
# Priority: CLI option > env var > adaptive (based on n_workers)
if _numba_threads is not None:
    # CLI option takes highest priority - user explicitly wants this many threads
    os.environ["NUMBA_NUM_THREADS"] = str(_numba_threads)
    os.environ["OMP_NUM_THREADS"] = str(_numba_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(_numba_threads)
elif "NUMBA_NUM_THREADS" not in os.environ:
    # Adaptive strategy based on worker count
    cpu_count = os.cpu_count() or 8  # Fallback to 8 if detection fails
    
    if _n_workers > 1:
        # Multiple workers: Set all thread limits to 1 to prevent nested parallelism
        # Each worker uses 1 thread, parallelism comes from multiple processes
        # This prevents the "Red Line" thrashing issue
        os.environ["NUMBA_NUM_THREADS"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
    elif _n_workers == 1:
        # Single worker: Allow full core usage for maximum speed
        # This enables Numba's internal parallelism (prange, parallel=True)
        os.environ["NUMBA_NUM_THREADS"] = str(cpu_count)
        os.environ["OMP_NUM_THREADS"] = str(cpu_count)
        os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_count)
    # If NUMBA_NUM_THREADS is already set in environment, keep it (allows override)

# Set threading layer (required for parallel Numba operations)
os.environ.setdefault("NUMBA_THREADING_LAYER", "threadsafe")

from empathy.clean_up.experiment.cli import main

if __name__ == "__main__":
    sys.exit(main(project_root=PROJECT_ROOT))
