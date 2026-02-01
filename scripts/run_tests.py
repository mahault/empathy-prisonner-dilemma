#!/usr/bin/env python
"""
Run all tests for the empathy research project.

Usage:
------
# Run all tests
python scripts/run_tests.py

# Run with verbose output
python scripts/run_tests.py -v

# Run specific test module
python scripts/run_tests.py --module prisoners_dilemma

# Run with coverage (if pytest-cov installed)
python scripts/run_tests.py --coverage
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def run_tests(
    verbose: bool = False,
    module: str = None,
    coverage: bool = False,
    extra_args: list = None,
) -> int:
    """Run pytest with the specified options."""
    cmd = [sys.executable, "-m", "pytest"]

    # Test paths
    test_paths = []
    if module == "prisoners_dilemma":
        test_paths = [
            "tests/test_tom.py",
            "tests/test_exploitability.py",
            "tests/test_full_simulation.py",
            "tests/test_integration.py",
        ]
    elif module == "clean_up":
        test_paths = [
            "src/empathy/clean_up/agent/tests/",
            "src/empathy/clean_up/experiment/tests/",
        ]
    elif module == "contracts":
        test_paths = ["tests/test_contracts.py"]
    else:
        # Run all tests
        test_paths = ["tests/", "src/empathy/clean_up/agent/tests/"]

    cmd.extend(test_paths)

    # Options
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-v")
        cmd.append("--tb=short")

    if coverage:
        cmd.extend(["--cov=empathy", "--cov-report=term-missing"])

    if extra_args:
        cmd.extend(extra_args)

    print(f"Running: {' '.join(cmd)}")
    print("=" * 60)

    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run tests for empathy research project"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose test output"
    )
    parser.add_argument(
        "--module",
        choices=["prisoners_dilemma", "clean_up", "contracts"],
        help="Run tests for specific module only",
    )
    parser.add_argument(
        "--coverage", action="store_true", help="Run with coverage report"
    )
    parser.add_argument(
        "extra_args",
        nargs="*",
        help="Additional arguments to pass to pytest",
    )

    args = parser.parse_args()

    return run_tests(
        verbose=args.verbose,
        module=args.module,
        coverage=args.coverage,
        extra_args=args.extra_args,
    )


if __name__ == "__main__":
    sys.exit(main())
