#!/usr/bin/env python
"""
Validate experiment configurations.

Usage:
    python scripts/validate_configs.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from empathy.clean_up.experiment.config import (
    get_experiment_config,
    validate_episode_count,
    create_experiment_grid,
)


def main():
    """Validate all experiment configs."""
    configs = [
        "smoke",
        "killer_test",
        "phase_diagram",
        "phase_diagram_baseline",
    ]

    print("=" * 60)
    print("Validating Experiment Configurations")
    print("=" * 60)

    all_passed = True

    for config_name in configs:
        print(f"\n[{config_name}]")
        try:
            config = get_experiment_config(config_name)
            grid = create_experiment_grid(config)
            
            # Validate episode count
            validate_episode_count(config)
            print(f"  ✓ Episode count: {len(grid)} (expected: {config.get('expected_episodes')})")
            
            # Check baseline explicitly disables social
            if config_name == "phase_diagram_baseline":
                sample = grid[0]
                social_flags = [
                    "social_enabled",
                ]
                all_disabled = all(
                    sample.get(flag) is False 
                    for flag in social_flags 
                    if flag in sample
                )
                # use_tom is auto-set during agent creation, so we just check the social flags
                if all_disabled:
                    print(f"  ✓ Baseline correctly disables all social modules")
                    print(f"    (use_tom will be auto-set to False during agent creation)")
                else:
                    print(f"  ✗ Baseline does NOT properly disable social modules!")
                    print(f"    Flags: {[(f, sample.get(f)) for f in social_flags]}")
                    all_passed = False
            
            # Check killer test has truth params
            if config_name == "killer_test":
                has_truth = all(
                    "spawn_probs_true" in row and row["spawn_probs_true"]
                    for row in grid
                )
                if has_truth:
                    print(f"  ✓ All grid rows have truth params")
                else:
                    print(f"  ✗ Some grid rows missing truth params!")
                    all_passed = False
            
            print(f"  ✓ Grid generation successful ({len(grid)} episodes)")
            
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            all_passed = False
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All configs validated successfully!")
        return 0
    else:
        print("✗ Some configs failed validation!")
        return 1


if __name__ == "__main__":
    sys.exit(main())



