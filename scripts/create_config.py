#!/usr/bin/env python
"""
Interactive experiment configuration creator.

Usage:
------
# Create a new config interactively
python scripts/create_config.py

# Create from a template
python scripts/create_config.py --template smoke

# List available configs
python scripts/create_config.py --list
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from empathy.clean_up.experiment.config import EXPERIMENT_CONFIGS

# Configs directory is in clean_up/configs/
CONFIGS_DIR = Path(__file__).parent.parent / "clean_up" / "configs"


def _prompt_text(prompt: str) -> str:
    """Prompt for text input."""
    return input(prompt).strip()


def _prompt_bool(prompt: str, current: bool) -> bool:
    """Prompt for boolean input."""
    while True:
        raw = _prompt_text(f"{prompt} [{current}]: ")
        if not raw:
            return current
        if raw.lower() in {"y", "yes", "true", "t", "1"}:
            return True
        if raw.lower() in {"n", "no", "false", "f", "0"}:
            return False
        print("Please enter yes/no or true/false.")


def _prompt_scalar(prompt: str, current: Any, caster: Any) -> Any:
    """Prompt for scalar input (int/float)."""
    while True:
        raw = _prompt_text(f"{prompt} [{current}]: ")
        if not raw:
            return current
        try:
            return caster(raw)
        except ValueError:
            print("Invalid value. Try again.")


def _prompt_json(prompt: str, current: Any) -> Any:
    """Prompt for JSON input."""
    while True:
        raw = _prompt_text(f"{prompt} (JSON, blank keeps current): ")
        if not raw:
            return current
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            print(f"Invalid JSON: {exc}")


def _prompt_value(key: str, current: Any) -> Any:
    """Prompt for a value based on its type."""
    if isinstance(current, bool):
        return _prompt_bool(key, current)
    if isinstance(current, int) and not isinstance(current, bool):
        return _prompt_scalar(key, current, int)
    if isinstance(current, float):
        return _prompt_scalar(key, current, float)
    if isinstance(current, str):
        raw = _prompt_text(f"{key} [{current}]: ")
        return current if not raw else raw
    if current is None:
        raw = _prompt_text(f"{key} [None]: ")
        return None if not raw else raw
    return _prompt_json(key, current)


def _edit_config_interactively(config: Dict[str, Any]) -> Dict[str, Any]:
    """Edit config values interactively."""
    print("\nEdit config values. Press Enter to keep current values.")
    for key in list(config.keys()):
        config[key] = _prompt_value(key, config[key])
    while True:
        add_more = _prompt_text("Add extra keys? [y/N]: ").lower()
        if add_more in {"y", "yes"}:
            new_key = _prompt_text("New key name: ")
            if not new_key:
                continue
            config[new_key] = _prompt_json(new_key, None)
            continue
        break
    return config


def _choose_template(template_name: Optional[str]) -> Dict[str, Any]:
    """Choose a template config to start from."""
    if template_name:
        if template_name not in EXPERIMENT_CONFIGS:
            raise ValueError(
                f"Unknown template: {template_name}. Available: {list(EXPERIMENT_CONFIGS.keys())}"
            )
        return json.loads(json.dumps(EXPERIMENT_CONFIGS[template_name]))

    if EXPERIMENT_CONFIGS:
        print("Available templates:")
        for name in sorted(EXPERIMENT_CONFIGS.keys()):
            print(f"- {name}")
        choice = _prompt_text("Template name (blank for empty): ")
        if choice:
            if choice not in EXPERIMENT_CONFIGS:
                raise ValueError(
                    f"Unknown template: {choice}. Available: {list(EXPERIMENT_CONFIGS.keys())}"
                )
            return json.loads(json.dumps(EXPERIMENT_CONFIGS[choice]))

    return {"name": "new_experiment", "description": "Describe the experiment"}


def create_config_interactively(
    template_name: Optional[str] = None, output_path: Optional[str] = None
) -> Path:
    """Create a new experiment config interactively."""
    config = _choose_template(template_name)
    config = _edit_config_interactively(config)

    name = config.get("name")
    if not isinstance(name, str) or not name.strip():
        name = _prompt_text("Config name: ")
        if not name:
            raise ValueError("Config name is required.")
        config["name"] = name

    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    default_path = CONFIGS_DIR / f"{name}.json"
    target_path = Path(output_path) if output_path else default_path
    if target_path.exists():
        overwrite = _prompt_text(f"{target_path} exists. Overwrite? [y/N]: ").lower()
        if overwrite not in {"y", "yes"}:
            raise ValueError("Aborted config creation.")

    target_path.write_text(json.dumps(config, indent=2, sort_keys=True))
    
    # Note about grid builders
    from empathy.clean_up.experiment.config.grid import _GRID_BUILDERS
    if config.get("name") not in _GRID_BUILDERS:
        print(
            "Note: this config name has no grid builder yet. "
            "Add a builder in experiment/config/grid.py to run it."
        )
    return target_path


def _list_configs() -> None:
    """List all available experiment configs."""
    if not EXPERIMENT_CONFIGS:
        print(f"No configs found in {CONFIGS_DIR}.")
        return
    print("Available configs:")
    for name in sorted(EXPERIMENT_CONFIGS.keys()):
        print(f"- {name}")


def main() -> int:
    """Main entry point for config creation script."""
    import argparse

    parser = argparse.ArgumentParser(description="Experiment config helper")
    parser.add_argument("--list", action="store_true", help="List available configs")
    parser.add_argument("--create", action="store_true", help="Create a new config")
    parser.add_argument("--template", type=str, help="Template name to copy")
    parser.add_argument("--output", type=str, help="Output path for new config")
    args = parser.parse_args()

    if args.list:
        _list_configs()
        return 0

    if args.create or not any([args.list, args.create, args.template, args.output]):
        try:
            path = create_config_interactively(args.template, args.output)
        except ValueError as exc:
            print(f"Error: {exc}")
            return 1
        print(f"Saved config to {path}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
