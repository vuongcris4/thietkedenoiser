"""
Config loader utility.

Loads YAML configs with defaults merging and CLI override support.

Usage:
    from config import load_config

    # Load from YAML
    cfg = load_config("configs/dae_resnet34.yaml")

    # Access nested values
    lr = cfg["training"]["lr"]

    # With CLI args
    cfg = load_config_from_args()  # --config configs/dae_resnet34.yaml --training.lr=0.001
"""
import os
import sys
import copy
import argparse
import yaml
from typing import Any, Dict, Optional


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Override values take precedence."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _set_nested(d: dict, key_path: str, value: Any) -> None:
    """Set a nested dict value using dot-separated key path.

    Example: _set_nested(cfg, "training.lr", 0.001)
    """
    keys = key_path.split(".")
    for k in keys[:-1]:
        if k not in d or not isinstance(d[k], dict):
            d[k] = {}
        d = d[k]
    d[keys[-1]] = value


def _parse_value(value_str: str) -> Any:
    """Parse a string value to its appropriate Python type."""
    # None
    if value_str.lower() in ("none", "null", "~"):
        return None
    # Bool
    if value_str.lower() in ("true", "yes"):
        return True
    if value_str.lower() in ("false", "no"):
        return False
    # Int
    try:
        return int(value_str)
    except ValueError:
        pass
    # Float
    try:
        return float(value_str)
    except ValueError:
        pass
    # List (comma separated)
    if "," in value_str:
        return [_parse_value(v.strip()) for v in value_str.split(",")]
    # String
    return value_str


def load_yaml(path: str) -> dict:
    """Load a single YAML file."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}


def load_config(config_path: str, overrides: Optional[Dict[str, Any]] = None) -> dict:
    """Load config from YAML with _base_ inheritance and optional overrides.

    Supports:
      - `_base_: "default.yaml"` for inheriting from a base config
      - Nested overrides via dict

    Args:
        config_path: Path to the YAML config file.
        overrides: Optional dict of dot-path overrides, e.g. {"training.lr": 0.001}.

    Returns:
        Merged configuration dict.
    """
    cfg = load_yaml(config_path)
    config_dir = os.path.dirname(os.path.abspath(config_path))

    # Handle _base_ inheritance
    if "_base_" in cfg:
        base_path = cfg.pop("_base_")
        if not os.path.isabs(base_path):
            base_path = os.path.join(config_dir, base_path)
        base_cfg = load_config(base_path)  # Recursive for chained bases
        cfg = _deep_merge(base_cfg, cfg)

    # Apply overrides
    if overrides:
        for key_path, value in overrides.items():
            _set_nested(cfg, key_path, value)

    return cfg


def load_config_from_args(args: Optional[argparse.Namespace] = None) -> dict:
    """Load config from CLI arguments.

    Expects:
      --config <path>      (required) Path to YAML config
      --override key=value (repeatable) Override specific config values
      --resume <path>      (optional) Path to checkpoint for resuming

    Returns:
        Merged configuration dict with any CLI overrides applied.
    """
    if args is None:
        parser = argparse.ArgumentParser(description="Train with YAML config")
        parser.add_argument("--config", type=str, required=True,
                            help="Path to YAML config file")
        parser.add_argument("--override", type=str, nargs="*", default=[],
                            help="Override config values: key=value (dot notation)")
        parser.add_argument("--resume", type=str, default=None,
                            help="Path to checkpoint to resume from")
        args = parser.parse_args()

    # Parse overrides
    overrides = {}
    override_list = getattr(args, "override", []) or []
    for ov in override_list:
        if "=" not in ov:
            print(f"Warning: ignoring malformed override '{ov}' (expected key=value)")
            continue
        key, val = ov.split("=", 1)
        overrides[key] = _parse_value(val)

    cfg = load_config(args.config, overrides)

    # Inject resume path
    if getattr(args, "resume", None):
        cfg["resume"] = args.resume

    return cfg


def cfg_to_flat(cfg: dict, prefix: str = "") -> dict:
    """Flatten nested config dict to dot-separated keys (for logging)."""
    flat = {}
    for k, v in cfg.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(cfg_to_flat(v, full_key))
        else:
            flat[full_key] = v
    return flat


def print_config(cfg: dict, title: str = "Configuration") -> None:
    """Pretty-print config."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    flat = cfg_to_flat(cfg)
    max_key_len = max(len(k) for k in flat.keys()) if flat else 0
    for k, v in sorted(flat.items()):
        print(f"  {k:<{max_key_len}}  = {v}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    # Quick test
    if len(sys.argv) > 1:
        cfg = load_config_from_args()
        print_config(cfg)
    else:
        print("Usage: python config.py --config configs/dae_resnet34.yaml")
        print("       python config.py --config configs/diffusion.yaml --override training.lr=0.001")
