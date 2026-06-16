from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path = "params.yaml") -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def ensure_parent(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved
