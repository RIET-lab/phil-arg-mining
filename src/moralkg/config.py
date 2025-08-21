from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import rootutils
import yaml

__all__ = ["Config"]

_ROOT = Path(rootutils.setup_root(__file__, indicator=".git"))


class Config:
    def __init__(self, config_path: str = "config.yaml"):
        self._config_path = config_path
        self._config: Dict[str, Any] = self._load_config()

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Config":
        return cls(config_path=str(path) if path is not None else "config.yaml")

    def _load_config(self) -> Dict[str, Any]:
        cfg_file = _ROOT / self._config_path
        if not cfg_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {cfg_file}")
        with cfg_file.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def get(self, key_path: str, default: Any = None) -> Any:
        keys = key_path.split(".") if key_path else []
        value: Any = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    @property
    def general(self) -> Dict[str, Any]:
        return self.get("general", {})

    @property
    def philpapers(self) -> Dict[str, Any]:
        return self.get("philpapers", {})

    @property
    def workshop(self) -> Dict[str, Any]:
        return self.get("workshop", {})

    @property
    def argmining(self) -> Dict[str, Any]:
        return self.get("argmining", {})

    @property
    def snowball(self) -> Dict[str, Any]:
        return self.get("snowball", {})

    @property
    def figures(self) -> Dict[str, Any]:
        return self.get("figures", {})
