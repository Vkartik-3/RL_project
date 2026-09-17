"""Local feature flags with deterministic percentage rollout."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from forgeline.core.errors import ConfigError


def stable_bucket(key: str, salt: str = "", buckets: int = 10_000) -> int:
    """Deterministic bucket in ``[0, buckets)`` from ``sha256(salt:key)``."""
    h = hashlib.sha256(f"{salt}:{key}".encode()).digest()
    return int.from_bytes(h[:8], "big") % buckets


@dataclass
class FeatureFlag:
    name: str
    enabled: bool = True
    percentage: float = 100.0  # 0..100 of keys for which the flag is on
    allow: set = field(default_factory=set)
    deny: set = field(default_factory=set)
    salt: str = ""
    description: str = ""

    def validate(self) -> None:
        if not 0.0 <= self.percentage <= 100.0:
            raise ConfigError(f"flag {self.name!r}: percentage must be in [0, 100]")

    def is_on(self, key: Optional[str] = None) -> bool:
        if not self.enabled:
            return False
        if key is not None:
            if key in self.deny:
                return False
            if key in self.allow:
                return True
        if self.percentage >= 100.0:
            return True
        if self.percentage <= 0.0 or key is None:
            return False
        return stable_bucket(key, self.salt or self.name) < self.percentage * 100  # buckets = 10_000

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "enabled": self.enabled, "percentage": self.percentage, "allow": sorted(self.allow),
                "deny": sorted(self.deny), "salt": self.salt, "description": self.description}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FeatureFlag":
        f = cls(name=d["name"], enabled=bool(d.get("enabled", True)), percentage=float(d.get("percentage", 100.0)),
                allow=set(d.get("allow", [])), deny=set(d.get("deny", [])), salt=d.get("salt", ""), description=d.get("description", ""))
        f.validate()
        return f


class FeatureFlags:
    """In-memory flag set with optional JSON persistence."""

    def __init__(self, path: Optional[str | os.PathLike] = None):
        self.path = Path(path) if path else None
        self._flags: Dict[str, FeatureFlag] = {}
        if self.path and self.path.exists():
            data = json.loads(self.path.read_text())
            self._flags = {f["name"]: FeatureFlag.from_dict(f) for f in data.get("flags", [])}

    def _save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump({"flags": [x.to_dict() for x in self._flags.values()]}, f, indent=2)
        os.replace(tmp, self.path)

    def set(self, flag: FeatureFlag) -> FeatureFlag:
        flag.validate()
        self._flags[flag.name] = flag
        self._save()
        return flag

    def get(self, name: str) -> Optional[FeatureFlag]:
        return self._flags.get(name)

    def is_on(self, name: str, key: Optional[str] = None, default: bool = False) -> bool:
        flag = self._flags.get(name)
        return flag.is_on(key) if flag is not None else default

    def disable(self, name: str) -> None:
        if name in self._flags:
            self._flags[name].enabled = False
            self._save()

    def enable(self, name: str) -> None:
        if name in self._flags:
            self._flags[name].enabled = True
            self._save()

    def names(self) -> list:
        return sorted(self._flags)
