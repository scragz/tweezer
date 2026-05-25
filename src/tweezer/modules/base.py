from __future__ import annotations
import dataclasses
from typing import Any
import numpy as np


@dataclasses.dataclass
class ParamSpec:
    type: type
    default: Any
    range: tuple | None
    description: str


class DSPModule:
    NAME: str = ""
    DESCRIPTION: str = ""
    PARAMS: dict[str, ParamSpec] = {}

    def __init__(self, **kwargs):
        self.params: dict[str, Any] = {}
        for name, spec in self.PARAMS.items():
            val = kwargs.get(name, spec.default)
            self.params[name] = self._coerce(name, val, spec)
        for key in kwargs:
            if key not in self.PARAMS:
                raise ValueError(f"{self.NAME}: unknown parameter '{key}'")

    def _coerce(self, name: str, val: Any, spec: ParamSpec) -> Any:
        if spec.type is bool:
            if isinstance(val, str):
                val = val.lower() in ("true", "1", "yes")
            else:
                val = bool(val)
        else:
            val = spec.type(val)
        if spec.range is not None:
            if spec.type in (int, float):
                lo, hi = spec.range
                if not (lo <= val <= hi):
                    raise ValueError(
                        f"{self.NAME}.{name}={val} out of range [{lo}, {hi}]"
                    )
            elif spec.type is str:
                if val not in spec.range:
                    raise ValueError(
                        f"{self.NAME}.{name}='{val}' not in {spec.range}"
                    )
        return val

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        raise NotImplementedError

    def describe(self) -> str:
        parts = [f"{k}={v}" for k, v in self.params.items()]
        return f"{self.NAME}({', '.join(parts)})"
