from .base import DSPModule, ParamSpec
from .ghost import Ghost
from .brr import BRR
from .interpolation import Interpolation
from .pmon import PMON
from .truncation import Truncation
from .codec import Codec
from .quantization import Quantization
from .mpc60 import MPC60

REGISTRY: dict[str, type[DSPModule]] = {
    "ghost": Ghost,
    "brr": BRR,
    "interp": Interpolation,
    "pmon": PMON,
    "trunc": Truncation,
    "codec": Codec,
    "quant": Quantization,
    "mpc60": MPC60,
}


def parse_module_arg(arg: str) -> DSPModule:
    """Parse 'ghost:bits=3,alpha=2.5' into an instantiated module."""
    if ":" in arg:
        name, rest = arg.split(":", 1)
        kwargs: dict[str, str] = {}
        for pair in rest.split(","):
            pair = pair.strip()
            if not pair:
                continue
            if "=" not in pair:
                raise ValueError(f"Expected key=value, got '{pair}' in '{arg}'")
            k, v = pair.split("=", 1)
            kwargs[k.strip()] = v.strip()
    else:
        name = arg
        kwargs = {}

    name = name.lower().strip()
    if name not in REGISTRY:
        available = ", ".join(sorted(REGISTRY))
        raise ValueError(f"Unknown module '{name}'. Available: {available}")

    return REGISTRY[name](**kwargs)
