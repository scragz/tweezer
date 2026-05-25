import json
import numpy as np
from .modules import DSPModule, REGISTRY, parse_module_arg

# ---------------------------------------------------------------------------
# Preset chains
# ---------------------------------------------------------------------------

PRESETS: dict[str, list[dict]] = {
    # --- Hardware references ---
    "sp1200": [
        {"module": "interp", "kernel": "nn", "ratio": 1.69},
        {"module": "quant", "bits": 12, "dither": 0.0},
    ],
    "sp1200-raw": [
        {"module": "interp", "kernel": "nn", "ratio": 1.69},
        {"module": "quant", "bits": 12},
        {"module": "trunc", "block_size": 512},
    ],
    "mpc60": [
        {"module": "mpc60", "ratio": 1.0, "aa_filter": "bessel", "saturation": 0.2},
        {"module": "quant", "bits": 12, "dither": 0.3, "dither_color": "pink"},
    ],
    "snes": [
        {"module": "brr", "filter_mode": 2},
        {"module": "interp", "kernel": "gaussian"},
    ],
    "snes-hot": [
        {"module": "brr", "filter_mode": 3, "block_size": 16},
        {"module": "interp", "kernel": "gaussian"},
    ],
    "snes-chaos": [
        {"module": "brr", "filter_mode": 3, "block_size": 8, "sequence": "random"},
        {"module": "interp", "kernel": "gaussian"},
    ],
    "psx": [
        {"module": "interp", "kernel": "zigzag", "ratio": 0.833},
        {"module": "pmon", "fold_threshold": 0.6},
    ],
    "psx-clean": [
        {"module": "interp", "kernel": "gaussian", "ratio": 0.833},
    ],
    "ym2612": [
        {"module": "ghost", "bits": 3, "alpha": 2.0, "threshold": 0.4},
    ],
    "ym2612-deep": [
        {"module": "ghost", "bits": 2, "alpha": 3.0, "threshold": 0.3, "detune": 7.0},
    ],
    # --- Codec degradation ---
    "mp3-128k": [
        {"module": "codec", "bitrate_pressure": 0.3, "phase_randomization": 0.3},
    ],
    "mp3-64k": [
        {"module": "codec", "bitrate_pressure": 1.0, "phase_randomization": 0.8},
    ],
    "mp3-hell": [
        {"module": "codec", "bitrate_pressure": 2.5, "phase_randomization": 0.95},
    ],
    "real28k": [
        {
            "module": "codec",
            "bitrate_pressure": 1.5,
            "phase_randomization": 1.0,
            "window_size": 2048,
            "celp_blend": 0.35,
            "lp_order": 10,
        },
    ],
    "real14k": [
        {
            "module": "codec",
            "bitrate_pressure": 2.0,
            "phase_randomization": 1.0,
            "window_size": 4096,
            "celp_blend": 0.65,
            "lp_order": 10,
        },
    ],
    "dissolve": [
        {
            "module": "codec",
            "bitrate_pressure": 0.5,
            "phase_randomization": 1.0,
            "window_size": 4096,
        },
    ],
    "formant-noise": [
        {"module": "codec", "celp_blend": 1.0, "bitrate_pressure": 0.0, "lp_order": 10},
    ],
    # --- Cross-section combinations (from the doc appendix) ---
    "ghost-trail": [
        {"module": "ghost", "bits": 3, "threshold": 0.3},
        {"module": "brr", "filter_mode": 3},
    ],
    "alias-smear": [
        {"module": "interp", "kernel": "nn", "ratio": 4.0},
        {"module": "codec", "bitrate_pressure": 0.5, "phase_randomization": 0.7},
    ],
    "blur-clicks": [
        {"module": "trunc", "block_size": 64},
        {"module": "brr", "filter_mode": 2, "gaussian": True},
    ],
    "fold-voice": [
        {"module": "pmon", "fold_threshold": 0.5},
        {"module": "codec", "celp_blend": 0.6},
    ],
    "harmonic-pit": [
        {"module": "quant", "bits": 4, "dither": 0.0},
        {"module": "mpc60", "ratio": 4.0},
    ],
    "ladder-ghost": [
        {"module": "ghost", "bits": 3, "threshold": 0.3},
        {"module": "brr", "filter_mode": 3, "gaussian": True},
    ],
    # --- Texture / noise characters ---
    "lo-fi": [
        {"module": "quant", "bits": 10},
        {"module": "ghost", "bits": 4},
    ],
    "crunch": [
        {"module": "quant", "bits": 6, "error_feedback": 0.5},
    ],
    "stutter": [
        {"module": "trunc", "block_size": 128, "alignment": "inverted"},
    ],
    "stutter-fine": [
        {"module": "trunc", "block_size": 32, "alignment": "random", "post_filter": "resonant"},
    ],
    "chaos": [
        {"module": "pmon", "fold_threshold": 0.3, "wrap_modulus": 64},
        {"module": "brr", "filter_mode": 3, "sequence": "random"},
        {"module": "ghost", "bits": 2},
    ],
    "vinyl": [
        {"module": "mpc60", "ratio": 0.98, "saturation": 0.15},
        {"module": "quant", "bits": 14, "dither": 0.5, "dither_color": "pink"},
    ],
    "tape": [
        {"module": "mpc60", "aa_filter": "bessel", "saturation": 0.3},
        {"module": "quant", "bits": 14, "dither_color": "pink"},
    ],
    "bit1": [
        {"module": "quant", "bits": 1},
    ],
    "bit4": [
        {"module": "quant", "bits": 4, "dither": 0.0},
    ],
    "bit4-drift": [
        {"module": "quant", "bits": 4, "dither": 0.0, "mod_rate": 2.5},
    ],
    "everything": [
        {"module": "ghost", "bits": 4, "threshold": 0.4},
        {"module": "brr", "filter_mode": 3, "sequence": "cycle"},
        {"module": "interp", "kernel": "zigzag"},
        {"module": "trunc", "block_size": 256},
        {"module": "codec", "bitrate_pressure": 1.0, "phase_randomization": 0.6},
        {"module": "quant", "bits": 8},
    ],
}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class Pipeline:
    def __init__(self, modules: list[DSPModule]):
        self.modules = modules

    @classmethod
    def from_args(cls, args: list[str]) -> "Pipeline":
        return cls([parse_module_arg(a) for a in args])

    @classmethod
    def from_json(cls, path: str) -> "Pipeline":
        with open(path) as f:
            data = json.load(f)
        modules = []
        for entry in data.get("chain", []):
            entry = dict(entry)
            name = entry.pop("module")
            if name not in REGISTRY:
                available = ", ".join(sorted(REGISTRY))
                raise ValueError(f"Unknown module '{name}'. Available: {available}")
            modules.append(REGISTRY[name](**entry))
        return cls(modules)

    @classmethod
    def from_preset(cls, name: str) -> "Pipeline":
        if name not in PRESETS:
            available = ", ".join(sorted(PRESETS))
            raise ValueError(f"Unknown preset '{name}'.\n\nAvailable: {available}")
        chain = PRESETS[name]
        modules = []
        for entry in chain:
            entry = dict(entry)
            mod_name = entry.pop("module")
            modules.append(REGISTRY[mod_name](**entry))
        return cls(modules)

    def process(
        self, audio: np.ndarray, sr: int, progress_callback=None
    ) -> np.ndarray:
        result = audio.copy()
        for i, mod in enumerate(self.modules):
            if progress_callback:
                progress_callback(i, mod)
            result = mod.process(result, sr)
        return result

    def to_json_dict(self) -> dict:
        chain = []
        for mod in self.modules:
            entry = {"module": mod.NAME}
            entry.update(mod.params)
            chain.append(entry)
        return {"chain": chain}
