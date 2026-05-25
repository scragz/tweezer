"""
PSX pitch modulation sign-extension fold.

When amplitude-modulated pitch exceeds fold_threshold, the pitch register
undergoes AND-mask wrap (simulating sign-extension from 16→32 bit), folding
to a non-linear position. Deterministic chaos: same envelope → same fold.
"""
import numpy as np
from scipy.signal import lfilter, butter, sosfilt
from .base import DSPModule, ParamSpec


class PMON(DSPModule):
    NAME = "pmon"
    DESCRIPTION = "PSX pitch fold — amplitude drives pitch/filter/position into chaotic wrap"
    PARAMS = {
        "fold_threshold": ParamSpec(
            float, 0.5, (0.0, 1.0), "Envelope level at which folding begins"
        ),
        "wrap_modulus": ParamSpec(
            int, 256, (2, 4096), "AND-mask modulus (controls fold destination range)"
        ),
        "clip_threshold": ParamSpec(
            float, 1.0, (0.1, 4.0), "Clip pitch register after fold"
        ),
        "depth": ParamSpec(
            float, 1.0, (0.0, 4.0), "Modulation depth (scales envelope influence)"
        ),
        "target": ParamSpec(
            str, "pitch", ("pitch", "filter", "position"),
            "What the folded envelope modulates",
        ),
    }

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        fold_threshold = self.params["fold_threshold"]
        wrap_modulus = self.params["wrap_modulus"]
        clip_threshold = self.params["clip_threshold"]
        depth = self.params["depth"]
        target = self.params["target"]
        n = len(audio)

        # Smoothed amplitude envelope (~10ms)
        tau = 0.01
        coeff = np.exp(-1.0 / (sr * tau))
        envelope = lfilter([1 - coeff], [1, -coeff], np.abs(audio))

        # Normalize to [0, 1]
        env_max = envelope.max()
        if env_max < 1e-10:
            return audio.copy()
        env_norm = envelope / env_max

        # PSX sign-extension fold: scale to integer pitch register range
        env_int = (env_norm * wrap_modulus).astype(int)
        fold_thresh_int = int(fold_threshold * wrap_modulus)

        # Above threshold: AND-mask wrap (non-monotonic — the interesting part)
        folded = np.where(
            env_int > fold_thresh_int,
            env_int & (wrap_modulus - 1),
            env_int,
        ).astype(float)

        # Convert back to [0, 1] modulation signal, scaled by depth
        mod = folded / wrap_modulus * depth

        if target == "pitch":
            # Variable-rate playback: mod controls instantaneous speed
            # Speed range: [0.5, 2.0] so there's always some motion
            speed = 0.5 + mod * 1.5
            speed = np.clip(speed, 0.1, clip_threshold * 2)
            # Build fractional read-index array
            read_idx = np.cumsum(speed)
            # Normalize so we span roughly the same duration
            read_idx = read_idx / read_idx[-1] * (n - 1)
            output = np.interp(read_idx, np.arange(n), audio)

        elif target == "filter":
            # Modulate a resonant lowpass cutoff per block
            block_size = max(64, sr // 100)
            sos_list = []
            output = np.zeros(n)
            for start in range(0, n, block_size):
                end = min(start + block_size, n)
                block_mod = mod[start:end].mean()
                # Cutoff sweeps 200Hz–18kHz
                cutoff_hz = 200 + block_mod * 17800
                cutoff_norm = min(0.99, cutoff_hz / (sr / 2))
                sos = butter(2, cutoff_norm, btype="low", output="sos")
                output[start:end] = sosfilt(sos, audio[start:end])

        else:  # position
            # Jump the read position based on folded value
            read_positions = (np.arange(n, dtype=float) + mod * n * 0.25) % n
            output = np.interp(read_positions, np.arange(n), audio)

        return output.astype(audio.dtype)
