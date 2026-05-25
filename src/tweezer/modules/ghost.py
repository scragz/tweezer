import numpy as np
from fractions import Fraction
from scipy.signal import lfilter, resample_poly
from .base import DSPModule, ParamSpec


class Ghost(DSPModule):
    NAME = "ghost"
    DESCRIPTION = "YM2612 ladder effect — quantized ghost signal rises as clean signal decays"
    PARAMS = {
        "bits": ParamSpec(int, 4, (1, 16), "Ghost bit depth (lower = grittier)"),
        "alpha": ParamSpec(float, 2.0, (0.1, 10.0), "Crossover curve exponent"),
        "delta": ParamSpec(float, 0.0, (-10.0, 10.0), "Phase offset in samples"),
        "detune": ParamSpec(float, 0.0, (-100.0, 100.0), "Ghost detune in cents"),
        "threshold": ParamSpec(float, 0.5, (0.001, 1.0), "Amplitude at which crossover begins"),
    }

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        bits = self.params["bits"]
        alpha = self.params["alpha"]
        delta = self.params["delta"]
        detune = self.params["detune"]
        threshold = self.params["threshold"]
        n = len(audio)

        # Build quantized ghost path
        levels = 2 ** bits
        lsb = 2.0 / levels
        ghost = np.round(audio / lsb) * lsb
        ghost = np.clip(ghost, -1.0, 1.0 - lsb)

        # Detune ghost via rational resampling
        if abs(detune) > 0.01:
            ratio = 2 ** (detune / 1200)
            frac = Fraction(ratio).limit_denominator(128)
            ghost_r = resample_poly(ghost, frac.numerator, frac.denominator)
            if len(ghost_r) >= n:
                ghost = ghost_r[:n]
            else:
                ghost = np.pad(ghost_r, (0, n - len(ghost_r)))

        # Phase offset via fractional sample delay
        if abs(delta) > 0.01:
            idx = np.clip(np.arange(n, dtype=float) + delta, 0, n - 1)
            ghost = np.interp(idx, np.arange(n), ghost)

        # Smoothed amplitude envelope — single-pole IIR (~10ms)
        tau = 0.01
        coeff = np.exp(-1.0 / (sr * tau))
        envelope = lfilter([1 - coeff], [1, -coeff], np.abs(audio))

        # Crossover: clean fades, ghost rises
        clean_gain = np.clip(envelope / threshold, 0.0, 1.0) ** alpha
        ghost_gain = 1.0 - clean_gain

        return audio * clean_gain + ghost * ghost_gain
