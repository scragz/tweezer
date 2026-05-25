"""
Undithered bit crushing — correlated quantization error as a harmonic generator.

Without dithering, the quantization error is a deterministic function of the signal,
producing harmonics tuned to the input. At low bit depths this is the dominant character.
Error feedback (delta-sigma style) noise-shapes the error spectrum.
"""
import numpy as np
from scipy.signal import lfilter
from .base import DSPModule, ParamSpec


class Quantization(DSPModule):
    NAME = "quant"
    DESCRIPTION = "Undithered bit crushing — correlated error produces signal-tuned harmonics"
    PARAMS = {
        "bits": ParamSpec(
            float, 8.0, (1.0, 16.0),
            "Bit depth (fractional = probabilistic rounding between floor/ceil)",
        ),
        "dither": ParamSpec(
            float, 0.0, (0.0, 1.0),
            "Dither amount (0=fully correlated harmonic error, 1=TPDF white noise)",
        ),
        "dither_color": ParamSpec(
            str, "white", ("white", "pink", "hpf"), "Dither noise spectrum"
        ),
        "error_feedback": ParamSpec(
            float, 0.0, (0.0, 1.0),
            "Error feedback gain — delta-sigma style noise shaping",
        ),
        "mod_rate": ParamSpec(
            float, 0.0, (0.0, 20.0),
            "Bit depth LFO rate in Hz (0=off, creates harmonic tremolo)",
        ),
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Pink noise state (Voss-McCartney, 16 generators)
        self._pink_state = np.zeros(16)

    def _gen_dither(self, n: int, lsb: float) -> np.ndarray:
        color = self.params["dither_color"]
        if color == "white":
            return np.random.uniform(-lsb / 2, lsb / 2, n)
        elif color == "pink":
            out = np.zeros(n)
            for i in range(n):
                k = np.random.randint(0, len(self._pink_state))
                self._pink_state[k] = np.random.uniform(-1.0, 1.0)
                out[i] = np.mean(self._pink_state)
            return out * (lsb / 2)
        else:  # hpf — highpass-filtered white noise (most audible at high freq)
            white = np.random.uniform(-lsb / 2, lsb / 2, n)
            # 1-pole highpass
            coeff = 0.9
            return lfilter([1, -1], [1, -coeff], white)

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        bits = self.params["bits"]
        dither_amt = self.params["dither"]
        feedback = self.params["error_feedback"]
        mod_rate = self.params["mod_rate"]
        n = len(audio)

        # Bit depth per sample (LFO modulation or constant)
        if mod_rate > 0:
            t = np.arange(n) / sr
            lfo = np.sin(2 * np.pi * mod_rate * t)
            bits_arr = np.clip(bits + lfo * 2.0, 1.0, 16.0)
        else:
            bits_arr = np.full(n, bits)

        # Fractional-bit stochastic rounding
        b_floor = np.floor(bits_arr).astype(int)
        b_ceil = b_floor + 1
        frac = bits_arr - b_floor
        use_ceil = np.random.random(n) < frac
        eff_bits = np.where(use_ceil, b_ceil, b_floor)
        eff_bits = np.maximum(eff_bits, 1)

        levels = (2.0 ** eff_bits).astype(float)
        lsb_arr = 2.0 / levels

        # Dither signal (scaled to mean LSB)
        mean_lsb = float(np.mean(lsb_arr))
        if dither_amt > 0:
            dither_sig = self._gen_dither(n, mean_lsb) * dither_amt
        else:
            dither_sig = None

        if feedback > 0:
            # Sequential error-feedback path (delta-sigma style)
            output = np.zeros(n)
            err = 0.0
            for i in range(n):
                lsb = lsb_arr[i]
                sample = audio[i] + err * feedback
                if dither_sig is not None:
                    sample += dither_sig[i]
                q = np.round(sample / lsb) * lsb
                q = np.clip(q, -1.0, 1.0 - lsb)
                err = audio[i] - q
                output[i] = q
        else:
            # Fully vectorized path (fast)
            inp = audio.copy()
            if dither_sig is not None:
                inp = inp + dither_sig
            output = np.round(inp / lsb_arr) * lsb_arr
            output = np.clip(output, -1.0, 1.0 - lsb_arr)

        return output
