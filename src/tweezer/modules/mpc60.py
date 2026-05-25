"""
Akai MPC60 linear interpolation pitch shift.

Linear interpolation produces a sinc-shaped frequency response with comb notches
that track the pitch ratio. The more extreme the transposition, the lower the notch
frequencies fall into the audible range.
"""
import numpy as np
from scipy.signal import butter, bessel, sosfilt
from .base import DSPModule, ParamSpec


class MPC60(DSPModule):
    NAME = "mpc60"
    DESCRIPTION = "MPC60 linear interpolation — sinc comb notches track pitch ratio"
    PARAMS = {
        "ratio": ParamSpec(
            float, 1.0, (0.063, 16.0),
            "Pitch ratio (>1 = pitch down, <1 = pitch up)",
        ),
        "aa_filter": ParamSpec(
            str, "butterworth", ("butterworth", "bessel", "none"),
            "Anti-alias filter before resampling (bessel = MPC60 linear-phase character)",
        ),
        "cutoff": ParamSpec(
            float, 0.9, (0.05, 0.99),
            "Output reconstruction filter cutoff (fraction of Nyquist)",
        ),
        "saturation": ParamSpec(
            float, 0.0, (0.0, 1.0),
            "Output stage soft-clipping saturation (MPC60 analog output warmth)",
        ),
    }

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        ratio = self.params["ratio"]
        aa_filter = self.params["aa_filter"]
        cutoff = self.params["cutoff"]
        saturation = self.params["saturation"]
        n = len(audio)
        nyquist = sr / 2

        # Anti-alias before downsampling
        if aa_filter != "none" and ratio > 1.0:
            aa_cutoff = min(0.98, 1.0 / ratio * 0.95)
            if aa_filter == "butterworth":
                sos = butter(4, aa_cutoff, btype="low", output="sos")
            else:  # bessel — linear phase, MPC60 character
                sos = bessel(4, aa_cutoff, btype="low", output="sos", norm="phase")
            audio = sosfilt(sos, audio)

        # Linear interpolation resampling (the sinc comb emerges from this inherently).
        # ratio > 1: pitch down (advance slowly through source, loops for continuity).
        # ratio < 1: pitch up (advance quickly, reading past source end wraps to start).
        # Always outputs n samples so the full duration is preserved.
        src_positions = (np.arange(n, dtype=float) / ratio) % n
        output = np.interp(src_positions, np.arange(n, dtype=float), audio)

        # Reconstruction lowpass
        if cutoff < 0.98:
            if aa_filter == "bessel":
                sos = bessel(4, cutoff, btype="low", output="sos", norm="phase")
            else:
                sos = butter(4, cutoff, btype="low", output="sos")
            output = sosfilt(sos, output)

        # Soft-clip saturation (MPC60 analog output stage)
        if saturation > 0.0:
            drive = 1.0 + saturation * 4.0
            output = np.tanh(output * drive) / drive

        return output
