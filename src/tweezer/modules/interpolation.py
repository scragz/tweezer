"""
Three hardware resampling kernels with distinct artifact signatures:
  nn      — nearest-neighbor (SP-1200): staircase aliasing, no smoothing
  gaussian — 4-point Gaussian (PSX SPU): warm rolloff with pre-cutoff presence
  zigzag  — 25-point irregular FIR (PSX XA-ADPCM): angular steps, 22kHz noise floor
"""
import numpy as np
from scipy.signal import butter, bessel, sosfilt, lfilter, resample_poly
from .base import DSPModule, ParamSpec


# ---------------------------------------------------------------------------
# PSX SPU 4-point Gaussian table (512 entries, Gaussian-shaped weights)
# Approximates the actual NOCASH-documented silicon table character.
# ---------------------------------------------------------------------------

def _build_psx_gaussian_table() -> np.ndarray:
    t = np.linspace(-2.0, 2.0, 512)
    # Gaussian σ tuned to approximate the PSX rolloff/presence character
    sigma = 0.88
    table = np.exp(-0.5 * (t / sigma) ** 2)
    # Scale to match PSX integer range; table sums to ~0x7F80 per NOCASH note
    table = (table * 32640).astype(np.int32)
    return table


_PSX_GAUSS = _build_psx_gaussian_table()


def _gaussian_interp(audio: np.ndarray, ratio: float, n_out: int) -> np.ndarray:
    n_in = len(audio)
    positions = np.arange(n_out, dtype=float) * ratio
    i_arr = positions.astype(int)
    frac_arr = positions - i_arr
    idx_arr = np.clip((frac_arr * 255).astype(int), 0, 255)

    i0 = np.clip(i_arr - 1, 0, n_in - 1)
    i1 = np.clip(i_arr,     0, n_in - 1)
    i2 = np.clip(i_arr + 1, 0, n_in - 1)
    i3 = np.clip(i_arr + 2, 0, n_in - 1)

    w0 = _PSX_GAUSS[0xFF - idx_arr].astype(float)
    w1 = _PSX_GAUSS[0x1FF - idx_arr].astype(float)
    w2 = _PSX_GAUSS[0x100 + idx_arr].astype(float)
    w3 = _PSX_GAUSS[idx_arr].astype(float)

    out = (w0 * audio[i0] + w1 * audio[i1] + w2 * audio[i2] + w3 * audio[i3]) / 32768.0
    return out


# ---------------------------------------------------------------------------
# PSX XA-ADPCM 25-point zigzag FIR
# Irregular coefficient spacing creates angular waveform steps and a
# persistent noise component at ~22kHz (half the 44.1kHz output rate).
# ---------------------------------------------------------------------------

_ZIGZAG_COEFFS = np.array([
    -0.004, 0.0,    0.019, 0.0,   -0.064, 0.0,    0.179, 0.0,
    -0.498, 0.0,    0.997, 0.0,    0.997, 0.0,   -0.498, 0.0,
     0.179, 0.0,   -0.064, 0.0,    0.019, 0.0,   -0.004, 0.0,
     0.0,
], dtype=float)
# Renormalize so DC gain ≈ 1
_ZIGZAG_COEFFS /= _ZIGZAG_COEFFS.sum() if _ZIGZAG_COEFFS.sum() != 0 else 1.0


class Interpolation(DSPModule):
    NAME = "interp"
    DESCRIPTION = (
        "Hardware resampling kernels — nn (SP-1200 aliasing), "
        "gaussian (PSX warmth), zigzag (PSX XA angular steps)"
    )
    PARAMS = {
        "kernel": ParamSpec(
            str, "nn", ("nn", "gaussian", "zigzag"), "Interpolation kernel type"
        ),
        "ratio": ParamSpec(
            float, 1.0, (0.063, 16.0),
            "Resample ratio (>1 = downsample/pitch down, <1 = upsample/pitch up)",
        ),
        "prefilter": ParamSpec(
            bool, False, None, "Apply anti-alias lowpass before downsampling"
        ),
        "postfilter": ParamSpec(
            bool, False, None, "Apply SSM2044-style ladder lowpass after resampling"
        ),
    }

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        kernel = self.params["kernel"]
        ratio = self.params["ratio"]
        prefilter = self.params["prefilter"]
        postfilter = self.params["postfilter"]
        n = len(audio)

        # Anti-alias before downsampling
        if prefilter and ratio > 1.0:
            cutoff = min(0.95, 1.0 / ratio)
            sos = butter(4, cutoff, btype="low", output="sos")
            audio = sosfilt(sos, audio)

        n_out = max(1, int(n / ratio))

        if kernel == "nn":
            indices = np.clip(
                np.floor(np.arange(n_out, dtype=float) * ratio).astype(int), 0, n - 1
            )
            output = audio[indices].astype(float)

        elif kernel == "gaussian":
            output = _gaussian_interp(audio, ratio, n_out)

        else:  # zigzag
            # Resample to n_out via linear interp first, then apply zigzag FIR
            linear = np.interp(
                np.arange(n_out, dtype=float) * ratio,
                np.arange(n, dtype=float),
                audio,
            )
            output = lfilter(_ZIGZAG_COEFFS, [1.0], linear)

        # Trim or zero-pad to original length
        if len(output) >= n:
            output = output[:n]
        else:
            output = np.pad(output, (0, n - len(output)))

        # SSM2044-style Butterworth post-filter (SP-1200 ladder character)
        if postfilter:
            cutoff = min(0.95, 1.0 / max(ratio, 1.0))
            sos = butter(4, cutoff, btype="low", output="sos")
            output = sosfilt(sos, output)

        return output
