"""
Three hardware resampling kernels with distinct artifact signatures:
  nn      — nearest-neighbor (SP-1200): staircase aliasing, no smoothing
  gaussian — 4-point Gaussian (PSX SPU): warm rolloff with pre-cutoff presence
  zigzag  — 25-point irregular FIR (PSX XA-ADPCM): angular steps, 22kHz noise floor

All kernels simulate recording at a lower sample rate (sr/ratio) and playing back
at the original rate. The full audio duration is always preserved — ratio controls
fidelity/aliasing character, not pitch or length.
"""
import numpy as np
from scipy.signal import butter, sosfilt, lfilter
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
    """Downsample `audio` to `n_out` samples using PSX 4-point Gaussian weights."""
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

    return (w0 * audio[i0] + w1 * audio[i1] + w2 * audio[i2] + w3 * audio[i3]) / 32768.0


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
_ZIGZAG_COEFFS /= _ZIGZAG_COEFFS.sum()


def _nn_upsample(audio: np.ndarray, n_out: int) -> np.ndarray:
    """Nearest-neighbor upsample — each input sample repeats to fill n_out."""
    n_in = len(audio)
    indices = np.clip(
        np.round(np.arange(n_out, dtype=float) * n_in / n_out).astype(int),
        0, n_in - 1,
    )
    return audio[indices]


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
            "Sample rate reduction factor: 1.69=SP-1200 (26kHz), 4.0=11kHz aliasing",
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

        if ratio <= 1.0:
            # ratio <= 1: upsample (higher fidelity than source) — minimal effect
            return audio.copy()

        # Optional pre-filter (band-limits before decimation, reduces aliasing)
        src = audio
        if prefilter:
            cutoff = min(0.95, 1.0 / ratio)
            sos = butter(4, cutoff, btype="low", output="sos")
            src = sosfilt(sos, src)

        # Step 1 — Downsample to n_down = n / ratio samples using the kernel
        n_down = max(2, int(n / ratio))

        if kernel == "nn":
            # Nearest-neighbor: skip samples (staircase aliasing, no smoothing)
            dn_indices = np.clip(
                np.floor(np.arange(n_down, dtype=float) * ratio).astype(int), 0, n - 1
            )
            downsampled = src[dn_indices].astype(float)

        elif kernel == "gaussian":
            # PSX 4-point Gaussian: smooth Gaussian-weighted downsample
            downsampled = _gaussian_interp(src, ratio, n_down)

        else:  # zigzag
            # Linearly decimate to n_down, then apply zigzag FIR character
            positions = np.arange(n_down, dtype=float) * ratio
            linear = np.interp(positions, np.arange(n, dtype=float), src)
            downsampled = lfilter(_ZIGZAG_COEFFS, [1.0], linear)

        # Step 2 — NN upsample back to n (sample-repeat = staircase waveform)
        # This is the authentic low-sample-rate playback artifact.
        output = _nn_upsample(downsampled, n)

        # SSM2044-style Butterworth post-filter (SP-1200 ladder character)
        if postfilter:
            cutoff = min(0.95, 1.0 / ratio)
            sos = butter(4, cutoff, btype="low", output="sos")
            output = sosfilt(sos, output)

        return output
