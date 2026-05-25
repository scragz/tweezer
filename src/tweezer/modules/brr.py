import numpy as np
from scipy.signal import lfilter, lfilter_zi, convolve
from .base import DSPModule, ParamSpec

# SNES BRR filter coefficients: (a, b) where y[n] = residual + a*y[n-1] + b*y[n-2]
_COEFFS = {
    0: (0.0,        0.0),
    1: (15 / 16,    0.0),
    2: (61 / 32,   -15 / 16),
    3: (115 / 64,  -13 / 16),
}

# SNES Gaussian post-filter kernel (4-point approximation)
_GAUSSIAN_KERNEL = np.array([0.0625, 0.4375, 0.4375, 0.0625])

# INT16 saturation range
_INT16_MAX = 32767
_INT16_MIN = -32768


def _apply_brr_block(audio_block: np.ndarray, mode: int, s1: float, s2: float):
    """
    Apply one BRR filter block with int16 saturation on state.
    Returns (output_block, new_s1, new_s2).
    Mode 3 near-instability is authentic only with saturating state.
    """
    a, b = _COEFFS[mode]
    out = np.empty_like(audio_block)
    for i, sample in enumerate(audio_block):
        val = sample + a * s1 + b * s2
        # Saturate state to int16 range (authentic SNES overflow behavior)
        clamped = float(np.clip(val * _INT16_MAX, _INT16_MIN, _INT16_MAX)) / _INT16_MAX
        out[i] = clamped
        s2 = s1
        s1 = clamped
    return out, s1, s2


class BRR(DSPModule):
    NAME = "brr"
    DESCRIPTION = "SNES BRR IIR cascade — near-unstable mode 3 produces ringing at block edges"
    PARAMS = {
        "filter_mode": ParamSpec(int, 2, (0, 3), "BRR filter mode (3=near-unstable)"),
        "block_size": ParamSpec(int, 16, (2, 512), "Block size in samples"),
        "sequence": ParamSpec(
            str, "fixed", ("fixed", "cycle", "random"), "Mode switching strategy per block"
        ),
        "gaussian": ParamSpec(bool, False, None, "Apply SNES Gaussian post-filter"),
        "gaussian_width": ParamSpec(
            float, 1.0, (0.1, 8.0), "Gaussian kernel width scale"
        ),
    }

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        filter_mode = self.params["filter_mode"]
        block_size = self.params["block_size"]
        sequence = self.params["sequence"]
        do_gaussian = self.params["gaussian"]
        gwidth = self.params["gaussian_width"]

        n = len(audio)
        output = np.zeros(n)
        s1, s2 = 0.0, 0.0
        cycle_idx = 0

        for start in range(0, n, block_size):
            end = min(start + block_size, n)
            block = audio[start:end]

            if sequence == "fixed":
                mode = filter_mode
            elif sequence == "cycle":
                mode = cycle_idx % 4
                cycle_idx += 1
            else:  # random
                mode = int(np.random.randint(0, 4))

            block_out, s1, s2 = _apply_brr_block(block, mode, s1, s2)
            output[start:end] = block_out

        if do_gaussian:
            if gwidth == 1.0:
                kernel = _GAUSSIAN_KERNEL
            else:
                size = max(4, int(4 * gwidth))
                x = np.linspace(-2, 2, size)
                kernel = np.exp(-0.5 * (x / gwidth) ** 2)
                kernel /= kernel.sum()
            output = convolve(output, kernel, mode="same")

        return output
