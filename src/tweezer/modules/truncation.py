"""
SP-1200 block truncation at non-zero-crossing points.

Sample playback truncated at coarse block boundaries regardless of waveform phase.
The click amplitude at each boundary equals the waveform amplitude at that point —
broadband transient, bandlimited by post-filter.
"""
import numpy as np
from scipy.signal import butter, sosfilt, iirpeak, lfilter
from .base import DSPModule, ParamSpec


class Truncation(DSPModule):
    NAME = "trunc"
    DESCRIPTION = "SP-1200 block truncation — clicks at non-zero-crossing block boundaries"
    PARAMS = {
        "block_size": ParamSpec(
            int, 512, (2, 16384), "Block size in samples (smaller = denser clicks)"
        ),
        "alignment": ParamSpec(
            str, "fixed", ("fixed", "random", "inverted"),
            "fixed=regular grid, random=jittered, inverted=aligned to peaks",
        ),
        "post_filter": ParamSpec(
            str, "none", ("none", "ladder", "resonant"),
            "Post-truncation filter (ladder=soft, resonant=tuned clicks)",
        ),
        "dc_accum": ParamSpec(
            float, 0.0, (0.0, 1.0), "DC offset accumulation factor per block"
        ),
        "click_gain": ParamSpec(
            float, 1.0, (0.0, 4.0), "Scale click amplitude relative to source"
        ),
    }

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        block_size = self.params["block_size"]
        alignment = self.params["alignment"]
        post_filter = self.params["post_filter"]
        dc_accum = self.params["dc_accum"]
        click_gain = self.params["click_gain"]
        n = len(audio)

        output = audio.copy()
        dc_level = 0.0

        # Build boundary list
        if alignment == "fixed":
            boundaries = list(range(0, n, block_size))
        elif alignment == "random":
            boundaries = []
            pos = 0
            while pos < n:
                jitter = int(np.random.randint(-block_size // 4, block_size // 4 + 1))
                boundaries.append(pos)
                pos += block_size + jitter
        else:  # inverted — find local maxima, then place boundaries there
            abs_audio = np.abs(audio)
            # Coarsely identify peaks within each block
            boundaries = []
            for start in range(0, n, block_size):
                end = min(start + block_size, n)
                if end > start:
                    local_max = start + int(np.argmax(abs_audio[start:end]))
                    boundaries.append(local_max)

        for b in boundaries:
            if b <= 0 or b >= n:
                continue
            # Click = step discontinuity at boundary
            click_val = output[b] * click_gain
            output[b] = 0.0  # force zero = creates jump from previous sample

            # Accumulate DC drift
            dc_level += click_val * dc_accum
            output[b:] = output[b:] + dc_level

        # Post-filter
        if post_filter == "ladder":
            # Fixed lowpass at ~8kHz approximates SSM2044 ladder character
            sos = butter(4, min(0.9, 8000 / (sr / 2)), btype="low", output="sos")
            output = sosfilt(sos, output)

        elif post_filter == "resonant":
            # Tuned bandpass — clicks ring at a frequency tied to block rate
            center_hz = min(sr / 2 * 0.9, sr / block_size * 2)
            center_norm = min(0.99, max(0.01, center_hz / (sr / 2)))
            b_coeff, a_coeff = iirpeak(center_norm, 8.0)
            output = lfilter(b_coeff, a_coeff, output)

        return output
