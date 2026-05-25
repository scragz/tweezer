"""
Psychoacoustic phase decoherence — MP3 / RealAudio artifact emulation.

Applies the full psychoacoustic masking pipeline: spectral decimation below
the masking threshold, phase randomization of decimated bins, temporal smearing
via STFT window length, and optional LPC/CELP resynthesis for RealAudio
voiced-noise character.
"""
import numpy as np
from .base import DSPModule, ParamSpec
from ._psychoacoustic import run_masking_pipeline, celp_artifact


class Codec(DSPModule):
    NAME = "codec"
    DESCRIPTION = (
        "Psychoacoustic codec artifacts — phase decoherence, spectral decimation, "
        "temporal smearing, CELP voiced-noise resynthesis"
    )
    PARAMS = {
        "bitrate_pressure": ParamSpec(
            float, 1.0, (0.0, 5.0),
            "Decimation aggressiveness (0=transparent, 1=64kbps, 2.5+=destroyed)",
        ),
        "phase_randomization": ParamSpec(
            float, 0.5, (0.0, 1.0), "Phase randomization of sub-threshold bins"
        ),
        "window_size": ParamSpec(
            int, 2048, (128, 16384),
            "STFT window size (larger = more temporal smearing / pre-echo)",
        ),
        "hop_size": ParamSpec(
            int, 512, (32, 4096), "STFT hop size (smaller = more overlap smearing)"
        ),
        "celp_blend": ParamSpec(
            float, 0.0, (0.0, 1.0),
            "LPC/CELP resynthesis blend — 1.0 = full RealAudio voiced-noise character",
        ),
        "lp_order": ParamSpec(
            int, 10, (4, 24),
            "LPC order for CELP mode (10-12 = RealAudio formant character)",
        ),
    }

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        window_size = self.params["window_size"]
        hop_size = self.params["hop_size"]
        bitrate_pressure = self.params["bitrate_pressure"]
        phase_randomization = self.params["phase_randomization"]
        celp_blend = self.params["celp_blend"]
        lp_order = self.params["lp_order"]

        # Ensure hop < window
        if hop_size >= window_size:
            hop_size = window_size // 4

        rng = np.random.default_rng()

        processed = run_masking_pipeline(
            audio, sr,
            window_size=window_size,
            hop_size=hop_size,
            bitrate_pressure=bitrate_pressure,
            phase_randomization=phase_randomization,
            rng=rng,
        )

        if celp_blend > 0.0:
            celp = celp_artifact(
                audio,
                frame_size=256,
                lp_order=lp_order,
                residual_noise_blend=celp_blend,
                rng=rng,
            )
            # Trim to processed length (STFT reconstruction may be shorter)
            n_out = len(processed)
            processed = (1 - celp_blend) * processed + celp_blend * celp[:n_out]

        return processed
