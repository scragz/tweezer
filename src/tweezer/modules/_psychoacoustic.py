"""
Psychoacoustic masking model — MPEG-1 Layer III approximation.
Based on Bark scale spreading function, tonality estimation, and spectral decimation.
Internal module; not in the REGISTRY.
"""
import numpy as np
from scipy.signal import get_window
from scipy.linalg import solve_toeplitz


# ---------------------------------------------------------------------------
# STFT helpers
# ---------------------------------------------------------------------------

def stft(audio: np.ndarray, window_size: int, hop_size: int, window: np.ndarray) -> np.ndarray:
    frames = []
    for i in range(0, len(audio) - window_size, hop_size):
        frame = audio[i : i + window_size] * window
        frames.append(np.fft.rfft(frame))
    return np.array(frames)


def istft(
    frames: np.ndarray, window_size: int, hop_size: int, window: np.ndarray
) -> np.ndarray:
    output_length = (len(frames) - 1) * hop_size + window_size
    output = np.zeros(output_length)
    norm = np.zeros(output_length)
    for i, frame in enumerate(frames):
        start = i * hop_size
        time_frame = np.fft.irfft(frame, window_size)
        output[start : start + window_size] += time_frame * window
        norm[start : start + window_size] += window ** 2
    norm = np.where(norm < 1e-8, 1e-8, norm)
    return output / norm


# ---------------------------------------------------------------------------
# Bark scale
# ---------------------------------------------------------------------------

def hz_to_bark(f: np.ndarray) -> np.ndarray:
    return 13 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500) ** 2)


def get_bark_bands(window_size: int, sample_rate: int):
    n_bins = window_size // 2 + 1
    freqs = np.fft.rfftfreq(window_size, 1 / sample_rate)
    bark_freqs = hz_to_bark(np.maximum(freqs, 1.0))
    bark_edges = np.linspace(0, 24, 25)
    band_assignments = np.clip(np.digitize(bark_freqs, bark_edges) - 1, 0, 23)
    return band_assignments, freqs, bark_freqs


# ---------------------------------------------------------------------------
# Masking model
# ---------------------------------------------------------------------------

def spreading_function(delta_z: np.ndarray) -> np.ndarray:
    return 15.81 + 7.5 * (delta_z + 0.474) - 17.5 * np.sqrt(1 + (delta_z + 0.474) ** 2)


def absolute_threshold(freqs: np.ndarray) -> np.ndarray:
    f_khz = np.maximum(freqs, 20) / 1000
    return (
        3.64 * f_khz ** -0.8
        - 6.5 * np.exp(-0.6 * (f_khz - 3.3) ** 2)
        + 1e-3 * f_khz ** 4
    )


def estimate_tonality(frames: np.ndarray, alpha: float = 0.9) -> np.ndarray:
    n_frames, n_bins = frames.shape
    tonality = np.zeros((n_frames, n_bins))
    for i in range(2, n_frames):
        angle_prev = np.angle(frames[i - 1])
        angle_prev2 = np.angle(frames[i - 2])
        predicted_phase = 2 * angle_prev - angle_prev2
        predicted_mag = np.abs(frames[i - 1])
        X_predicted = predicted_mag * np.exp(1j * predicted_phase)
        error = np.abs(frames[i] - X_predicted) ** 2
        signal = np.abs(frames[i]) ** 2 + 1e-10
        t = 1 - np.clip(error / signal, 0, 1)
        if i > 2:
            tonality[i] = alpha * tonality[i - 1] + (1 - alpha) * t
        else:
            tonality[i] = t
    return tonality


def compute_masking_threshold(
    frame_db: np.ndarray,
    bark_freqs: np.ndarray,
    tonality_frame: np.ndarray,
    freqs: np.ndarray,
) -> np.ndarray:
    """Fast version: Bark-domain convolution approximation."""
    bark_grid = np.linspace(0, 24, 512)
    frame_db_bark = np.interp(bark_grid, bark_freqs, frame_db)
    tonality_bark = np.interp(bark_grid, bark_freqs, tonality_frame)

    masking_offset = tonality_bark * (-14.5) + (1 - tonality_bark) * (-5.5)

    delta_z = bark_grid - bark_grid[256]
    sf_kernel = 10 ** (spreading_function(delta_z) / 10)

    masker_linear = 10 ** ((frame_db_bark + masking_offset) / 10)
    threshold_linear = np.convolve(masker_linear, sf_kernel, mode="same")
    threshold_db = 10 * np.log10(threshold_linear + 1e-10)

    threshold_db = np.interp(bark_freqs, bark_grid, threshold_db)
    return np.maximum(threshold_db, absolute_threshold(freqs))


def apply_temporal_masking(
    thresholds: np.ndarray,
    hop_size: int,
    sample_rate: int,
    pre_mask_ms: float = 20,
    post_mask_ms: float = 200,
) -> np.ndarray:
    n_frames, n_bins = thresholds.shape
    frame_duration_ms = (hop_size / sample_rate) * 1000
    pre_frames = int(pre_mask_ms / frame_duration_ms)
    post_frames = int(post_mask_ms / frame_duration_ms)

    smoothed = thresholds.copy()
    for i in range(n_frames):
        for j in range(1, post_frames + 1):
            if i + j >= n_frames:
                break
            t_ms = j * frame_duration_ms
            decay_db = 10 * np.log10(t_ms / 1)
            smoothed[i + j] = np.maximum(smoothed[i + j], thresholds[i] - decay_db)
        for j in range(1, pre_frames + 1):
            if i - j < 0:
                break
            t_ms = j * frame_duration_ms
            decay_db = 20 * (t_ms / pre_mask_ms)
            smoothed[i - j] = np.maximum(smoothed[i - j], thresholds[i] - decay_db)
    return smoothed


def apply_masking_artifacts(
    frames: np.ndarray,
    thresholds: np.ndarray,
    bitrate_pressure: float = 1.0,
    phase_randomization: float = 1.0,
    quantization_bits_min: int = 1,
    rng=None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    n_frames, n_bins = frames.shape
    output = frames.copy()

    for i in range(n_frames):
        magnitudes = np.abs(frames[i])
        phases = np.angle(frames[i])
        mag_db = 20 * np.log10(magnitudes + 1e-10)
        smr = mag_db - thresholds[i]
        # Multiplier of 6: pressure=1→6dB shift (moderate), 2.5→15dB (aggressive), 5→30dB (extreme)
        effective_smr = smr - (bitrate_pressure * 6)

        below = effective_smr < 0
        depth = np.where(below, -effective_smr, 0.0)

        # Zero bins deep below threshold
        deep = below & (depth > 40)
        output[i, deep] = 0

        # Coarse-quantize + phase-randomize bins just below threshold
        mid = below & ~deep
        if mid.any():
            mags = magnitudes[mid]
            deps = depth[mid]
            bits = np.maximum(
                quantization_bits_min,
                quantization_bits_min + ((40 - deps) / 10).astype(int),
            )
            steps = mags / (2.0 ** bits)
            q_mags = np.round(mags / np.maximum(steps, 1e-10)) * steps
            blend = np.minimum(1.0, deps / 20) * phase_randomization
            phase_noise = rng.uniform(-np.pi, np.pi, mid.sum())
            new_phases = (1 - blend) * phases[mid] + blend * phase_noise
            output[i, mid] = q_mags * np.exp(1j * new_phases)

    return output


# ---------------------------------------------------------------------------
# LPC / CELP resynthesis
# ---------------------------------------------------------------------------

def lpc_analysis(frame: np.ndarray, order: int = 10):
    r = np.correlate(frame, frame, mode="full")
    r = r[len(frame) - 1 :][:order + 1] / len(frame)
    if r[0] < 1e-10:
        return np.zeros(order), frame.copy()
    try:
        coeffs = solve_toeplitz(r[:order], r[1 : order + 1])
    except (np.linalg.LinAlgError, ValueError):
        return np.zeros(order), frame.copy()
    if not np.all(np.isfinite(coeffs)):
        return np.zeros(order), frame.copy()
    residual = np.zeros_like(frame)
    for n in range(order, len(frame)):
        prediction = np.dot(coeffs, frame[n - order : n][::-1])
        residual[n] = frame[n] - prediction
    return coeffs, residual


def lpc_synthesis(coeffs: np.ndarray, excitation: np.ndarray) -> np.ndarray:
    order = len(coeffs)
    output = np.zeros_like(excitation)
    for n in range(order, len(excitation)):
        prediction = np.dot(coeffs, output[n - order : n][::-1])
        val = excitation[n] + prediction
        # Clamp runaway synthesis (unstable LPC filter on pathological input)
        output[n] = val if abs(val) < 10.0 else np.sign(val) * 10.0
    return output


def celp_artifact(
    audio: np.ndarray,
    frame_size: int = 256,
    lp_order: int = 10,
    residual_noise_blend: float = 1.0,
    rng=None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    output = np.zeros_like(audio)
    for i in range(0, len(audio) - frame_size, frame_size):
        frame = audio[i : i + frame_size]
        coeffs, residual = lpc_analysis(frame, lp_order)
        noise = rng.normal(0, np.std(residual) + 1e-10, len(residual))
        blended = (1 - residual_noise_blend) * residual + residual_noise_blend * noise
        output[i : i + frame_size] = lpc_synthesis(coeffs, blended)
    return output


# ---------------------------------------------------------------------------
# Public interface for codec.py
# ---------------------------------------------------------------------------

def run_masking_pipeline(
    audio: np.ndarray,
    sr: int,
    window_size: int,
    hop_size: int,
    bitrate_pressure: float,
    phase_randomization: float,
    rng=None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    window = get_window("hann", window_size)
    frames = stft(audio, window_size, hop_size, window)
    if len(frames) == 0:
        return audio.copy()

    n_frames, n_bins = frames.shape
    _, freqs, bark_freqs = get_bark_bands(window_size, sr)
    tonality = estimate_tonality(frames)

    thresholds = np.zeros((n_frames, n_bins))
    for i in range(n_frames):
        mag_db = 20 * np.log10(np.abs(frames[i]) + 1e-10)
        thresholds[i] = compute_masking_threshold(mag_db, bark_freqs, tonality[i], freqs)

    thresholds = apply_temporal_masking(thresholds, hop_size, sr)

    processed = apply_masking_artifacts(
        frames,
        thresholds,
        bitrate_pressure=bitrate_pressure,
        phase_randomization=phase_randomization,
        rng=rng,
    )

    output = istft(processed, window_size, hop_size, window)
    return output[: len(audio)]
