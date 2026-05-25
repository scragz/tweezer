import re
import numpy as np
import soundfile as sf
from pathlib import Path


def read_audio(path: str, mono: bool = False) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float64", always_2d=True)
    if mono:
        audio = audio.mean(axis=1)
    return audio, sr


def write_audio(path: str, audio: np.ndarray, sr: int):
    audio = np.clip(audio, -1.0, 1.0)
    ext = Path(path).suffix.lower()
    subtype = {".wav": "PCM_24", ".flac": "PCM_24", ".aiff": "PCM_24", ".aif": "PCM_24"}.get(
        ext, "PCM_24"
    )
    sf.write(path, audio, sr, subtype=subtype)


def next_output_path(input_path: str) -> str:
    p = Path(input_path)
    stem = p.stem
    parent = p.parent
    pattern = re.compile(rf"^{re.escape(stem)}\.tweezer-(\d+)\.wav$")
    existing = [int(m.group(1)) for f in parent.iterdir() if (m := pattern.match(f.name))]
    n = max(existing, default=0) + 1
    return str(parent / f"{stem}.tweezer-{n:02d}.wav")


def process_stereo(audio: np.ndarray, sr: int, pipeline, progress_callback=None) -> np.ndarray:
    if audio.ndim == 1:
        return pipeline.process(audio, sr, progress_callback)
    channels = []
    for ch in range(audio.shape[1]):
        cb = progress_callback if ch == 0 else None
        channels.append(pipeline.process(audio[:, ch], sr, cb))
    return np.stack(channels, axis=1)
