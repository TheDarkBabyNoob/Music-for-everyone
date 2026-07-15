from fractions import Fraction

import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import resample_poly


def record_audio(duration_seconds: float, sample_rate: int = 44100) -> np.ndarray:
    frames = int(round(duration_seconds * sample_rate))
    recording = sd.rec(frames, samplerate=sample_rate, channels=1, dtype="float64")
    sd.wait()
    return np.asarray(recording, dtype=np.float64).reshape(-1)


def load_audio_file(path: str, target_sample_rate: int = 44100) -> tuple[np.ndarray, int]:
    audio, native_sample_rate = sf.read(path, dtype="float64", always_2d=False)
    audio = np.asarray(audio, dtype=np.float64)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if native_sample_rate != target_sample_rate:
        ratio = Fraction(target_sample_rate, native_sample_rate)
        audio = resample_poly(audio, ratio.numerator, ratio.denominator)
        audio = np.asarray(audio, dtype=np.float64)

    return audio.reshape(-1), target_sample_rate
