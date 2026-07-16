"""Isolate the main melodic line from a full mix using Demucs.

Drums and bass are discarded; vocals + other (which covers lead instruments
like trumpet, guitar, piano, synths) are summed back together, since a
melody that Demucs doesn't recognize as "vocals" often lands in "other"
instead.
"""

import numpy as np
import torch
from demucs.api import Separator

_separator: Separator | None = None


def _get_separator() -> Separator:
    global _separator
    if _separator is None:
        _separator = Separator(model="htdemucs")
    return _separator


def isolate_melody(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
    """Remove drums/bass via Demucs source separation.

    Returns (mono_melody_audio, sample_rate) at Demucs' native sample rate
    (44100 Hz).
    """
    separator = _get_separator()
    wav = torch.as_tensor(audio, dtype=torch.float32).reshape(1, -1)
    stereo = wav.expand(2, -1).contiguous()

    _, stems = separator.separate_tensor(stereo, sr=sample_rate)
    melody = stems["vocals"] + stems["other"]
    mono = melody.mean(dim=0).numpy().astype(np.float64)
    return mono, separator.samplerate
