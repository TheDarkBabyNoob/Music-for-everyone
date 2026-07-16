"""Isolate the main melodic line from a full mix using Demucs.

Drums and bass are always discarded. Between vocals and other, we pick
whichever actually carries the melody instead of always summing both:

- A song with real singing has vocals energy far above "other" (guitar/
  keys/backing) — summing them back in would reintroduce exactly the
  backing noise we're trying to remove, which is what was making
  transcriptions of vocal tracks noisy and octave-confused.
- A solo instrumental melody (e.g. a trumpet recording) that Demucs
  doesn't recognize as "vocals" lands almost entirely in "other", with
  vocals near silent — so we fall back to combining both in that case.
"""

import numpy as np
import torch
from demucs.api import Separator

_separator: Separator | None = None

# If vocals carry at least this fraction of "other"'s energy, treat vocals
# as the real melody source and use them alone rather than summing with
# "other". Calibrated against a real vocal track (vocals ~5x "other") and a
# synthetic instrumental-only clip (vocals ~0.008x "other").
_VOCALS_DOMINANCE_RATIO = 0.15


def _get_separator() -> Separator:
    global _separator
    if _separator is None:
        _separator = Separator(model="htdemucs")
    return _separator


def isolate_melody(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
    """Remove drums/bass (and backing instruments, if vocals are present)
    via Demucs source separation.

    Returns (mono_melody_audio, sample_rate) at Demucs' native sample rate
    (44100 Hz).
    """
    separator = _get_separator()
    wav = torch.as_tensor(audio, dtype=torch.float32).reshape(1, -1)
    stereo = wav.expand(2, -1).contiguous()

    _, stems = separator.separate_tensor(stereo, sr=sample_rate)

    vocals_energy = float(stems["vocals"].abs().mean())
    other_energy = float(stems["other"].abs().mean())
    if vocals_energy > _VOCALS_DOMINANCE_RATIO * other_energy:
        melody = stems["vocals"]
    else:
        melody = stems["vocals"] + stems["other"]

    mono = melody.mean(dim=0).numpy().astype(np.float64)
    return mono, separator.samplerate
