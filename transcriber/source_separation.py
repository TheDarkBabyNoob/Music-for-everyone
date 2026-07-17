"""Isolate the main melodic line from a full mix using Demucs.

Bass is always discarded — it's a separate part from the lead melody, and
including it would confuse the monophonic pitch tracker with the wrong
notes. Between vocals and other, we pick whichever actually carries the
melody instead of always summing both:

- A song with real singing has vocals energy far above "other" (guitar/
  keys/backing) — summing them back in would reintroduce exactly the
  backing noise we're trying to remove, which is what was making
  transcriptions of vocal tracks noisy and octave-confused.
- A solo instrumental melody (e.g. a trumpet recording) that Demucs
  doesn't recognize as "vocals" lands almost entirely in "other", with
  vocals near silent — so we fall back to combining both in that case.

Drums are NOT discarded wholesale. Demucs' "drums" stem is a learned
classification, not a literal pitch test, and it regularly misclassifies
pitched content (synth stabs, plucked/percussive synths, bleed from other
instruments) as drums — on a real test track, 44% of the "drums" stem's
energy turned out to be harmonic/pitched, not real percussion. So instead
of dropping that whole stem, harmonic-percussive source separation (HPSS,
a signal-processing technique that separates sustained/pitched content
from noisy transients based on actual spectral shape, not an instrument
classifier) recovers any pitched material from it.

That recovered material is kept as a *separate* signal rather than mixed
back into the main melody audio: summing two independently-pitched signals
creates real interference for a monophonic pitch tracker (two competing
tones fighting over one pitch estimate per instant), which made things
worse, not better, in testing. Instead, the caller transcribes each signal
on its own and only uses the recovered one to fill in gaps where the main
signal has nothing — see pitch_detection.merge_note_events.
"""

from dataclasses import dataclass

import librosa
import numpy as np
import torch
from demucs.api import Separator

_separator: Separator | None = None

# If vocals carry at least this fraction of "other"'s energy, treat vocals
# as the real melody source and use them alone rather than summing with
# "other". Calibrated against a real vocal track (vocals ~5x "other") and a
# synthetic instrumental-only clip (vocals ~0.008x "other").
_VOCALS_DOMINANCE_RATIO = 0.15


@dataclass
class IsolatedMelody:
    primary: np.ndarray  # vocals, or vocals+other for instrumental melodies
    recovered: np.ndarray  # pitched content salvaged out of the "drums" stem
    sample_rate: int


def _get_separator() -> Separator:
    global _separator
    if _separator is None:
        _separator = Separator(model="htdemucs")
    return _separator


def isolate_melody(audio: np.ndarray, sample_rate: int) -> IsolatedMelody:
    """Separate out real percussion and bass via Demucs + HPSS."""
    separator = _get_separator()
    wav = torch.as_tensor(audio, dtype=torch.float32).reshape(1, -1)
    stereo = wav.expand(2, -1).contiguous()

    _, stems = separator.separate_tensor(stereo, sr=sample_rate)

    vocals_energy = float(stems["vocals"].abs().mean())
    other_energy = float(stems["other"].abs().mean())
    if vocals_energy > _VOCALS_DOMINANCE_RATIO * other_energy:
        primary = stems["vocals"]
    else:
        primary = stems["vocals"] + stems["other"]
    primary_mono = primary.mean(dim=0).numpy().astype(np.float64)

    drums_mono = stems["drums"].mean(dim=0).numpy().astype(np.float64)
    recovered_mono, _drums_percussive = librosa.effects.hpss(drums_mono)

    return IsolatedMelody(primary=primary_mono, recovered=recovered_mono,
                           sample_rate=separator.samplerate)
