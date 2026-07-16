"""Local, offline tempo and key estimation (no external services).

Tempo uses librosa's onset-strength beat tracker. Key uses chroma energy
correlated against the Krumhansl-Schmuckler major/minor key profiles, the
standard textbook approach to key-finding from pitch-class content.
"""

import librosa
import numpy as np

_PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Schmuckler key profiles (relative pitch-class weights, starting at
# the tonic).
_MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
_MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)


def estimate_tempo(audio: np.ndarray, sample_rate: int) -> float:
    """Return an estimated tempo in BPM."""
    tempo, _ = librosa.beat.beat_track(y=audio.astype(np.float32), sr=sample_rate)
    value = float(np.atleast_1d(tempo)[0])
    return value if value > 0 else 120.0


def estimate_key(audio: np.ndarray, sample_rate: int) -> tuple[str, str]:
    """Return (tonic_name, mode) e.g. ("G", "major"), estimated by correlating
    averaged chroma energy against the Krumhansl-Schmuckler key profiles."""
    chroma = librosa.feature.chroma_cqt(y=audio.astype(np.float32), sr=sample_rate)
    profile = chroma.mean(axis=1)

    best_score = -np.inf
    best_tonic = "C"
    best_mode = "major"

    for shift in range(12):
        for mode, template in (("major", _MAJOR_PROFILE), ("minor", _MINOR_PROFILE)):
            rotated = np.roll(template, shift)
            score = float(np.corrcoef(profile, rotated)[0, 1])
            if score > best_score:
                best_score = score
                best_tonic = _PITCH_CLASSES[shift]
                best_mode = mode

    return best_tonic, best_mode
