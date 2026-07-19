"""Synthesize a rough piano-like audio preview of a transcribed score.

Not a real piano sample/soundfont (that would mean bundling a large
soundfont file or an external synth) — just additive synthesis (a
fundamental plus a few decaying harmonics with a percussive envelope) that's
good enough to let you check the transcription sounds roughly like the
source, without needing to read the notation or play it yourself.
"""

import numpy as np
from music21 import stream

# (harmonic number, relative amplitude) — decaying overtone series gives a
# plucked/struck-string character rather than a pure sine beep.
_HARMONICS = [(1, 1.0), (2, 0.5), (3, 0.25), (4, 0.125), (5, 0.06), (6, 0.03)]
_ATTACK_SECONDS = 0.005
_DECAY_RATE = 3.0  # higher = faster decay


def _synthesize_tone(midi: float, duration_seconds: float, sample_rate: int) -> np.ndarray:
    freq = 440.0 * 2.0 ** ((midi - 69) / 12.0)
    n = max(1, int(duration_seconds * sample_rate))
    t = np.arange(n) / sample_rate

    wave = np.zeros(n)
    for harmonic, amplitude in _HARMONICS:
        wave += amplitude * np.sin(2 * np.pi * freq * harmonic * t)

    envelope = np.exp(-_DECAY_RATE * t / max(duration_seconds, 0.05))
    attack_samples = min(n, int(_ATTACK_SECONDS * sample_rate))
    if attack_samples > 0:
        envelope[:attack_samples] *= np.linspace(0.0, 1.0, attack_samples)
    wave *= envelope

    return wave


def _synthesize_part(part: stream.Part, seconds_per_quarter: float, sample_rate: int) -> np.ndarray:
    elements = list(part.flatten().notesAndRests)
    total_seconds = sum(float(el.quarterLength) for el in elements) * seconds_per_quarter
    buffer = np.zeros(max(1, int(total_seconds * sample_rate)) + sample_rate)  # pad tail

    cursor_samples = 0
    for element in elements:
        duration_seconds = float(element.quarterLength) * seconds_per_quarter
        n_samples = int(duration_seconds * sample_rate)

        if element.isNote:
            tone = _synthesize_tone(element.pitch.midi, duration_seconds, sample_rate)
            end = cursor_samples + len(tone)
            if end > len(buffer):
                buffer = np.pad(buffer, (0, end - len(buffer)))
            buffer[cursor_samples:end] += tone

        cursor_samples += n_samples

    return buffer[:cursor_samples + sample_rate // 4]


def synthesize_score(score: stream.Score, bpm: float, sample_rate: int = 44100) -> np.ndarray:
    """Render every part in the score (mixed together if there's more than
    one, e.g. a two-voice duet) to a mono audio buffer at the given tempo."""
    seconds_per_quarter = 60.0 / bpm
    part_buffers = [_synthesize_part(part, seconds_per_quarter, sample_rate) for part in score.parts]

    length = max(len(b) for b in part_buffers)
    buffer = np.zeros(length)
    for part_buffer in part_buffers:
        buffer[: len(part_buffer)] += part_buffer

    peak = np.max(np.abs(buffer))
    if peak > 0:
        buffer = buffer / peak * 0.7

    return buffer.astype(np.float32)
