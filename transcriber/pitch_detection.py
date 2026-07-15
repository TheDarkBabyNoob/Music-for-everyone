"""Monophonic pitch detection and note segmentation using the YIN algorithm.

No external DSP library is used here: aubio (the obvious choice) does not build
on current Python/numpy, so YIN is implemented directly on top of numpy/scipy.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class NoteEvent:
    midi: float  # fractional MIDI pitch (rounded to nearest int before notating)
    start: float  # seconds
    duration: float  # seconds


def _yin_frame_pitch(frame: np.ndarray, sample_rate: int, fmin: float, fmax: float,
                      threshold: float) -> float:
    """Return the fundamental frequency of one frame in Hz, or 0.0 if unvoiced."""
    n = len(frame)
    tau_max = min(n // 2, int(sample_rate / fmin))
    tau_min = max(1, int(sample_rate / fmax))

    diff = np.zeros(tau_max)
    for tau in range(tau_min, tau_max):
        delta = frame[: n - tau] - frame[tau:n]
        diff[tau] = np.dot(delta, delta)

    cmnd = np.ones(tau_max)
    running_sum = 0.0
    for tau in range(tau_min, tau_max):
        running_sum += diff[tau]
        cmnd[tau] = diff[tau] * tau / running_sum if running_sum > 0 else 1.0

    tau_estimate = -1
    for tau in range(tau_min, tau_max):
        if cmnd[tau] < threshold:
            while tau + 1 < tau_max and cmnd[tau + 1] < cmnd[tau]:
                tau += 1
            tau_estimate = tau
            break

    if tau_estimate == -1:
        return 0.0

    # Parabolic interpolation around tau_estimate for sub-sample precision.
    t = tau_estimate
    if 0 < t < tau_max - 1:
        s0, s1, s2 = cmnd[t - 1], cmnd[t], cmnd[t + 1]
        denom = 2 * s1 - s2 - s0
        better_tau = t + (0 if denom == 0 else 0.5 * (s0 - s2) / denom)
    else:
        better_tau = t

    if better_tau <= 0:
        return 0.0
    return sample_rate / better_tau


def freq_to_midi(freq: float) -> float:
    return 69.0 + 12.0 * np.log2(freq / 440.0)


def detect_notes(audio: np.ndarray, sample_rate: int, fmin: float = 80.0,
                  fmax: float = 1200.0, frame_size: int = 2048, hop_size: int = 512,
                  yin_threshold: float = 0.15, silence_rms: float = 0.01,
                  min_note_seconds: float = 0.08,
                  pitch_merge_semitones: float = 0.6) -> list[NoteEvent]:
    """Analyze mono audio and return a list of detected NoteEvents (rests are the
    gaps between consecutive events, so silence is implicit)."""
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float64)

    n_frames = 1 + max(0, (len(audio) - frame_size) // hop_size)
    frame_midi = np.zeros(n_frames)
    frame_voiced = np.zeros(n_frames, dtype=bool)

    for i in range(n_frames):
        start = i * hop_size
        frame = audio[start:start + frame_size]
        if len(frame) < frame_size:
            frame = np.pad(frame, (0, frame_size - len(frame)))

        rms = np.sqrt(np.mean(frame ** 2))
        if rms < silence_rms:
            continue

        freq = _yin_frame_pitch(frame, sample_rate, fmin, fmax, yin_threshold)
        if freq <= 0:
            continue

        frame_midi[i] = freq_to_midi(freq)
        frame_voiced[i] = True

    # Group consecutive voiced frames with a stable pitch into note events.
    events: list[NoteEvent] = []
    frame_duration = hop_size / sample_rate

    i = 0
    while i < n_frames:
        if not frame_voiced[i]:
            i += 1
            continue

        j = i
        pitches = [frame_midi[i]]
        while j + 1 < n_frames and frame_voiced[j + 1] and \
                abs(frame_midi[j + 1] - np.median(pitches)) <= pitch_merge_semitones:
            j += 1
            pitches.append(frame_midi[j])

        start_time = i * frame_duration
        duration = (j - i + 1) * frame_duration
        if duration >= min_note_seconds:
            events.append(NoteEvent(midi=float(np.median(pitches)), start=start_time,
                                     duration=duration))
        i = j + 1

    return events
