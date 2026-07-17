"""Monophonic pitch detection and note segmentation.

Uses librosa's pYIN (probabilistic YIN, Mauch & Dixon 2014) rather than a
hand-rolled YIN implementation. Plain YIN is prone to octave errors and
drops quiet/breathy content under a hard silence threshold; pYIN's
Viterbi-decoded voiced/unvoiced detection and octave-aware candidate
selection are specifically built to fix both, and it's already available
since librosa is a dependency for tempo/key detection.
"""

from dataclasses import dataclass

import librosa
import numpy as np


@dataclass
class NoteEvent:
    midi: float  # fractional MIDI pitch (rounded to nearest int before notating)
    start: float  # seconds
    duration: float  # seconds


def freq_to_midi(freq: float) -> float:
    return 69.0 + 12.0 * np.log2(freq / 440.0)


def _bridge_small_gaps(events: list[NoteEvent], max_gap_seconds: float) -> list[NoteEvent]:
    """Real singing/playing is mostly legato. A brief gap between two
    detected notes is almost always a pitch-tracking seam (a consonant, a
    breath, a momentary dip in voicing confidence) rather than an
    intentional rest, so close it by extending the earlier note up to the
    next one instead of leaving a silent hole. Only gaps longer than
    max_gap_seconds are trusted as real rests."""
    if not events:
        return events

    bridged = [events[0]]
    for event in events[1:]:
        previous = bridged[-1]
        gap = event.start - (previous.start + previous.duration)
        if 0 < gap <= max_gap_seconds:
            previous.duration += gap
        bridged.append(event)
    return bridged


def merge_note_events(primary: list[NoteEvent], secondary: list[NoteEvent]) -> list[NoteEvent]:
    """Fill gaps in `primary` using notes from `secondary`, without touching
    anything primary already covers.

    Used to add back pitched content recovered from Demucs' "drums" stem
    (see source_separation.isolate_melody) only where the main signal has
    nothing — filling real silence rather than competing with it. Summing
    the two audio signals before transcribing instead (so the pitch
    tracker sees both at once) made results worse: two simultaneous pitched
    sources fight over one pitch estimate per instant.
    """
    if not secondary:
        return primary
    if not primary:
        return secondary

    combined = list(primary)
    for event in secondary:
        event_end = event.start + event.duration
        overlaps = any(
            event.start < p.start + p.duration and event_end > p.start
            for p in primary
        )
        if not overlaps:
            combined.append(event)

    combined.sort(key=lambda e: e.start)
    return combined


def detect_notes(audio: np.ndarray, sample_rate: int, fmin: float = 65.4,
                  fmax: float = 1046.5, frame_size: int = 2048, hop_size: int = 512,
                  min_note_seconds: float = 0.06,
                  pitch_merge_semitones: float = 0.6,
                  max_voiced_gap_seconds: float = 0.08,
                  bridge_gap_seconds: float = 0.18) -> list[NoteEvent]:
    """Analyze mono audio and return a list of detected NoteEvents (rests are
    the gaps between consecutive events, so silence is implicit).

    Default fmin/fmax span C2-C6, wide enough for both vocals and typical
    lead instruments (e.g. trumpet).
    """
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)

    f0, voiced_flag, _voiced_probs = librosa.pyin(
        audio,
        fmin=fmin,
        fmax=fmax,
        sr=sample_rate,
        frame_length=frame_size,
        hop_length=hop_size,
    )
    n_frames = len(f0)
    frame_midi = np.where(voiced_flag, freq_to_midi(f0), 0.0)
    frame_duration = hop_size / sample_rate
    max_gap_frames = max(0, round(max_voiced_gap_seconds / frame_duration))

    # Group voiced frames into note events, allowing brief unvoiced gaps
    # (a few ms of consonant/breath noise mid-syllable) to stay part of the
    # same note rather than splitting it.
    events: list[NoteEvent] = []
    i = 0
    while i < n_frames:
        if not voiced_flag[i]:
            i += 1
            continue

        j = i
        pitches = [frame_midi[i]]
        last_voiced = i
        while j + 1 < n_frames:
            if voiced_flag[j + 1]:
                if abs(frame_midi[j + 1] - np.median(pitches)) > pitch_merge_semitones:
                    break
                pitches.append(frame_midi[j + 1])
                last_voiced = j + 1
                j += 1
            elif j + 1 - last_voiced <= max_gap_frames:
                j += 1
            else:
                break

        start_time = i * frame_duration
        duration = (last_voiced - i + 1) * frame_duration
        if duration >= min_note_seconds:
            events.append(NoteEvent(midi=float(np.median(pitches)), start=start_time,
                                     duration=duration))
        i = j + 1

    return _bridge_small_gaps(events, bridge_gap_seconds)
