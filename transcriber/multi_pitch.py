"""Two-voice pitch detection for duets: two independent monophonic lines
played simultaneously (e.g. two instruments in harmony), as opposed to a
single instrument playing chords.

A single autocorrelation-based tracker (pitch_detection.detect_notes, using
pYIN) locks onto one pitch per instant and is actively misleading on
simultaneous two-note intervals: mixing two sustained tones a perfect fifth
apart, pYIN reports the dyad's implied "missing fundamental" (an octave
below the lower note) rather than either real pitch — a well known
psychoacoustic effect, not a bug in that implementation.

This instead works in the frequency domain: for each frame, score every
candidate pitch by summing the STFT magnitude at its fundamental and first
few harmonics (a simplified version of the classical harmonic-sum-spectrum
approach to multi-F0 estimation, e.g. Klapuri 2006), take the best-scoring
candidate as voice 1, cancel its harmonic content out of that frame's
spectrum, then repeat to find voice 2 in what's left. Both real pitches
show up directly as spectral peaks even when the combined waveform's
periodicity does not, so this sidesteps the missing-fundamental problem
pYIN runs into.

This is a heuristic, not a full polyphonic transcription model (which
would need supervised training data this project doesn't have and can't
easily get) — expect it to do reasonably well on two clearly distinct
simultaneous lines and to struggle on dense/unclear harmony, the same way
any lightweight DSP approach would.
"""

from dataclasses import dataclass

import librosa
import numpy as np

from transcriber.pitch_detection import NoteEvent, _bridge_small_gaps, _split_on_onsets

_N_HARMONICS = 6
_CANCEL_FACTOR = 0.05  # how much of voice 1's harmonic energy survives cancellation
_CANCEL_BIN_RADIUS = 2  # cancel bins neighboring each harmonic too, not just the exact
# one: real tones (vibrato, natural pitch drift within a note) spread energy across
# nearby bins, unlike a stationary synthetic test tone, so cancelling only the exact
# bin leaves enough real-world residue behind that it gets mistaken for a second voice
_MIN_MIDI, _MAX_MIDI = 36, 96  # C2-C7, wide enough for most duet material


def _midi_to_hz(midi: np.ndarray) -> np.ndarray:
    return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)


def _salience_spectrogram(magnitude: np.ndarray, sample_rate: int, frame_size: int,
                           candidates_midi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """magnitude: (n_bins, n_frames). Returns (salience, bin_idx) where
    salience is (n_candidates, n_frames) and bin_idx (n_candidates,
    n_harmonics) maps each candidate to its harmonics' spectrogram rows."""
    candidate_hz = _midi_to_hz(candidates_midi)
    harmonic_freqs = candidate_hz[:, None] * np.arange(1, _N_HARMONICS + 1)[None, :]
    bin_idx = np.clip(
        np.round(harmonic_freqs / (sample_rate / frame_size)).astype(int), 0, magnitude.shape[0] - 1
    )
    weights = 1.0 / np.arange(1, _N_HARMONICS + 1)

    harmonic_mags = magnitude[bin_idx]  # (n_candidates, n_harmonics, n_frames)
    salience = (harmonic_mags * weights[None, :, None]).sum(axis=1)  # (n_candidates, n_frames)
    return salience, bin_idx


def _track_voice(magnitude: np.ndarray, sample_rate: int, frame_size: int,
                  candidates_midi: np.ndarray,
                  salience_threshold: float | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (midi_per_frame, voiced_flag, magnitude_with_this_voice_cancelled,
    best_salience_value). If salience_threshold is None, every frame is
    provisionally marked voiced and the caller is expected to threshold
    best_salience_value itself (used to calibrate voice 1's threshold from
    its own salience distribution before applying it to voice 2 — see
    detect_notes_two_voices)."""
    salience, bin_idx = _salience_spectrogram(magnitude, sample_rate, frame_size, candidates_midi)
    best = np.argmax(salience, axis=0)  # (n_frames,)
    best_value = np.take_along_axis(salience, best[None, :], axis=0)[0]

    midi_per_frame = candidates_midi[best].astype(np.float64)
    voiced = np.ones_like(best_value, dtype=bool) if salience_threshold is None \
        else best_value > salience_threshold

    n_frames = magnitude.shape[1]
    chosen_bins = bin_idx[best]  # (n_frames, n_harmonics)
    frame_indices = np.arange(n_frames)[:, None]
    cancelled = magnitude.copy()
    for offset in range(-_CANCEL_BIN_RADIUS, _CANCEL_BIN_RADIUS + 1):
        offset_bins = np.clip(chosen_bins + offset, 0, magnitude.shape[0] - 1)
        cancelled[offset_bins, frame_indices] *= _CANCEL_FACTOR

    return midi_per_frame, voiced, cancelled, best_value


def _frames_to_events(midi_per_frame: np.ndarray, voiced: np.ndarray, frame_duration: float,
                       pitch_merge_semitones: float, min_note_seconds: float,
                       max_voiced_gap_seconds: float) -> list[NoteEvent]:
    n_frames = len(midi_per_frame)
    max_gap_frames = max(0, round(max_voiced_gap_seconds / frame_duration))

    events: list[NoteEvent] = []
    i = 0
    while i < n_frames:
        if not voiced[i]:
            i += 1
            continue

        j = i
        pitches = [midi_per_frame[i]]
        last_voiced = i
        while j + 1 < n_frames:
            if voiced[j + 1]:
                if abs(midi_per_frame[j + 1] - np.median(pitches)) > pitch_merge_semitones:
                    break
                pitches.append(midi_per_frame[j + 1])
                last_voiced = j + 1
                j += 1
            elif j + 1 - last_voiced <= max_gap_frames:
                j += 1
            else:
                break

        start_time = i * frame_duration
        duration = (last_voiced - i + 1) * frame_duration
        if duration >= min_note_seconds:
            events.append(NoteEvent(midi=float(np.median(pitches)), start=start_time, duration=duration))
        i = j + 1

    return events


def _merge_adjacent_same_pitch(events: list[NoteEvent], pitch_tolerance: float = 0.6,
                                gap_tolerance_seconds: float = 0.02) -> list[NoteEvent]:
    """Onset detection runs on the full mixed signal, so a sustained voice
    can pick up false re-attacks from the OTHER voice's note onsets,
    fragmenting one held note into several adjacent same-pitch pieces.
    Re-merge consecutive events that are essentially touching and at the
    same pitch — a real repeated note (pitch_detection.detect_notes'
    onset-splitting case) has an audible gap or is a separately intended
    attack, not a near-zero seam."""
    if not events:
        return events

    merged = [events[0]]
    for event in events[1:]:
        previous = merged[-1]
        gap = event.start - (previous.start + previous.duration)
        if abs(event.midi - previous.midi) <= pitch_tolerance and gap <= gap_tolerance_seconds:
            previous.duration = event.start + event.duration - previous.start
        else:
            merged.append(event)
    return merged


def detect_notes_two_voices(audio: np.ndarray, sample_rate: int, frame_size: int = 4096,
                             hop_size: int = 512, silence_ratio: float = 0.15,
                             pitch_merge_semitones: float = 0.6, min_note_seconds: float = 0.06,
                             max_voiced_gap_seconds: float = 0.08,
                             bridge_gap_seconds: float = 0.18) -> tuple[list[NoteEvent], list[NoteEvent]]:
    """Detect two simultaneous monophonic voices (e.g. a duet) and return
    (voice1_events, voice2_events), each independently segmented into notes
    the same way pitch_detection.detect_notes does."""
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)

    stft = np.abs(librosa.stft(audio, n_fft=frame_size, hop_length=hop_size))
    candidates_midi = np.arange(_MIN_MIDI, _MAX_MIDI + 1)

    # Calibrate an absolute salience threshold from voice 1's own detected
    # strength, rather than against the raw total spectral energy: real
    # recordings carry a lot of broadband noise/reverb/timbre energy a
    # clean synthetic test tone doesn't, so "at least 15% of everything in
    # the spectrum" is a sensible bar for a pure sine wave and an
    # impossible one for real audio — voice 2 would never clear it even
    # when it's a perfectly legitimate, clearly present note.
    _, _, _, provisional_salience = _track_voice(stft, sample_rate, frame_size, candidates_midi, None)
    reference_level = np.median(provisional_salience[provisional_salience > 0])
    salience_threshold = silence_ratio * reference_level

    midi1, voiced1, cancelled, _ = _track_voice(
        stft, sample_rate, frame_size, candidates_midi, salience_threshold
    )
    midi2, voiced2, _, _ = _track_voice(
        cancelled, sample_rate, frame_size, candidates_midi, salience_threshold
    )

    frame_duration = hop_size / sample_rate
    onset_times = librosa.onset.onset_detect(y=audio, sr=sample_rate, hop_length=hop_size, units="time")

    voice_events = []
    for midi_track, voiced in ((midi1, voiced1), (midi2, voiced2)):
        events = _frames_to_events(
            midi_track, voiced, frame_duration, pitch_merge_semitones,
            min_note_seconds, max_voiced_gap_seconds,
        )
        events = _split_on_onsets(events, onset_times, min_note_seconds)
        events = _bridge_small_gaps(events, bridge_gap_seconds)
        events = _merge_adjacent_same_pitch(events)
        voice_events.append(events)

    return voice_events[0], voice_events[1]
