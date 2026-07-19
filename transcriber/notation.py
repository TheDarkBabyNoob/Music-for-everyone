"""Turn detected note events into sheet music (MusicXML) via music21.

Supports two output keys:
  - "concert": pitches written exactly as heard.
  - "trumpet": pitches transposed up a major second, the standard Bb trumpet
    transposition, and labelled with a Trumpet instrument.
"""

from music21 import instrument, meter, note, stream, tempo
from music21 import key as m21key

from transcriber.pitch_detection import NoteEvent

# Quantize note starts/durations to the nearest 16th note.
QUANTIZE_UNIT = 0.25  # quarterLength


def _quantize(quarter_length: float) -> float:
    steps = max(1, round(quarter_length / QUANTIZE_UNIT))
    return steps * QUANTIZE_UNIT


def _build_part(events: list[NoteEvent], bpm: float, key_signature: tuple[str, str] | None,
                 part_name: str | None = None) -> stream.Part:
    part = stream.Part()
    if part_name is not None:
        part.partName = part_name
    # Plain text rather than a numeric MetronomeMark: the latter renders its
    # note-value icon using a special embedded music font that our pure-Python
    # SVG-to-PDF pipeline (svglib/reportlab, no system Cairo/MuseScore) can't
    # resolve, so it shows up as a solid black box instead of a note glyph.
    part.append(tempo.MetronomeMark(text=f"Quarter = {bpm:.0f}"))
    part.append(meter.TimeSignature("4/4"))
    if key_signature is not None:
        tonic, mode = key_signature
        part.append(m21key.Key(tonic, mode))

    seconds_per_quarter = 60.0 / bpm
    cursor = 0.0  # quarterLength position already written

    for event in events:
        event_start_ql = event.start / seconds_per_quarter
        gap = event_start_ql - cursor
        if gap >= QUANTIZE_UNIT / 2:
            rest_len = _quantize(gap)
            part.append(note.Rest(quarterLength=rest_len))
            cursor += rest_len

        duration_ql = _quantize(event.duration / seconds_per_quarter)
        n = note.Note()
        n.pitch.midi = round(event.midi)
        n.quarterLength = duration_ql
        part.append(n)
        cursor += duration_ql

    return part


def _apply_instrument(part: stream.Part, instrument_key: str) -> stream.Part:
    if instrument_key == "trumpet":
        part.insert(0, instrument.Trumpet())
        part = part.transpose("M2")
    return part


def events_to_stream(events: list[NoteEvent], bpm: float = 120.0,
                      instrument_key: str = "concert",
                      key_signature: tuple[str, str] | None = None) -> stream.Score:
    """Build a single-part music21 Score from detected note events.

    `instrument_key` is "concert" (no transposition) or "trumpet" (written Bb
    trumpet part, transposed up a major second from concert pitch).

    `key_signature`, if given, is a (tonic_name, mode) pair, e.g. ("G", "major"),
    describing the CONCERT-PITCH key. It is inserted before any transposition so
    a trumpet part's key signature transposes along with its notes.
    """
    if instrument_key not in ("concert", "trumpet"):
        raise ValueError(f"Unknown instrument_key: {instrument_key!r}")

    part = _build_part(events, bpm, key_signature)
    part = _apply_instrument(part, instrument_key)

    score = stream.Score()
    score.append(part)
    return score


def voice_events_to_stream(voice_events: list[list[NoteEvent]], bpm: float = 120.0,
                            instrument_key: str = "concert",
                            key_signature: tuple[str, str] | None = None) -> stream.Score:
    """Build a multi-part music21 Score, one staff per voice — for duets/
    harmony where two (or more) independent melodic lines were detected
    separately (see transcriber.multi_pitch). Same instrument_key/
    key_signature handling as events_to_stream, applied to every part.
    """
    if instrument_key not in ("concert", "trumpet"):
        raise ValueError(f"Unknown instrument_key: {instrument_key!r}")

    score = stream.Score()
    for i, events in enumerate(voice_events, start=1):
        part = _build_part(events, bpm, key_signature, part_name=f"Voice {i}")
        part = _apply_instrument(part, instrument_key)
        score.append(part)
    return score


def export_musicxml(score: stream.Score, output_path: str) -> str:
    score.write("musicxml", fp=output_path)
    return output_path
