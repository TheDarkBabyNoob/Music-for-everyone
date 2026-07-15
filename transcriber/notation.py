"""Turn detected note events into sheet music (MusicXML) via music21.

Supports two output keys:
  - "concert": pitches written exactly as heard.
  - "trumpet": pitches transposed up a major second, the standard Bb trumpet
    transposition, and labelled with a Trumpet instrument.
"""

from music21 import instrument, meter, note, stream, tempo

from transcriber.pitch_detection import NoteEvent

# Quantize note starts/durations to the nearest 16th note.
QUANTIZE_UNIT = 0.25  # quarterLength


def _quantize(quarter_length: float) -> float:
    steps = max(1, round(quarter_length / QUANTIZE_UNIT))
    return steps * QUANTIZE_UNIT


def events_to_stream(events: list[NoteEvent], bpm: float = 120.0,
                      key: str = "concert") -> stream.Score:
    """Build a music21 Score from detected note events.

    `key` is "concert" (no transposition) or "trumpet" (written Bb trumpet part,
    transposed up a major second from concert pitch).
    """
    if key not in ("concert", "trumpet"):
        raise ValueError(f"Unknown key: {key!r}")

    part = stream.Part()
    part.append(tempo.MetronomeMark(number=bpm))
    part.append(meter.TimeSignature("4/4"))
    if key == "trumpet":
        part.insert(0, instrument.Trumpet())

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

    if key == "trumpet":
        part = part.transpose("M2")

    score = stream.Score()
    score.append(part)
    return score


def export_musicxml(score: stream.Score, output_path: str) -> str:
    score.write("musicxml", fp=output_path)
    return output_path
