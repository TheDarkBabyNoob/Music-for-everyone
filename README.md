Music for Everyone — Audio to Sheet Music
=================================

Desktop app that listens to (or loads) monophonic audio, detects the melody's
pitch and rhythm, and writes it out as sheet music (MusicXML). Two output
keys are supported:

- **Concert Pitch** — written exactly as heard.
- **Bb Trumpet** — written a major second higher, the standard transposition
  for Bb trumpet.

Pitch detection uses a from-scratch YIN implementation on top of numpy/scipy
(no aubio — it doesn't build on current Python). The GUI is plain Tkinter
(ships with Python, no PySimpleGUI license issues).

Setup
-----

```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Run
---

```bash
source venv/bin/activate
python main.py
```

Usage
-----

1. Either **Record from Microphone** for the given duration, or **Load Audio
   File...** (.wav / .flac / .ogg).
2. Set the **Tempo (BPM)** you want durations quantized against, and pick
   **Concert Pitch** or **Bb Trumpet**.
3. Click **Transcribe**.
4. Click **Save Sheet Music...** to write a `.musicxml` file. Open it in
   MuseScore (free), Finale, or Sibelius to view/print/play it back.

Limitations
-----------

- Monophonic only — one melodic line at a time (chords/polyphony will not
  transcribe cleanly).
- Rhythm is quantized to 16th notes at the tempo you specify; the app does
  not estimate tempo automatically.
- mp3 files aren't supported (soundfile has no mp3 decoder); convert to wav
  first if needed.
