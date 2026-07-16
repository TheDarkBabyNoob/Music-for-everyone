Music for Everyone — Audio to Sheet Music
=================================

Desktop app (Tkinter/customtkinter GUI) that listens to (or loads) audio,
isolates the main melody, detects its pitch and rhythm, and writes it out as
sheet music (MusicXML). Two output keys are supported:

- **Concert Pitch** — written exactly as heard.
- **Bb Trumpet** — written a major second higher, the standard transposition
  for Bb trumpet.

Transcription itself is entirely local — no cloud APIs, no accounts. The one
optional network feature is importing audio from a YouTube/Spotify link.

How it works
------------

- **Import from URL** (optional): paste a YouTube link and it downloads the
  audio directly (via `yt-dlp`). Paste a Spotify *track* link and it reads
  the public metadata Spotify serves for link previews (song title + artist,
  no login or API key) to find and download the matching audio from
  YouTube — Spotify's own streams are DRM-protected and there's no
  legitimate way to pull audio from them directly, so this is the same
  approach tools like spotDL use. The status bar shows exactly what got
  downloaded so you can confirm it matched the right track.
- **Melody isolation** (optional, on by default): runs [Demucs](https://github.com/facebookresearch/demucs)
  (Meta's open-source source-separation model) to split the audio into
  vocals/drums/bass/other stems, then discards drums and bass before
  transcribing. This is what lets it pull a melody out of a full mix instead
  of tripping over drum hits. Downloads a ~80MB model the first time it runs
  (cached after), and adds real processing time — turn it off in Settings for
  an already-clean solo recording.
- **Pitch detection** uses a from-scratch YIN implementation on top of
  numpy/scipy (no aubio — it doesn't build on current Python).
- **Tempo & key** are auto-detected locally with `librosa` (beat tracking +
  Krumhansl-Schmuckler key profile correlation) right after you record,
  load, or import audio, and pre-fill the Tempo field and the sheet music's
  key signature. You can still edit the tempo by hand before transcribing.

Setup
-----

```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Note: this installs PyTorch and Demucs for melody isolation, which is a
sizeable download (roughly 1GB with dependencies). If you'd rather skip that,
remove `torch` and `demucs` from `requirements.txt` and leave the "Isolate
melody" checkbox off — everything else still works.

Run
---

```bash
source venv/bin/activate
python main.py
```

On macOS there's also **Music for Everyone.app** for double-clicking instead
of using the terminal.

Usage
-----

1. **Record from Microphone**, **Load Audio File...** (.wav / .flac / .ogg),
   or paste a **YouTube/Spotify link** and click **Import from URL**. Tempo
   and key are auto-detected right after.
2. Adjust **Tempo (BPM)** if needed, pick **Concert Pitch** or **Bb
   Trumpet**, and leave **Isolate melody** checked if the recording has
   drums/other instruments behind the melody.
3. Click **Transcribe**.
4. Click **Save Sheet Music...** to write a `.musicxml` file. Open it in
   MuseScore (free), Finale, or Sibelius to view/print/play it back.

Limitations
-----------

- Monophonic only — one melodic *line* at a time. Melody isolation separates
  the lead line from the backing track, but if the lead itself is
  polyphonic (chords, harmony vocals) only one pitch per moment gets
  transcribed.
- Rhythm is quantized to 16th notes at the detected/entered tempo.
- mp3 files aren't supported directly (soundfile has no mp3 decoder), but
  importing from a URL sidesteps that since it downloads straight to wav.
- Spotify import matches by title/artist search on YouTube — it isn't
  Spotify's actual audio (impossible without circumventing DRM, which this
  project won't do), so very obscure tracks or unusual remixes may not
  match exactly. The status bar always shows what it actually found so you
  can double check.
- No general "identify this recording" (Shazam-style) lookup — that needs an
  acoustic fingerprint database and wouldn't help transcribe your own
  playing anyway, which is the app's main use case. Song lookup only works
  when you already have a YouTube/Spotify link.
