import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from music21.stream import Score

import audio_io
from transcriber.notation import events_to_stream, export_musicxml
from transcriber.pitch_detection import detect_notes


class SheetMusicApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Music for Everyone — Audio to Sheet Music")
        self.root.geometry("480x420")
        self.root.resizable(True, True)

        self.audio: np.ndarray | None = None
        self.sample_rate: int | None = None
        self.score: Score | None = None

        self.duration_var = tk.StringVar(value="8")
        self.bpm_var = tk.StringVar(value="120")
        self.instrument_var = tk.StringVar(value="Concert Pitch")
        self.status_var = tk.StringVar(value="Load or record audio to begin.")

        self._build_widgets()
        self.root.update_idletasks()
        self.root.minsize(self.root.winfo_reqwidth(), self.root.winfo_reqheight())

    def _build_widgets(self) -> None:
        main = ttk.Frame(self.root, padding=18)
        main.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)

        ttk.Label(main, text="Duration (seconds)").grid(row=0, column=0, sticky="w")
        duration_entry = ttk.Entry(main, textvariable=self.duration_var, width=10)
        duration_entry.grid(row=0, column=1, sticky="w", padx=(12, 0))

        self.record_button = ttk.Button(
            main,
            text="Record from Microphone",
            command=self.record_from_microphone,
        )
        self.record_button.grid(row=0, column=2, sticky="e", padx=(12, 0))

        self.load_button = ttk.Button(
            main,
            text="Load Audio File...",
            command=self.load_audio_file,
        )
        self.load_button.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(18, 0))

        ttk.Separator(main).grid(row=2, column=0, columnspan=3, sticky="ew", pady=18)

        ttk.Label(main, text="Tempo (BPM)").grid(row=3, column=0, sticky="w")
        bpm_entry = ttk.Entry(main, textvariable=self.bpm_var, width=10)
        bpm_entry.grid(row=3, column=1, sticky="w", padx=(12, 0))

        ttk.Label(main, text="Instrument/Key").grid(row=4, column=0, sticky="w", pady=(14, 0))
        instrument_combo = ttk.Combobox(
            main,
            textvariable=self.instrument_var,
            values=("Concert Pitch", "Bb Trumpet"),
            state="readonly",
            width=18,
        )
        instrument_combo.grid(row=4, column=1, columnspan=2, sticky="w", padx=(12, 0), pady=(14, 0))

        self.transcribe_button = ttk.Button(
            main,
            text="Transcribe",
            command=self.transcribe,
            state="disabled",
        )
        self.transcribe_button.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(24, 0))

        self.save_button = ttk.Button(
            main,
            text="Save Sheet Music...",
            command=self.save_sheet_music,
            state="disabled",
        )
        self.save_button.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(12, 0))

        ttk.Separator(main).grid(row=7, column=0, columnspan=3, sticky="ew", pady=18)

        status_label = ttk.Label(
            main,
            textvariable=self.status_var,
            wraplength=420,
            justify="left",
        )
        status_label.grid(row=8, column=0, columnspan=3, sticky="ew")

    def run(self) -> None:
        self.root.mainloop()

    def record_from_microphone(self) -> None:
        try:
            duration = float(self.duration_var.get())
            if duration <= 0:
                raise ValueError("Duration must be greater than zero.")
        except ValueError as exc:
            messagebox.showerror("Invalid Duration", str(exc))
            return

        self.record_button.configure(state="disabled")
        self.status_var.set("Recording...")
        self.root.update_idletasks()

        try:
            self.audio = audio_io.record_audio(duration, 44100)
            self.sample_rate = 44100
            self.score = None
            self.transcribe_button.configure(state="normal")
            self.save_button.configure(state="disabled")
            self.status_var.set(f"Recorded {duration:.1f}s of audio.")
        except Exception as exc:
            self.status_var.set("Recording failed.")
            messagebox.showerror("Recording Failed", str(exc))
        finally:
            self.record_button.configure(state="normal")

    def load_audio_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Load Audio File",
            filetypes=(
                ("Audio files", "*.wav *.flac *.ogg"),
                ("WAV files", "*.wav"),
                ("FLAC files", "*.flac"),
                ("Ogg files", "*.ogg"),
                ("All files", "*.*"),
            ),
        )
        if not path:
            return

        try:
            self.audio, self.sample_rate = audio_io.load_audio_file(path, 44100)
            self.score = None
            self.transcribe_button.configure(state="normal")
            self.save_button.configure(state="disabled")
            duration = len(self.audio) / self.sample_rate
            self.status_var.set(f"Loaded {Path(path).name} ({duration:.1f}s).")
        except Exception as exc:
            self.status_var.set("Loading failed.")
            messagebox.showerror("Loading Failed", str(exc))

    def transcribe(self) -> None:
        if self.audio is None or self.sample_rate is None:
            messagebox.showerror("No Audio", "Load or record audio before transcribing.")
            return

        previous_status = self.status_var.get()
        self.status_var.set("Transcribing...")
        self.root.update_idletasks()

        try:
            bpm = float(self.bpm_var.get())
            if bpm <= 0:
                raise ValueError("Tempo must be greater than zero.")

            key = self._notation_key()
            events = detect_notes(self.audio, self.sample_rate)
            self.score = events_to_stream(events, bpm=bpm, key=key)
            self.save_button.configure(state="normal")
            self.status_var.set(f"Transcription complete: detected {len(events)} notes.")
        except Exception as exc:
            self.status_var.set(previous_status)
            messagebox.showerror("Transcription Failed", str(exc))

    def save_sheet_music(self) -> None:
        if self.score is None:
            messagebox.showerror("No Score", "Transcribe audio before saving sheet music.")
            return

        path = filedialog.asksaveasfilename(
            title="Save Sheet Music",
            defaultextension=".musicxml",
            filetypes=(
                ("MusicXML files", "*.musicxml *.xml"),
                ("All files", "*.*"),
            ),
        )
        if not path:
            return

        try:
            saved_path = export_musicxml(self.score, path)
            self.status_var.set(f"Saved sheet music to {saved_path}.")
            messagebox.showinfo(
                "Sheet Music Saved",
                f"Saved to {saved_path}\n\nOpen it in MuseScore, Finale, or Sibelius.",
            )
        except Exception as exc:
            messagebox.showerror("Save Failed", str(exc))

    def _notation_key(self) -> str:
        if self.instrument_var.get() == "Bb Trumpet":
            return "trumpet"
        return "concert"


if __name__ == "__main__":
    SheetMusicApp().run()
