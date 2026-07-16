from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk
import numpy as np
from music21.stream import Score

import audio_io
from transcriber.audio_analysis import estimate_key, estimate_tempo
from transcriber.notation import events_to_stream, export_musicxml
from transcriber.pitch_detection import detect_notes
from transcriber.source_separation import isolate_melody


ctk.set_appearance_mode("system")
ctk.set_default_color_theme("blue")


class SheetMusicApp:
    def __init__(self) -> None:
        self.root = ctk.CTk()
        self.root.title("Music for Everyone — Audio to Sheet Music")
        self.root.geometry("460x600")
        self.root.resizable(True, True)

        self.audio: np.ndarray | None = None
        self.sample_rate: int | None = None
        self.score: Score | None = None
        self.detected_key_signature: tuple[str, str] | None = None

        self.duration_var = ctk.StringVar(value="8")
        self.bpm_var = ctk.StringVar(value="120")
        self.instrument_var = ctk.StringVar(value="Concert Pitch")
        self.isolate_var = ctk.BooleanVar(value=True)
        self.status_var = ctk.StringVar(value="Load or record audio to begin.")

        self._build_widgets()
        self.root.update_idletasks()
        self.root.minsize(self.root.winfo_reqwidth(), self.root.winfo_reqheight())

    def _build_widgets(self) -> None:
        main = ctk.CTkFrame(self.root, fg_color="transparent")
        main.grid(row=0, column=0, sticky="nsew", padx=24, pady=24)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=1)

        title_label = ctk.CTkLabel(
            main,
            text="Music for Everyone",
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        title_label.grid(row=0, column=0, sticky="w")

        subtitle_label = ctk.CTkLabel(
            main,
            text="Audio to Sheet Music",
            font=ctk.CTkFont(size=13),
            text_color=("gray35", "gray70"),
        )
        subtitle_label.grid(row=1, column=0, sticky="w", pady=(2, 22))

        audio_card = ctk.CTkFrame(main, corner_radius=12, fg_color=("gray90", "gray17"))
        audio_card.grid(row=2, column=0, sticky="ew")
        audio_card.columnconfigure(2, weight=1)

        audio_heading = ctk.CTkLabel(
            audio_card,
            text="Audio Source",
            font=ctk.CTkFont(size=15, weight="bold"),
        )
        audio_heading.grid(row=0, column=0, columnspan=3, sticky="w", padx=18, pady=(16, 12))

        ctk.CTkLabel(audio_card, text="Duration (sec)").grid(
            row=1,
            column=0,
            sticky="w",
            padx=(18, 10),
            pady=(0, 14),
        )
        duration_entry = ctk.CTkEntry(audio_card, textvariable=self.duration_var, width=64)
        duration_entry.grid(row=1, column=1, sticky="w", pady=(0, 14))

        self.record_button = ctk.CTkButton(
            audio_card,
            text="Record from Microphone",
            command=self.record_from_microphone,
        )
        self.record_button.grid(
            row=1,
            column=2,
            sticky="ew",
            padx=(12, 18),
            pady=(0, 14),
        )

        self.load_button = ctk.CTkButton(
            audio_card,
            text="Load Audio File...",
            command=self.load_audio_file,
            fg_color="transparent",
            border_width=2,
            text_color=("gray10", "gray90"),
        )
        self.load_button.grid(
            row=2,
            column=0,
            columnspan=3,
            sticky="ew",
            padx=18,
            pady=(0, 18),
        )

        settings_card = ctk.CTkFrame(main, corner_radius=12, fg_color=("gray90", "gray17"))
        settings_card.grid(row=3, column=0, sticky="ew", pady=(18, 0))
        settings_card.columnconfigure(1, weight=1)

        settings_heading = ctk.CTkLabel(
            settings_card,
            text="Settings",
            font=ctk.CTkFont(size=15, weight="bold"),
        )
        settings_heading.grid(row=0, column=0, columnspan=2, sticky="w", padx=18, pady=(16, 12))

        ctk.CTkLabel(settings_card, text="Tempo (BPM)").grid(
            row=1,
            column=0,
            sticky="w",
            padx=(18, 14),
            pady=(0, 14),
        )
        bpm_entry = ctk.CTkEntry(settings_card, textvariable=self.bpm_var, width=84)
        bpm_entry.grid(row=1, column=1, sticky="w", padx=(0, 18), pady=(0, 14))

        ctk.CTkLabel(settings_card, text="Instrument").grid(
            row=2,
            column=0,
            sticky="w",
            padx=(18, 14),
            pady=(0, 18),
        )
        instrument_combo = ctk.CTkComboBox(
            settings_card,
            variable=self.instrument_var,
            values=["Concert Pitch", "Bb Trumpet"],
            state="readonly",
            width=180,
        )
        instrument_combo.grid(row=2, column=1, sticky="w", padx=(0, 18), pady=(0, 18))

        isolate_checkbox = ctk.CTkCheckBox(
            settings_card,
            text="Isolate melody (remove drums/bass)",
            variable=self.isolate_var,
            onvalue=True,
            offvalue=False,
        )
        isolate_checkbox.grid(
            row=3,
            column=0,
            columnspan=2,
            sticky="w",
            padx=18,
            pady=(0, 8),
        )

        isolate_hint = ctk.CTkLabel(
            settings_card,
            text=(
                "Uses AI source separation (Demucs). Slower, and downloads a model "
                "the first time it runs. Turn off for a clean solo recording."
            ),
            font=ctk.CTkFont(size=11),
            text_color=("gray45", "gray60"),
            wraplength=330,
            justify="left",
        )
        isolate_hint.grid(
            row=4,
            column=0,
            columnspan=2,
            sticky="w",
            padx=18,
            pady=(0, 18),
        )

        self.transcribe_button = ctk.CTkButton(
            main,
            text="Transcribe",
            command=self.transcribe,
            state="disabled",
            height=44,
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.transcribe_button.grid(row=4, column=0, sticky="ew", pady=(24, 0))

        self.save_button = ctk.CTkButton(
            main,
            text="Save Sheet Music...",
            command=self.save_sheet_music,
            state="disabled",
            fg_color="transparent",
            border_width=2,
            text_color=("gray10", "gray90"),
        )
        self.save_button.grid(row=5, column=0, sticky="ew", pady=(12, 0))

        self.progress_bar = ctk.CTkProgressBar(main, mode="indeterminate")
        self.progress_bar.grid(row=6, column=0, sticky="ew", pady=(24, 10))
        self.progress_bar.grid_remove()

        status_label = ctk.CTkLabel(
            main,
            textvariable=self.status_var,
            wraplength=400,
            justify="left",
            anchor="w",
            font=ctk.CTkFont(size=12),
            text_color=("gray35", "gray70"),
        )
        status_label.grid(row=7, column=0, sticky="ew")

    def run(self) -> None:
        self.root.mainloop()

    def _analyze_tempo_and_key(self) -> str:
        """Best-effort local tempo/key detection (librosa, fully offline).

        Updates bpm_var and detected_key_signature as a side effect. Returns a
        short phrase to append to the status message, or '' if detection failed.
        """
        try:
            bpm = estimate_tempo(self.audio, self.sample_rate)
            tonic, mode = estimate_key(self.audio, self.sample_rate)
        except Exception:
            self.detected_key_signature = None
            return ""

        self.bpm_var.set(f"{bpm:.0f}")
        self.detected_key_signature = (tonic, mode)
        return f" Detected ~{bpm:.0f} BPM, key of {tonic} {mode}."

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
        self.progress_bar.grid()
        self.progress_bar.start()
        self.root.update_idletasks()

        try:
            self.audio = audio_io.record_audio(duration, 44100)
            self.sample_rate = 44100
            self.score = None
            self.transcribe_button.configure(state="normal")
            self.save_button.configure(state="disabled")
            self.status_var.set("Analyzing tempo & key...")
            self.root.update_idletasks()
            suffix = self._analyze_tempo_and_key()
            self.status_var.set(f"Recorded {duration:.1f}s of audio.{suffix}")
        except Exception as exc:
            self.status_var.set("Recording failed.")
            messagebox.showerror("Recording Failed", str(exc))
        finally:
            self.progress_bar.stop()
            self.progress_bar.grid_remove()
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

        self.load_button.configure(state="disabled")
        self.status_var.set("Loading...")
        self.progress_bar.grid()
        self.progress_bar.start()
        self.root.update_idletasks()

        try:
            self.audio, self.sample_rate = audio_io.load_audio_file(path, 44100)
            self.score = None
            self.transcribe_button.configure(state="normal")
            self.save_button.configure(state="disabled")
            duration = len(self.audio) / self.sample_rate
            self.status_var.set("Analyzing tempo & key...")
            self.root.update_idletasks()
            suffix = self._analyze_tempo_and_key()
            self.status_var.set(f"Loaded {Path(path).name} ({duration:.1f}s).{suffix}")
        except Exception as exc:
            self.status_var.set("Loading failed.")
            messagebox.showerror("Loading Failed", str(exc))
        finally:
            self.progress_bar.stop()
            self.progress_bar.grid_remove()
            self.load_button.configure(state="normal")

    def transcribe(self) -> None:
        if self.audio is None or self.sample_rate is None:
            messagebox.showerror("No Audio", "Load or record audio before transcribing.")
            return

        previous_status = self.status_var.get()
        self.status_var.set("Transcribing...")
        self.transcribe_button.configure(state="disabled")
        self.progress_bar.grid()
        self.progress_bar.start()
        self.root.update_idletasks()

        try:
            bpm = float(self.bpm_var.get())
            if bpm <= 0:
                raise ValueError("Tempo must be greater than zero.")

            key = self._notation_key()
            if self.isolate_var.get() is True:
                self.status_var.set("Isolating melody (this can take a while the first time)...")
                self.root.update_idletasks()
                melody_audio, melody_sr = isolate_melody(self.audio, self.sample_rate)
            else:
                melody_audio, melody_sr = self.audio, self.sample_rate

            self.status_var.set("Transcribing...")
            events = detect_notes(melody_audio, melody_sr)
            self.score = events_to_stream(
                events,
                bpm=bpm,
                instrument_key=key,
                key_signature=self.detected_key_signature,
            )
            self.save_button.configure(state="normal")
            self.status_var.set(f"Transcription complete: detected {len(events)} notes.")
        except Exception as exc:
            self.status_var.set(previous_status)
            messagebox.showerror("Transcription Failed", str(exc))
        finally:
            self.progress_bar.stop()
            self.progress_bar.grid_remove()
            if self.audio is not None and self.sample_rate is not None:
                self.transcribe_button.configure(state="normal")

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
