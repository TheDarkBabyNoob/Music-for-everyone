import io
import shutil
import tempfile
import threading
import time
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk
import numpy as np
import sounddevice as sd
from music21.stream import Score
from PIL import Image

import audio_io
from transcriber.audio_analysis import estimate_key, estimate_tempo
from transcriber.multi_pitch import detect_notes_two_voices
from transcriber.notation import events_to_stream, export_musicxml, voice_events_to_stream
from transcriber.pitch_detection import detect_notes, merge_note_events
from transcriber.playback import synthesize_score
from transcriber.rendering import render_score
from transcriber.source_separation import isolate_melody
from transcriber.url_import import import_from_url


ctk.set_appearance_mode("system")
ctk.set_default_color_theme("blue")


class SheetMusicApp:
    def __init__(self) -> None:
        self.root = ctk.CTk()
        self.root.title("Music for Everyone — Audio to Sheet Music")
        self.root.geometry("1040x740")
        self.root.resizable(True, True)

        self.audio: np.ndarray | None = None
        self.sample_rate: int | None = None
        self.score: Score | None = None
        self.detected_key_signature: tuple[str, str] | None = None
        self.rendered_pdf_path: str | None = None
        self.last_bpm: float | None = None
        self._preview_ctk_image: ctk.CTkImage | None = None

        self.duration_var = ctk.StringVar(value="8")
        self.url_var = ctk.StringVar(value="")
        self.bpm_var = ctk.StringVar(value="120")
        self.instrument_var = ctk.StringVar(value="Concert Pitch")
        self.isolate_var = ctk.BooleanVar(value=True)
        self.multi_voice_var = ctk.BooleanVar(value=False)
        self.status_var = ctk.StringVar(value="Load or record audio to begin.")

        self._build_widgets()
        self.root.update_idletasks()
        self.root.minsize(self.root.winfo_reqwidth(), self.root.winfo_reqheight())

    def _build_widgets(self) -> None:
        main = ctk.CTkFrame(self.root, fg_color="transparent")
        main.grid(row=0, column=0, sticky="nsew", padx=24, pady=24)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=0)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        left_column = ctk.CTkFrame(main, fg_color="transparent", width=420)
        left_column.grid(row=0, column=0, sticky="n")
        left_column.columnconfigure(0, weight=1)

        right_column = ctk.CTkFrame(main, fg_color="transparent")
        right_column.grid(row=0, column=1, sticky="nsew", padx=(24, 0))
        right_column.columnconfigure(0, weight=1)
        right_column.rowconfigure(0, weight=1)

        title_label = ctk.CTkLabel(
            left_column,
            text="Music for Everyone",
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        title_label.grid(row=0, column=0, sticky="w")

        subtitle_label = ctk.CTkLabel(
            left_column,
            text="Audio to Sheet Music",
            font=ctk.CTkFont(size=13),
            text_color=("gray35", "gray70"),
        )
        subtitle_label.grid(row=1, column=0, sticky="w", pady=(2, 22))

        audio_card = ctk.CTkFrame(left_column, corner_radius=12, fg_color=("gray90", "gray17"))
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

        url_entry = ctk.CTkEntry(
            audio_card,
            textvariable=self.url_var,
            placeholder_text="YouTube or Spotify link",
        )
        url_entry.grid(row=3, column=0, columnspan=2, sticky="ew", padx=(18, 10), pady=(0, 18))

        self.import_button = ctk.CTkButton(
            audio_card,
            text="Import from URL",
            command=self.import_from_url_clicked,
            fg_color="transparent",
            border_width=2,
            text_color=("gray10", "gray90"),
        )
        self.import_button.grid(row=3, column=2, sticky="ew", padx=(0, 18), pady=(0, 18))

        settings_card = ctk.CTkFrame(left_column, corner_radius=12, fg_color=("gray90", "gray17"))
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
            pady=(0, 8),
        )

        multi_voice_checkbox = ctk.CTkCheckBox(
            settings_card,
            text="Advanced: detect two voices (duet/harmony)",
            variable=self.multi_voice_var,
            onvalue=True,
            offvalue=False,
        )
        multi_voice_checkbox.grid(
            row=5,
            column=0,
            columnspan=2,
            sticky="w",
            padx=18,
            pady=(0, 8),
        )

        multi_voice_hint = ctk.CTkLabel(
            settings_card,
            text=(
                "For two independent simultaneous melodic lines (e.g. a duet), "
                "written as two staves instead of one. Heuristic, not a trained "
                "model: the primary line is usually solid, the second line is "
                "rougher on dense/busy recordings. Roughly doubles processing "
                "time; leave off for a single melody/vocal line."
            ),
            font=ctk.CTkFont(size=11),
            text_color=("gray45", "gray60"),
            wraplength=330,
            justify="left",
        )
        multi_voice_hint.grid(
            row=6,
            column=0,
            columnspan=2,
            sticky="w",
            padx=18,
            pady=(0, 18),
        )

        self.transcribe_button = ctk.CTkButton(
            left_column,
            text="Transcribe",
            command=self.transcribe,
            state="disabled",
            height=44,
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.transcribe_button.grid(row=4, column=0, sticky="ew", pady=(24, 0))

        buttons_row = ctk.CTkFrame(left_column, fg_color="transparent")
        buttons_row.grid(row=5, column=0, sticky="ew", pady=(12, 0))
        buttons_row.columnconfigure(0, weight=1)
        buttons_row.columnconfigure(1, weight=1)

        self.save_button = ctk.CTkButton(
            buttons_row,
            text="Save MusicXML...",
            command=self.save_sheet_music,
            state="disabled",
            fg_color="transparent",
            border_width=2,
            text_color=("gray10", "gray90"),
        )
        self.save_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        self.export_pdf_button = ctk.CTkButton(
            buttons_row,
            text="Export PDF...",
            command=self.export_pdf,
            state="disabled",
        )
        self.export_pdf_button.grid(row=0, column=1, sticky="ew", padx=(6, 0))

        playback_row = ctk.CTkFrame(left_column, fg_color="transparent")
        playback_row.grid(row=6, column=0, sticky="ew", pady=(12, 0))
        playback_row.columnconfigure(0, weight=1)
        playback_row.columnconfigure(1, weight=0)

        self.play_button = ctk.CTkButton(
            playback_row,
            text="▶  Play Preview (piano)",
            command=self.play_preview,
            state="disabled",
        )
        self.play_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        self.stop_button = ctk.CTkButton(
            playback_row,
            text="Stop",
            command=self.stop_preview,
            state="disabled",
            width=70,
            fg_color="transparent",
            border_width=2,
            text_color=("gray10", "gray90"),
        )
        self.stop_button.grid(row=0, column=1, sticky="ew", padx=(6, 0))

        self.progress_bar = ctk.CTkProgressBar(left_column, mode="determinate")
        self.progress_bar.set(0)
        self.progress_bar.grid(row=7, column=0, sticky="ew", pady=(24, 10))
        self.progress_bar.grid_remove()

        status_label = ctk.CTkLabel(
            left_column,
            textvariable=self.status_var,
            wraplength=400,
            justify="left",
            anchor="w",
            font=ctk.CTkFont(size=12),
            text_color=("gray35", "gray70"),
        )
        status_label.grid(row=8, column=0, sticky="ew")

        preview_card = ctk.CTkFrame(right_column, corner_radius=12, fg_color=("gray90", "gray17"))
        preview_card.grid(row=0, column=0, sticky="nsew")
        preview_card.columnconfigure(0, weight=1)
        preview_card.rowconfigure(0, weight=1)

        self.preview_label = ctk.CTkLabel(
            preview_card,
            text="Transcribe audio to see a sheet music preview.",
            font=ctk.CTkFont(size=13),
            text_color=("gray45", "gray60"),
            wraplength=440,
            justify="center",
        )
        self.preview_label.grid(row=0, column=0, padx=24, pady=24, sticky="nsew")

        self.preview_caption = ctk.CTkLabel(
            preview_card,
            text="",
            font=ctk.CTkFont(size=11),
            text_color=("gray45", "gray60"),
        )
        self.preview_caption.grid(row=1, column=0, padx=18, pady=(0, 16))

    def run(self) -> None:
        self.root.mainloop()

    def _run_async(self, work, on_done, status_text: str, lock_widgets: list,
                    estimated_seconds: float | None = None) -> None:
        """Run `work(report_progress)` on a background thread so the Tk main
        loop (and the progress bar) keeps running during slow operations like
        Demucs separation or a YouTube download. `work` should call
        `report_progress(fraction)` (0.0-1.0) at meaningful checkpoints so the
        bar shows real pipeline stages rather than an ambiguous animation.

        If `estimated_seconds` is given (e.g. a fixed recording duration), a
        timer independently advances the bar toward 0.85 over that many
        seconds, since we know that wait's length in advance even though
        `work` itself can't report progress mid-call.

        `on_done(result, error)` is called back on the main thread once
        finished, with exactly one of result/error set. `lock_widgets` are
        disabled for the duration (and re-enabled after) to prevent starting
        a second operation — e.g. loading new audio — while one that
        reads/writes self.audio is still running on another thread.
        """
        for widget in lock_widgets:
            widget.configure(state="disabled")
        self.status_var.set(status_text)
        self.progress_bar.set(0)
        self.progress_bar.grid()

        timer_state = {"active": True}

        def report_progress(fraction: float) -> None:
            self.root.after(0, lambda fraction=fraction: self.progress_bar.set(fraction))

        if estimated_seconds and estimated_seconds > 0:
            start_time = time.monotonic()

            def tick() -> None:
                if not timer_state["active"]:
                    return
                elapsed = time.monotonic() - start_time
                self.progress_bar.set(min(0.85, elapsed / estimated_seconds * 0.85))
                self.root.after(100, tick)

            tick()

        def worker() -> None:
            try:
                result = work(report_progress)
            except Exception as exc:
                # Bind exc as a default arg: except-clause variables are
                # cleared when the block exits, so a plain closure over exc
                # would see it unbound by the time this lambda actually runs.
                self.root.after(
                    0, lambda exc=exc: self._finish_async(lock_widgets, timer_state, on_done, None, exc)
                )
            else:
                self.root.after(
                    0, lambda: self._finish_async(lock_widgets, timer_state, on_done, result, None)
                )

        threading.Thread(target=worker, daemon=True).start()

    def _finish_async(self, lock_widgets: list, timer_state: dict, on_done, result, error) -> None:
        timer_state["active"] = False
        self.progress_bar.grid_remove()
        for widget in lock_widgets:
            widget.configure(state="normal")
        on_done(result, error)

    def _analyze_tempo_and_key(self, audio: np.ndarray, sample_rate: int) -> tuple[float, tuple[str, str] | None]:
        """Best-effort local tempo/key detection (librosa, fully offline). Pure
        computation, safe to call off the main thread — does not touch any
        Tk widgets or variables."""
        try:
            bpm = estimate_tempo(audio, sample_rate)
            tonic, mode = estimate_key(audio, sample_rate)
        except Exception:
            return 120.0, None
        return bpm, (tonic, mode)

    def _apply_detected_tempo_key(self, bpm: float, key_signature: tuple[str, str] | None) -> str:
        """Update bpm_var/detected_key_signature and return a status suffix.
        Must run on the main thread."""
        if key_signature is None:
            return ""
        self.bpm_var.set(f"{bpm:.0f}")
        self.detected_key_signature = key_signature
        tonic, mode = key_signature
        return f" Detected ~{bpm:.0f} BPM, key of {tonic} {mode}."

    def record_from_microphone(self) -> None:
        try:
            duration = float(self.duration_var.get())
            if duration <= 0:
                raise ValueError("Duration must be greater than zero.")
        except ValueError as exc:
            messagebox.showerror("Invalid Duration", str(exc))
            return

        def work(report_progress):
            audio = audio_io.record_audio(duration, 44100)
            report_progress(0.9)
            bpm, key_signature = self._analyze_tempo_and_key(audio, 44100)
            report_progress(1.0)
            return audio, bpm, key_signature

        def on_done(result, error) -> None:
            if error is not None:
                self.status_var.set("Recording failed.")
                messagebox.showerror("Recording Failed", str(error))
                return
            audio, bpm, key_signature = result
            self.audio = audio
            self.sample_rate = 44100
            self.score = None
            self.transcribe_button.configure(state="normal")
            self.save_button.configure(state="disabled")
            suffix = self._apply_detected_tempo_key(bpm, key_signature)
            self.status_var.set(f"Recorded {duration:.1f}s of audio.{suffix}")

        self._run_async(
            work, on_done, "Recording...",
            [self.record_button, self.load_button, self.import_button],
            estimated_seconds=duration,
        )

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

        def work(report_progress):
            report_progress(0.1)
            audio, sample_rate = audio_io.load_audio_file(path, 44100)
            report_progress(0.6)
            bpm, key_signature = self._analyze_tempo_and_key(audio, sample_rate)
            report_progress(1.0)
            return audio, sample_rate, bpm, key_signature

        def on_done(result, error) -> None:
            if error is not None:
                self.status_var.set("Loading failed.")
                messagebox.showerror("Loading Failed", str(error))
                return
            audio, sample_rate, bpm, key_signature = result
            self.audio = audio
            self.sample_rate = sample_rate
            self.score = None
            self.transcribe_button.configure(state="normal")
            self.save_button.configure(state="disabled")
            duration = len(audio) / sample_rate
            suffix = self._apply_detected_tempo_key(bpm, key_signature)
            self.status_var.set(f"Loaded {Path(path).name} ({duration:.1f}s).{suffix}")

        self._run_async(
            work, on_done, "Loading...",
            [self.record_button, self.load_button, self.import_button],
        )

    def import_from_url_clicked(self) -> None:
        url = self.url_var.get().strip()
        if not url:
            messagebox.showerror("No Link", "Paste a YouTube or Spotify link first.")
            return

        def work(report_progress):
            result = import_from_url(url, 44100, progress_callback=report_progress)
            bpm, key_signature = self._analyze_tempo_and_key(result.audio, result.sample_rate)
            report_progress(1.0)
            return result, bpm, key_signature

        def on_done(result, error) -> None:
            if error is not None:
                self.status_var.set("Import failed.")
                messagebox.showerror("Import Failed", str(error))
                return
            import_result, bpm, key_signature = result
            self.audio = import_result.audio
            self.sample_rate = import_result.sample_rate
            self.score = None
            self.transcribe_button.configure(state="normal")
            self.save_button.configure(state="disabled")
            duration = len(self.audio) / self.sample_rate
            suffix = self._apply_detected_tempo_key(bpm, key_signature)
            self.status_var.set(f"Imported '{import_result.label}' ({duration:.1f}s).{suffix}")

        self._run_async(
            work, on_done, "Importing audio (this can take a moment)...",
            [self.record_button, self.load_button, self.import_button],
        )

    def transcribe(self) -> None:
        if self.audio is None or self.sample_rate is None:
            messagebox.showerror("No Audio", "Load or record audio before transcribing.")
            return

        try:
            bpm = float(self.bpm_var.get())
            if bpm <= 0:
                raise ValueError("Tempo must be greater than zero.")
        except ValueError as exc:
            messagebox.showerror("Invalid Tempo", str(exc))
            return

        key = self._notation_key()
        key_signature = self.detected_key_signature
        audio, sample_rate = self.audio, self.sample_rate
        isolate = self.isolate_var.get() is True
        multi_voice = self.multi_voice_var.get() is True

        def work(report_progress):
            report_progress(0.05)
            if isolate:
                isolated = isolate_melody(audio, sample_rate)
                report_progress(0.4)
                melody_audio, melody_sr = isolated.primary, isolated.sample_rate
            else:
                melody_audio, melody_sr = audio, sample_rate
                report_progress(0.4)

            if multi_voice:
                voice1, voice2 = detect_notes_two_voices(melody_audio, melody_sr)
                report_progress(0.7)
                score = voice_events_to_stream(
                    [voice1, voice2], bpm=bpm, instrument_key=key, key_signature=key_signature,
                )
                note_count = len(voice1) + len(voice2)
            else:
                events = detect_notes(melody_audio, melody_sr)
                if isolate:
                    recovered_events = detect_notes(isolated.recovered, isolated.sample_rate)
                    events = merge_note_events(events, recovered_events)
                report_progress(0.7)
                score = events_to_stream(
                    events, bpm=bpm, instrument_key=key, key_signature=key_signature,
                )
                note_count = len(events)
            report_progress(0.8)

            pdf_handle = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
            pdf_handle.close()
            rendered = render_score(score, pdf_handle.name)
            report_progress(1.0)

            return score, note_count, rendered

        def on_done(result, error) -> None:
            if error is not None:
                self.status_var.set("Transcription failed.")
                messagebox.showerror("Transcription Failed", str(error))
                return
            score, note_count, rendered = result
            self.score = score
            self.last_bpm = bpm
            self.rendered_pdf_path = rendered.pdf_path
            self._display_preview(rendered.preview_png, rendered.page_count)
            self.save_button.configure(state="normal")
            self.export_pdf_button.configure(state="normal")
            self.play_button.configure(state="normal")
            self.status_var.set(f"Transcription complete: detected {note_count} notes.")

        if isolate:
            status_text = "Isolating melody (this can take a while the first time)..."
        elif multi_voice:
            status_text = "Detecting two voices..."
        else:
            status_text = "Transcribing..."
        self._run_async(
            work, on_done, status_text,
            [self.record_button, self.load_button, self.import_button, self.transcribe_button],
        )

    def _display_preview(self, preview_png: bytes, page_count: int) -> None:
        image = Image.open(io.BytesIO(preview_png))
        max_width, max_height = 560, 650
        scale = min(max_width / image.width, max_height / image.height, 1.0)
        display_size = (round(image.width * scale), round(image.height * scale))

        self._preview_ctk_image = ctk.CTkImage(
            light_image=image, dark_image=image, size=display_size
        )
        self.preview_label.configure(image=self._preview_ctk_image, text="")
        if page_count > 1:
            self.preview_caption.configure(text=f"Page 1 of {page_count} (full score in the exported PDF)")
        else:
            self.preview_caption.configure(text="")

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

    def export_pdf(self) -> None:
        if self.rendered_pdf_path is None:
            messagebox.showerror("No Score", "Transcribe audio before exporting a PDF.")
            return

        path = filedialog.asksaveasfilename(
            title="Export PDF",
            defaultextension=".pdf",
            filetypes=(
                ("PDF files", "*.pdf"),
                ("All files", "*.*"),
            ),
        )
        if not path:
            return

        try:
            shutil.copyfile(self.rendered_pdf_path, path)
            self.status_var.set(f"Saved PDF to {path}.")
            messagebox.showinfo("PDF Exported", f"Saved to {path}")
        except Exception as exc:
            messagebox.showerror("Export Failed", str(exc))

    def play_preview(self) -> None:
        if self.score is None or self.last_bpm is None:
            messagebox.showerror("No Score", "Transcribe audio before playing a preview.")
            return

        score = self.score
        bpm = self.last_bpm

        def work(report_progress):
            report_progress(0.3)
            audio = synthesize_score(score, bpm)
            report_progress(1.0)
            return audio

        def on_done(result, error) -> None:
            if error is not None:
                messagebox.showerror("Playback Failed", str(error))
                return
            sd.play(result, 44100)
            self.stop_button.configure(state="normal")
            self.status_var.set("Playing preview (rough piano tones — for checking the transcription, not a real piano).")

        self._run_async(work, on_done, "Rendering preview audio...", [self.play_button])

    def stop_preview(self) -> None:
        sd.stop()
        self.stop_button.configure(state="disabled")
        self.status_var.set("Playback stopped.")

    def _notation_key(self) -> str:
        if self.instrument_var.get() == "Bb Trumpet":
            return "trumpet"
        return "concert"


if __name__ == "__main__":
    SheetMusicApp().run()
