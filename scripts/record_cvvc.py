from __future__ import annotations

import csv
import math
import queue
import re
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RECORDING_LIST_PATH = PROJECT_ROOT / "data" / "recording_lists" / "mandarin_cvvc_test.txt"

FINAL_DIR = PROJECT_ROOT / "data" / "new_recordings"
CACHE_DIR = PROJECT_ROOT / "data" / "recording_cache"

FINAL_INDEX_CSV = FINAL_DIR / "recording_index.csv"
CACHE_INDEX_CSV = CACHE_DIR / "recording_index_cache.csv"

SAMPLE_RATE = 48000
CHANNELS = 1
DTYPE = "float32"

WAVEFORM_SECONDS = 4.0


def safe_filename(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[^\w\u4e00-\u9fff\-]+", "_", text)
    text = text.strip("_")
    return text or "recording"


def parse_aliases(line: str) -> list[str]:
    """
    a_ba_pa_ta -> ["a", "ba", "pa", "ta"]
    """
    line = line.strip()
    if not line:
        return []

    parts = [p.strip() for p in line.split("_")]
    return [p for p in parts if p]


def hz_to_note_name(freq: float) -> str:
    if freq <= 0 or math.isnan(freq):
        return "N/A"

    note_names = ["C", "C#", "D", "D#", "E", "F",
                  "F#", "G", "G#", "A", "A#", "B"]

    midi = round(69 + 12 * math.log2(freq / 440.0))
    note = note_names[midi % 12]
    octave = midi // 12 - 1
    return f"{note}{octave}"


def note_to_freq(note: str) -> float:
    note = note.strip().upper()

    mapping = {
        "C": 0,
        "C#": 1,
        "DB": 1,
        "D": 2,
        "D#": 3,
        "EB": 3,
        "E": 4,
        "F": 5,
        "F#": 6,
        "GB": 6,
        "G": 7,
        "G#": 8,
        "AB": 8,
        "A": 9,
        "A#": 10,
        "BB": 10,
        "B": 11,
    }

    match = re.match(r"^([A-G]#?|[A-G]B)(-?\d+)$", note)
    if not match:
        raise ValueError(f"Invalid note: {note}")

    name = match.group(1)
    octave = int(match.group(2))

    midi = (octave + 1) * 12 + mapping[name]
    return 440.0 * (2 ** ((midi - 69) / 12))


def play_tone(freq: float, duration: float = 0.8, sr: int = SAMPLE_RATE) -> None:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    wave = 0.25 * np.sin(2 * np.pi * freq * t)
    fade_len = min(int(sr * 0.03), len(wave) // 2)
    if fade_len > 0:
        fade_in = np.linspace(0, 1, fade_len)
        fade_out = np.linspace(1, 0, fade_len)
        wave[:fade_len] *= fade_in
        wave[-fade_len:] *= fade_out

    sd.play(wave.astype(np.float32), sr)


def estimate_average_pitch_hz(audio: np.ndarray, sr: int) -> float | None:
    if audio.ndim > 1:
        y = audio[:, 0]
    else:
        y = audio

    if len(y) < sr * 0.1:
        return None

    y = y.astype(np.float32)

    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C6"),
            sr=sr,
        )
    except Exception:
        return None

    if f0 is None:
        return None

    valid = f0[~np.isnan(f0)]

    if len(valid) == 0:
        return None

    return float(np.mean(valid))


def load_recording_list(path: Path) -> list["RecordingLine"]:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "a_ba_pa_ta\n"
            "i_bi_pi_ti\n"
            "u_bu_pu_tu\n",
            encoding="utf-8",
        )

    lines: list[RecordingLine] = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        text = raw_line.strip()
        if not text or text.startswith("#"):
            continue

        aliases = parse_aliases(text)
        if aliases:
            lines.append(RecordingLine(text=text, aliases=aliases))

    if not lines:
        raise ValueError(f"录音表为空：{path}")

    return lines


@dataclass
class RecordingLine:
    text: str
    aliases: list[str]


@dataclass
class PendingTake:
    audio: np.ndarray
    stop_ms: float
    marks_ms: list[float]
    line: RecordingLine
    line_index: int
    take: int
    avg_pitch_hz: float | None


class CVVCRecorderApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("SinoOto CVVC Recorder")

        FINAL_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        self.lines = load_recording_list(RECORDING_LIST_PATH)

        self.line_index = 0
        self.take_counter = 1
        self.current_alias_index = 0

        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self.stream: sd.InputStream | None = None
        self.recording = False
        self.start_time_perf = 0.0

        self.space_start_marks_ms: list[float] = []
        self.audio_chunks: list[np.ndarray] = []
        self.audio_lock = threading.Lock()

        self.pending_take: PendingTake | None = None

        self.saved_pitch_values: list[float] = []
        self.load_existing_pitch_stats()

        self.build_ui()
        self.update_display()
        self.schedule_waveform_update()

    @property
    def current_line(self) -> RecordingLine:
        return self.lines[self.line_index]

    def build_ui(self) -> None:
        self.root.geometry("1200x720")

        main = tk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left = tk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        center = tk.Frame(main)
        center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right = tk.Frame(main)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))

        tk.Label(left, text="Recording List", font=("Arial", 14, "bold")).pack(anchor="w")

        self.line_listbox = tk.Listbox(left, width=30, height=32, font=("Arial", 12))
        self.line_listbox.pack(fill=tk.Y, expand=True)

        for line in self.lines:
            self.line_listbox.insert(tk.END, line.text)

        self.line_listbox.bind("<<ListboxSelect>>", self.on_select_line)

        tk.Label(center, text="SinoOto CVVC Recorder", font=("Arial", 22, "bold")).pack(pady=5)

        self.line_label = tk.Label(center, text="", font=("Arial", 30, "bold"), fg="#222222")
        self.line_label.pack(pady=10)

        self.alias_label = tk.Label(center, text="", font=("Arial", 24), fg="#0066aa")
        self.alias_label.pack(pady=5)

        self.instruction_label = tk.Label(
            center,
            text="Space = mark current alias START",
            font=("Arial", 14),
            fg="#555555",
        )
        self.instruction_label.pack(pady=3)

        self.progress_label = tk.Label(center, text="", font=("Arial", 14))
        self.progress_label.pack(pady=3)

        self.status_label = tk.Label(center, text="", font=("Arial", 14), fg="#555555")
        self.status_label.pack(pady=3)

        self.wave_canvas = tk.Canvas(center, height=190, bg="#111111")
        self.wave_canvas.pack(fill=tk.X, pady=12)

        stat_frame = tk.LabelFrame(center, text="Pitch Stats", font=("Arial", 12, "bold"))
        stat_frame.pack(fill=tk.X, pady=5)

        self.current_pitch_label = tk.Label(stat_frame, text="Current take avg pitch: N/A", font=("Arial", 13))
        self.current_pitch_label.pack(anchor="w", padx=10, pady=2)

        self.all_pitch_label = tk.Label(stat_frame, text="All saved wav avg pitch: N/A", font=("Arial", 13))
        self.all_pitch_label.pack(anchor="w", padx=10, pady=2)

        button_frame = tk.Frame(center)
        button_frame.pack(pady=10)

        self.start_button = tk.Button(
            button_frame,
            text="Start",
            font=("Arial", 15),
            width=10,
            command=self.start_recording,
        )
        self.start_button.grid(row=0, column=0, padx=4)

        self.stop_button = tk.Button(
            button_frame,
            text="Stop",
            font=("Arial", 15),
            width=10,
            command=self.stop_recording,
            state=tk.DISABLED,
        )
        self.stop_button.grid(row=0, column=1, padx=4)

        self.save_button = tk.Button(
            button_frame,
            text="Save",
            font=("Arial", 15),
            width=10,
            command=self.save_pending_final,
            state=tk.DISABLED,
        )
        self.save_button.grid(row=0, column=2, padx=4)

        self.cache_button = tk.Button(
            button_frame,
            text="Cache",
            font=("Arial", 15),
            width=10,
            command=self.save_pending_cache,
            state=tk.DISABLED,
        )
        self.cache_button.grid(row=0, column=3, padx=4)

        self.redo_button = tk.Button(
            button_frame,
            text="Redo",
            font=("Arial", 15),
            width=10,
            command=self.redo_current,
        )
        self.redo_button.grid(row=0, column=4, padx=4)

        self.next_button = tk.Button(
            button_frame,
            text="Next",
            font=("Arial", 15),
            width=10,
            command=self.next_line,
        )
        self.next_button.grid(row=0, column=5, padx=4)

        tk.Label(right, text="Reference Pitch", font=("Arial", 14, "bold")).pack(anchor="w")

        self.note_var = tk.StringVar(value="C4")

        note_entry_frame = tk.Frame(right)
        note_entry_frame.pack(anchor="w", pady=5)

        tk.Entry(note_entry_frame, textvariable=self.note_var, width=8, font=("Arial", 13)).pack(side=tk.LEFT)

        tk.Button(
            note_entry_frame,
            text="Play",
            font=("Arial", 12),
            command=self.play_selected_note,
        ).pack(side=tk.LEFT, padx=5)

        piano_frame = tk.LabelFrame(right, text="Piano Notes")
        piano_frame.pack(fill=tk.X, pady=8)

        notes = [
            "C3", "D3", "E3", "F3", "G3", "A3", "B3",
            "C4", "D4", "E4", "F4", "G4", "A4", "B4",
            "C5",
        ]

        for i, note in enumerate(notes):
            btn = tk.Button(
                piano_frame,
                text=note,
                width=5,
                command=lambda n=note: self.play_note_button(n),
            )
            btn.grid(row=i // 3, column=i % 3, padx=2, pady=2)

        sharp_frame = tk.LabelFrame(right, text="Sharps")
        sharp_frame.pack(fill=tk.X, pady=8)

        sharps = ["C#3", "D#3", "F#3", "G#3", "A#3", "C#4", "D#4", "F#4", "G#4", "A#4"]
        for i, note in enumerate(sharps):
            btn = tk.Button(
                sharp_frame,
                text=note,
                width=5,
                command=lambda n=note: self.play_note_button(n),
            )
            btn.grid(row=i // 2, column=i % 2, padx=2, pady=2)

        help_text = (
            "Controls:\n"
            "Start: begin current line\n"
            "Space: mark alias START\n"
            "Stop: finish take\n"
            "Save: save to new_recordings\n"
            "Cache: save to recording_cache\n"
            "Redo: discard pending take\n"
            "Click list item: select / overwrite"
        )

        tk.Label(right, text=help_text, justify=tk.LEFT, font=("Arial", 11), fg="#555555").pack(anchor="w", pady=10)

        self.root.bind("<space>", self.on_space)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def load_existing_pitch_stats(self) -> None:
        if not FINAL_DIR.exists():
            return

        for wav_path in FINAL_DIR.glob("*.wav"):
            try:
                audio, sr = sf.read(wav_path, dtype="float32")
                pitch = estimate_average_pitch_hz(audio, sr)
                if pitch is not None:
                    self.saved_pitch_values.append(pitch)
            except Exception:
                continue

    def update_display(self) -> None:
        line = self.current_line

        self.line_listbox.selection_clear(0, tk.END)
        self.line_listbox.selection_set(self.line_index)
        self.line_listbox.see(self.line_index)

        self.line_label.config(text=line.text)

        if self.current_alias_index < len(line.aliases):
            current_alias = line.aliases[self.current_alias_index]
        else:
            current_alias = "All aliases marked"

        self.alias_label.config(text=f"Current: {current_alias}")

        self.progress_label.config(
            text=f"Line {self.line_index + 1}/{len(self.lines)} | "
                 f"Alias {min(self.current_alias_index + 1, len(line.aliases))}/{len(line.aliases)} | "
                 f"Take {self.take_counter}"
        )

        if self.saved_pitch_values:
            avg_all = float(np.mean(self.saved_pitch_values))
            self.all_pitch_label.config(
                text=f"All saved wav avg pitch: {avg_all:.2f} Hz ({hz_to_note_name(avg_all)})"
            )
        else:
            self.all_pitch_label.config(text="All saved wav avg pitch: N/A")

    def on_select_line(self, event=None) -> None:
        if self.recording:
            return

        selection = self.line_listbox.curselection()
        if not selection:
            return

        idx = int(selection[0])
        self.line_index = idx
        self.current_alias_index = 0
        self.pending_take = None

        self.save_button.config(state=tk.DISABLED)
        self.cache_button.config(state=tk.DISABLED)
        self.status_label.config(text="Selected line. Press Start.", fg="#555555")
        self.current_pitch_label.config(text="Current take avg pitch: N/A")

        self.update_display()

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(status)
        self.audio_queue.put(indata.copy())

    def consume_audio_queue(self) -> None:
        while self.recording:
            try:
                chunk = self.audio_queue.get(timeout=0.1)
                with self.audio_lock:
                    self.audio_chunks.append(chunk)
            except queue.Empty:
                continue

    def start_recording(self) -> None:
        if self.recording:
            return

        self.pending_take = None
        self.current_alias_index = 0
        self.space_start_marks_ms = []

        with self.audio_lock:
            self.audio_chunks = []

        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        try:
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                callback=self.audio_callback,
            )
            self.stream.start()
        except Exception as e:
            messagebox.showerror("Recording error", f"无法打开麦克风：\n{e}")
            return

        self.recording = True
        self.start_time_perf = time.perf_counter()

        self.consumer_thread = threading.Thread(target=self.consume_audio_queue, daemon=True)
        self.consumer_thread.start()

        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.save_button.config(state=tk.DISABLED)
        self.cache_button.config(state=tk.DISABLED)

        self.status_label.config(
            text="Recording. Press Space when each alias STARTS.",
            fg="#aa0000",
        )
        self.current_pitch_label.config(text="Current take avg pitch: recording...")

        self.update_display()

    def on_space(self, event=None) -> None:
        if not self.recording:
            return

        line = self.current_line

        if self.current_alias_index >= len(line.aliases):
            self.status_label.config(
                text="All start markers already recorded. Finish reading, then press Stop.",
                fg="#aa7700",
            )
            return

        mark_ms = (time.perf_counter() - self.start_time_perf) * 1000.0
        self.space_start_marks_ms.append(mark_ms)

        alias = line.aliases[self.current_alias_index]
        print(f"Start mark {alias}: {mark_ms:.1f} ms")

        self.current_alias_index += 1

        if self.current_alias_index >= len(line.aliases):
            self.status_label.config(
                text="Last alias start marked. Finish the sound, then press Stop.",
                fg="#007700",
            )

        self.update_display()

    def stop_stream(self) -> float:
        stop_ms = (time.perf_counter() - self.start_time_perf) * 1000.0

        self.recording = False

        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

        time.sleep(0.1)

        while not self.audio_queue.empty():
            try:
                chunk = self.audio_queue.get_nowait()
                with self.audio_lock:
                    self.audio_chunks.append(chunk)
            except queue.Empty:
                break

        return stop_ms

    def stop_recording(self) -> None:
        if not self.recording:
            return

        stop_ms = self.stop_stream()

        with self.audio_lock:
            if not self.audio_chunks:
                messagebox.showerror("No audio", "没有录到音频。")
                self.reset_after_stop()
                return
            audio = np.concatenate(self.audio_chunks, axis=0)

        line = self.current_line

        if len(self.space_start_marks_ms) < len(line.aliases):
            messagebox.showwarning(
                "Not enough start markers",
                f"这句需要 {len(line.aliases)} 次 Space，但你只按了 {len(self.space_start_marks_ms)} 次。\n"
                f"这条不会直接保存，可以 Redo。",
            )

        pitch = estimate_average_pitch_hz(audio, SAMPLE_RATE)

        self.pending_take = PendingTake(
            audio=audio,
            stop_ms=stop_ms,
            marks_ms=list(self.space_start_marks_ms),
            line=line,
            line_index=self.line_index,
            take=self.take_counter,
            avg_pitch_hz=pitch,
        )

        if pitch is None:
            self.current_pitch_label.config(text="Current take avg pitch: N/A")
        else:
            self.current_pitch_label.config(
                text=f"Current take avg pitch: {pitch:.2f} Hz ({hz_to_note_name(pitch)})"
            )

        self.stop_button.config(state=tk.DISABLED)
        self.start_button.config(state=tk.NORMAL)
        self.save_button.config(state=tk.NORMAL)
        self.cache_button.config(state=tk.NORMAL)

        self.status_label.config(
            text="Stopped. Choose Save or Cache, or Redo.",
            fg="#555555",
        )

    def reset_after_stop(self) -> None:
        self.stop_button.config(state=tk.DISABLED)
        self.start_button.config(state=tk.NORMAL)
        self.save_button.config(state=tk.DISABLED)
        self.cache_button.config(state=tk.DISABLED)
        self.recording = False

    def make_take_filename(self, line: RecordingLine, take: int) -> str:
        return f"{safe_filename(line.text)}_take{take:03d}.wav"

    def save_pending_final(self) -> None:
        if self.pending_take is None:
            return

        self.save_pending_to(
            target_dir=FINAL_DIR,
            index_csv=FINAL_INDEX_CSV,
            is_final=True,
        )

    def save_pending_cache(self) -> None:
        if self.pending_take is None:
            return

        self.save_pending_to(
            target_dir=CACHE_DIR,
            index_csv=CACHE_INDEX_CSV,
            is_final=False,
        )

    def save_pending_to(self, target_dir: Path, index_csv: Path, is_final: bool) -> None:
        take = self.pending_take
        if take is None:
            return

        target_dir.mkdir(parents=True, exist_ok=True)

        filename = self.make_take_filename(take.line, take.take)
        wav_path = target_dir / filename

        if wav_path.exists():
            overwrite = messagebox.askyesno(
                "Overwrite?",
                f"{filename} 已经存在。\n要覆盖吗？",
            )
            if not overwrite:
                return

        sf.write(wav_path, take.audio, SAMPLE_RATE)

        self.rewrite_index_for_line(
            index_csv=index_csv,
            wav_filename=filename,
            take=take,
        )

        if is_final and take.avg_pitch_hz is not None:
            self.saved_pitch_values.append(take.avg_pitch_hz)

        kind = "Saved" if is_final else "Cached"
        self.status_label.config(text=f"{kind}: {filename}", fg="#007700")

        print(f"{kind} wav: {wav_path}")
        print(f"Updated index: {index_csv}")

        self.pending_take = None
        self.save_button.config(state=tk.DISABLED)
        self.cache_button.config(state=tk.DISABLED)

        self.take_counter += 1
        self.update_display()

        if is_final:
            self.next_line()

    def rewrite_index_for_line(self, index_csv: Path, wav_filename: str, take: PendingTake) -> None:
        """
        覆盖重录时，删除同一个 line_text + take 的旧记录，再写入新记录。
        """
        index_csv.parent.mkdir(parents=True, exist_ok=True)

        rows: list[dict] = []

        if index_csv.exists():
            with index_csv.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    same_line = row.get("line_text") == take.line.text
                    same_take = str(row.get("take")) == str(take.take)
                    if not (same_line and same_take):
                        rows.append(row)

        fieldnames = [
            "wav",
            "alias",
            "rough_start_ms",
            "rough_end_ms",
            "line_text",
            "alias_index",
            "take",
            "avg_pitch_hz",
        ]

        marks = take.marks_ms

        for i, alias in enumerate(take.line.aliases):
            if i >= len(marks):
                break

            start_ms = marks[i]

            if i + 1 < len(marks):
                end_ms = marks[i + 1]
            else:
                end_ms = take.stop_ms

            rows.append({
                "wav": wav_filename,
                "alias": alias,
                "rough_start_ms": f"{start_ms:.3f}",
                "rough_end_ms": f"{end_ms:.3f}",
                "line_text": take.line.text,
                "alias_index": i,
                "take": take.take,
                "avg_pitch_hz": "" if take.avg_pitch_hz is None else f"{take.avg_pitch_hz:.3f}",
            })

        with index_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def redo_current(self) -> None:
        if self.recording:
            self.stop_stream()

        self.pending_take = None
        self.current_alias_index = 0
        self.space_start_marks_ms = []

        with self.audio_lock:
            self.audio_chunks = []

        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.DISABLED)
        self.cache_button.config(state=tk.DISABLED)

        self.current_pitch_label.config(text="Current take avg pitch: N/A")
        self.status_label.config(text="Redo current line. Press Start.", fg="#555555")

        self.update_display()

    def next_line(self) -> None:
        if self.recording:
            return

        if self.line_index + 1 >= len(self.lines):
            self.status_label.config(text="All lines finished.", fg="#007700")
            messagebox.showinfo("Done", "录音表已经录完。")
            return

        self.line_index += 1
        self.current_alias_index = 0
        self.pending_take = None

        self.save_button.config(state=tk.DISABLED)
        self.cache_button.config(state=tk.DISABLED)
        self.current_pitch_label.config(text="Current take avg pitch: N/A")
        self.status_label.config(text="Next line selected. Press Start.", fg="#555555")

        self.update_display()

    def play_selected_note(self) -> None:
        note = self.note_var.get().strip()
        try:
            freq = note_to_freq(note)
        except ValueError as e:
            messagebox.showerror("Invalid note", str(e))
            return

        play_tone(freq)

    def play_note_button(self, note: str) -> None:
        self.note_var.set(note)
        play_tone(note_to_freq(note))

    def schedule_waveform_update(self) -> None:
        self.draw_waveform()
        self.root.after(80, self.schedule_waveform_update)

    def draw_waveform(self) -> None:
        canvas = self.wave_canvas
        canvas.delete("all")

        width = canvas.winfo_width()
        height = canvas.winfo_height()

        if width <= 10 or height <= 10:
            return

        canvas.create_line(0, height // 2, width, height // 2, fill="#333333")

        with self.audio_lock:
            if not self.audio_chunks:
                return

            recent_audio = np.concatenate(self.audio_chunks, axis=0)

        if recent_audio.ndim > 1:
            y = recent_audio[:, 0]
        else:
            y = recent_audio

        max_samples = int(SAMPLE_RATE * WAVEFORM_SECONDS)
        if len(y) > max_samples:
            y = y[-max_samples:]

        if len(y) == 0:
            return

        # 下采样到画布宽度
        step = max(1, len(y) // width)
        y_small = y[::step]

        if len(y_small) > width:
            y_small = y_small[:width]

        max_amp = max(0.01, float(np.max(np.abs(y_small))))

        points = []
        for x, sample in enumerate(y_small):
            normalized = float(sample) / max_amp
            y_pos = height / 2 - normalized * (height * 0.42)
            points.append((x, y_pos))

        for i in range(1, len(points)):
            x1, y1 = points[i - 1]
            x2, y2 = points[i]
            canvas.create_line(x1, y1, x2, y2, fill="#66ccff")

        # 画 Space start markers
        if self.recording:
            elapsed_ms = (time.perf_counter() - self.start_time_perf) * 1000.0
        elif self.pending_take is not None:
            elapsed_ms = self.pending_take.stop_ms
        else:
            elapsed_ms = None

        if elapsed_ms and elapsed_ms > 0:
            visible_ms = WAVEFORM_SECONDS * 1000.0
            window_start_ms = max(0.0, elapsed_ms - visible_ms)

            for idx, mark_ms in enumerate(self.space_start_marks_ms):
                if mark_ms < window_start_ms:
                    continue
                x = int((mark_ms - window_start_ms) / visible_ms * width)
                canvas.create_line(x, 0, x, height, fill="#ffcc66")
                canvas.create_text(
                    x + 4,
                    12,
                    text=str(idx + 1),
                    fill="#ffcc66",
                    anchor="nw",
                )

    def on_close(self) -> None:
        if self.recording:
            self.stop_stream()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = CVVCRecorderApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()