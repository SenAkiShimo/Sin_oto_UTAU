from pathlib import Path
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

import numpy as np
import soundfile as sf
import sounddevice as sd

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sin_oto_utau.oto_parser import read_oto


class OtoDebugGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("SinoOto OTO Debug Viewer")
        self.root.geometry("1280x760")

        self.voicebank_dir: Path | None = None
        self.oto_entries = []
        self.entries_by_wav: dict[str, list] = {}

        self.current_wav_name: str | None = None
        self.current_entry_index: int = 0

        self.current_audio: np.ndarray | None = None
        self.current_sr: int | None = None

        self.zoom_ms = tk.DoubleVar(value=1600.0)
        self.show_all_entries = tk.BooleanVar(value=True)

        self.build_ui()

    def build_ui(self):
        main = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(main, padding=8)
        center = ttk.Frame(main, padding=8)

        main.add(left, weight=1)
        main.add(center, weight=4)

        # left panel
        ttk.Button(left, text="Choose Voicebank Folder", command=self.choose_folder).pack(fill=tk.X)

        self.folder_label = ttk.Label(left, text="No folder selected", wraplength=260)
        self.folder_label.pack(fill=tk.X, pady=(8, 12))

        ttk.Label(left, text="WAV files").pack(anchor="w")
        self.wav_list = tk.Listbox(left, height=16, exportselection=False)
        self.wav_list.pack(fill=tk.BOTH, expand=True)
        self.wav_list.bind("<<ListboxSelect>>", self.on_select_wav)

        ttk.Label(left, text="OTO entries in selected WAV").pack(anchor="w", pady=(10, 0))
        self.entry_list = tk.Listbox(left, height=18, exportselection=False)
        self.entry_list.pack(fill=tk.BOTH, expand=True)
        self.entry_list.bind("<<ListboxSelect>>", self.on_select_entry)

        nav = ttk.Frame(left)
        nav.pack(fill=tk.X, pady=8)

        ttk.Button(nav, text="Prev", command=self.prev_entry).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(nav, text="Next", command=self.next_entry).pack(side=tk.LEFT, expand=True, fill=tk.X)

        ttk.Button(left, text="Play WAV", command=self.play_current_wav).pack(fill=tk.X, pady=(4, 0))
        ttk.Button(left, text="Play Visible Range", command=self.play_visible_range).pack(fill=tk.X, pady=(4, 0))

        # center panel
        top_controls = ttk.Frame(center)
        top_controls.pack(fill=tk.X)

        ttk.Label(top_controls, text="Zoom around selected entry (ms):").pack(side=tk.LEFT)

        zoom_scale = ttk.Scale(
            top_controls,
            from_=300,
            to=6000,
            variable=self.zoom_ms,
            orient=tk.HORIZONTAL,
            command=lambda _: self.redraw()
        )
        zoom_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)

        self.zoom_label = ttk.Label(top_controls, text="1600 ms")
        self.zoom_label.pack(side=tk.LEFT)

        ttk.Checkbutton(
            top_controls,
            text="Show all entries in this wav",
            variable=self.show_all_entries,
            command=self.redraw
        ).pack(side=tk.LEFT, padx=12)

        self.info_label = ttk.Label(center, text="Choose a voicebank folder to start.")
        self.info_label.pack(fill=tk.X, pady=6)

        self.figure = Figure(figsize=(9, 5), dpi=100)
        self.ax = self.figure.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.figure, master=center)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        toolbar_frame = ttk.Frame(center)
        toolbar_frame.pack(fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

        self.root.bind("<Left>", lambda e: self.prev_entry())
        self.root.bind("<Right>", lambda e: self.next_entry())
        self.root.bind("<space>", lambda e: self.play_visible_range())

    def choose_folder(self):
        folder = filedialog.askdirectory(title="Choose voicebank folder containing oto.ini and wav files")
        if not folder:
            return

        self.load_voicebank(Path(folder))

    def load_voicebank(self, folder: Path):
        oto_path = folder / "oto.ini"

        if not oto_path.exists():
            messagebox.showerror("Missing oto.ini", f"找不到 oto.ini:\n{oto_path}")
            return

        try:
            entries = read_oto(oto_path)
        except Exception as e:
            messagebox.showerror("Read oto failed", str(e))
            return

        if not entries:
            messagebox.showerror("Empty oto.ini", "oto.ini 没有读到任何 entry。")
            return

        self.voicebank_dir = folder
        self.oto_entries = entries
        self.entries_by_wav.clear()

        for e in entries:
            self.entries_by_wav.setdefault(e.wav, []).append(e)

        self.folder_label.config(text=str(folder))

        self.wav_list.delete(0, tk.END)
        for wav_name in sorted(self.entries_by_wav.keys()):
            wav_path = folder / wav_name
            label = wav_name
            if not wav_path.exists():
                label += "  [missing wav]"
            self.wav_list.insert(tk.END, label)

        self.entry_list.delete(0, tk.END)
        self.ax.clear()
        self.canvas.draw()

        if self.wav_list.size() > 0:
            self.wav_list.selection_set(0)
            self.on_select_wav()

    def on_select_wav(self, event=None):
        if self.voicebank_dir is None:
            return

        selection = self.wav_list.curselection()
        if not selection:
            return

        raw = self.wav_list.get(selection[0])
        wav_name = raw.replace("  [missing wav]", "")
        self.current_wav_name = wav_name
        self.current_entry_index = 0

        wav_path = self.voicebank_dir / wav_name

        if not wav_path.exists():
            messagebox.showerror("Missing wav", f"找不到 wav:\n{wav_path}")
            return

        try:
            self.current_audio, self.current_sr = self.load_wav(wav_path)
        except Exception as e:
            messagebox.showerror("Read wav failed", str(e))
            return

        self.entry_list.delete(0, tk.END)
        entries = self.entries_by_wav.get(wav_name, [])

        for i, e in enumerate(entries):
            text = (
                f"{i}: {e.alias} | "
                f"off={e.offset:.1f}, con={e.consonant:.1f}, "
                f"cut={e.cutoff:.1f}, pre={e.preutterance:.1f}, ovl={e.overlap:.1f}"
            )
            self.entry_list.insert(tk.END, text)

        if entries:
            self.entry_list.selection_set(0)

        self.redraw()

    def on_select_entry(self, event=None):
        selection = self.entry_list.curselection()
        if not selection:
            return

        self.current_entry_index = int(selection[0])
        self.redraw()

    def load_wav(self, path: Path):
        y, sr = sf.read(path, dtype="float32")

        if y.ndim > 1:
            y = y[:, 0]

        return y, sr

    def get_current_entries(self):
        if self.current_wav_name is None:
            return []
        return self.entries_by_wav.get(self.current_wav_name, [])

    def get_current_entry(self):
        entries = self.get_current_entries()
        if not entries:
            return None

        index = max(0, min(self.current_entry_index, len(entries) - 1))
        return entries[index]

    def redraw(self):
        self.zoom_label.config(text=f"{self.zoom_ms.get():.0f} ms")

        self.ax.clear()

        if self.current_audio is None or self.current_sr is None:
            self.canvas.draw()
            return

        entry = self.get_current_entry()
        if entry is None:
            self.canvas.draw()
            return

        y = self.current_audio
        sr = self.current_sr
        duration_ms = len(y) / sr * 1000.0

        offset = float(entry.offset)
        pre_ms = offset + float(entry.preutterance)

        zoom = float(self.zoom_ms.get())
        view_start_ms = max(0.0, pre_ms - zoom / 2.0)
        view_end_ms = min(duration_ms, pre_ms + zoom / 2.0)

        if view_end_ms <= view_start_ms:
            view_start_ms = 0
            view_end_ms = duration_ms

        start_sample = self.ms_to_sample(view_start_ms, sr)
        end_sample = self.ms_to_sample(view_end_ms, sr)

        y_view = y[start_sample:end_sample]

        if len(y_view) == 0:
            self.canvas.draw()
            return

        peak = np.max(np.abs(y_view))
        if peak > 0:
            y_view = y_view / peak

        x_ms = np.linspace(view_start_ms, view_end_ms, len(y_view))

        self.ax.plot(x_ms, y_view, linewidth=0.7)
        self.ax.axhline(0, linewidth=0.6)

        if self.show_all_entries.get():
            for other in self.get_current_entries():
                self.draw_entry_markers(other, alpha=0.35, label_prefix="")
        else:
            self.draw_entry_markers(entry, alpha=1.0, label_prefix="")

        self.draw_entry_markers(entry, alpha=1.0, label_prefix="selected ")

        self.ax.set_xlim(view_start_ms, view_end_ms)
        self.ax.set_ylim(-1.1, 1.1)

        self.ax.set_title(
            f"{self.current_wav_name} | alias={entry.alias} | "
            f"entry {self.current_entry_index + 1}/{len(self.get_current_entries())}"
        )
        self.ax.set_xlabel("Time (ms)")
        self.ax.set_ylabel("Amplitude")

        handles, labels = self.ax.get_legend_handles_labels()
        unique = {}
        for h, l in zip(handles, labels):
            if l not in unique:
                unique[l] = h

        if unique:
            self.ax.legend(unique.values(), unique.keys(), loc="upper right", fontsize=8)

        self.figure.tight_layout()
        self.canvas.draw()

        self.update_info_label(entry, duration_ms)

    def draw_entry_markers(self, entry, alpha: float = 1.0, label_prefix: str = ""):
        duration_ms = len(self.current_audio) / self.current_sr * 1000.0

        offset = float(entry.offset)
        consonant_end = offset + float(entry.consonant)
        pre = offset + float(entry.preutterance)
        overlap = offset + float(entry.overlap)
        end = self.oto_cutoff_to_end_ms(offset, float(entry.cutoff), duration_ms)

        end = max(0.0, min(duration_ms, end))

        self.ax.axvline(offset, linestyle="-", linewidth=1.8, alpha=alpha, label=label_prefix + "offset")
        self.ax.axvline(overlap, linestyle="--", linewidth=1.4, alpha=alpha, label=label_prefix + "overlap")
        self.ax.axvline(pre, linestyle="--", linewidth=1.4, alpha=alpha, label=label_prefix + "preutterance")
        self.ax.axvline(consonant_end, linestyle=":", linewidth=1.6, alpha=alpha, label=label_prefix + "consonant end")
        self.ax.axvline(end, linestyle="-.", linewidth=1.4, alpha=alpha, label=label_prefix + "cutoff/end")

        self.ax.axvspan(offset, consonant_end, alpha=0.04 * alpha)
        self.ax.axvspan(offset, end, alpha=0.025 * alpha)

        self.ax.text(
            offset,
            0.92,
            entry.alias,
            rotation=90,
            fontsize=8,
            alpha=alpha,
            verticalalignment="top",
        )

    def update_info_label(self, entry, duration_ms: float):
        end_ms = self.oto_cutoff_to_end_ms(entry.offset, entry.cutoff, duration_ms)

        info = (
            f"wav={entry.wav} | alias={entry.alias} | "
            f"offset={entry.offset:.2f} | "
            f"consonant={entry.consonant:.2f} | "
            f"cutoff={entry.cutoff:.2f} -> end={end_ms:.2f}ms | "
            f"preutterance={entry.preutterance:.2f} | "
            f"overlap={entry.overlap:.2f}"
        )
        self.info_label.config(text=info)

    def prev_entry(self):
        entries = self.get_current_entries()
        if not entries:
            return

        self.current_entry_index = max(0, self.current_entry_index - 1)
        self.entry_list.selection_clear(0, tk.END)
        self.entry_list.selection_set(self.current_entry_index)
        self.entry_list.see(self.current_entry_index)
        self.redraw()

    def next_entry(self):
        entries = self.get_current_entries()
        if not entries:
            return

        self.current_entry_index = min(len(entries) - 1, self.current_entry_index + 1)
        self.entry_list.selection_clear(0, tk.END)
        self.entry_list.selection_set(self.current_entry_index)
        self.entry_list.see(self.current_entry_index)
        self.redraw()

    def play_current_wav(self):
        if self.current_audio is None or self.current_sr is None:
            return

        sd.stop()
        sd.play(self.current_audio, self.current_sr)

    def play_visible_range(self):
        if self.current_audio is None or self.current_sr is None:
            return

        entry = self.get_current_entry()
        if entry is None:
            return

        y = self.current_audio
        sr = self.current_sr
        duration_ms = len(y) / sr * 1000.0

        pre_ms = float(entry.offset) + float(entry.preutterance)
        zoom = float(self.zoom_ms.get())

        view_start_ms = max(0.0, pre_ms - zoom / 2.0)
        view_end_ms = min(duration_ms, pre_ms + zoom / 2.0)

        start = self.ms_to_sample(view_start_ms, sr)
        end = self.ms_to_sample(view_end_ms, sr)

        if end <= start:
            return

        sd.stop()
        sd.play(y[start:end], sr)

    @staticmethod
    def ms_to_sample(ms: float, sr: int) -> int:
        return int(ms / 1000.0 * sr)

    @staticmethod
    def oto_cutoff_to_end_ms(offset: float, cutoff: float, wav_duration_ms: float) -> float:
        if cutoff < 0:
            return wav_duration_ms + cutoff
        return offset + cutoff


def main():
    root = tk.Tk()
    app = OtoDebugGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()