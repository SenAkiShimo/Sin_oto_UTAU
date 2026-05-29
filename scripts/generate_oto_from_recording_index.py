from pathlib import Path
import argparse
import joblib
import pandas as pd
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sin_oto_utau.oto_parser import OtoEntry, write_oto
from sin_oto_utau.audio_features import (
    load_wav_mono,
    slice_audio_ms,
    extract_audio_features_from_array,
)
from sin_oto_utau.languages.mandarin import MandarinProfile
from sin_oto_utau.rules import fix_entry


FEATURE_COLUMNS = [
    "alias",
    "initial",
    "final",
    "initial_type",
    "final_type",
    "syllable_type",
    "sample_rate",
    "full_duration_ms",
    "entry_window_duration_ms",
    "local_duration_ms",
    "local_rms",
    "local_onset_ms",
    "local_end_ms",
    "local_spectral_centroid",
    "local_zero_crossing_rate",
]

def estimate_real_onset_ms(
    y: np.ndarray,
    sr: int,
    rough_start_ms: float,
    rough_end_ms: float,
    ignore_after_marker_ms: float = 250.0,
    pre_margin_ms: float = 25.0,
    min_sustain_ms: float = 90.0,
) -> float:
    
    duration_ms = len(y) / sr * 1000.0

    search_start_ms = max(0.0, rough_start_ms + ignore_after_marker_ms)
    search_end_ms = min(duration_ms, rough_end_ms)

    if search_end_ms <= search_start_ms + 80:
        return max(0.0, rough_start_ms + ignore_after_marker_ms)

    start_sample = int(search_start_ms / 1000.0 * sr)
    end_sample = int(search_end_ms / 1000.0 * sr)

    segment = y[start_sample:end_sample]

    if len(segment) == 0:
        return max(0.0, rough_start_ms + ignore_after_marker_ms)

    if segment.ndim > 1:
        segment = segment[:, 0]

    segment = segment - np.mean(segment)

    frame_ms = 15.0
    hop_ms = 5.0

    frame_len = max(1, int(sr * frame_ms / 1000.0))
    hop_len = max(1, int(sr * hop_ms / 1000.0))

    rms_values = []

    for i in range(0, max(1, len(segment) - frame_len), hop_len):
        frame = segment[i:i + frame_len]
        if len(frame) == 0:
            continue
        rms = float(np.sqrt(np.mean(frame ** 2)))
        rms_values.append(rms)

    if not rms_values:
        return max(0.0, rough_start_ms + ignore_after_marker_ms)

    rms = np.array(rms_values)

    smooth_width = 5
    if len(rms) >= smooth_width:
        kernel = np.ones(smooth_width) / smooth_width
        rms_smooth = np.convolve(rms, kernel, mode="same")
    else:
        rms_smooth = rms

    noise_floor = float(np.percentile(rms_smooth, 15))
    mid_energy = float(np.percentile(rms_smooth, 70))
    high_energy = float(np.percentile(rms_smooth, 95))
    peak = float(np.max(rms_smooth))

    if peak < 0.003:
        return max(0.0, rough_start_ms + ignore_after_marker_ms)

    if high_energy <= noise_floor * 1.5:
        return max(0.0, rough_start_ms + ignore_after_marker_ms)

    main_threshold = noise_floor + (high_energy - noise_floor) * 0.55

    backtrack_threshold = noise_floor + (high_energy - noise_floor) * 0.22

    main_threshold = max(main_threshold, 0.006)
    backtrack_threshold = max(backtrack_threshold, 0.0035)

    sustain_frames = max(1, int(min_sustain_ms / hop_ms))

    main_idx = None

    for idx in range(0, len(rms_smooth) - sustain_frames + 1):
        window = rms_smooth[idx:idx + sustain_frames]

        if np.mean(window > main_threshold) >= 0.75:
            main_idx = idx
            break

    if main_idx is None:
        peak_idx = int(np.argmax(rms_smooth))
        main_idx = max(0, peak_idx - sustain_frames)

    onset_idx = main_idx

    for idx in range(main_idx, -1, -1):
        if rms_smooth[idx] < backtrack_threshold:
            onset_idx = idx + 1
            break
        onset_idx = idx

    onset_ms = search_start_ms + onset_idx * hop_ms

    return max(0.0, onset_ms - pre_margin_ms)

def predict_entry(model, recording_dir: Path, row: pd.Series, next_row=None) -> OtoEntry:
    wav_name = str(row["wav"])
    alias = str(row["alias"])

    rough_start_ms = float(row["rough_start_ms"])
    rough_end_ms = float(row["rough_end_ms"])

    wav_path = recording_dir / wav_name

    y, sr = load_wav_mono(wav_path)
    full_duration_ms = len(y) / sr * 1000.0

    window_start_ms = max(0.0, rough_start_ms)
    window_end_ms = min(full_duration_ms, rough_end_ms + 180.0)

    local_y = slice_audio_ms(y, sr, window_start_ms, window_end_ms)
    audio = extract_audio_features_from_array(local_y, sr)

    profile = MandarinProfile()
    alias_info = profile.analyze_alias(alias)

    feature_row = {
        "alias": alias,
        "initial": alias_info.initial or "",
        "final": alias_info.final or "",
        "initial_type": alias_info.initial_type,
        "final_type": alias_info.final_type,
        "syllable_type": alias_info.syllable_type,

        "sample_rate": sr,
        "full_duration_ms": full_duration_ms,
        "entry_window_duration_ms": window_end_ms - window_start_ms,
        "local_duration_ms": audio["duration_ms"],
        "local_rms": audio["rms"],
        "local_onset_ms": audio["onset_ms"],
        "local_end_ms": audio["end_ms"],
        "local_spectral_centroid": audio["spectral_centroid"],
        "local_zero_crossing_rate": audio["zero_crossing_rate"],
    }

    x = pd.DataFrame([feature_row], columns=FEATURE_COLUMNS)

    prediction = model.predict(x)[0]

    consonant, _predicted_cutoff, preutterance, overlap = prediction

    offset = estimate_real_onset_ms(
        y=y,
        sr=sr,
        rough_start_ms=rough_start_ms,
        rough_end_ms=rough_end_ms,
        ignore_after_marker_ms=250.0,
        pre_margin_ms=25.0,
        min_sustain_ms=90.0,
    )

    if next_row is not None and str(next_row["wav"]) == wav_name:
        next_start_ms = float(next_row["rough_start_ms"])

        safe_min_end_ms = offset + max(float(consonant), float(preutterance), float(overlap)) + 80.0
        cutoff_end_ms = max(safe_min_end_ms, next_start_ms - 40.0)
    else:

        cutoff_end_ms = min(full_duration_ms, rough_end_ms + 150.0)

    cutoff_end_ms = max(0.0, min(full_duration_ms, cutoff_end_ms))

    cutoff = cutoff_end_ms - full_duration_ms

    entry = OtoEntry(
        wav=wav_name,
        alias=alias,
        offset=float(offset),
        consonant=float(consonant),
        cutoff=float(cutoff),
        preutterance=float(preutterance),
        overlap=float(overlap),
    )

    return fix_entry(entry)


def generate(recording_dir: str, model_path: str, output_path: str) -> None:
    recording_dir = Path(recording_dir)
    model_path = Path(model_path)
    output_path = Path(output_path)

    index_path = recording_dir / "recording_index.csv"

    if not index_path.exists():
        raise FileNotFoundError(f"Missing recording_index.csv: {index_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")

    model = joblib.load(model_path)
    df = pd.read_csv(index_path)

    required_columns = [
        "wav",
        "alias",
        "rough_start_ms",
        "rough_end_ms",
        "alias_index",
    ]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"recording_index.csv 缺少列：{col}")

    df["rough_start_ms"] = pd.to_numeric(df["rough_start_ms"], errors="coerce")
    df["rough_end_ms"] = pd.to_numeric(df["rough_end_ms"], errors="coerce")
    df["alias_index"] = pd.to_numeric(df["alias_index"], errors="coerce")

    df = df.dropna(subset=["wav", "alias", "rough_start_ms", "rough_end_ms", "alias_index"])

    if df.empty:
        raise ValueError("recording_index.csv 没有可用数据。")

    df = df.sort_values(["wav", "alias_index"]).reset_index(drop=True)

    entries = []
    skipped = 0

    for wav_name, group in df.groupby("wav", sort=False):
        group = group.sort_values("alias_index").reset_index(drop=True)

        wav_path = recording_dir / str(wav_name)

        if not wav_path.exists():
            print(f"Skipped wav group because wav file is missing: {wav_path}")
            skipped += len(group)
            continue

        print(f"Processing wav: {wav_name} ({len(group)} entries)")

        for i, row in group.iterrows():
            if i + 1 < len(group):
                next_row = group.iloc[i + 1]
            else:
                next_row = None

            try:
                entry = predict_entry(
                    model=model,
                    recording_dir=recording_dir,
                    row=row,
                    next_row=next_row,
                )

                entries.append(entry)

                print(
                    f"Generated: {entry.wav}={entry.alias},"
                    f"{entry.offset:.3f},"
                    f"{entry.consonant:.3f},"
                    f"{entry.cutoff:.3f},"
                    f"{entry.preutterance:.3f},"
                    f"{entry.overlap:.3f}"
                )

            except Exception as e:
                skipped += 1
                print(
                    f"Skipped row wav={row.get('wav')} alias={row.get('alias')} "
                    f"because of error: {e}"
                )

    if not entries:
        raise RuntimeError("没有成功生成任何 oto entry。请检查 wav、recording_index.csv 和模型。")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_oto(entries, output_path)

    print(f"Saved oto.ini: {output_path}")
    print(f"Total generated entries: {len(entries)}")
    print(f"Skipped entries: {skipped}")




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recording-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    generate(
        recording_dir=args.recording_dir,
        model_path=args.model,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()