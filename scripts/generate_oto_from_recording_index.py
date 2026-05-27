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
    ignore_after_marker_ms: float = 60.0,
    pre_margin_ms: float = 20.0,
    min_sustain_ms: float = 35.0,
) -> float:

    duration_ms = len(y) / sr * 1000.0

    search_start_ms = max(0.0, rough_start_ms + ignore_after_marker_ms)
    search_end_ms = min(duration_ms, rough_end_ms)

    if search_end_ms <= search_start_ms + 50:
        return max(0.0, rough_start_ms)

    start_sample = int(search_start_ms / 1000.0 * sr)
    end_sample = int(search_end_ms / 1000.0 * sr)

    segment = y[start_sample:end_sample]

    if len(segment) == 0:
        return max(0.0, rough_start_ms)

    if segment.ndim > 1:
        segment = segment[:, 0]

    frame_ms = 10.0
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
        return max(0.0, rough_start_ms)

    rms_values = np.array(rms_values)

    noise_floor = float(np.percentile(rms_values, 20))
    peak = float(np.percentile(rms_values, 95))

    if peak <= noise_floor * 1.2:
        return max(0.0, rough_start_ms)

    threshold = noise_floor + (peak - noise_floor) * 0.45

    sustain_frames = max(1, int(min_sustain_ms / hop_ms))

    for idx in range(0, len(rms_values) - sustain_frames + 1):
        window = rms_values[idx:idx + sustain_frames]

        if np.all(window > threshold):
            onset_ms = search_start_ms + idx * hop_ms
            return max(0.0, onset_ms - pre_margin_ms)

    active = np.where(rms_values > threshold)[0]

    if len(active) > 0:
        onset_ms = search_start_ms + int(active[0]) * hop_ms
        return max(0.0, onset_ms - pre_margin_ms)

    return max(0.0, rough_start_ms)

def predict_entry(model, recording_dir: Path, row: pd.Series) -> OtoEntry:
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

    consonant, cutoff, preutterance, overlap = prediction


    offset = estimate_real_onset_ms(
        y=y,
        sr=sr,
        rough_start_ms=rough_start_ms,
        rough_end_ms=rough_end_ms,
        ignore_after_marker_ms=180.0,
        pre_margin_ms=20.0,
        min_sustain_ms=80.0,
    )

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

    model = joblib.load(model_path)
    df = pd.read_csv(index_path)

    required = ["wav", "alias", "rough_start_ms", "rough_end_ms"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"recording_index.csv 缺少列：{col}")

    entries = []

    for _, row in df.iterrows():
        try:
            entry = predict_entry(model, recording_dir, row)
            entries.append(entry)
            print(f"Generated: {entry.wav}={entry.alias}")
        except Exception as e:
            print(f"Skipped row because of error: {e}")

    write_oto(entries, output_path)
    print(f"Saved oto.ini: {output_path}")
    print(f"Total entries: {len(entries)}")


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