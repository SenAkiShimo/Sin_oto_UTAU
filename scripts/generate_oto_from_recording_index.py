from pathlib import Path
import argparse
import joblib
import pandas as pd
import sys

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


def predict_entry(model, recording_dir: Path, row: pd.Series) -> OtoEntry:
    wav_name = str(row["wav"])
    alias = str(row["alias"])

    rough_start_ms = float(row["rough_start_ms"])
    rough_end_ms = float(row["rough_end_ms"])

    wav_path = recording_dir / wav_name

    y, sr = load_wav_mono(wav_path)
    full_duration_ms = len(y) / sr * 1000.0

    # 在 rough_start 附近往前留一点，方便检测真实起音
    window_start_ms = max(0.0, rough_start_ms - 120.0)
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

    # offset 不靠 AI 猜，而是 rough_start + 局部 onset
    offset = window_start_ms + audio["onset_ms"] - 30.0
    offset = max(0.0, offset)

    # cutoff 先用 AI，后面再加规则优化
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