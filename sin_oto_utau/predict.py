from pathlib import Path
import joblib
import pandas as pd

from sin_oto_utau.oto_parser import OtoEntry, write_oto
from sin_oto_utau.audio_features import extract_audio_features
from sin_oto_utau.languages.mandarin import MandarinProfile
from sin_oto_utau.rules import fix_entry


FEATURE_COLUMNS = [
    "alias",
    "initial",
    "final",
    "initial_type",
    "final_type",
    "syllable_type",
    "duration_ms",
    "rms",
    "onset_ms",
    "end_ms",
    "sample_rate",
    "spectral_centroid",
    "zero_crossing_rate",
]


def predict_one(model, wav_path: Path, alias: str) -> OtoEntry:
    profile = MandarinProfile()
    alias_info = profile.analyze_alias(alias)
    audio = extract_audio_features(wav_path)

    row = {
        "alias": alias,

        "initial": alias_info.initial or "",
        "final": alias_info.final or "",
        "initial_type": alias_info.initial_type,
        "final_type": alias_info.final_type,
        "syllable_type": alias_info.syllable_type,

        "duration_ms": audio["duration_ms"],
        "rms": audio["rms"],
        "onset_ms": audio["onset_ms"],
        "end_ms": audio["end_ms"],
        "sample_rate": audio["sample_rate"],
        "spectral_centroid": audio["spectral_centroid"],
        "zero_crossing_rate": audio["zero_crossing_rate"],
    }

    x = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    prediction = model.predict(x)[0]

    offset, consonant, cutoff, preutterance, overlap = prediction

    entry = OtoEntry(
        wav=wav_path.name,
        alias=alias,
        offset=float(offset),
        consonant=float(consonant),
        cutoff=float(cutoff),
        preutterance=float(preutterance),
        overlap=float(overlap),
    )

    return fix_entry(entry)


def predict_from_labels_csv(
    model_path: str,
    labels_csv: str,
    output_oto: str,
) -> None:
    model = joblib.load(model_path)
    labels = pd.read_csv(labels_csv)

    if "wav" not in labels.columns or "alias" not in labels.columns:
        raise ValueError("labels.csv 必须包含 wav 和 alias 两列")

    labels_dir = Path(labels_csv).parent
    entries = []

    for _, row in labels.iterrows():
        wav_name = str(row["wav"])
        alias = str(row["alias"])

        wav_path = labels_dir / wav_name

        if not wav_path.exists():
            print(f"跳过，找不到 wav：{wav_path}")
            continue

        try:
            entry = predict_one(model, wav_path, alias)
            entries.append(entry)
            print(f"生成：{entry.wav}={entry.alias}")
        except Exception as e:
            print(f"跳过，预测失败：{wav_path}，原因：{e}")

    write_oto(entries, output_oto)
    print(f"完成，生成 oto：{output_oto}")