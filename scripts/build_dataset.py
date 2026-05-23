from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from sin_oto_utau.oto_parser import read_oto
from sin_oto_utau.audio_features import extract_entry_audio_features
from sin_oto_utau.languages.mandarin import MandarinProfile


def build_dataset(raw_banks_dir: str, output_csv: str) -> None:
    raw_banks_path = Path(raw_banks_dir)
    profile = MandarinProfile()

    rows = []
    oto_files = list(raw_banks_path.rglob("oto.ini"))

    if not oto_files:
        print(f"没有找到 oto.ini：{raw_banks_dir}")
        return

    total_entries = 0
    skipped_missing_wav = 0
    skipped_audio_error = 0

    for oto_path in oto_files:
        bank_dir = oto_path.parent
        bank_name = bank_dir.name

        print(f"读取：{oto_path}")

        entries = read_oto(oto_path)

        for entry in entries:
            total_entries += 1

            wav_path = bank_dir / entry.wav

            if not wav_path.exists():
                skipped_missing_wav += 1
                print(f"跳过，找不到 wav：{wav_path}")
                continue

            try:
                audio = extract_entry_audio_features(
                    path=wav_path,
                    offset_ms=entry.offset,
                    consonant_ms=entry.consonant,
                    preutterance_ms=entry.preutterance,
                )
            except Exception as e:
                skipped_audio_error += 1
                print(f"跳过，读取局部音频失败：{wav_path}，原因：{e}")
                continue

            alias_info = profile.analyze_alias(entry.alias)

            rows.append({
                "bank": bank_name,
                "oto_path": str(oto_path),
                "wav": entry.wav,
                "alias": entry.alias,
                "wav_path": str(wav_path),

                "language": alias_info.language,
                "initial": alias_info.initial or "",
                "final": alias_info.final or "",
                "initial_type": alias_info.initial_type,
                "final_type": alias_info.final_type,
                "syllable_type": alias_info.syllable_type,

                # 局部音频窗口信息
                "sample_rate": audio["sample_rate"],
                "full_duration_ms": audio["full_duration_ms"],
                "entry_window_start_ms": audio["entry_window_start_ms"],
                "entry_window_end_ms": audio["entry_window_end_ms"],
                "entry_window_duration_ms": audio["entry_window_duration_ms"],

                # 局部音频特征
                "local_duration_ms": audio["local_duration_ms"],
                "local_rms": audio["local_rms"],
                "local_onset_ms": audio["local_onset_ms"],
                "absolute_onset_ms": audio["absolute_onset_ms"],
                "local_end_ms": audio["local_end_ms"],
                "absolute_end_ms": audio["absolute_end_ms"],
                "local_spectral_centroid": audio["local_spectral_centroid"],
                "local_zero_crossing_rate": audio["local_zero_crossing_rate"],
                "offset_to_detected_onset_ms": audio["offset_to_detected_onset_ms"],

                # 训练目标：原 oto 参数
                "offset": entry.offset,
                "consonant": entry.consonant,
                "cutoff": entry.cutoff,
                "preutterance": entry.preutterance,
                "overlap": entry.overlap,
            })

    if not rows:
        print("没有生成任何数据。请检查 oto.ini 和 wav 是否对应。")
        return

    df = pd.DataFrame(rows)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("完成。")
    print(f"找到 oto.ini 数量：{len(oto_files)}")
    print(f"oto entry 总数：{total_entries}")
    print(f"成功生成训练数据：{len(df)}")
    print(f"找不到 wav 跳过：{skipped_missing_wav}")
    print(f"音频读取失败跳过：{skipped_audio_error}")
    print(f"保存到：{output_path}")


if __name__ == "__main__":
    build_dataset("data/raw_banks", "data/dataset.csv")