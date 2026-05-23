from pathlib import Path
import numpy as np
import librosa


def load_wav_mono(path: str | Path, sr: int | None = None):
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y, sr


def ms_to_sample(ms: float, sr: int) -> int:
    return int(ms / 1000.0 * sr)


def sample_to_ms(sample: int, sr: int) -> float:
    return sample / sr * 1000.0


def slice_audio_ms(
    y: np.ndarray,
    sr: int,
    start_ms: float,
    end_ms: float,
) -> np.ndarray:
    start_sample = max(0, ms_to_sample(start_ms, sr))
    end_sample = min(len(y), ms_to_sample(end_ms, sr))

    if end_sample <= start_sample:
        return np.array([], dtype=y.dtype)

    return y[start_sample:end_sample]


def get_duration_ms(y: np.ndarray, sr: int) -> float:
    return len(y) / sr * 1000.0


def get_rms(y: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    return float(np.sqrt(np.mean(y ** 2)))


def estimate_onset_ms(y: np.ndarray, sr: int) -> float:
    """
    在传入的音频片段内部估计起音点。
    返回值是：相对于这个片段开头的毫秒数。
    """
    if len(y) == 0:
        return 0.0

    frame_length = max(1, int(sr * 0.02))
    hop_length = max(1, int(sr * 0.005))

    rms = librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length,
    )[0]

    if len(rms) == 0:
        return 0.0

    noise_floor = float(np.percentile(rms, 20))
    peak = float(np.max(rms))

    if peak <= 0:
        return 0.0

    threshold = noise_floor + (peak - noise_floor) * 0.15
    active = np.where(rms > threshold)[0]

    if len(active) == 0:
        return 0.0

    onset_frame = int(active[0])
    onset_sample = onset_frame * hop_length

    return sample_to_ms(onset_sample, sr)


def estimate_end_ms(y: np.ndarray, sr: int) -> float:
    """
    在传入的音频片段内部估计结束点。
    返回值是：相对于这个片段开头的毫秒数。
    """
    if len(y) == 0:
        return 0.0

    frame_length = max(1, int(sr * 0.02))
    hop_length = max(1, int(sr * 0.005))

    rms = librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length,
    )[0]

    if len(rms) == 0:
        return get_duration_ms(y, sr)

    noise_floor = float(np.percentile(rms, 20))
    peak = float(np.max(rms))

    if peak <= 0:
        return get_duration_ms(y, sr)

    threshold = noise_floor + (peak - noise_floor) * 0.10
    active = np.where(rms > threshold)[0]

    if len(active) == 0:
        return get_duration_ms(y, sr)

    last_active_frame = int(active[-1])
    last_active_sample = last_active_frame * hop_length

    return sample_to_ms(last_active_sample, sr)


def spectral_centroid_mean(y: np.ndarray, sr: int) -> float:
    if len(y) == 0:
        return 0.0

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

    if len(centroid) == 0:
        return 0.0

    return float(np.mean(centroid))


def zero_crossing_rate_mean(y: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0

    zcr = librosa.feature.zero_crossing_rate(y)[0]

    if len(zcr) == 0:
        return 0.0

    return float(np.mean(zcr))


def extract_audio_features_from_array(y: np.ndarray, sr: int) -> dict:
    duration_ms = get_duration_ms(y, sr)

    return {
        "duration_ms": duration_ms,
        "rms": get_rms(y),
        "onset_ms": estimate_onset_ms(y, sr),
        "end_ms": estimate_end_ms(y, sr),
        "spectral_centroid": spectral_centroid_mean(y, sr),
        "zero_crossing_rate": zero_crossing_rate_mean(y),
    }


def extract_audio_features(path: str | Path) -> dict:
    """
    对整条 wav 提取特征。
    """
    y, sr = load_wav_mono(path)

    features = extract_audio_features_from_array(y, sr)
    features["sample_rate"] = sr
    features["full_duration_ms"] = get_duration_ms(y, sr)

    return features


def extract_entry_audio_features(
    path: str | Path,
    offset_ms: float,
    consonant_ms: float,
    preutterance_ms: float,
    left_context_ms: float = 200.0,
    right_context_ms: float = 700.0,
) -> dict:
    """
    对一条 oto entry 附近的局部音频提取特征。

    注意：
    - offset_ms 是这条 entry 在原 wav 里的 offset
    - 返回的 local_onset_ms 是相对于局部片段开头的起音
    - absolute_onset_ms 是换算回原 wav 的起音位置
    """
    y, sr = load_wav_mono(path)
    full_duration_ms = get_duration_ms(y, sr)

    start_ms = max(0.0, offset_ms - left_context_ms)

    # 右边至少覆盖 consonant 和 preutterance，再多留一点元音区域
    useful_width = max(
        right_context_ms,
        consonant_ms + 400.0,
        preutterance_ms + 400.0,
    )

    end_ms = min(full_duration_ms, offset_ms + useful_width)

    local_y = slice_audio_ms(y, sr, start_ms, end_ms)

    local_features = extract_audio_features_from_array(local_y, sr)

    local_onset_ms = local_features["onset_ms"]
    absolute_onset_ms = start_ms + local_onset_ms

    return {
        "sample_rate": sr,
        "full_duration_ms": full_duration_ms,

        "entry_window_start_ms": start_ms,
        "entry_window_end_ms": end_ms,
        "entry_window_duration_ms": end_ms - start_ms,

        "local_duration_ms": local_features["duration_ms"],
        "local_rms": local_features["rms"],
        "local_onset_ms": local_onset_ms,
        "absolute_onset_ms": absolute_onset_ms,
        "local_end_ms": local_features["end_ms"],
        "absolute_end_ms": start_ms + local_features["end_ms"],

        "local_spectral_centroid": local_features["spectral_centroid"],
        "local_zero_crossing_rate": local_features["zero_crossing_rate"],

        # 这个很重要：模型可以学习人工 offset 和自动检测起音之间的差距
        "offset_to_detected_onset_ms": offset_ms - absolute_onset_ms,
    }