from pathlib import Path
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMNS = [
    "consonant",
    "cutoff",
    "preutterance",
    "overlap",
]


TEXT_COLUMN = "alias"

CATEGORICAL_COLUMNS = [
    "initial",
    "final",
    "initial_type",
    "final_type",
    "syllable_type",
]


NUMERIC_COLUMNS = [
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


def train_model(dataset_csv: str, model_path: str) -> None:
    df = pd.read_csv(dataset_csv)

    needed = [TEXT_COLUMN] + CATEGORICAL_COLUMNS + NUMERIC_COLUMNS + TARGET_COLUMNS
    df = df.dropna(subset=needed)

    if len(df) < 10:
        print("数据太少了，至少建议有 10 条以上 oto 数据。")
        return

    x = df[[TEXT_COLUMN] + CATEGORICAL_COLUMNS + NUMERIC_COLUMNS]
    y = df[TARGET_COLUMNS].astype(float)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.15,
        random_state=42,
    )

    preprocess = ColumnTransformer(
        transformers=[
            (
                "alias_text",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(1, 4),
                ),
                TEXT_COLUMN,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_COLUMNS,
            ),
            (
                "numeric",
                StandardScaler(),
                NUMERIC_COLUMNS,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "regressor",
                MultiOutputRegressor(
                    RandomForestRegressor(
                        n_estimators=300,
                        random_state=42,
                        n_jobs=-1,
                        min_samples_leaf=2,
                    )
                ),
            ),
        ]
    )

    model.fit(x_train, y_train)

    pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, pred, multioutput="raw_values")

    print("验证集平均绝对误差：")
    for name, err in zip(TARGET_COLUMNS, mae):
        print(f"{name}: {err:.3f} ms")

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)
    print(f"模型已保存：{model_path}")


if __name__ == "__main__":
    train_model("data/dataset.csv", "models/oto_model.joblib")