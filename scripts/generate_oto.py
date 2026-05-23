from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from sin_oto_utau.predict import predict_from_labels_csv


if __name__ == "__main__":
    predict_from_labels_csv(
        model_path="models/oto_model.joblib",
        labels_csv="data/new_recordings/labels.csv",
        output_oto="data/output/oto.ini",
    )