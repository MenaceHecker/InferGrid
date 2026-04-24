import pickle
from pathlib import Path

import numpy as np


class SklearnLoader:
    def __init__(self, model_path: str) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        with open(path, "rb") as f:
            payload = pickle.load(f)

        self.pipeline = payload["pipeline"]
        self.classes: list[str] = list(payload["classes"])

    def predict(self, text: str) -> tuple[str, float]:
        """
        Returns (predicted_class_label, confidence_score).
        """
        proba = self.pipeline.predict_proba([text])[0]
        class_idx = int(np.argmax(proba))
        return self.classes[class_idx], float(proba[class_idx])