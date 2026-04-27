"""
Train a TF-IDF + Logistic Regression classifier on the 20 Newsgroups dataset
and save it to models/classifier.pkl.

"""

import pickle
from pathlib import Path

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

OUTPUT_DIR = Path(__file__).parent.parent / "models"
OUTPUT_PATH = OUTPUT_DIR / "classifier.pkl"
DATA_HOME = Path(__file__).parent.parent / "data" / "sklearn"


def train() -> None:
    print("Fetching 20 Newsgroups dataset...")
    data = fetch_20newsgroups(
        subset="train",
        remove=("headers", "footers", "quotes"),
        data_home=DATA_HOME,
    )

    print(f"Training on {len(data.data)} samples, {len(data.target_names)} classes...")
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=30_000, sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=1000, C=5.0, solver="lbfgs")),
        ]
    )
    pipeline.fit(data.data, data.target)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump({"pipeline": pipeline, "classes": data.target_names}, f)

    print(f"Model saved to {OUTPUT_PATH}")
    print(f"Classes: {data.target_names}")


if __name__ == "__main__":
    train()
