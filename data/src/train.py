import argparse
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def main(data_path: str, output_path: str) -> None:
    df = pd.read_csv(data_path)

    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("CSV must contain 'text' and 'label' columns")

    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )
    model.fit(df["text"].astype(str), df["label"].astype(int))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    dump(model, output)
    print(f"Model saved to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sentiments.csv")
    parser.add_argument("--out", default="models/sentiment.joblib")
    args = parser.parse_args()
    main(args.data, args.out)
