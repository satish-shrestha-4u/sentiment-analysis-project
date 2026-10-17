from pathlib import Path

import pytest

from src.predict import load_model, predict_texts


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "sentiment.joblib"
)


@pytest.fixture(scope="module")
def classifier():
    """Load the trained model once for these tests."""
    return load_model(str(MODEL_PATH))


@pytest.mark.parametrize(
    ("text", "expected_label"),
    [
        ("I absolutely loved it", 1),  # positive
        ("That was awful", 0),         # negative
    ],
)
def test_predict_texts(classifier, text, expected_label):
    labels, probabilities = predict_texts(classifier, [text])

    assert labels == [expected_label]
    assert probabilities[0] is not None
    assert 0.0 <= probabilities[0] <= 1.0