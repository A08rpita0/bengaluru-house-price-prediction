import json

import pytest

from house_price import config
from house_price.predictor import HousePricePredictor, ValidationError, validate_inputs


def test_training_writes_artifact_and_metadata(trained_dir):
    assert (trained_dir / config.MODEL_FILENAME).exists()
    assert (trained_dir / config.MODEL_CARD_FILENAME).exists()
    meta = json.loads((trained_dir / config.METADATA_FILENAME).read_text(encoding="utf-8"))
    for key in [
        "model_name",
        "cv_results",
        "test_metrics",
        "prediction_range",
        "known_locations",
        "sklearn_version",
        "trained_at",
        "data",
    ]:
        assert key in meta
    assert set(meta["test_metrics"]) == {"honest", "original_protocol"}
    assert len(meta["cv_results"]) == 3
    assert meta["model_name"] == min(meta["cv_results"], key=lambda r: r["cv_mae_mean"])["model"]
    assert len(meta["data"]["sha256"]) == 64
    # Honest test set keeps every feature-valid row; the comparison set also drops price outliers.
    assert meta["test_metrics"]["honest"]["n"] == meta["data"]["test_rows"]
    assert meta["test_metrics"]["original_protocol"]["n"] <= meta["data"]["test_rows"]
    assert meta["prediction_range"]["low_ratio"] < 1 < meta["prediction_range"]["high_ratio"]


def test_predictor_known_and_unknown_location(model_path):
    p = HousePricePredictor(model_path)
    assert "Whitefield" in p.locations()
    known = p.predict("whitefield", 1200, 2, 2)
    assert known["location_known"] is True
    assert known["estimate_lakhs"] > 0
    assert known["range_lakhs"]["low"] < known["estimate_lakhs"] < known["range_lakhs"]["high"]
    unknown = p.predict("Somewhere New", 1200, 2, 2)
    assert unknown["location_known"] is False


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"location": "", "total_sqft": 1000, "bath": 2, "bhk": 2}, "location"),
        ({"location": "X", "total_sqft": "abc", "bath": 2, "bhk": 2}, "total_sqft"),
        ({"location": "X", "total_sqft": 100, "bath": 2, "bhk": 2}, "total_sqft"),
        ({"location": "X", "total_sqft": 500, "bath": 2, "bhk": 2}, "total_sqft"),  # < 300 sqft per bhk
        ({"location": "X", "total_sqft": 1000, "bath": 2, "bhk": 2.5}, "bhk"),
        ({"location": "X", "total_sqft": 1000, "bath": 2, "bhk": 0}, "bhk"),
        ({"location": "X", "total_sqft": 1000, "bath": 4, "bhk": 2}, "bath"),  # > bhk + 1
        ({"location": "X", "total_sqft": 1000, "bath": None, "bhk": 2}, "bath"),
        ({"location": "X", "total_sqft": True, "bath": 2, "bhk": 2}, "total_sqft"),
    ],
)
def test_validation_errors(kwargs, field):
    with pytest.raises(ValidationError) as exc:
        validate_inputs(**kwargs)
    assert field in exc.value.errors


def test_validation_accepts_numeric_strings():
    assert validate_inputs(" Whitefield ", "1200", "2", "2") == {
        "location": "Whitefield",
        "total_sqft": 1200.0,
        "bath": 2,
        "bhk": 2,
    }
