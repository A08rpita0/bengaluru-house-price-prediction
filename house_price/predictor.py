"""Inference wrapper: loads the trained pipeline once and validates requests."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from house_price import config
from house_price.cleaning import normalize_location


class ValidationError(ValueError):
    """Raised with a ``{field: message}`` mapping when inputs are invalid."""

    def __init__(self, errors: dict[str, str]):
        super().__init__("; ".join(f"{k}: {v}" for k, v in errors.items()))
        self.errors = errors


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    return None


def validate_inputs(location: Any, total_sqft: Any, bath: Any, bhk: Any) -> dict[str, Any]:
    """Return cleaned inputs or raise ValidationError. Mirrors the training feature rules."""
    errors: dict[str, str] = {}

    loc = location.strip() if isinstance(location, str) else ""
    if not loc:
        errors["location"] = "Location is required."
    elif len(loc) > 100:
        errors["location"] = "Location must be at most 100 characters."

    sqft = _as_number(total_sqft)
    if sqft is None:
        errors["total_sqft"] = "Area must be a number."
    elif not config.MIN_SQFT <= sqft <= config.MAX_SQFT:
        errors["total_sqft"] = f"Area must be between {config.MIN_SQFT} and {config.MAX_SQFT} sq ft."

    bhk_n = _as_number(bhk)
    if bhk_n is None or not bhk_n.is_integer():
        errors["bhk"] = "BHK must be a whole number."
    elif not config.MIN_BHK <= bhk_n <= config.MAX_BHK:
        errors["bhk"] = f"BHK must be between {config.MIN_BHK} and {config.MAX_BHK}."

    bath_n = _as_number(bath)
    if bath_n is None or not bath_n.is_integer():
        errors["bath"] = "Bathrooms must be a whole number."
    elif bath_n < config.MIN_BATH:
        errors["bath"] = f"Bathrooms must be at least {config.MIN_BATH}."

    if not errors:
        if bath_n > bhk_n + config.MAX_EXTRA_BATHS:
            errors["bath"] = f"Bathrooms can be at most BHK + {config.MAX_EXTRA_BATHS} ({int(bhk_n) + 1})."
        if sqft / bhk_n < config.MIN_SQFT_PER_BHK:
            errors["total_sqft"] = (
                f"Area must be at least {config.MIN_SQFT_PER_BHK} sq ft per bedroom "
                f"({int(config.MIN_SQFT_PER_BHK * bhk_n)} sq ft for {int(bhk_n)} BHK)."
            )
    if errors:
        raise ValidationError(errors)
    return {"location": loc, "total_sqft": sqft, "bath": int(bath_n), "bhk": int(bhk_n)}


class HousePricePredictor:
    def __init__(self, model_path: Path | str = config.MODELS_DIR / config.MODEL_FILENAME):
        model_path = Path(model_path)
        self.model = joblib.load(model_path)
        self.metadata: dict[str, Any] = json.loads(
            (model_path.parent / config.METADATA_FILENAME).read_text(encoding="utf-8")
        )
        self._known = {normalize_location(name) for name in self.metadata["known_locations"]}

    @property
    def model_name(self) -> str:
        return self.metadata["model_name"]

    def locations(self) -> list[str]:
        return list(self.metadata["known_locations"])

    def is_known_location(self, location: str) -> bool:
        return normalize_location(location) in self._known

    def predict(self, location: Any, total_sqft: Any, bath: Any, bhk: Any) -> dict[str, Any]:
        clean = validate_inputs(location, total_sqft, bath, bhk)
        frame = pd.DataFrame([clean], columns=config.FEATURES)
        estimate = float(self.model.predict(frame)[0])
        rng = self.metadata["prediction_range"]
        return {
            "estimate_lakhs": round(estimate, 2),
            "range_lakhs": {
                "low": round(estimate * rng["low_ratio"], 2),
                "high": round(estimate * rng["high_ratio"], 2),
                "coverage": round(rng["test_coverage"], 3),  # observed on the held-out test set
            },
            "price_per_sqft_inr": round(estimate * 100_000 / clean["total_sqft"]),
            "location_known": self.is_known_location(clean["location"]),
            "inputs": clean,
        }
