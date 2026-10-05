"""Shared constants for cleaning, training, validation and serving."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "Bengaluru_House_Data.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_FILENAME = "house_price_model.joblib"
METADATA_FILENAME = "metadata.json"
MODEL_CARD_FILENAME = "MODEL_CARD.md"

FEATURES = ["location", "total_sqft", "bath", "bhk"]
TARGET = "price"  # lakhs of INR (1 lakh = 100,000 INR)

RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# A location needs MORE than this many training rows to get its own category.
RARE_LOCATION_MAX_COUNT = 10
OTHER_LOCATION = "other"

# Feature-based plausibility rules (applied to every row and enforced by the API).
MIN_SQFT_PER_BHK = 300
MAX_EXTRA_BATHS = 1  # bath <= bhk + 1
MIN_SQFT = 300
MAX_SQFT = 30_000
MIN_BHK = 1
MAX_BHK = 10
MIN_BATH = 1

# Prediction interval built from out-of-fold actual/predicted ratios.
RANGE_LOW_QUANTILE = 0.10
RANGE_HIGH_QUANTILE = 0.90
