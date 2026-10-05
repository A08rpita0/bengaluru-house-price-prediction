"""Pure data-cleaning functions.

Two kinds of filtering are kept strictly apart:

* Feature-based rules (parsing, sqft-per-BHK, bath count) look only at inputs, so
  they can be applied to any row, including the test set and live requests.
* Target-based outlier filters look at ``price``. They are applied to the
  training split only; applying them to evaluation data would remove exactly
  the hard cases and inflate the metrics.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from house_price import config

_RANGE_RE = re.compile(r"^\s*([\d.]+)\s*-\s*([\d.]+)\s*$")


def parse_sqft(value: object) -> float:
    """Parse ``total_sqft``: plain numbers, ``"a - b"`` ranges (midpoint); unit strings -> NaN."""
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value) if np.isfinite(value) else np.nan
    text = str(value).strip()
    match = _RANGE_RE.match(text)
    if match:
        try:
            return (float(match.group(1)) + float(match.group(2))) / 2
        except ValueError:
            return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan  # e.g. "34.46Sq. Meter", "4125Perch": ambiguous units, dropped


def parse_bhk(size: object) -> float:
    """Parse bedroom count from ``size`` values such as ``"2 BHK"`` or ``"4 Bedroom"``."""
    if size is None or (isinstance(size, float) and np.isnan(size)):
        return np.nan
    match = re.match(r"^\s*(\d+)", str(size))
    return float(match.group(1)) if match else np.nan


def normalize_location(value: object) -> str:
    """Canonical key for a location: trimmed, single-spaced, lower-case."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return " ".join(str(value).split()).lower()


def prepare_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise parsing of the raw Kaggle CSV into ``location, total_sqft, bath, bhk, price``.

    Drops columns the model does not use (area_type, availability, society, balcony)
    and rows with missing or unparseable values. No cross-row statistics are used,
    so this is safe to run before the train/test split.
    """
    out = pd.DataFrame(
        {
            "location": df["location"].map(lambda v: " ".join(str(v).split()) if pd.notna(v) else np.nan),
            "total_sqft": df["total_sqft"].map(parse_sqft),
            "bath": pd.to_numeric(df["bath"], errors="coerce"),
            "bhk": df["size"].map(parse_bhk),
            "price": pd.to_numeric(df["price"], errors="coerce"),
        }
    )
    out = out.replace("", np.nan).dropna().reset_index(drop=True)
    out["bath"] = out["bath"].astype(int)
    out["bhk"] = out["bhk"].astype(int)
    return out


def feature_rule_mask(df: pd.DataFrame) -> pd.Series:
    """True for rows that pass the input-only plausibility rules."""
    return (
        (df["bhk"] >= config.MIN_BHK)
        & (df["bath"] >= config.MIN_BATH)
        & (df["total_sqft"] / df["bhk"] >= config.MIN_SQFT_PER_BHK)
        & (df["bath"] <= df["bhk"] + config.MAX_EXTRA_BATHS)
    )


def apply_feature_rules(df: pd.DataFrame) -> pd.DataFrame:
    return df[feature_rule_mask(df)].reset_index(drop=True)


def fit_location_vocabulary(locations: pd.Series, max_rare_count: int = config.RARE_LOCATION_MAX_COUNT) -> set[str]:
    """Normalized locations with more than ``max_rare_count`` rows. Fit on TRAINING data only."""
    counts = locations.map(normalize_location).value_counts()
    return set(counts[counts > max_rare_count].index)


def bucket_locations(locations: pd.Series, vocabulary: set[str]) -> pd.Series:
    """Map each location to its normalized key, or ``"other"`` if outside the vocabulary."""
    keys = locations.map(normalize_location)
    return keys.where(keys.isin(vocabulary), config.OTHER_LOCATION)


class LocationBucketer(BaseEstimator, TransformerMixin):
    """Pipeline step: learns the location vocabulary in ``fit`` and buckets the rest as ``other``."""

    def __init__(self, max_rare_count: int = config.RARE_LOCATION_MAX_COUNT):
        self.max_rare_count = max_rare_count

    def fit(self, X: pd.DataFrame, y=None):
        self.vocabulary_ = fit_location_vocabulary(X["location"], self.max_rare_count)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["location"] = bucket_locations(X["location"], self.vocabulary_)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(config.FEATURES, dtype=object)


def remove_target_outliers(train: pd.DataFrame, vocabulary: set[str]) -> pd.DataFrame:
    """Price-based outlier removal. TRAINING SPLIT ONLY.

    1. Per location bucket, keep rows whose price-per-sqft lies within mean +/- 1 std.
    2. Within a location, drop n-BHK rows whose price-per-sqft is below the mean
       price-per-sqft of (n-1)-BHK homes there (when that group has > 5 rows).
    """
    df = train.copy()
    df["_loc"] = bucket_locations(df["location"], vocabulary)
    df["_pps"] = df["price"] * 100_000 / df["total_sqft"]

    grouped = df.groupby("_loc")["_pps"]
    mean, std = grouped.transform("mean"), grouped.transform("std").fillna(0)
    df = df[(df["_pps"] > mean - std) & (df["_pps"] <= mean + std)]

    drop_idx: list = []
    for _, loc_df in df.groupby("_loc"):
        stats = loc_df.groupby("bhk")["_pps"].agg(["mean", "count"])
        for bhk, bhk_df in loc_df.groupby("bhk"):
            prev = stats.loc[bhk - 1] if (bhk - 1) in stats.index else None
            if prev is not None and prev["count"] > 5:
                drop_idx.extend(bhk_df.index[bhk_df["_pps"] < prev["mean"]])
    df = df.drop(index=drop_idx)
    return df.drop(columns=["_loc", "_pps"]).reset_index(drop=True)
