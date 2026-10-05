import math

import pandas as pd
import pytest

from house_price.cleaning import (
    LocationBucketer,
    apply_feature_rules,
    bucket_locations,
    fit_location_vocabulary,
    parse_bhk,
    parse_sqft,
    prepare_raw,
    remove_target_outliers,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("1056", 1056.0), ("2100 - 2850", 2475.0), (" 1000-1200 ", 1100.0), (1200, 1200.0)],
)
def test_parse_sqft_numbers_and_ranges(raw, expected):
    assert parse_sqft(raw) == expected


@pytest.mark.parametrize("raw", ["34.46Sq. Meter", "4125Perch", "1Grounds", "", None])
def test_parse_sqft_units_are_nan(raw):
    assert math.isnan(parse_sqft(raw))


@pytest.mark.parametrize(("raw", "expected"), [("4 Bedroom", 4), ("2 BHK", 2), ("1 RK", 1), ("10 BHK", 10)])
def test_parse_bhk(raw, expected):
    assert parse_bhk(raw) == expected


def test_parse_bhk_missing():
    assert math.isnan(parse_bhk(None))
    assert math.isnan(parse_bhk("Studio"))


def test_prepare_raw_drops_unparseable_rows():
    raw = pd.DataFrame(
        {
            "area_type": ["a"] * 3,
            "availability": ["x"] * 3,
            "location": [" Whitefield ", "Indiranagar", None],
            "size": ["2 BHK", "3 Bedroom", "2 BHK"],
            "society": ["s"] * 3,
            "total_sqft": ["1000 - 1200", "34.46Sq. Meter", "900"],
            "bath": [2, 3, 2],
            "balcony": [1, 1, 1],
            "price": [50, 90, 40],
        }
    )
    out = prepare_raw(raw)
    assert list(out.columns) == ["location", "total_sqft", "bath", "bhk", "price"]
    assert len(out) == 1
    assert out.iloc[0]["location"] == "Whitefield"
    assert out.iloc[0]["total_sqft"] == 1100


def test_feature_rules():
    df = pd.DataFrame(
        {
            "location": ["a"] * 4,
            "total_sqft": [1200, 500, 1200, 1200],
            "bath": [2, 2, 4, 3],
            "bhk": [2, 2, 2, 2],
            "price": [1, 1, 1, 1],
        }
    )
    out = apply_feature_rules(df)
    # row 1: 250 sqft/bhk (<300); row 2: bath 4 >= bhk + 2
    assert out["total_sqft"].tolist() == [1200, 1200]
    assert out["bath"].tolist() == [2, 3]


def test_location_vocabulary_is_fitted_on_train_only():
    train = pd.Series(["Whitefield"] * 11 + ["Rare Place"] * 3)
    test = pd.Series(["whitefield", "Rare Place", "Only In Test"])
    vocab = fit_location_vocabulary(train, max_rare_count=10)
    assert vocab == {"whitefield"}
    assert bucket_locations(test, vocab).tolist() == ["whitefield", "other", "other"]

    bucketer = LocationBucketer(max_rare_count=10).fit(pd.DataFrame({"location": train}))
    out = bucketer.transform(pd.DataFrame({"location": test}))
    assert "only in test" not in bucketer.vocabulary_
    assert out["location"].tolist() == ["whitefield", "other", "other"]


def _pps_frame(prices):
    return pd.DataFrame(
        {
            "location": ["Whitefield"] * len(prices),
            "total_sqft": [1000] * len(prices),
            "bath": 2,
            "bhk": 2,
            "price": prices,
        }
    )


def test_target_outlier_filter_removes_extreme_prices():
    df = _pps_frame([50, 52, 48, 51, 49, 300])
    out = remove_target_outliers(df, {"whitefield"})
    assert 300 not in out["price"].tolist()
    assert len(out) < len(df)


def test_target_filter_is_not_part_of_inference_pipeline():
    """The fitted pipeline must keep every test row: only the bucketer touches rows, and it never drops them."""
    test = _pps_frame([50, 300])
    bucketer = LocationBucketer(max_rare_count=0).fit(test)
    assert len(bucketer.transform(test)) == len(test)
