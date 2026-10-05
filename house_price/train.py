"""Train, compare and export the house price model.

Usage: python -m house_price.train [--data PATH] [--out DIR] [--sample N]

Protocol: parse rows -> split (fixed seed) -> feature rules on both splits ->
target-based outlier removal on the TRAIN split only -> 5-fold CV on train
(outlier removal re-done inside each fold, validation folds left untouched)
-> refit every candidate on train -> evaluate on the untouched test split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from house_price import __version__, config
from house_price.cleaning import (
    LocationBucketer,
    apply_feature_rules,
    fit_location_vocabulary,
    prepare_raw,
    remove_target_outliers,
)

log = logging.getLogger(__name__)
NUMERIC = ["total_sqft", "bath", "bhk"]


def build_pipeline(estimator, log_sqft: bool = False) -> Pipeline:
    """One-hot location + scaled numerics. ``log_sqft`` makes a log-target model log-log in area,
    which keeps predictions bounded for very large inputs."""
    if log_sqft:
        numeric = [
            ("log_sqft", make_pipeline(FunctionTransformer(np.log), StandardScaler()), ["total_sqft"]),
            ("rooms", StandardScaler(), ["bath", "bhk"]),
        ]
    else:
        numeric = [("numeric", StandardScaler(), NUMERIC)]
    pre = ColumnTransformer(
        [("location", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["location"]), *numeric]
    )
    return Pipeline([("bucket", LocationBucketer()), ("pre", pre), ("model", estimator)])


def build_tree_pipeline(estimator) -> Pipeline:
    """Trees use native categorical splits on an ordinal-encoded location (much faster than one-hot)."""
    pre = ColumnTransformer(
        [
            (
                "location",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1),
                ["location"],
            ),
            ("numeric", "passthrough", NUMERIC),
        ]
    )
    return Pipeline([("bucket", LocationBucketer()), ("pre", pre), ("model", estimator)])


def _log_target(regressor) -> TransformedTargetRegressor:
    return TransformedTargetRegressor(regressor=regressor, func=np.log1p, inverse_func=np.expm1)


def candidate_models() -> dict[str, Pipeline]:
    return {
        "linear_regression": build_pipeline(LinearRegression()),
        "ridge_log_target": build_pipeline(_log_target(Ridge(alpha=1.0)), log_sqft=True),
        "hist_gradient_boosting": build_tree_pipeline(
            _log_target(
                HistGradientBoostingRegressor(
                    categorical_features=[0], max_iter=300, learning_rate=0.05, random_state=config.RANDOM_SEED
                )
            )
        ),
    }


MODEL_DESCRIPTIONS = {
    "linear_regression": "LinearRegression on one-hot location + scaled numerics (baseline)",
    "ridge_log_target": "Ridge(alpha=1) on log1p(price) with log(total_sqft) (log-log model)",
    "hist_gradient_boosting": "HistGradientBoostingRegressor on log1p(price)",
}


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ape = np.abs(y_pred - y_true) / y_true
    return {
        "mae_lakhs": float(mean_absolute_error(y_true, y_pred)),
        "rmse_lakhs": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(ape.mean()),
        "median_ape": float(np.median(ape)),
        "n": int(len(y_true)),
    }


def _fit_filtered(pipe: Pipeline, train: pd.DataFrame) -> Pipeline:
    vocab = fit_location_vocabulary(train["location"])
    filtered = remove_target_outliers(train, vocab)
    return pipe.fit(filtered[config.FEATURES], filtered[config.TARGET])


def cross_validate(name: str, train: pd.DataFrame) -> tuple[dict, np.ndarray]:
    """K-fold CV where outlier removal is fitted per training fold; validation folds stay raw."""
    kf = KFold(config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)
    oof = np.full(len(train), np.nan)
    fold_mae, fold_r2 = [], []
    for tr_idx, va_idx in kf.split(train):
        pipe = _fit_filtered(candidate_models()[name], train.iloc[tr_idx])
        va = train.iloc[va_idx]
        pred = pipe.predict(va[config.FEATURES])
        oof[va_idx] = pred
        fold_mae.append(mean_absolute_error(va[config.TARGET], pred))
        fold_r2.append(r2_score(va[config.TARGET], pred))
    row = {
        "model": name,
        "description": MODEL_DESCRIPTIONS[name],
        "cv_mae_mean": float(np.mean(fold_mae)),
        "cv_mae_std": float(np.std(fold_mae)),
        "cv_r2_mean": float(np.mean(fold_r2)),
        "cv_r2_std": float(np.std(fold_r2)),
    }
    return row, oof


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def train(data_path: Path = config.DATA_PATH, out_dir: Path = config.MODELS_DIR, sample: int | None = None) -> dict:
    data_path, out_dir = Path(data_path), Path(out_dir)
    raw = pd.read_csv(data_path)
    parsed = prepare_raw(raw)
    if sample:
        parsed = parsed.sample(n=min(sample, len(parsed)), random_state=config.RANDOM_SEED).reset_index(drop=True)
    log.info("raw rows=%d parsed rows=%d", len(raw), len(parsed))

    train_df, test_df = train_test_split(parsed, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED)
    train_df = apply_feature_rules(train_df)
    test_df = apply_feature_rules(test_df)

    cv_rows, oof_preds = [], {}
    for name in candidate_models():
        row, oof = cross_validate(name, train_df)
        cv_rows.append(row)
        oof_preds[name] = oof
        log.info("CV %-24s MAE %.2f +/- %.2f  R2 %.3f", name, row["cv_mae_mean"], row["cv_mae_std"], row["cv_r2_mean"])
    best = min(cv_rows, key=lambda r: r["cv_mae_mean"])["model"]

    # Prediction range from out-of-fold actual/predicted ratios (training split only, never the test set).
    ratio = train_df[config.TARGET].to_numpy() / oof_preds[best]
    q_low = float(np.quantile(ratio, config.RANGE_LOW_QUANTILE))
    q_high = float(np.quantile(ratio, config.RANGE_HIGH_QUANTILE))

    vocab = fit_location_vocabulary(train_df["location"])
    train_filtered = remove_target_outliers(train_df, vocab)
    # Comparison test set: the same price-based filters applied to test rows, showing the
    # effect of filtering the whole dataset before splitting.
    test_filtered = remove_target_outliers(test_df, vocab)

    test_results, fitted = {}, {}
    for name, pipe in candidate_models().items():
        pipe.fit(train_filtered[config.FEATURES], train_filtered[config.TARGET])
        fitted[name] = pipe
        test_results[name] = {
            "honest": regression_metrics(test_df[config.TARGET], pipe.predict(test_df[config.FEATURES])),
            "original_protocol": regression_metrics(
                test_filtered[config.TARGET], pipe.predict(test_filtered[config.FEATURES])
            ),
        }
    model = fitted[best]
    test_pred = model.predict(test_df[config.FEATURES])
    actual = test_df[config.TARGET].to_numpy()
    coverage = float(np.mean((actual >= test_pred * q_low) & (actual <= test_pred * q_high)))

    # Display name per known location: the most frequent spelling of each normalized key.
    keyed = train_df.assign(_k=train_df["location"].str.lower())
    display = keyed.groupby("_k")["location"].agg(lambda s: s.value_counts().index[0])
    known = sorted((display[k] for k in model.named_steps["bucket"].vocabulary_ if k in display.index), key=str.lower)

    metadata = {
        "model_name": best,
        "model_description": MODEL_DESCRIPTIONS[best],
        "package_version": __version__,
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "data": {
            "file": data_path.name,
            "sha256": sha256_of(data_path),
            "raw_rows": int(len(raw)),
            "parsed_rows": int(len(parsed)),
            "sample": sample,
            "train_rows_after_feature_rules": int(len(train_df)),
            "train_rows_after_outlier_removal": int(len(train_filtered)),
            "test_rows": int(len(test_df)),
            "test_rows_original_protocol": int(len(test_filtered)),
        },
        "split": {"test_size": config.TEST_SIZE, "random_seed": config.RANDOM_SEED, "cv_folds": config.CV_FOLDS},
        "features": config.FEATURES,
        "target": "price (lakhs INR)",
        "cv_results": cv_rows,
        "test_metrics": test_results[best],
        "test_metrics_all_models": test_results,
        "prediction_range": {
            "method": "out-of-fold actual/predicted ratio quantiles on the training split",
            "low_quantile": config.RANGE_LOW_QUANTILE,
            "high_quantile": config.RANGE_HIGH_QUANTILE,
            "low_ratio": q_low,
            "high_ratio": q_high,
            "test_coverage": coverage,
        },
        "validation_rules": {
            "min_sqft": config.MIN_SQFT,
            "max_sqft": config.MAX_SQFT,
            "min_sqft_per_bhk": config.MIN_SQFT_PER_BHK,
            "min_bhk": config.MIN_BHK,
            "max_bhk": config.MAX_BHK,
            "min_bath": config.MIN_BATH,
            "max_extra_baths": config.MAX_EXTRA_BATHS,
        },
        "known_locations": known,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / config.MODEL_FILENAME)
    (out_dir / config.METADATA_FILENAME).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (out_dir / config.MODEL_CARD_FILENAME).write_text(render_model_card(metadata), encoding="utf-8")
    log.info("best=%s saved to %s", best, out_dir)
    return metadata


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def render_model_card(m: dict) -> str:
    h, o = m["test_metrics"]["honest"], m["test_metrics"]["original_protocol"]
    r, d = m["prediction_range"], m["data"]
    cv = "\n".join(
        f"| {c['model']} | {c['cv_mae_mean']:.2f} ± {c['cv_mae_std']:.2f} "
        f"| {c['cv_r2_mean']:.3f} ± {c['cv_r2_std']:.3f} |"
        for c in m["cv_results"]
    )
    lines = [
        "# Model Card: Bengaluru House Price Estimator",
        "",
        f"_Generated by `python -m house_price.train` on {m['trained_at']}. Do not edit by hand._",
        "",
        "## Model",
        f"- **Selected model:** `{m['model_name']}`: {m['model_description']}",
        "- **Inputs:** location (free text), total_sqft, bath, bhk. **Output:** price in lakhs INR.",
        f"- **Selection rule:** lowest {m['split']['cv_folds']}-fold CV MAE on the training split.",
        f"- scikit-learn {m['sklearn_version']}; data SHA-256 `{d['sha256'][:16]}...`",
        "",
        "## Data",
        f'Kaggle "Bengaluru House price data": {d["raw_rows"]} raw rows, {d["parsed_rows"]} after parsing. '
        f"Train {d['train_rows_after_feature_rules']} rows ({d['train_rows_after_outlier_removal']} after "
        f"price-based outlier removal), test {d['test_rows']} rows.",
        "",
        "## Cross-validation (training split; validation folds not outlier-filtered)",
        "| Model | MAE (lakhs) | R² |",
        "|---|---|---|",
        cv,
        "",
        "## Held-out test performance (selected model)",
        "| Protocol | MAE (lakhs) | RMSE (lakhs) | R² | MAPE | Median APE | n |",
        "|---|---|---|---|---|---|---|",
        f"| Honest (feature rules only) | {h['mae_lakhs']:.2f} | {h['rmse_lakhs']:.2f} | {h['r2']:.3f} | "
        f"{_pct(h['mape'])} | {_pct(h['median_ape'])} | {h['n']} |",
        f"| Test rows also price-filtered (comparison only) | {o['mae_lakhs']:.2f} | "
        f"{o['rmse_lakhs']:.2f} | {o['r2']:.3f} | {_pct(o['mape'])} | {_pct(o['median_ape'])} | {o['n']} |",
        "",
        "The second row is for methodology comparison only. It is optimistic: "
        "the filter uses the price being predicted.",
        "",
        "## Prediction range",
        f"Estimate x [{r['low_ratio']:.3f}, {r['high_ratio']:.3f}] (the {_pct(r['low_quantile'])} and "
        f"{_pct(r['high_quantile'])} quantiles of out-of-fold actual/predicted ratios). "
        f"Observed coverage on the test set: {_pct(r['test_coverage'])}.",
        "",
        "## Limitations",
        "- Listing prices from a single, undated Kaggle snapshot, not transaction prices.",
        "- Only four features; ignores age, floor, amenities, area type and exact geography.",
        f"- Locations with {config.RARE_LOCATION_MAX_COUNT} or fewer training rows and unseen locations share one "
        '"other" category; the API flags unseen locations.',
        "- Not suitable for valuation, lending or any financial decision.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Bengaluru house price model.")
    parser.add_argument("--data", type=Path, default=config.DATA_PATH)
    parser.add_argument("--out", type=Path, default=config.MODELS_DIR)
    parser.add_argument("--sample", type=int, default=None, help="train on a random subsample (smoke tests)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    meta = train(args.data, args.out, args.sample)
    h = meta["test_metrics"]["honest"]
    print(f"selected={meta['model_name']} test MAE={h['mae_lakhs']:.2f} R2={h['r2']:.3f} MAPE={h['mape']:.3f}")


if __name__ == "__main__":
    main()
