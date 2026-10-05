from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from house_price import config
from house_price.train import train

COLUMNS = ["area_type", "availability", "location", "size", "society", "total_sqft", "bath", "balcony", "price"]


def synthetic_csv(path: Path, n: int = 400, seed: int = 0) -> Path:
    """Small synthetic dataset in the Kaggle schema, so tests do not need the real CSV."""
    rng = np.random.default_rng(seed)
    locs = {"Whitefield": 6000, "Indiranagar": 12000, "Electronic City": 4000, "Hebbal": 8000}
    rows = []
    for _ in range(n):
        loc = rng.choice(list(locs))
        bhk = int(rng.integers(1, 5))
        sqft = float(rng.integers(450, 650)) * bhk
        bath = int(rng.integers(1, bhk + 2))
        pps = locs[loc] * rng.normal(1.0, 0.1)
        rows.append(
            [
                "Super built-up  Area",
                "Ready To Move",
                loc,
                f"{bhk} BHK",
                "",
                str(sqft),
                bath,
                1,
                round(pps * sqft / 100_000, 2),
            ]
        )
    rows.append(["Plot  Area", "Ready To Move", "Whitefield", "3 Bedroom", "", "2100 - 2850", 3, 1, 150.0])
    rows.append(["Plot  Area", "Ready To Move", "Whitefield", "2 BHK", "", "34.46Sq. Meter", 2, 1, 40.0])
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def trained_dir(tmp_path_factory) -> Path:
    """Train on synthetic data so API tests are independent of the committed artifact."""
    root = tmp_path_factory.mktemp("model")
    csv = synthetic_csv(root / "data.csv")
    train(csv, root / "models")
    return root / "models"


@pytest.fixture(scope="session")
def model_path(trained_dir) -> Path:
    return trained_dir / config.MODEL_FILENAME
