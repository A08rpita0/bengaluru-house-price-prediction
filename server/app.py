"""Flask API serving the house price model.

Run locally:  python -m server.app
Production:   gunicorn server.wsgi:app
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

from house_price import __version__, config
from house_price.predictor import HousePricePredictor, ValidationError

DEFAULT_ORIGINS = "http://localhost:8000,http://127.0.0.1:8000"
log = logging.getLogger("server")


def _origins(raw: str) -> list[str]:
    return [o.strip().rstrip("/") for o in raw.split(",") if o.strip()]


def create_app(overrides: dict[str, Any] | None = None) -> Flask:
    app = Flask(__name__)
    app.config.update(
        MODEL_PATH=os.environ.get("MODEL_PATH", str(config.MODELS_DIR / config.MODEL_FILENAME)),
        ALLOWED_ORIGINS=_origins(os.environ.get("ALLOWED_ORIGINS", DEFAULT_ORIGINS)),
        JSON_SORT_KEYS=False,
    )
    app.config.update(overrides or {})
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    CORS(app, origins=app.config["ALLOWED_ORIGINS"], methods=["GET", "POST"], allow_headers=["Content-Type"])

    predictor = HousePricePredictor(Path(app.config["MODEL_PATH"]))
    app.extensions["predictor"] = predictor
    log.info("loaded model=%s from %s", predictor.model_name, app.config["MODEL_PATH"])

    @app.get("/health")
    def health():
        return jsonify(status="ok", model_loaded=True, model_name=predictor.model_name, version=__version__)

    @app.get("/api/locations")
    def locations():
        locs = predictor.locations()
        return jsonify(locations=locs, count=len(locs))

    @app.get("/api/model")
    def model_info():
        m = predictor.metadata
        return jsonify(
            model_name=m["model_name"],
            model_description=m["model_description"],
            trained_at=m["trained_at"],
            data=m["data"],
            cv_results=m["cv_results"],
            test_metrics=m["test_metrics"],
            prediction_range=m["prediction_range"],
            validation_rules=m["validation_rules"],
        )

    @app.post("/api/predict")
    def predict():
        payload = request.get_json(silent=True) if request.is_json else request.form.to_dict()
        if not isinstance(payload, dict):
            return jsonify(error="invalid_request", message="Send a JSON object."), 400
        try:
            result = predictor.predict(
                payload.get("location"), payload.get("total_sqft"), payload.get("bath"), payload.get("bhk")
            )
        except ValidationError as exc:
            return jsonify(error="validation_error", message="Invalid input.", fields=exc.errors), 400
        log.info("predict %s -> %.2f lakhs", result["inputs"], result["estimate_lakhs"])
        return jsonify(result)

    @app.errorhandler(HTTPException)
    def http_error(exc: HTTPException):
        return jsonify(error=exc.name.lower().replace(" ", "_"), message=exc.description), exc.code

    @app.errorhandler(Exception)
    def unhandled(exc: Exception):
        log.exception("unhandled error")
        return jsonify(error="internal_error", message="Unexpected server error."), 500

    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")))
