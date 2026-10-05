import pytest

from server.app import create_app

ALLOWED = "http://localhost:8000"


@pytest.fixture()
def client(model_path):
    app = create_app({"MODEL_PATH": str(model_path), "ALLOWED_ORIGINS": [ALLOWED], "TESTING": True})
    return app.test_client()


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    body = res.get_json()
    assert body["status"] == "ok" and body["model_loaded"] is True and body["model_name"]


def test_locations(client):
    body = client.get("/api/locations").get_json()
    assert "Whitefield" in body["locations"]
    assert body["count"] == len(body["locations"])


def test_model_info(client):
    body = client.get("/api/model").get_json()
    assert {"cv_results", "test_metrics", "prediction_range"} <= set(body)


def test_predict_json(client):
    res = client.post("/api/predict", json={"location": "Whitefield", "total_sqft": 1200, "bath": 2, "bhk": 2})
    assert res.status_code == 200
    body = res.get_json()
    assert body["estimate_lakhs"] > 0 and body["location_known"] is True
    assert body["price_per_sqft_inr"] > 0


def test_predict_form_encoded(client):
    res = client.post("/api/predict", data={"location": "Whitefield", "total_sqft": "1200", "bath": "2", "bhk": "2"})
    assert res.status_code == 200


def test_predict_missing_fields(client):
    res = client.post("/api/predict", json={"location": "Whitefield"})
    assert res.status_code == 400
    body = res.get_json()
    assert body["error"] == "validation_error"
    assert {"total_sqft", "bath", "bhk"} <= set(body["fields"])


def test_predict_invalid_values(client):
    res = client.post("/api/predict", json={"location": "Whitefield", "total_sqft": 1200, "bath": 5, "bhk": 2})
    assert res.status_code == 400
    assert "bath" in res.get_json()["fields"]


def test_predict_non_object_json(client):
    res = client.post("/api/predict", data="[1,2]", content_type="application/json")
    assert res.status_code == 400


def test_predict_get_not_allowed(client):
    res = client.get("/api/predict")
    assert res.status_code == 405
    assert res.get_json()["error"] == "method_not_allowed"


def test_404_is_json(client):
    res = client.get("/nope")
    assert res.status_code == 404
    assert res.get_json()["error"] == "not_found"


def test_cors_allowed_origin(client):
    res = client.get("/api/locations", headers={"Origin": ALLOWED})
    assert res.headers.get("Access-Control-Allow-Origin") == ALLOWED


def test_cors_disallowed_origin(client):
    res = client.get("/api/locations", headers={"Origin": "https://evil.example"})
    assert "Access-Control-Allow-Origin" not in res.headers


def test_cors_preflight(client):
    res = client.options(
        "/api/predict",
        headers={
            "Origin": ALLOWED,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )
    assert res.headers.get("Access-Control-Allow-Origin") == ALLOWED
