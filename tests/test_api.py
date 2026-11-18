from __future__ import annotations

import os
import tempfile
from pathlib import Path

# FileStore legge la variabile all'import: deve essere impostata prima di app.main.
TEST_DATA_DIR = Path(tempfile.mkdtemp(prefix="sensorlab-api-tests-"))
os.environ["SENSORLAB_DATA_DIR"] = str(TEST_DATA_DIR)

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


def test_health_and_safe_defaults() -> None:
    with TestClient(app) as client:
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

        response = client.get("/api/config")
        assert response.status_code == 200
        config = response.json()
        assert config["hardware"]["source"] == "simulation"
        assert config["hardware"]["allow_real_actuators"] is False
        assert all(rule["enabled"] is False for rule in config["actuator_rules"])


def test_simulated_sensor_lifecycle() -> None:
    with TestClient(app) as client:
        response = client.post("/api/sensors/connect")
        assert response.status_code == 200
        assert response.json()["connected"] is True

        response = client.post("/api/sensors/sample")
        assert response.status_code == 200
        reading = response.json()
        assert reading["source"] == "simulation"
        assert {"temperature_c", "humidity_pct", "vibration_g", "light_lux"} <= set(reading["values"])

        latest = client.get("/api/sensors/latest")
        assert latest.status_code == 200
        assert latest.json()["id"]

        history = client.get("/api/sensors/history", params={"limit": 10})
        assert history.status_code == 200
        assert len(history.json()) >= 1

        response = client.post("/api/sensors/disconnect")
        assert response.status_code == 200
        assert response.json()["running"] is False


def test_hardware_probe_has_no_runtime_side_effects() -> None:
    with TestClient(app) as client:
        before = client.get("/api/sensors/history", params={"limit": 1000}).json()

        response = client.post("/api/hardware/test")

        assert response.status_code == 200, response.text
        assert response.json()["success"] is True
        assert response.json()["reading"]["values"]
        assert response.json()["status"]["running"] is False
        after = client.get("/api/sensors/history", params={"limit": 1000}).json()
        assert len(after) == len(before)


def test_linear_training_prediction_and_actuator_dry_run() -> None:
    with TestClient(app) as client:
        dataset_response = client.post(
            "/api/datasets/demo",
            json={"problem_type": "regression", "rows": 100, "seed": 7},
        )
        assert dataset_response.status_code == 200, dataset_response.text
        dataset = dataset_response.json()
        assert dataset["rows"] == 100
        assert dataset["suggested_target"] == "target_energy_w"

        training_payload = {
            "dataset_id": dataset["id"],
            "name": "Test lineare",
            "target": "target_energy_w",
            "features": ["temperature_c", "humidity_pct", "vibration_g", "light_lux"],
            "problem_type": "regression",
            "algorithm": "linear_regression",
            "test_size": 0.2,
            "random_state": 7,
        }
        training_response = client.post("/api/models/train", json=training_payload)
        assert training_response.status_code == 200, training_response.text
        model = training_response.json()
        assert model["algorithm"] == "linear_regression"
        assert {"r2", "mae", "rmse"} <= set(model["metrics"])

        activate_response = client.post(f"/api/models/{model['id']}/activate")
        assert activate_response.status_code == 200
        assert activate_response.json()["active"] is True

        rule_payload = {
            "name": "LED test sempre attivo",
            "enabled": True,
            "actuator": "led",
            "operator": "gt",
            "threshold": -1000000,
            "command_on": '{"actuator":"led","value":1}',
            "command_off": '{"actuator":"led","value":0}',
            "cooldown_seconds": 0,
        }
        rule_response = client.post("/api/actuators/rules", json=rule_payload)
        assert rule_response.status_code == 200, rule_response.text

        prediction_response = client.post(
            "/api/predict",
            json={
                "values": {
                    "temperature_c": 27.5,
                    "humidity_pct": 55.0,
                    "vibration_g": 0.2,
                    "light_lux": 450,
                },
                "evaluate_actuators": True,
            },
        )
        assert prediction_response.status_code == 200, prediction_response.text
        prediction = prediction_response.json()
        assert prediction["model_id"] == model["id"]
        assert isinstance(prediction["prediction"], (int, float))

        events_response = client.get("/api/actuators/events")
        assert events_response.status_code == 200
        events = events_response.json()
        assert events
        assert events[0]["actuator"] == "led"
        assert events[0]["execution_mode"] == "simulated"
        assert events[0]["success"] is True

        disabled_rule = rule_response.json()
        disabled_rule["enabled"] = False
        disable_response = client.put(
            "/api/actuators/rules",
            json={"rules": [disabled_rule]},
        )
        assert disable_response.status_code == 200, disable_response.text
        events_after_disable = client.get("/api/actuators/events").json()
        assert events_after_disable[0]["command"] == '{"actuator":"led","value":0}'
        assert "disabilitata" in events_after_disable[0]["message"].lower()


def test_classification_cannot_use_linear_regression() -> None:
    with TestClient(app) as client:
        dataset_response = client.post(
            "/api/datasets/demo",
            json={"problem_type": "classification", "rows": 80, "seed": 11},
        )
        assert dataset_response.status_code == 200
        dataset = dataset_response.json()

        response = client.post(
            "/api/models/train",
            json={
                "dataset_id": dataset["id"],
                "name": "Combinazione non valida",
                "target": "target_state",
                "features": ["temperature_c", "humidity_pct", "vibration_g"],
                "problem_type": "classification",
                "algorithm": "linear_regression",
            },
        )
        assert response.status_code == 422
        assert "XGBoost" in response.text


def test_xgboost_classification_training_and_prediction() -> None:
    with TestClient(app) as client:
        dataset_response = client.post(
            "/api/datasets/demo",
            json={"problem_type": "classification", "rows": 140, "seed": 17},
        )
        assert dataset_response.status_code == 200, dataset_response.text
        dataset = dataset_response.json()

        training_response = client.post(
            "/api/models/train",
            json={
                "dataset_id": dataset["id"],
                "name": "Test classificatore XGBoost",
                "target": "target_state",
                "features": ["temperature_c", "humidity_pct", "vibration_g", "light_lux"],
                "problem_type": "classification",
                "algorithm": "xgboost",
                "test_size": 0.2,
                "random_state": 17,
                "hyperparameters": {
                    "n_estimators": 12,
                    "max_depth": 2,
                    "learning_rate": 0.15,
                },
            },
        )
        assert training_response.status_code == 200, training_response.text
        model = training_response.json()
        assert model["algorithm"] == "xgboost"
        assert {"accuracy", "precision_weighted", "recall_weighted", "f1_weighted"} <= set(model["metrics"])

        prediction_response = client.post(
            "/api/predict",
            json={
                "model_id": model["id"],
                "values": {
                    "temperature_c": 31.0,
                    "humidity_pct": 70.0,
                    "vibration_g": 1.2,
                    "light_lux": 250,
                },
            },
        )
        assert prediction_response.status_code == 200, prediction_response.text
        prediction = prediction_response.json()
        assert prediction["prediction"] in {"normal", "attention", "alarm"}
        assert prediction["probabilities"]

