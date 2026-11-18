from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import joblib
import pandas as pd
from fastapi import HTTPException
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from .datasets import DatasetRepository
from .schemas import ModelInfo, PredictionResult, TrainingRequest
from .storage import ConfigRepository, FileStore, is_safe_identifier


class ModelService:
    def __init__(self, store: FileStore, datasets: DatasetRepository, configs: ConfigRepository) -> None:
        self.store = store
        self.datasets = datasets
        self.configs = configs
        self._artifact_cache: dict[str, dict[str, Any]] = {}

    def train(self, request: TrainingRequest) -> ModelInfo:
        frame = self.datasets.load_frame(request.dataset_id)
        missing = [column for column in [*request.features, request.target] if column not in frame.columns]
        if missing:
            raise HTTPException(status_code=422, detail=f"Colonne non trovate nel dataset: {', '.join(missing)}")
        working = frame[[*request.features, request.target]].dropna(subset=[request.target])
        if len(working) < 20:
            raise HTTPException(status_code=422, detail="Servono almeno 20 righe complete per il training")
        try:
            features = working[request.features].apply(pd.to_numeric, errors="raise")
        except Exception as exc:
            raise HTTPException(status_code=422, detail="Le feature devono essere numeriche") from exc
        target = working[request.target]
        stratify = None
        label_encoder: LabelEncoder | None = None
        if request.problem_type == "regression":
            try:
                target = pd.to_numeric(target, errors="raise")
            except Exception as exc:
                raise HTTPException(status_code=422, detail="Il target di regressione deve essere numerico") from exc
        else:
            if target.nunique() < 2:
                raise HTTPException(status_code=422, detail="Per classificare servono almeno due classi")
            if target.value_counts().min() < 2:
                raise HTTPException(status_code=422, detail="Ogni classe deve avere almeno due esempi")
            label_encoder = LabelEncoder()
            target = pd.Series(label_encoder.fit_transform(target.astype(str)), index=target.index)
            stratify = target
        try:
            x_train, x_test, y_train, y_test = train_test_split(
                features, target, test_size=request.test_size,
                random_state=request.random_state, stratify=stratify,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Suddivisione train/test non possibile: {exc}",
            ) from exc
        estimator, effective_params = self._build_estimator(request, y_train)
        pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("estimator", estimator)])
        try:
            pipeline.fit(x_train, y_train)
            predicted = pipeline.predict(x_test)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Training non riuscito: {exc}") from exc
        if request.problem_type == "regression":
            metrics = {
                "r2": self._finite(r2_score(y_test, predicted)),
                "mae": self._finite(mean_absolute_error(y_test, predicted)),
                "rmse": self._finite(math.sqrt(mean_squared_error(y_test, predicted))),
            }
        else:
            assert label_encoder is not None
            labels = list(range(len(label_encoder.classes_)))
            metrics = {
                "accuracy": self._finite(accuracy_score(y_test, predicted)),
                "precision_weighted": self._finite(precision_score(y_test, predicted, average="weighted", zero_division=0)),
                "recall_weighted": self._finite(recall_score(y_test, predicted, average="weighted", zero_division=0)),
                "f1_weighted": self._finite(f1_score(y_test, predicted, average="weighted", zero_division=0)),
                "labels": label_encoder.classes_.tolist(),
                "confusion_matrix": confusion_matrix(y_test, predicted, labels=labels).tolist(),
            }
        model_id = f"model-{uuid4().hex[:10]}"
        details: dict[str, Any] = {}
        if request.algorithm == "linear_regression":
            fitted = pipeline.named_steps["estimator"]
            details = {
                "intercept": self._finite(fitted.intercept_),
                "coefficients": {name: self._finite(value) for name, value in zip(request.features, fitted.coef_)},
            }
        artifact = {"pipeline": pipeline, "label_encoder": label_encoder, "features": request.features, "problem_type": request.problem_type}
        self.store.ensure()
        joblib.dump(artifact, self.store.models_dir / f"{model_id}.joblib")
        info = ModelInfo(
            id=model_id, name=request.name, algorithm=request.algorithm,
            problem_type=request.problem_type, target=request.target, features=request.features,
            dataset_id=request.dataset_id, train_rows=len(x_train), test_rows=len(x_test),
            metrics=metrics, hyperparameters=effective_params, details=details,
            created_at=datetime.now(timezone.utc), active=False,
        )
        self.store.save_json(self.store.models_dir / f"{model_id}.json", info.model_dump(mode="json"))
        self._artifact_cache[model_id] = artifact
        return info

    @staticmethod
    def _build_estimator(request: TrainingRequest, y_train: pd.Series) -> tuple[Any, dict[str, Any]]:
        if request.algorithm == "linear_regression":
            allowed = {"fit_intercept", "positive"}
            params = {key: value for key, value in request.hyperparameters.items() if key in allowed}
            return LinearRegression(**params), params
        try:
            from xgboost import XGBClassifier, XGBRegressor
        except ImportError as exc:
            raise HTTPException(status_code=503, detail="XGBoost non è installato. Esegui: pip install -r requirements.txt") from exc
        defaults: dict[str, Any] = {
            "n_estimators": 160, "max_depth": 4, "learning_rate": 0.06,
            "subsample": 0.9, "colsample_bytree": 0.9,
            "random_state": request.random_state, "n_jobs": 2,
        }
        allowed = {"n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "min_child_weight", "reg_alpha", "reg_lambda"}
        defaults.update({key: value for key, value in request.hyperparameters.items() if key in allowed})
        if request.problem_type == "classification":
            defaults["eval_metric"] = "logloss" if int(pd.Series(y_train).nunique()) == 2 else "mlogloss"
            return XGBClassifier(**defaults), defaults
        defaults.update({"objective": "reg:squarederror", "eval_metric": "rmse"})
        return XGBRegressor(**defaults), defaults

    def list(self) -> list[ModelInfo]:
        self.store.ensure()
        active_id = self.configs.get().active_model_id
        result: list[ModelInfo] = []
        for path in self.store.models_dir.glob("*.json"):
            try:
                info = ModelInfo.model_validate(self.store.load_json(path))
                info.active = info.id == active_id
                result.append(info)
            except Exception:
                continue
        return sorted(result, key=lambda item: item.created_at, reverse=True)

    def get(self, model_id: str) -> ModelInfo:
        if not is_safe_identifier(model_id):
            raise HTTPException(status_code=422, detail="ID modello non valido")
        raw = self.store.load_json(self.store.models_dir / f"{model_id}.json")
        if raw is None:
            raise HTTPException(status_code=404, detail="Modello non trovato")
        info = ModelInfo.model_validate(raw)
        info.active = self.configs.get().active_model_id == model_id
        return info

    def activate(self, model_id: str) -> ModelInfo:
        info = self.get(model_id)
        config = self.configs.get()
        config.active_model_id = model_id
        self.configs.save(config)
        info.active = True
        return info

    def predict(self, values: dict[str, float | int | bool], model_id: str | None = None) -> PredictionResult:
        chosen_id = model_id or self.configs.get().active_model_id
        if not chosen_id:
            raise HTTPException(status_code=409, detail="Nessun modello attivo")
        info = self.get(chosen_id)
        missing = [feature for feature in info.features if feature not in values]
        if missing:
            raise HTTPException(status_code=422, detail=f"Valori mancanti per: {', '.join(missing)}")
        try:
            feature_values = {feature: float(values[feature]) for feature in info.features}
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="Le feature di inferenza devono essere numeriche") from exc
        artifact = self._load_artifact(chosen_id)
        frame = pd.DataFrame([feature_values], columns=info.features)
        raw_prediction = artifact["pipeline"].predict(frame)[0]
        probabilities = None
        if info.problem_type == "classification":
            encoder: LabelEncoder = artifact["label_encoder"]
            prediction: str | float = str(encoder.inverse_transform([int(raw_prediction)])[0])
            if hasattr(artifact["pipeline"], "predict_proba"):
                raw_probabilities = artifact["pipeline"].predict_proba(frame)[0]
                probabilities = {str(label): round(float(value), 6) for label, value in zip(encoder.classes_, raw_probabilities)}
        else:
            prediction = self._finite(raw_prediction)
        return PredictionResult(
            model_id=info.id, model_name=info.name, algorithm=info.algorithm,
            problem_type=info.problem_type, prediction=prediction,
            probabilities=probabilities, used_features=feature_values,
        )

    def _load_artifact(self, model_id: str) -> dict[str, Any]:
        if model_id not in self._artifact_cache:
            path = self.store.models_dir / f"{model_id}.joblib"
            if not path.exists():
                raise HTTPException(status_code=404, detail="File del modello non trovato")
            self._artifact_cache[model_id] = joblib.load(path)
        return self._artifact_cache[model_id]

    @staticmethod
    def _finite(value: Any) -> float:
        result = float(value)
        return result if math.isfinite(result) else 0.0
