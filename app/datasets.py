from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from fastapi import HTTPException, UploadFile

from .schemas import DatasetInfo, DemoDatasetRequest
from .storage import FileStore, is_safe_identifier, safe_stem


MAX_UPLOAD_BYTES = 20 * 1024 * 1024


class DatasetRepository:
    def __init__(self, store: FileStore) -> None:
        self.store = store

    async def save_upload(self, upload: UploadFile) -> DatasetInfo:
        if not upload.filename or not upload.filename.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="Carica un file in formato CSV")
        raw = await upload.read(MAX_UPLOAD_BYTES + 1)
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Il CSV supera il limite di 20 MB")
        try:
            frame = pd.read_csv(io.BytesIO(raw))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"CSV non leggibile: {exc}") from exc
        return self._save_frame(frame, upload.filename)

    def create_demo(self, request: DemoDatasetRequest) -> DatasetInfo:
        rng = np.random.default_rng(request.seed)
        temperature = rng.normal(26, 4.8, request.rows).clip(12, 44)
        humidity = rng.normal(57, 14, request.rows).clip(15, 98)
        vibration = rng.gamma(1.6, 0.32, request.rows).clip(0, 3.5)
        light = rng.normal(470, 210, request.rows).clip(0, 1200)
        energy = 18 + temperature * 2.5 + humidity * 0.32 + vibration * 15 - light * 0.012 + rng.normal(0, 4.5, request.rows)
        frame = pd.DataFrame({
            "temperature_c": temperature.round(2),
            "humidity_pct": humidity.round(2),
            "vibration_g": vibration.round(3),
            "light_lux": light.round(0),
        })
        if request.problem_type == "regression":
            frame["target_energy_w"] = energy.round(2)
            name = "demo-regressione.csv"
        else:
            risk = temperature * 0.8 + humidity * 0.12 + vibration * 13 - light * 0.002
            frame["target_state"] = pd.cut(
                risk, bins=[-np.inf, 28, 39, np.inf],
                labels=["normal", "attention", "alarm"],
            ).astype(str)
            name = "demo-classificazione.csv"
        return self._save_frame(frame, name, request.problem_type)

    def _save_frame(self, frame: pd.DataFrame, original_name: str, problem_hint: str | None = None) -> DatasetInfo:
        if frame.empty:
            raise HTTPException(status_code=400, detail="Il dataset è vuoto")
        if len(frame.columns) < 2:
            raise HTTPException(status_code=400, detail="Il dataset deve contenere almeno una feature e un target")
        dataset_id = f"{safe_stem(original_name)}-{uuid4().hex[:8]}"
        self.store.ensure()
        frame.to_csv(self.store.datasets_dir / f"{dataset_id}.csv", index=False)
        target = str(frame.columns[-1])
        if problem_hint is None:
            series = frame[target]
            problem_hint = "classification" if (not pd.api.types.is_numeric_dtype(series) or series.nunique(dropna=True) <= 10) else "regression"
        info = DatasetInfo(
            id=dataset_id, name=Path(original_name).stem, filename=original_name,
            rows=len(frame), columns=[str(column) for column in frame.columns],
            dtypes={str(column): str(dtype) for column, dtype in frame.dtypes.items()},
            preview=self._preview(frame), suggested_target=target,
            suggested_problem_type=problem_hint, created_at=datetime.now(timezone.utc),
        )
        self.store.save_json(self.store.datasets_dir / f"{dataset_id}.json", info.model_dump(mode="json"))
        return info

    def list(self) -> list[DatasetInfo]:
        self.store.ensure()
        result: list[DatasetInfo] = []
        for path in self.store.datasets_dir.glob("*.json"):
            try:
                result.append(DatasetInfo.model_validate(self.store.load_json(path)))
            except Exception:
                continue
        return sorted(result, key=lambda item: item.created_at, reverse=True)

    def get(self, dataset_id: str) -> DatasetInfo:
        if not is_safe_identifier(dataset_id):
            raise HTTPException(status_code=422, detail="ID dataset non valido")
        raw = self.store.load_json(self.store.datasets_dir / f"{dataset_id}.json")
        if raw is None:
            raise HTTPException(status_code=404, detail="Dataset non trovato")
        return DatasetInfo.model_validate(raw)

    def csv_path(self, dataset_id: str) -> Path:
        self.get(dataset_id)
        path = self.store.datasets_dir / f"{dataset_id}.csv"
        if not path.exists():
            raise HTTPException(status_code=404, detail="File del dataset non trovato")
        return path

    def load_frame(self, dataset_id: str) -> pd.DataFrame:
        return pd.read_csv(self.csv_path(dataset_id))

    @staticmethod
    def _preview(frame: pd.DataFrame) -> list[dict[str, Any]]:
        preview = frame.head(8).astype(object)
        return preview.where(pd.notna(preview), None).to_dict(orient="records")
