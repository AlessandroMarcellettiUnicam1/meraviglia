from __future__ import annotations

from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .actuators import ActuatorReleaseError, ActuatorService
from .datasets import DatasetRepository
from .ml import ModelService
from .schemas import (
    ActuatorEvent, ActuatorRule, AppConfig, DatasetInfo, DemoDatasetRequest,
    ModelInfo, PredictionRequest, PredictionResult, RuntimeStatus,
    SensorReading, TrainingRequest,
)
from .sensors import SensorService
from .storage import ConfigRepository, FileStore


store = FileStore()
configs = ConfigRepository(store)
datasets = DatasetRepository(store)
models = ModelService(store, datasets, configs)
actuators = ActuatorService(store, configs)
prediction_history: deque[PredictionResult] = deque(maxlen=1000)
last_actuator_events: list[ActuatorEvent] = []


async def process_reading(reading: SensorReading) -> None:
    global last_actuator_events
    if not configs.get().active_model_id:
        return
    try:
        prediction = models.predict(reading.values)
        prediction_history.append(prediction)
        last_actuator_events = await actuators.evaluate(prediction)
    except HTTPException as exc:
        # Hardware fields and model features may intentionally be configured in two steps.
        sensors.last_error = str(exc.detail)


sensors = SensorService(configs.get, process_reading)
actuators.sensor_service = sensors


async def release_actuators_or_409(
    *,
    rule_ids: list[str] | None = None,
    reason: str,
    config_override: AppConfig | None = None,
) -> list[ActuatorEvent]:
    try:
        return await actuators.release_active(
            rule_ids=rule_ids,
            reason=reason,
            config_override=config_override,
            strict=True,
        )
    except ActuatorReleaseError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "Impossibile confermare lo spegnimento degli attuatori; configurazione invariata",
                "active_rule_ids": list(exc.active_rule_ids),
            },
        ) from exc


@asynccontextmanager
async def lifespan(_: FastAPI):
    store.ensure()
    if configs.get().hardware.auto_start:
        try:
            await sensors.start()
        except HTTPException:
            pass
    yield
    await actuators.release_active(reason="arresto applicazione", strict=False)
    await sensors.stop()


app = FastAPI(
    title="SensorLab EDU API",
    description="Sensori Arduino, machine learning e attuatori in un flusso didattico.",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "SensorLab EDU", "version": app.version, "data_directory": str(store.root)}


@app.get("/api/status")
def overall_status() -> dict[str, Any]:
    return {"runtime": sensors.status(), "active_model_id": configs.get().active_model_id, "latest_prediction": prediction_history[-1] if prediction_history else None}


@app.get("/api/config", response_model=AppConfig)
def get_config() -> AppConfig:
    return configs.get()


@app.put("/api/config", response_model=AppConfig)
async def update_config(config: AppConfig) -> AppConfig:
    current = configs.get()
    if actuators.active_rule_ids:
        await release_actuators_or_409(
            reason="aggiornamento configurazione",
            config_override=current,
        )
    saved = configs.save(config)
    await sensors.restart_if_running()
    return saved


@app.get("/api/runtime", response_model=RuntimeStatus)
@app.get("/api/sensors/status", response_model=RuntimeStatus)
def runtime_status() -> RuntimeStatus:
    return sensors.status()


@app.post("/api/runtime/start", response_model=RuntimeStatus)
@app.post("/api/sensors/connect", response_model=RuntimeStatus)
async def start_runtime() -> RuntimeStatus:
    return await sensors.start()


@app.post("/api/runtime/stop", response_model=RuntimeStatus)
@app.post("/api/sensors/disconnect", response_model=RuntimeStatus)
async def stop_runtime() -> RuntimeStatus:
    if actuators.active_rule_ids:
        await release_actuators_or_409(reason="arresto acquisizione")
    return await sensors.stop()


@app.post("/api/runtime/sample", response_model=SensorReading)
@app.post("/api/sensors/sample", response_model=SensorReading)
async def take_sample() -> SensorReading:
    return await sensors.sample()


@app.post("/api/hardware/test")
async def test_hardware() -> dict[str, Any]:
    reading = await sensors.probe()
    return {"success": True, "reading": reading, "status": sensors.status()}


@app.post("/api/sensors/ingest", response_model=SensorReading)
async def ingest_sensor_payload(values: dict[str, Any] = Body(...)) -> SensorReading:
    """HTTP bridge useful before or instead of a direct serial connection."""
    return await sensors.ingest(values)


@app.get("/api/sensors/readings", response_model=list[SensorReading])
@app.get("/api/sensors/history", response_model=list[SensorReading])
@app.get("/api/readings", response_model=list[SensorReading])
def readings(limit: int = Query(default=120, ge=1, le=1000)) -> list[SensorReading]:
    return list(sensors.history)[-limit:]


@app.get("/api/sensors/latest", response_model=SensorReading)
@app.get("/api/readings/latest", response_model=SensorReading)
def latest_reading() -> SensorReading:
    if not sensors.history:
        raise HTTPException(status_code=404, detail="Nessuna lettura disponibile")
    return sensors.history[-1]


@app.get("/api/predictions/latest")
def latest_prediction() -> dict[str, Any]:
    return {"prediction": prediction_history[-1] if prediction_history else None, "actuator_events": last_actuator_events}


@app.get("/api/predictions/history", response_model=list[PredictionResult])
def predictions(limit: int = Query(default=120, ge=1, le=1000)) -> list[PredictionResult]:
    return list(prediction_history)[-limit:]


@app.post("/api/datasets/upload", response_model=DatasetInfo)
@app.post("/api/training/dataset", response_model=DatasetInfo)
async def upload_dataset(file: UploadFile = File(...)) -> DatasetInfo:
    return await datasets.save_upload(file)


@app.post("/api/datasets/demo", response_model=DatasetInfo)
@app.post("/api/training/demo", response_model=DatasetInfo)
def create_demo_dataset(request: DemoDatasetRequest) -> DatasetInfo:
    return datasets.create_demo(request)


@app.get("/api/datasets", response_model=list[DatasetInfo])
def list_datasets() -> list[DatasetInfo]:
    return datasets.list()


@app.get("/api/datasets/{dataset_id}", response_model=DatasetInfo)
def get_dataset(dataset_id: str) -> DatasetInfo:
    return datasets.get(dataset_id)


@app.get("/api/datasets/{dataset_id}/download")
def download_dataset(dataset_id: str) -> FileResponse:
    info = datasets.get(dataset_id)
    return FileResponse(datasets.csv_path(dataset_id), media_type="text/csv", filename=info.filename)


@app.post("/api/models/train", response_model=ModelInfo)
@app.post("/api/training/train", response_model=ModelInfo)
def train_model(request: TrainingRequest) -> ModelInfo:
    return models.train(request)


@app.get("/api/models", response_model=list[ModelInfo])
def list_models() -> list[ModelInfo]:
    return models.list()


@app.get("/api/models/{model_id}", response_model=ModelInfo)
def get_model(model_id: str) -> ModelInfo:
    return models.get(model_id)


@app.post("/api/models/{model_id}/activate", response_model=ModelInfo)
async def activate_model(model_id: str) -> ModelInfo:
    if actuators.active_rule_ids:
        await release_actuators_or_409(reason="cambio modello attivo")
    return models.activate(model_id)


@app.post("/api/predict", response_model=PredictionResult)
async def predict(request: PredictionRequest) -> PredictionResult:
    if request.values is not None:
        values = request.values
    elif sensors.history:
        values = sensors.history[-1].values
    else:
        raise HTTPException(status_code=409, detail="Fornisci values o acquisisci prima una lettura")
    result = models.predict(values, request.model_id)
    prediction_history.append(result)
    if request.evaluate_actuators:
        await actuators.evaluate(result)
    return result


@app.get("/api/actuators/rules", response_model=list[ActuatorRule])
def list_rules() -> list[ActuatorRule]:
    return configs.get().actuator_rules


@app.put("/api/actuators/rules", response_model=list[ActuatorRule])
async def replace_rules(payload: Any = Body(...)) -> list[ActuatorRule]:
    raw_rules = payload.get("rules") if isinstance(payload, dict) and "rules" in payload else payload
    if not isinstance(raw_rules, list):
        raise HTTPException(status_code=422, detail="Invia una lista di regole o {rules: [...]}")
    rules = [ActuatorRule.model_validate(item) for item in raw_rules]
    if len({rule.id for rule in rules}) != len(rules):
        raise HTTPException(status_code=422, detail="Gli ID delle regole devono essere univoci")
    config = configs.get()
    incoming = {rule.id: rule for rule in rules}
    current = {rule.id: rule for rule in config.actuator_rules}
    release_ids = [
        rule_id
        for rule_id in actuators.active_rule_ids
        if rule_id not in incoming
        or not incoming[rule_id].enabled
        or incoming[rule_id] != current.get(rule_id)
    ]
    if release_ids:
        await release_actuators_or_409(
            rule_ids=release_ids,
            reason="regola disabilitata, rimossa o modificata",
            config_override=config,
        )
    config.actuator_rules = rules
    configs.save(config)
    return rules


@app.post("/api/actuators/rules", response_model=ActuatorRule)
def create_rule(rule: ActuatorRule) -> ActuatorRule:
    config = configs.get()
    if any(existing.id == rule.id for existing in config.actuator_rules):
        raise HTTPException(status_code=409, detail="Esiste già una regola con questo ID")
    config.actuator_rules.append(rule)
    configs.save(config)
    return rule


@app.put("/api/actuators/rules/{rule_id}", response_model=ActuatorRule)
async def update_rule(rule_id: str, rule: ActuatorRule) -> ActuatorRule:
    config = configs.get()
    for index, existing in enumerate(config.actuator_rules):
        if existing.id == rule_id:
            rule.id = rule_id
            if rule_id in actuators.active_rule_ids and rule != existing:
                await release_actuators_or_409(
                    rule_ids=[rule_id],
                    reason="regola modificata",
                    config_override=config,
                )
            config.actuator_rules[index] = rule
            configs.save(config)
            return rule
    raise HTTPException(status_code=404, detail="Regola non trovata")


@app.delete("/api/actuators/rules/{rule_id}")
async def delete_rule(rule_id: str) -> dict[str, bool]:
    config = configs.get()
    remaining = [rule for rule in config.actuator_rules if rule.id != rule_id]
    if len(remaining) == len(config.actuator_rules):
        raise HTTPException(status_code=404, detail="Regola non trovata")
    if rule_id in actuators.active_rule_ids:
        await release_actuators_or_409(
            rule_ids=[rule_id],
            reason="regola eliminata",
            config_override=config,
        )
    config.actuator_rules = remaining
    configs.save(config)
    return {"deleted": True}


@app.get("/api/actuators/events", response_model=list[ActuatorEvent])
@app.get("/api/events", response_model=list[ActuatorEvent])
def actuator_events(limit: int = Query(default=100, ge=1, le=1000)) -> list[ActuatorEvent]:
    return actuators.recent_events(limit)


STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False, response_model=None)
def index() -> FileResponse | dict[str, str]:
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "SensorLab EDU API", "docs": "/docs"}
