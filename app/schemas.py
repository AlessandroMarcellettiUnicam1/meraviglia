from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SensorSource(str, Enum):
    simulation = "simulation"
    serial = "serial"


class SensorValueType(str, Enum):
    float = "float"
    integer = "integer"
    boolean = "boolean"


class SensorField(BaseModel):
    name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    label: str
    unit: str = ""
    value_type: SensorValueType = SensorValueType.float
    source_key: str | None = None
    simulation_min: float = 0.0
    simulation_max: float = 100.0
    precision: int = Field(default=2, ge=0, le=6)

    @model_validator(mode="after")
    def validate_range(self) -> "SensorField":
        if self.simulation_max <= self.simulation_min:
            raise ValueError("simulation_max deve essere maggiore di simulation_min")
        if not self.source_key:
            self.source_key = self.name
        return self


class SerialConfig(BaseModel):
    port: str = "COM3"
    baud_rate: int = Field(default=115200, ge=300, le=4_000_000)
    timeout_seconds: float = Field(default=2.0, gt=0, le=30)
    encoding: str = "utf-8"
    line_ending: str = "\n"
    input_format: Literal["json"] = "json"


class HardwareConfig(BaseModel):
    source: SensorSource = SensorSource.simulation
    sampling_interval_seconds: float = Field(default=1.0, ge=0.1, le=3600)
    auto_start: bool = False
    allow_real_actuators: bool = False
    simulation_seed: int = 42
    serial: SerialConfig = Field(default_factory=SerialConfig)
    sensors: list[SensorField] = Field(
        default_factory=lambda: [
            SensorField(name="temperature_c", source_key="temperature", label="Temperatura", unit="°C", simulation_min=18, simulation_max=38),
            SensorField(name="humidity_pct", source_key="humidity", label="Umidità", unit="%", simulation_min=30, simulation_max=85),
            SensorField(name="vibration_g", source_key="vibration", label="Vibrazione", unit="g", simulation_min=0, simulation_max=2.5, precision=3),
            SensorField(name="light_lux", source_key="light", label="Luce", unit="lux", simulation_min=80, simulation_max=900, precision=0),
        ]
    )

    @field_validator("sensors")
    @classmethod
    def unique_sensor_names(cls, value: list[SensorField]) -> list[SensorField]:
        if not value:
            raise ValueError("Configura almeno un sensore")
        names = [sensor.name for sensor in value]
        if len(names) != len(set(names)):
            raise ValueError("I nomi dei sensori devono essere univoci")
        return value


class RuleOperator(str, Enum):
    greater_than = "gt"
    greater_or_equal = "gte"
    less_than = "lt"
    less_or_equal = "lte"
    equal = "eq"
    not_equal = "ne"


class ActuatorRule(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    name: str
    enabled: bool = False
    actuator: str
    operator: RuleOperator = RuleOperator.greater_than
    threshold: float | str
    command_on: str = Field(validation_alias=AliasChoices("command_on", "command"))
    command_off: str | None = None
    pin: int | str | None = None
    cooldown_seconds: float = Field(default=5.0, ge=0, le=86_400)


class AppConfig(BaseModel):
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    active_model_id: str | None = None
    actuator_rules: list[ActuatorRule] = Field(
        default_factory=lambda: [
            ActuatorRule(
                name="Ventola dimostrativa", enabled=False, actuator="fan",
                operator=RuleOperator.greater_than, threshold=30,
                command_on='{"actuator":"fan","value":1}',
                command_off='{"actuator":"fan","value":0}', cooldown_seconds=10,
            )
        ]
    )


class SensorReading(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    timestamp: datetime = Field(default_factory=utc_now)
    source: SensorSource
    values: dict[str, float | int | bool]


class RuntimeStatus(BaseModel):
    running: bool
    connected: bool
    source: SensorSource
    samples_count: int
    last_read_at: datetime | None = None
    last_error: str | None = None


class DatasetInfo(BaseModel):
    id: str
    name: str
    filename: str
    rows: int
    columns: list[str]
    dtypes: dict[str, str]
    preview: list[dict[str, Any]]
    suggested_target: str | None = None
    suggested_problem_type: Literal["regression", "classification"] | None = None
    created_at: datetime


class DemoDatasetRequest(BaseModel):
    problem_type: Literal["regression", "classification"] = Field(
        default="regression", validation_alias=AliasChoices("problem_type", "task")
    )
    rows: int = Field(default=320, ge=40, le=10_000)
    seed: int = 42


class TrainingRequest(BaseModel):
    dataset_id: str
    name: str = Field(default="Modello didattico", min_length=1, max_length=100)
    target: str
    features: list[str] = Field(min_length=1)
    problem_type: Literal["regression", "classification"] = Field(
        validation_alias=AliasChoices("problem_type", "task")
    )
    algorithm: Literal["linear_regression", "xgboost"]
    test_size: float = Field(default=0.2, ge=0.1, le=0.4)
    random_state: int = 42
    hyperparameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_training(self) -> "TrainingRequest":
        if self.problem_type == "classification" and self.algorithm == "linear_regression":
            raise ValueError("Linear Regression produce valori continui: per la classificazione scegli XGBoost")
        if self.target in self.features:
            raise ValueError("La variabile target non può essere anche una feature")
        if len(self.features) != len(set(self.features)):
            raise ValueError("Le feature non possono essere duplicate")
        return self


class ModelInfo(BaseModel):
    id: str
    name: str
    algorithm: Literal["linear_regression", "xgboost"]
    problem_type: Literal["regression", "classification"]
    target: str
    features: list[str]
    dataset_id: str
    train_rows: int
    test_rows: int
    metrics: dict[str, Any]
    hyperparameters: dict[str, Any]
    details: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    active: bool = False


class PredictionRequest(BaseModel):
    values: dict[str, float | int | bool] | None = Field(
        default=None, validation_alias=AliasChoices("values", "features")
    )
    model_id: str | None = None
    evaluate_actuators: bool = False


class PredictionResult(BaseModel):
    model_id: str
    model_name: str
    algorithm: str
    problem_type: str
    prediction: float | int | str
    probabilities: dict[str, float] | None = None
    used_features: dict[str, float]
    created_at: datetime = Field(default_factory=utc_now)


class ActuatorEvent(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    timestamp: datetime = Field(default_factory=utc_now)
    rule_id: str
    rule_name: str
    actuator: str
    prediction: float | int | str
    threshold: float | str
    command: str
    execution_mode: Literal["simulated", "serial"]
    success: bool
    message: str
