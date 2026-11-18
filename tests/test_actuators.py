from __future__ import annotations

import asyncio
from pathlib import Path

from app.actuators import ActuatorService
from app.schemas import ActuatorRule, PredictionResult, RuleOperator
from app.storage import ConfigRepository, FileStore


def prediction(value: float) -> PredictionResult:
    return PredictionResult(
        model_id="model-test",
        model_name="Modello test",
        algorithm="linear_regression",
        problem_type="regression",
        prediction=value,
        used_features={"temperature_c": value},
    )


def test_actuator_commands_follow_edges_and_release(tmp_path: Path) -> None:
    store = FileStore(tmp_path)
    configs = ConfigRepository(store)
    config = configs.get()
    config.actuator_rules = [
        ActuatorRule(
            id="fan-test",
            name="Ventola test",
            enabled=True,
            actuator="fan",
            operator=RuleOperator.greater_than,
            threshold=10,
            command_on='{"actuator":"fan","value":1}',
            command_off='{"actuator":"fan","value":0}',
            cooldown_seconds=0,
        )
    ]
    configs.save(config)
    service = ActuatorService(store, configs)

    first_on = asyncio.run(service.evaluate(prediction(20)))
    repeated_on = asyncio.run(service.evaluate(prediction(21)))
    automatic_off = asyncio.run(service.evaluate(prediction(5)))
    second_on = asyncio.run(service.evaluate(prediction(20)))
    explicit_off = asyncio.run(service.release_active(reason="test"))

    assert [event.command for event in first_on] == ['{"actuator":"fan","value":1}']
    assert repeated_on == []
    assert [event.command for event in automatic_off] == ['{"actuator":"fan","value":0}']
    assert [event.command for event in second_on] == ['{"actuator":"fan","value":1}']
    assert [event.command for event in explicit_off] == ['{"actuator":"fan","value":0}']
    assert service.active_rule_ids == ()
