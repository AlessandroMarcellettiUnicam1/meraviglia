from __future__ import annotations

from collections import deque

from app.schemas import AppConfig
from app.sensors import SerialConnector


class FakeSerial:
    def __init__(self, *lines: bytes) -> None:
        self.lines = deque(lines)
        self.timeout = 0.0

    def readline(self) -> bytes:
        return self.lines.popleft() if self.lines else b""


def test_serial_connector_skips_actuator_ack_before_sensor_sample() -> None:
    config = AppConfig()
    connector = SerialConnector(config)
    connector.serial = FakeSerial(
        b'{"type":"command_ack","ok":true,"actuator":"led","value":1}\n',
        b'{"temperature":24.5,"humidity":51.0,"vibration":0.2,"light":420}\n',
    )
    connector.connected = True

    values = connector.read()

    assert values == {
        "temperature_c": 24.5,
        "humidity_pct": 51.0,
        "vibration_g": 0.2,
        "light_lux": 420.0,
    }
