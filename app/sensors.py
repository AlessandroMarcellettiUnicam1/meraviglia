from __future__ import annotations

import asyncio
import inspect
import json
import math
import random
import time
from collections import deque
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

from fastapi import HTTPException

from .schemas import AppConfig, RuntimeStatus, SensorReading, SensorSource, SensorValueType


class SensorConnector:
    connected = False

    def connect(self) -> None:
        self.connected = True

    def close(self) -> None:
        self.connected = False

    def read(self) -> dict[str, float | int | bool]:
        raise NotImplementedError

    def write(self, command: str) -> None:
        raise RuntimeError("Il connettore non supporta comandi")


class SimulationConnector(SensorConnector):
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.random = random.Random(config.hardware.simulation_seed)
        self.step = 0
        self.connected = False

    def read(self) -> dict[str, float | int | bool]:
        if not self.connected:
            raise RuntimeError("Simulatore non avviato")
        self.step += 1
        values: dict[str, float | int | bool] = {}
        for index, field in enumerate(self.config.hardware.sensors):
            low, high = field.simulation_min, field.simulation_max
            midpoint, amplitude = (low + high) / 2, (high - low) / 2
            phase = self.step / (12 + index * 3) + index * 1.7
            raw = midpoint + amplitude * (0.58 * math.sin(phase) + self.random.uniform(-0.13, 0.13))
            raw = min(high, max(low, raw))
            if field.value_type == SensorValueType.boolean:
                value: float | int | bool = raw >= midpoint
            elif field.value_type == SensorValueType.integer:
                value = int(round(raw))
            else:
                value = round(raw, field.precision)
            values[field.name] = value
        return values


class SerialConnector(SensorConnector):
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.serial: Any = None
        self.connected = False

    def connect(self) -> None:
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError("pyserial non è installato. Esegui: pip install -r requirements.txt") from exc
        serial_config = self.config.hardware.serial
        try:
            self.serial = serial.Serial(
                port=serial_config.port, baudrate=serial_config.baud_rate,
                timeout=serial_config.timeout_seconds,
            )
        except Exception as exc:
            raise RuntimeError(f"Connessione seriale non riuscita: {exc}") from exc
        self.connected = True

    def close(self) -> None:
        if self.serial is not None:
            self.serial.close()
        self.serial = None
        self.connected = False

    def read(self) -> dict[str, float | int | bool]:
        if not self.connected or self.serial is None:
            raise RuntimeError("Porta seriale non connessa")
        settings = self.config.hardware.serial
        deadline = time.monotonic() + settings.timeout_seconds
        last_problem = "Arduino non ha inviato una riga sensori"
        while time.monotonic() < deadline:
            # An acknowledgement for a previous actuator command can arrive
            # immediately before the next sensor sample. Keep reading within
            # the configured timeout instead of treating it as a bad sample.
            self.serial.timeout = max(0.05, deadline - time.monotonic())
            raw = self.serial.readline()
            if not raw:
                break
            try:
                payload = json.loads(raw.decode(settings.encoding).strip())
            except Exception as exc:
                last_problem = f"riga JSON non valida ({exc})"
                continue
            if not isinstance(payload, dict):
                last_problem = "la riga non è un oggetto JSON"
                continue

            missing = [
                field.source_key or field.name
                for field in self.config.hardware.sensors
                if (field.source_key or field.name) not in payload
            ]
            if missing:
                last_problem = f"chiavi mancanti: {', '.join(missing)}"
                continue

            result: dict[str, float | int | bool] = {}
            try:
                for field in self.config.hardware.sensors:
                    key = field.source_key or field.name
                    value = payload[key]
                    if field.value_type == SensorValueType.boolean:
                        result[field.name] = bool(value)
                    elif field.value_type == SensorValueType.integer:
                        result[field.name] = int(value)
                    else:
                        result[field.name] = float(value)
            except (TypeError, ValueError) as exc:
                last_problem = f"valore sensore non numerico ({exc})"
                continue
            return result
        raise RuntimeError(f"Timeout lettura seriale: {last_problem}")

    def write(self, command: str) -> None:
        if not self.connected or self.serial is None:
            raise RuntimeError("Porta seriale non connessa")
        settings = self.config.hardware.serial
        self.serial.write((command + settings.line_ending).encode(settings.encoding))
        self.serial.flush()


SampleCallback = Callable[[SensorReading], Any | Awaitable[Any]]


class SensorService:
    def __init__(self, config_provider: Callable[[], AppConfig], on_sample: SampleCallback | None = None) -> None:
        self.config_provider = config_provider
        self.on_sample = on_sample
        self.connector: SensorConnector | None = None
        self.running = False
        self.task: asyncio.Task[None] | None = None
        self.history: deque[SensorReading] = deque(maxlen=1000)
        self.samples_count = 0
        self.last_read_at: datetime | None = None
        self.last_error: str | None = None
        self._lock = asyncio.Lock()
        self._sample_lock = asyncio.Lock()

    async def start(self) -> RuntimeStatus:
        async with self._lock:
            if self.running:
                return self.status()
            self.connector = self._make_connector(self.config_provider())
            try:
                await asyncio.to_thread(self.connector.connect)
            except Exception as exc:
                self.connector = None
                self.last_error = str(exc)
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            self.running = True
            self.last_error = None
            self.task = asyncio.create_task(self._loop(), name="sensor-sampling-loop")
            return self.status()

    async def stop(self) -> RuntimeStatus:
        async with self._lock:
            self.running = False
            task, self.task = self.task, None
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            if self.connector:
                await asyncio.to_thread(self.connector.close)
            self.connector = None
            return self.status()

    async def restart_if_running(self) -> RuntimeStatus:
        was_running = self.running
        await self.stop()
        return await self.start() if was_running else self.status()

    async def sample(self) -> SensorReading:
        async with self._sample_lock:
            if not self.running or not self.connector:
                raise HTTPException(status_code=409, detail="Avvia prima l'acquisizione sensori")
            try:
                values = await asyncio.to_thread(self.connector.read)
                reading = SensorReading(source=self.config_provider().hardware.source, values=values)
                await self._record(reading)
                self.last_error = None
                return reading
            except HTTPException:
                raise
            except Exception as exc:
                self.last_error = str(exc)
                raise HTTPException(status_code=503, detail=str(exc)) from exc

    async def probe(self) -> SensorReading:
        """Read one sample without history, inference or actuator side effects."""
        config = self.config_provider()
        async with self._sample_lock:
            temporary_connector: SensorConnector | None = None
            connector = self.connector if self.running and self.connector else None
            try:
                if connector is None:
                    temporary_connector = self._make_connector(config)
                    await asyncio.to_thread(temporary_connector.connect)
                    connector = temporary_connector
                values = await asyncio.to_thread(connector.read)
                self.last_error = None
                return SensorReading(source=config.hardware.source, values=values)
            except Exception as exc:
                self.last_error = str(exc)
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            finally:
                if temporary_connector is not None:
                    await asyncio.to_thread(temporary_connector.close)

    async def ingest(self, values: dict[str, Any]) -> SensorReading:
        config = self.config_provider()
        converted: dict[str, float | int | bool] = {}
        for field in config.hardware.sensors:
            key = field.source_key or field.name
            if field.name in values:
                value = values[field.name]
            elif key in values:
                value = values[key]
            else:
                raise HTTPException(status_code=422, detail=f"Valore mancante: {field.name}")
            try:
                if field.value_type == SensorValueType.boolean:
                    converted[field.name] = bool(value)
                elif field.value_type == SensorValueType.integer:
                    converted[field.name] = int(value)
                else:
                    converted[field.name] = float(value)
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=422, detail=f"Valore non valido per {field.name}") from exc
        reading = SensorReading(source=config.hardware.source, values=converted)
        await self._record(reading)
        return reading

    async def send_command(self, command: str) -> None:
        if not self.connector:
            raise RuntimeError("Connettore sensori non attivo")
        await asyncio.to_thread(self.connector.write, command)

    def status(self) -> RuntimeStatus:
        config = self.config_provider()
        return RuntimeStatus(
            running=self.running, connected=bool(self.connector and self.connector.connected),
            source=config.hardware.source, samples_count=self.samples_count,
            last_read_at=self.last_read_at, last_error=self.last_error,
        )

    async def _loop(self) -> None:
        while self.running:
            started = asyncio.get_running_loop().time()
            try:
                await self.sample()
            except HTTPException:
                pass
            interval = self.config_provider().hardware.sampling_interval_seconds
            await asyncio.sleep(max(0.05, interval - (asyncio.get_running_loop().time() - started)))

    async def _record(self, reading: SensorReading) -> None:
        self.history.append(reading)
        self.samples_count += 1
        self.last_read_at = reading.timestamp
        if self.on_sample:
            result = self.on_sample(reading)
            if inspect.isawaitable(result):
                await result

    @staticmethod
    def _make_connector(config: AppConfig) -> SensorConnector:
        return SerialConnector(config) if config.hardware.source == SensorSource.serial else SimulationConnector(config)
