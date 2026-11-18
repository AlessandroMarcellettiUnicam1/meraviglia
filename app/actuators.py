from __future__ import annotations

import asyncio
import operator
import time
from dataclasses import dataclass
from typing import Any, Iterable, Literal

from .schemas import (
    ActuatorEvent,
    ActuatorRule,
    AppConfig,
    PredictionResult,
    RuleOperator,
)
from .storage import ConfigRepository, FileStore


COMPARATORS = {
    RuleOperator.greater_than: operator.gt,
    RuleOperator.greater_or_equal: operator.ge,
    RuleOperator.less_than: operator.lt,
    RuleOperator.less_or_equal: operator.le,
    RuleOperator.equal: operator.eq,
    RuleOperator.not_equal: operator.ne,
}


@dataclass
class RuleRuntimeState:
    """Snapshot needed to undo the exact command that activated a rule."""

    active: bool = False
    rule: ActuatorRule | None = None
    last_prediction: float | int | str | None = None
    execution_mode: Literal["simulated", "serial"] = "simulated"
    last_on_at: float = float("-inf")


class ActuatorReleaseError(RuntimeError):
    """Raised when at least one physical actuator cannot be released safely."""

    def __init__(
        self,
        active_rule_ids: tuple[str, ...],
        events: list[ActuatorEvent],
    ) -> None:
        self.active_rule_ids = active_rule_ids
        self.events = events
        super().__init__(
            "Rilascio attuatori non riuscito per: " + ", ".join(active_rule_ids)
        )


class ActuatorService:
    def __init__(self, store: FileStore, configs: ConfigRepository) -> None:
        self.store = store
        self.configs = configs
        self.sensor_service: Any = None
        self._states: dict[str, RuleRuntimeState] = {}
        self._transition_lock = asyncio.Lock()

    @property
    def active_rule_ids(self) -> tuple[str, ...]:
        """IDs still considered active; useful before a destructive config change."""

        return tuple(
            rule_id for rule_id, state in self._states.items() if state.active
        )

    async def evaluate(
        self,
        prediction: PredictionResult,
        *,
        config_override: AppConfig | None = None,
    ) -> list[ActuatorEvent]:
        """Apply edge-triggered rules to one prediction.

        ``command_on`` is sent only on a false -> true transition. ``command_off``
        is sent only on true -> false. A real activation without ``command_off``
        is rejected and logged, because the service would have no safe release path.
        """

        config = config_override or self.configs.get()
        async with self._transition_lock:
            events = await self._release_stale_states_locked(config)
            real_mode = self._is_real_mode(config)

            for rule in config.actuator_rules:
                if not rule.enabled:
                    continue

                state = self._states.setdefault(rule.id, RuleRuntimeState())
                if state.active and state.rule != rule:
                    # The caller should release before editing, but this prevents an
                    # edited rule from losing the old command_off contract.
                    event = await self._release_state_locked(
                        rule.id,
                        state,
                        real_mode=state.execution_mode == "serial" or real_mode,
                        reason="configurazione della regola modificata",
                    )
                    if event:
                        events.append(event)
                    if state.active:
                        continue

                matched = self._matches(rule, prediction.prediction)
                if matched == state.active:
                    if state.active:
                        state.last_prediction = prediction.prediction
                    continue

                if matched:
                    now = time.monotonic()
                    if now - state.last_on_at < rule.cooldown_seconds:
                        continue
                    if real_mode and not rule.command_off:
                        state.last_on_at = now
                        events.append(
                            self._log_event(
                                rule=rule,
                                prediction=prediction.prediction,
                                command=rule.command_on,
                                execution_mode="serial",
                                success=False,
                                message=(
                                    "Attivazione bloccata: in modalità reale "
                                    "command_off è obbligatorio"
                                ),
                            )
                        )
                        continue

                    success, message = await self._execute_command(
                        rule.command_on, real_mode=real_mode
                    )
                    if success:
                        state.active = True
                        state.rule = rule.model_copy(deep=True)
                        state.last_prediction = prediction.prediction
                        state.execution_mode = "serial" if real_mode else "simulated"
                        state.last_on_at = now
                    events.append(
                        self._log_event(
                            rule=rule,
                            prediction=prediction.prediction,
                            command=rule.command_on,
                            execution_mode="serial" if real_mode else "simulated",
                            success=success,
                            message=message,
                        )
                    )
                    continue

                event = await self._release_state_locked(
                    rule.id,
                    state,
                    real_mode=state.execution_mode == "serial" or real_mode,
                    reason="la condizione non è più vera",
                )
                if event:
                    events.append(event)
            return events

    async def release_active(
        self,
        *,
        rule_ids: Iterable[str] | None = None,
        config_override: AppConfig | None = None,
        reason: str = "rilascio richiesto",
        strict: bool = True,
    ) -> list[ActuatorEvent]:
        """Release active rules before disable/delete/reconfigure/stop.

        Call this *before* persisting the new configuration and before closing the
        serial connector. ``config_override`` should be the old/current config.
        With the safe default ``strict=True`` a failed physical release raises
        :class:`ActuatorReleaseError`; every attempt is still written to the event
        log. Dry-run releases are logged too and always clear their logical state.
        """

        config = config_override or self.configs.get()
        selected = set(rule_ids) if rule_ids is not None else None
        async with self._transition_lock:
            events: list[ActuatorEvent] = []
            configured = {rule.id: rule for rule in config.actuator_rules}
            effective_real_mode = self._is_real_mode(config)

            for rule_id, state in tuple(self._states.items()):
                if not state.active or (selected is not None and rule_id not in selected):
                    continue
                # Prefer the activation snapshot. It contains the command_off that
                # corresponds to the command_on which actually ran.
                if state.rule is None and rule_id in configured:
                    state.rule = configured[rule_id].model_copy(deep=True)
                event = await self._release_state_locked(
                    rule_id,
                    state,
                    real_mode=(
                        state.execution_mode == "serial" or effective_real_mode
                    ),
                    reason=reason,
                )
                if event:
                    events.append(event)

            remaining = tuple(
                rule_id
                for rule_id, state in self._states.items()
                if state.active and (selected is None or rule_id in selected)
            )
            if remaining and strict:
                raise ActuatorReleaseError(remaining, events)
            return events

    async def _release_stale_states_locked(
        self, config: AppConfig
    ) -> list[ActuatorEvent]:
        """Defensive fallback if a caller changed config without release_active."""

        enabled_ids = {rule.id for rule in config.actuator_rules if rule.enabled}
        current_real_mode = self._is_real_mode(config)
        events: list[ActuatorEvent] = []
        for rule_id, state in tuple(self._states.items()):
            if not state.active or rule_id in enabled_ids:
                continue
            event = await self._release_state_locked(
                rule_id,
                state,
                real_mode=state.execution_mode == "serial" or current_real_mode,
                reason="regola disabilitata o rimossa",
            )
            if event:
                events.append(event)
        return events

    async def _release_state_locked(
        self,
        rule_id: str,
        state: RuleRuntimeState,
        *,
        real_mode: bool,
        reason: str,
    ) -> ActuatorEvent | None:
        if not state.active:
            return None
        rule = state.rule
        if rule is None:
            # This should never occur for states created by evaluate. Keep the
            # state active so a strict caller cannot mistake it for released.
            return None

        command = rule.command_off or ""
        prediction = (
            state.last_prediction if state.last_prediction is not None else "release"
        )
        if not command:
            if real_mode:
                return self._log_event(
                    rule=rule,
                    prediction=prediction,
                    command="",
                    execution_mode="serial",
                    success=False,
                    message=(
                        f"Rilascio bloccato ({reason}): command_off mancante; "
                        "lo stato resta attivo"
                    ),
                )
            state.active = False
            state.rule = None
            return self._log_event(
                rule=rule,
                prediction=prediction,
                command="",
                execution_mode="simulated",
                success=True,
                message=f"Rilascio simulato ({reason}); command_off non configurato",
            )

        success, base_message = await self._execute_command(
            command, real_mode=real_mode
        )
        if success:
            state.active = False
            state.rule = None
        return self._log_event(
            rule=rule,
            prediction=prediction,
            command=command,
            execution_mode="serial" if real_mode else "simulated",
            success=success,
            message=f"{base_message} ({reason})",
        )

    async def _execute_command(
        self, command: str, *, real_mode: bool
    ) -> tuple[bool, str]:
        if not real_mode:
            return True, "Comando simulato: nessun hardware è stato azionato"
        if self.sensor_service is None:
            return False, "Invio comando fallito: servizio sensori non disponibile"
        try:
            await self.sensor_service.send_command(command)
            return True, "Comando inviato alla porta seriale"
        except Exception as exc:
            return False, f"Invio comando fallito: {exc}"

    def _log_event(
        self,
        *,
        rule: ActuatorRule,
        prediction: float | int | str,
        command: str,
        execution_mode: Literal["simulated", "serial"],
        success: bool,
        message: str,
    ) -> ActuatorEvent:
        event = ActuatorEvent(
            rule_id=rule.id,
            rule_name=rule.name,
            actuator=rule.actuator,
            prediction=prediction,
            threshold=rule.threshold,
            command=command,
            execution_mode=execution_mode,
            success=success,
            message=message,
        )
        self.store.append_jsonl(
            self.store.events_path, event.model_dump(mode="json")
        )
        return event

    @staticmethod
    def _is_real_mode(config: AppConfig) -> bool:
        return (
            config.hardware.allow_real_actuators
            and config.hardware.source.value == "serial"
        )

    @staticmethod
    def _matches(rule: ActuatorRule, prediction: float | int | str) -> bool:
        left: Any = prediction
        right: Any = rule.threshold
        if isinstance(left, (int, float)) and not isinstance(left, bool):
            try:
                right = float(right)
            except (TypeError, ValueError):
                return False
        else:
            left, right = str(left), str(right)
            if rule.operator not in {RuleOperator.equal, RuleOperator.not_equal}:
                return False
        return bool(COMPARATORS[rule.operator](left, right))

    def recent_events(self, limit: int = 100) -> list[ActuatorEvent]:
        return [
            ActuatorEvent.model_validate(item)
            for item in self.store.read_jsonl(self.store.events_path, limit)
        ]
