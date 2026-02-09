import json
from functools import cached_property
from typing import Any, ClassVar
from uuid import uuid4

from agents import Agent, AgentOutputSchema, Runner, SQLiteSession
from jsonschema import ValidationError as JsonSchemaValidationError
from jsonschema import validate as validate_json_schema
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from ..utils import no_trace
from .base_mocker import BaseMocker


class StatefulMocker(BaseMocker):
    """A mocker that conditions outputs on simulation state and updates that state in-place."""

    prompt_template: ClassVar[str] = (
        """You are an assistant helping with testing another AI agent. Your job is to """
        """mock the output for a tool based on the description of the tool (provided below), """
        """the additional context (if any), and the provided simulation state fields. """
        """Only respond with what the tool might output, without any extraneous phrases """
        """like "Here's the output".\n"""
        """<description>{description}</description>\n"""
        """<additional context>{context}</additional context>\n"""
        """<state fields>{state_fields}</state fields>\n"""
    )

    update_prompt_template: ClassVar[str] = (
        """You are updating a simulation state for testing another AI agent. """
        """Given the tool description, the tool output, and the prior state, """
        """return a JSON object with only the fields that should be updated. """
        """Only include keys from the allowed state fields. Use null to clear a field. """
        """Return JSON only with no extra text.\n"""
        """<description>{description}</description>\n"""
        """<additional context>{context}</additional context>\n"""
        """<allowed state fields>{state_fields}</allowed state fields>\n"""
    )

    def __init__(
        self,
        state_fields: list[str],
        model: str | None = "gpt-5-mini",
        context: str | None = None,
        state_model: type[BaseModel] | None = None,
        state_schema: dict[str, Any] | None = None,
        output_model: type[BaseModel] | None = None,
        output_schema: dict[str, Any] | None = None,
        max_validation_retries: int = 2,
        read_only_state: bool = False,
    ) -> None:
        """
        Initializes the StatefulMocker.

        Args:
            state_fields (list[str]): The simulation_state fields to condition on and update.
            model (str | None): The AI model to use for simulating tool behavior and updates.
            context (str | None): Additional context to guide behavior of the mocker.
            state_model (type[BaseModel] | None): Optional Pydantic model for state snapshot validation.
            state_schema (dict[str, Any] | None): Optional JSON schema for state snapshot validation.
            output_model (type[BaseModel] | None): Optional Pydantic model for tool output validation.
            output_schema (dict[str, Any] | None): Optional JSON schema for tool output validation.
            max_validation_retries (int): Maximum retries when output validation fails.
            read_only_state (bool): If True, skip state updates and keep simulation_state unchanged.
        """
        super().__init__()
        if state_model is not None and state_schema is not None:
            raise ValueError("Only one of state_model or state_schema may be provided.")
        if output_model is not None and output_schema is not None:
            raise ValueError("Only one of output_model or output_schema may be provided.")
        if max_validation_retries < 0:
            raise ValueError("max_validation_retries must be >= 0.")
        self.name = f"stateful-mocker-{uuid4().hex}"
        self.state_fields = list(state_fields)
        self.model = model
        self.context = context
        self.state_model = state_model
        self.state_schema = state_schema
        self.output_model = output_model
        self.output_schema = output_schema
        self.max_validation_retries = max_validation_retries
        self.read_only_state = read_only_state
        self._session = SQLiteSession(self.name)
        self._update_session = SQLiteSession(f"{self.name}-updates")

    @cached_property
    def agent(self) -> Agent:
        return Agent(
            name=self.name,
            instructions=self.prompt_template.format(
                description=self._func_doc,
                context=self.context or "",
                state_fields=", ".join(self.state_fields),
            ),
            model=self.model,
            output_type=AgentOutputSchema(self.output_type, strict_json_schema=False)
            if self.output_type is not None
            else None,
        )

    @cached_property
    def update_agent(self) -> Agent:
        return Agent(
            name=f"{self.name}-updates",
            instructions=self.update_prompt_template.format(
                description=self._func_doc,
                context=self.context or "",
                state_fields=", ".join(self.state_fields),
            ),
            model=self.model,
        )

    def _snapshot_state(self, simulation_state: dict[str, Any]) -> dict[str, Any]:
        snapshot = {field: simulation_state.get(field) for field in self.state_fields}
        self._validate_state_snapshot(snapshot)
        return snapshot

    def _validate_state_snapshot(self, snapshot: dict[str, Any]) -> None:
        if self.state_model is not None:
            try:
                self.state_model.model_validate(snapshot)
            except PydanticValidationError as exc:
                raise ValueError(f"State snapshot validation failed: {exc}") from exc
        elif self.state_schema is not None:
            try:
                validate_json_schema(instance=snapshot, schema=self.state_schema)
            except JsonSchemaValidationError as exc:
                raise ValueError(f"State snapshot validation failed: {exc.message}") from exc

    def _validate_output(self, output: Any) -> str | None:
        if self.output_model is not None:
            try:
                if isinstance(output, BaseModel):
                    return None
                self.output_model.model_validate(output)
            except PydanticValidationError as exc:
                return str(exc)
        elif self.output_schema is not None:
            try:
                validate_json_schema(instance=output, schema=self.output_schema)
            except JsonSchemaValidationError as exc:
                return exc.message
        return None

    def _parse_state_updates(self, raw_output: Any) -> dict[str, Any] | None:
        if isinstance(raw_output, dict):
            return raw_output
        text = str(raw_output).strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return None
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None

    def _apply_state_updates(self, simulation_state: dict[str, Any], updates: dict[str, Any]) -> None:
        for field in self.state_fields:
            if field in updates:
                simulation_state[field] = updates[field]

    def _run_with_validation(
        self,
        simulation_state: dict[str, Any],
        run_sync: bool,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        state_snapshot = self._snapshot_state(simulation_state)
        validation_errors: list[str] = []
        attempts = self.max_validation_retries + 1

        for _ in range(attempts):
            agent_input = str(
                {
                    "args": args,
                    "kwargs": kwargs,
                    "state": state_snapshot,
                    "validation_errors": validation_errors,
                }
            )
            with no_trace():
                if run_sync:
                    result = Runner.run_sync(self.agent, agent_input, session=self._session)
                else:
                    raise RuntimeError("Async path should call _arun_with_validation.")
            output = result.final_output
            error = self._validate_output(output)
            if error is None:
                return output
            validation_errors = [error]

        raise ValueError(f"Output validation failed after {attempts} attempts: {validation_errors[-1]}")

    async def _arun_with_validation(
        self,
        simulation_state: dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        state_snapshot = self._snapshot_state(simulation_state)
        validation_errors: list[str] = []
        attempts = self.max_validation_retries + 1

        for _ in range(attempts):
            agent_input = str(
                {
                    "args": args,
                    "kwargs": kwargs,
                    "state": state_snapshot,
                    "validation_errors": validation_errors,
                }
            )
            with no_trace():
                result = await Runner.run(self.agent, agent_input, session=self._session)
            output = result.final_output
            error = self._validate_output(output)
            if error is None:
                return output
            validation_errors = [error]

        raise ValueError(f"Output validation failed after {attempts} attempts: {validation_errors[-1]}")

    def _run(self, simulation_state: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        output = self._run_with_validation(simulation_state, True, *args, **kwargs)
        if self.read_only_state:
            return output
        state_snapshot = self._snapshot_state(simulation_state)

        update_input = str(
            {
                "args": args,
                "kwargs": kwargs,
                "output": output,
                "state": state_snapshot,
            }
        )
        with no_trace():
            update_result = Runner.run_sync(
                self.update_agent,
                update_input,
                session=self._update_session,
            )
        updates = self._parse_state_updates(update_result.final_output)
        if isinstance(updates, dict):
            self._apply_state_updates(simulation_state, updates)

        return output

    async def _arun(self, simulation_state: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        output = await self._arun_with_validation(simulation_state, *args, **kwargs)
        if self.read_only_state:
            return output
        state_snapshot = self._snapshot_state(simulation_state)

        update_input = str(
            {
                "args": args,
                "kwargs": kwargs,
                "output": output,
                "state": state_snapshot,
            }
        )
        with no_trace():
            update_result = await Runner.run(
                self.update_agent,
                update_input,
                session=self._update_session,
            )
        updates = self._parse_state_updates(update_result.final_output)
        if isinstance(updates, dict):
            self._apply_state_updates(simulation_state, updates)

        return output

    def serialize(self) -> dict[str, str]:
        return {
            "model": self.model or "None",
            "context": self.context or "None",
            "state_fields": ", ".join(self.state_fields),
            "state_model": self.state_model.__name__ if self.state_model is not None else "None",
            "state_schema": json.dumps(self.state_schema) if self.state_schema is not None else "None",
            "output_model": self.output_model.__name__ if self.output_model is not None else "None",
            "output_schema": json.dumps(self.output_schema) if self.output_schema is not None else "None",
            "max_validation_retries": str(self.max_validation_retries),
            "read_only_state": str(self.read_only_state),
        }
