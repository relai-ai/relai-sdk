from __future__ import annotations

from dataclasses import dataclass

import pytest
from pydantic import BaseModel

from relai.mocker.stateful_mocker import StatefulMocker


@dataclass
class DummyResult:
    final_output: object


class DummyValidationModel(BaseModel):
    pass


@pytest.mark.unit
def test_stateful_mocker_updates_state_in_place(mocker) -> None:
    state = {"count": 1, "status": "old", "extra": 5}
    mocker_instance = StatefulMocker(state_fields=["count", "status"], model="test-model")

    run_sync = mocker.patch(
        "relai.mocker.stateful_mocker.Runner.run_sync",
        side_effect=[
            DummyResult(final_output="tool-output"),
            DummyResult(final_output='{"count": 2, "status": "new", "extra": "ignored"}'),
        ],
    )

    output = mocker_instance._run(state, 10, flag=True)

    assert output == "tool-output"
    assert state["count"] == 2
    assert state["status"] == "new"
    assert state["extra"] == 5
    assert run_sync.call_count == 2


@pytest.mark.unit
def test_stateful_mocker_ignores_invalid_update_payload(mocker) -> None:
    state = {"count": 3, "status": "steady"}
    mocker_instance = StatefulMocker(state_fields=["count", "status"], model="test-model", max_validation_retries=0)

    mocker.patch(
        "relai.mocker.stateful_mocker.Runner.run_sync",
        side_effect=[
            DummyResult(final_output="tool-output"),
            DummyResult(final_output="not-json"),
        ],
    )

    with pytest.raises(ValueError, match="State update validation failed"):
        mocker_instance._run(state)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_stateful_mocker_updates_state_async(mocker) -> None:
    state = {"count": 0, "status": "idle"}
    mocker_instance = StatefulMocker(state_fields=["count", "status"], model="test-model")

    mocker.patch(
        "relai.mocker.stateful_mocker.Runner.run",
        new=mocker.AsyncMock(
            side_effect=[
                DummyResult(final_output="tool-output"),
                DummyResult(final_output={"count": 1}),
            ]
        ),
    )

    output = await mocker_instance._arun(state, "arg")

    assert output == "tool-output"
    assert state == {"count": 1, "status": "idle"}


@pytest.mark.unit
def test_stateful_mocker_serialize_includes_constructor_args() -> None:
    mocker_instance = StatefulMocker(
        state_fields=["count", "status"],
        model="test-model",
        context="extra-context",
        state_schema={"type": "object"},
        output_schema={"type": "string"},
        max_validation_retries=3,
        read_only_state=True,
    )

    serialized = mocker_instance.serialize()

    assert serialized["model"] == "test-model"
    assert serialized["context"] == "extra-context"
    assert serialized["state_fields"] == "count, status"
    assert serialized["state_schema"] == '{"type": "object"}'
    assert serialized["output_schema"] == '{"type": "string"}'
    assert serialized["max_validation_retries"] == "3"
    assert serialized["read_only_state"] == "True"


@pytest.mark.unit
def test_stateful_mocker_read_only_skips_state_update(mocker) -> None:
    state = {"count": 1, "status": "old"}
    mocker_instance = StatefulMocker(state_fields=["count", "status"], read_only_state=True)

    run_sync = mocker.patch(
        "relai.mocker.stateful_mocker.Runner.run_sync",
        return_value=DummyResult(final_output="tool-output"),
    )

    output = mocker_instance._run(state)

    assert output == "tool-output"
    assert state == {"count": 1, "status": "old"}
    assert run_sync.call_count == 1


@pytest.mark.unit
def test_stateful_mocker_retries_on_output_validation_failure(mocker) -> None:
    state = {"count": 1, "status": "old"}
    output_schema = {"type": "string"}
    mocker_instance = StatefulMocker(
        state_fields=["count", "status"],
        output_schema=output_schema,
        max_validation_retries=1,
        read_only_state=True,
    )

    run_sync = mocker.patch(
        "relai.mocker.stateful_mocker.Runner.run_sync",
        side_effect=[
            DummyResult(final_output={"not": "a string"}),
            DummyResult(final_output='"tool-output"'),
        ],
    )

    output = mocker_instance._run(state)

    assert output == "tool-output"
    assert run_sync.call_count == 2


@pytest.mark.unit
def test_stateful_mocker_retries_on_update_validation_failure(mocker) -> None:
    state = {"count": 1, "status": "old"}
    state_schema = {
        "type": "object",
        "properties": {
            "count": {"type": "integer"},
            "status": {"type": "string"},
        },
        "required": ["count", "status"],
    }
    mocker_instance = StatefulMocker(
        state_fields=["count", "status"],
        state_schema=state_schema,
        max_validation_retries=1,
    )

    run_sync = mocker.patch(
        "relai.mocker.stateful_mocker.Runner.run_sync",
        side_effect=[
            DummyResult(final_output="tool-output"),
            DummyResult(final_output='{"count": "oops"}'),
            DummyResult(final_output='{"count": 2}'),
        ],
    )

    output = mocker_instance._run(state)

    assert output == "tool-output"
    assert state == {"count": 2, "status": "old"}
    assert run_sync.call_count == 3


@pytest.mark.unit
def test_stateful_mocker_rejects_conflicting_state_spec() -> None:
    with pytest.raises(ValueError, match="state_model or state_schema"):
        StatefulMocker(
            state_fields=["count"],
            state_model=DummyValidationModel,
            state_schema={"type": "object"},
        )


@pytest.mark.unit
def test_stateful_mocker_rejects_conflicting_output_spec() -> None:
    with pytest.raises(ValueError, match="output_model or output_schema"):
        StatefulMocker(
            state_fields=["count"],
            output_model=DummyValidationModel,
            output_schema={"type": "string"},
        )
