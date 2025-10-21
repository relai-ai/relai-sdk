from __future__ import annotations

import pytest

from relai.mocker.base_mocker import BaseMocker
from relai.simulator import _simulate, simulated


class SyncDummyMocker(BaseMocker):
    def __init__(self, result: str) -> None:
        super().__init__()
        self.result = result
        self.run_called_with: tuple[tuple[object, ...], dict[str, object]] | None = None

    def _run(self, *args: object, **kwargs: object) -> str:
        self.run_called_with = (args, kwargs)
        return self.result


class AsyncDummyMocker(BaseMocker):
    def __init__(self, result: str) -> None:
        super().__init__()
        self.result = result
        self.arun_called_with: tuple[tuple[object, ...], dict[str, object]] | None = None

    def _run(self, *args: object, **kwargs: object) -> str:
        raise AssertionError("_run should not be invoked for async mocker")

    async def _arun(self, *args: object, **kwargs: object) -> str:
        self.arun_called_with = (args, kwargs)
        return self.result


@simulated
def decorated_sync(a: int, b: int) -> str:
    """Combine two integers into a string."""
    return f"{a}-{b}"


@simulated
async def decorated_async(prefix: str) -> str:
    """Append '-async' to the provided prefix."""
    return f"{prefix}-async"


def test_simulated_decorator_sync_support() -> None:
    # Ensure normal execution path remains unchanged outside simulation mode.
    assert decorated_sync(1, 2) == "1-2"

    mocker = SyncDummyMocker(result="mocked-value")
    func_name = f"{decorated_sync.__module__}.{decorated_sync.__qualname__}"

    with _simulate({func_name: mocker}):
        result = decorated_sync(10, 20)

    assert result == "mocked-value"
    assert mocker.run_called_with == ((10, 20), {})
    assert mocker.func_doc == "Combine two integers into a string."
    assert mocker.output_type is str


@pytest.mark.asyncio
async def test_simulated_decorator_async_support() -> None:
    # Ensure normal execution path remains unchanged outside simulation mode.
    assert await decorated_async("value") == "value-async"

    mocker = AsyncDummyMocker(result="async-mocked")
    func_name = f"{decorated_async.__module__}.{decorated_async.__qualname__}"

    with _simulate({func_name: mocker}):
        result = await decorated_async("prefix")

    assert result == "async-mocked"
    assert mocker.arun_called_with == (("prefix",), {})
    assert mocker.func_doc == "Append '-async' to the provided prefix."
    assert mocker.output_type is str
