import pytest
from pytest_mock import MockerFixture

from relai import AsyncRELAI, RELAI
from relai._exceptions import ContextLengthExceededError


@pytest.mark.unit
@pytest.mark.asyncio
async def test_relai_client_initialization_arg_api_key(set_env_vars):
    # Test with API key provided as an argument
    async with AsyncRELAI(api_key="arg_api_key") as client:
        assert client.api_key == "arg_api_key"
        assert client.api_url == "https://test-api.relai.ai"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_relai_client_initialization_env_api_key(set_env_vars):
    # Test with API key provided through environment variable
    async with AsyncRELAI() as client:
        assert client.api_key == "relai-org-test-key"
        assert client.api_url == "https://test-api.relai.ai"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_relai_client_initialization_no_api_key():
    # Test without API key (should not raise an error as client is not making any requests)
    async with AsyncRELAI():
        pass


@pytest.mark.unit
@pytest.mark.asyncio
async def test_relai_client_health_endpoint(relai_client: AsyncRELAI, mocker: MockerFixture):
    mocker.patch(
        "relai._client.AsyncRELAI._request", new=mocker.AsyncMock(return_value={"message": "Server Healthy!"})
    )
    # Test the health endpoint URL
    assert await relai_client.is_connected()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_poll_maestro_task_propagates_context_length_exceeded_error(
    relai_client: AsyncRELAI, mocker: MockerFixture
):
    mocker.patch.object(
        relai_client,
        "_post",
        new=mocker.AsyncMock(return_value={"result": {"status": "FAILURE", "error": "Context Length Exceeded"}}),
    )

    with pytest.raises(ContextLengthExceededError) as exc_info:
        await relai_client.poll_maestro_task("task-123", timeout=1)

    assert str(exc_info.value) == "Maestro task failed: Context Length Exceeded"


@pytest.mark.unit
def test_sync_poll_maestro_task_propagates_context_length_exceeded_error(set_env_vars, mocker: MockerFixture):
    with RELAI() as client:
        mocker.patch.object(
            client,
            "_post",
            return_value={"result": {"status": "FAILURE", "error": "Context Length Exceeded"}},
        )

        with pytest.raises(ContextLengthExceededError) as exc_info:
            client.poll_maestro_task("task-123", timeout=1)

    assert str(exc_info.value) == "Maestro task failed: Context Length Exceeded"
