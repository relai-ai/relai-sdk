import pytest
from pytest_mock import MockerFixture

from relai import AsyncRELAI, RELAIError


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
    # Test without API key (should raise an error)
    with pytest.raises(RELAIError):
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
