import httpx
import pytest
from pytest_mock import MockerFixture

from relai import AsyncRELAI
from relai._client import is_context_length_exceeded_error, retry_if_not_context_length_exceeded
from relai._exceptions import RELAIError


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
def test_is_context_length_exceeded_error_for_httpx_413_detail():
    request = httpx.Request("POST", "https://test-api.relai.ai/api/v1/enterprise/evaluator/")
    response = httpx.Response(413, request=request, json={"detail": "Context Length Exceeded"})
    error = httpx.HTTPStatusError("context length exceeded", request=request, response=response)
    wrapped_error = RELAIError("HTTP error occurred: 413 - context length exceeded")
    wrapped_error.__cause__ = error

    assert is_context_length_exceeded_error(wrapped_error) is True
    assert retry_if_not_context_length_exceeded(wrapped_error) is False


@pytest.mark.unit
def test_is_context_length_exceeded_error_requires_exact_413_detail():
    request = httpx.Request("POST", "https://test-api.relai.ai/api/v1/enterprise/evaluator/")
    response = httpx.Response(413, request=request, json={"detail": "Different Error"})
    error = httpx.HTTPStatusError("different error", request=request, response=response)
    wrapped_error = RELAIError("HTTP error occurred: 413 - different error")
    wrapped_error.__cause__ = error

    assert is_context_length_exceeded_error(wrapped_error) is False
    assert retry_if_not_context_length_exceeded(wrapped_error) is True


@pytest.mark.unit
def test_is_context_length_exceeded_error_for_async_relai_error_message():
    error = RELAIError(
        'HTTP error occurred: 413 - Request Entity Too Large\nResponse body: {"detail": "Context Length Exceeded"}'
    )

    assert is_context_length_exceeded_error(error) is True
    assert retry_if_not_context_length_exceeded(error) is False


@pytest.mark.unit
def test_is_context_length_exceeded_error_ignores_non_json_response_body():
    error = RELAIError("HTTP error occurred: 413 - Request Entity Too Large\nResponse body: not-json")

    assert is_context_length_exceeded_error(error) is False
    assert retry_if_not_context_length_exceeded(error) is True
