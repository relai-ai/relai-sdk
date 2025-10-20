import asyncio
import os
import time
from abc import ABC
from typing import Any, Optional

import aiohttp
import httpx

from ._exceptions import RELAIError
from .schema.visual import ConfigOptVizSchema, GraphOptVizSchema


class BaseRELAI(ABC):
    def __init__(self, api_key: Optional[str] = None, api_url: Optional[str] = None):
        if api_key is None:
            api_key = os.getenv("RELAI_API_KEY")
        if api_key is None:
            raise RELAIError(
                "API key must be provided either as an argument or through the `RELAI_API_KEY` environment variable."
            )
        self._api_key = api_key
        if api_url is None:
            api_url = os.getenv("RELAI_API_URL", "https://api.relai.ai")
        self._api_url = api_url
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Token {self._api_key}",
        }


class RELAI(BaseRELAI):
    """
    A client for interacting with the RELAI Platform.

    Args:
        api_key (Optional[str]): Your RELAI API key. You can find this on RELAI Platform:
            Enterprise > Access Management > API Keys. API key can also be set
            as an environment variable `RELAI_API_KEY`.
        api_url (Optional[str]): The base URL for the RELAI API. Defaults to "https://api.relai.ai".
            API URL can also be set as an environment variable `RELAI_API_URL`.

    Raises:
        RELAIError: If the API key is not provided or cannot be found in the environment variables.
    """

    def __init__(self, api_key: Optional[str] = None, api_url: Optional[str] = None):
        super().__init__(api_key=api_key, api_url=api_url)
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Token {self._api_key}",
        }
        self._client = httpx.Client(base_url=self._api_url, headers=self._headers)

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._client.close()

    def _request(self, method: str, url: str, **kwargs: Any) -> Any:
        """Performs an HTTP request and handles common errors."""
        try:
            response = self._client.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            # Re-raise with more context from the response body if possible
            raise RELAIError(f"HTTP error occurred: {e.response.status_code} - {e.response.text}") from e
        except httpx.RequestError as e:
            raise RELAIError(f"Request failed: {e}") from e

    def _get(self, *args, **kwargs) -> Any:
        return self._request("GET", *args, **kwargs)

    def _post(self, *args, **kwargs) -> Any:
        return self._request("POST", *args, **kwargs)

    def _put(self, *args, **kwargs) -> Any:
        return self._request("PUT", *args, **kwargs)

    def is_connected(self) -> bool:
        """Checks if the client can connect to the RELAI API."""

        try:
            self._get("health/")
        except RELAIError:
            return False
        else:
            return True

    def get_benchmark_keywords(self) -> list[str]:
        """
        Retrieves the list of benchmark keywords from the RELAI API.

        Returns:
            list[str]: A list of benchmark keywords.
        """
        response = self._get("api/v1/benchmark/keywords/")
        return response

    def get_benchmark(self, benchmark_id: str) -> list[dict[str, Any]]:
        """
        Retrieves the benchmark data for a given benchmark ID.

        Args:
            benchmark_id (str): The ID of the benchmark to retrieve.

        Returns:
            list[dict[str, Any]]: Samples from the benchmark.

        Raises:
            RELAIError: If the request fails or the benchmark is not found.
        """
        details_url = f"api/v1/benchmarks/all/{benchmark_id}/"

        metadata = self._get(details_url)

        samples_data = self._get(f"api/v1/benchmarks/all/{benchmark_id}/samples/", params={"no_pagination": "true"})

        samples = []
        for s in samples_data:
            split = s.get("split", "All").lower().capitalize()
            sample = {
                "sample_uuid": s["sample_uuid"],
                "split": split,
                **metadata.get("hyperparameters", {}),  # Fields like `style_rubrics`, `format_rubrics`
                **s["column_values"],
            }
            samples.append(sample)
        return samples

    def get_evaluator_response(self, evaluator_name: str, benchmark_id: str, sample_id: str, **kwargs: Any) -> Any:
        """
        Calls a RELAI evaluator with the specified name and parameters.

        Args:
            evaluator_name (str): The name of the evaluator to call.
            **kwargs: Additional parameters to pass to the evaluator.

        Returns:
            Any: The response from the evaluator.

        Raises:
            RELAIError: If the request fails or the evaluator is not found.
        """
        payload = {"evaluator_type": evaluator_name, "benchmark_id": benchmark_id, "sample_id": sample_id, **kwargs}
        response = self._post("api/v1/enterprise/evaluator/", json=payload)
        return response["evaluation_result"]["response"]

    def _execute_maestro_task(self, url: str, data: dict, timeout: int = 1800) -> dict[str, Any]:
        """Executes a Maestro task by POSTing data and polling for the result."""
        response = self._post(url, json=data)
        task_id = response["result"]["task_id"]
        return self.poll_maestro_task(task_id, timeout=timeout)

    def poll_maestro_task(self, task_id: str, timeout: int = 1800) -> Any:
        """
        Poll the status of a Maestro until it is complete or there is an error.
        If and when the task is complete, return the output.

        Args:
            task_id (str): The ID of the Maestro task to poll.
            timeout (int): The maximum time to wait for the task to complete (in seconds).

        Returns:
            Any: The output of the Maestro task if it is complete.

        Raises:
            RELAIError: If the polling request fails.
        """
        payload = {"task_id": task_id}
        sleep_interval = 0.25  # in seconds
        start = time.time()

        while True:
            elapsed = time.time() - start
            if elapsed > timeout:
                raise RELAIError(f"Maestro task timed out after {timeout:.2f} seconds")
            response = self._post("api/v1/maestro/poll-task/", json=payload)
            status = response["result"]["status"]
            if status == "PENDING":
                time.sleep(sleep_interval)
                sleep_interval = min(sleep_interval * 2, 10)  # Exponential backoff, max 10 seconds
                continue
            elif status == "SUCCESS":
                return response["result"]
            else:
                raise RELAIError(f"Maestro task failed: {status}")

    def optimize_structure(self, data: dict) -> str:
        """
        Optimizes the structure of an agent using the RELAI platform.

        Args:
            data (dict): The data required for structure optimization.

        Returns:
            str: The suggestion for optimizing the agent's structure.

        Raises:
            RELAIError: If the optimization request fails.
        """
        response = self._execute_maestro_task("api/v1/maestro/optimize-structure/", data)
        return response["response"]["suggestion"]

    def propose_values(self, data: dict) -> tuple[str, dict[str, str]]:
        """
        Proposes new values for the agent's parameters.

        Args:
            data (dict): The data required for proposing new values.

        Returns:
            tuple[str, dict[str, str]]: A tuple containing the analysis of agent implementation
                and the proposed parameter values.

        Raises:
            RELAIError: If the proposal request fails.
        """
        response = self._execute_maestro_task("api/v1/maestro/propose-values/", data)
        analysis = response["response"]["analysis"]
        proposed_values = response["response"]["proposed_values"]
        return analysis, proposed_values

    def review_values(self, data: dict) -> dict[str, Any]:
        """
        Reviews proposed values for the agent's parameters.

        Args:
            data (dict): The data required for reviewing proposed values.

        Returns:
            dict[str, Any]: The review of the proposed values.

        Raises::
            RELAIError: If the review request fails.
        """
        response = self._execute_maestro_task("api/v1/maestro/review-values/", data)
        return response["response"]["review"]

    def select_version(self, data: dict) -> int:
        """
        Selects the version to evaluate or apply

        Args:
            data (dict): The data to use for selecting the version.

        Returns:
            int: The ID of the selected version.

        Raises:
            RELAIError: If the selection request fails.
        """
        response = self._execute_maestro_task("api/v1/maestro/select-version/", data)
        return response["response"]["version"]

    def upload_trace(self, data: dict[str, Any]) -> Any:
        """
        Uploads OpenInference execution logs of a run to the RELAI platform.

        Args:
            data (dict[str, Any]): A dictionary containing execution log data to upload.

        Returns:
            str: The id for the uploaded execution logs.

        Raises:
            RELAIError: If the upload request fails.
        """
        response = self._post("/api/v1/traces/runs/", json=data)
        return response["trace_uuid"]

    def upload_run(
        self,
        group_id: str,
        trace_id: str,
        agent_inputs: dict[str, Any] | None = None,
        agent_outputs: dict[str, Any] | None = None,
        extras: dict[str, Any] | None = None,
        serialized_simulation_config: dict[str, Any] | None = None,
    ) -> None:
        agent_inputs = agent_inputs or {}
        agent_outputs = agent_outputs or {}
        extras = extras or {}
        serialized_simulation_config = serialized_simulation_config or {}
        self._post(
            "api/v1/agents/agent-runs/",
            json={
                "group_id": group_id,
                "trace": trace_id,
                "agent_inputs": agent_inputs,
                "agent_outputs": agent_outputs,
                "extras": extras,
                "serialized_simulation_config": serialized_simulation_config,
            },
        )

    def upload_critico_log(
        self,
        trace_id: str,
        evaluator_logs: list[dict[str, Any]],
        aggregate_score: float,
        aggregate_feedback: str,
    ) -> None:
        """
        Submits evaluation data to the RELAI platform.

        Args:
            evaluation_data (dict): The evaluation data to submit.

        Returns:
            Any: The response from the submission.

        Raises:
            RELAIError: If the submission fails.
        """
        self._post(
            "api/v1/agents/critico-logs/",
            json={
                "trace": trace_id,
                "evaluator_logs": evaluator_logs,
                "aggregate_score": aggregate_score,
                "aggregate_feedback": aggregate_feedback,
            },
        )

    def update_config_opt_visual(self, config_viz: ConfigOptVizSchema, uuid: str | None = None) -> str:
        """
        Updates the configuration optimization visualization data on the RELAI platform.

        Args:
            config_viz (ConfigOptVizSchema): The configuration optimization visualization data to update.
            uuid (str, optional): The UUID of the configuration optimization visualization to update. If not provided, a new visualization will be created.

        Returns:
            str: The uuid of the updated or created configuration optimization visualization.

        Raises:
            RELAIError: If the update request fails.
        """

        if uuid is None:
            response = self._post("api/v1/maestro/config-optimizer/", json=config_viz.model_dump())
            return response["config_opt_viz_util_uuid"]
        else:
            self._put(f"api/v1/maestro/config-optimizer/{uuid}/", json=config_viz.model_dump())
            return uuid

    def update_graph_opt_visual(self, graph_vis: GraphOptVizSchema, uuid: str | None = None) -> str:
        """
        Updates the graph optimization visualization data on the RELAI platform.

        Args:
            graph_vis (GraphOptVizSchema): The graph optimization visualization data to update.
            uuid (str, optional): The UUID of the graph optimization visualization to update. If not provided, a new visualization will be created.

        Returns:
            str: The uuid of the updated or created graph optimization visualization.

        Raises:
            RELAIError: If the update request fails.
        """

        if uuid is None:
            response = self._post("api/v1/maestro/graph-optimizer/", json=graph_vis.model_dump())
            return response["graph_opt_viz_util_uuid"]
        else:
            self._put(f"api/v1/maestro/graph-optimizer/{uuid}/", json=graph_vis.model_dump())
            return uuid

    def get_persona_set(self, persona_set_id: str) -> list[str]:
        """
        Retrieves the details of a persona set by its ID.

        Args:
            persona_set_id (str): The ID of the persona set to retrieve.

        Returns:
            list[str]: A list of system prompts in the persona set.
        """
        response = self._get(f"api/v1/persona/persona-sets/{persona_set_id}/")
        return response["system_prompts"]

    def get_custom_evaluator(self, evaluator_id: str) -> dict[str, Any]:
        """
        Retrieves the details of a custom evaluator by its ID.

        Args:
            evaluator_id (str): The ID of the evaluator to retrieve.

        Returns:
            dict[str, Any]: A dictionary containing the name and parameters of the evaluator.
        """
        response = self._get(f"api/v1/benchmarks/functions/{evaluator_id}/")
        return {"name": response["name"], "parameters": response["parameters"]}


class AsyncRELAI(BaseRELAI):
    """
    A client for interacting with the RELAI Platform.

    Args:
        api_key (Optional[str]): Your RELAI API key. You can find this on RELAI Platform:
            Enterprise > Access Management > API Keys. API key can also be set
            as an environment variable `RELAI_API_KEY`.
        api_url (Optional[str]): The base URL for the RELAI API. Defaults to "https://api.relai.ai".
            API URL can also be set as an environment variable `RELAI_API_URL`.

    Raises:
        RELAIError: If the API key is not provided or cannot be found in the environment variables.
    """

    def __init__(self, api_key: Optional[str] = None, api_url: Optional[str] = None):
        super().__init__(api_key=api_key, api_url=api_url)
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Token {self._api_key}",
            "Accept-Encoding": "gzip, deflate",
        }
        self._client = aiohttp.ClientSession(base_url=self._api_url, headers=self._headers)

    async def close(self):
        await self._client.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.close()

    async def _request(self, method: str, url: str, **kwargs: Any) -> Any:
        """Performs an HTTP request and handles common errors."""
        try:
            async with self._client.request(method, url, **kwargs) as response:
                try:
                    response.raise_for_status()
                except aiohttp.ClientResponseError as e:
                    # Try to include response body in the error message
                    try:
                        error_text = await response.text()
                    except Exception:
                        error_text = "<failed to read response text>"
                    raise RELAIError(
                        f"HTTP error occurred: {e.status} - {e.message}\nResponse body: {error_text}"
                    ) from e
                return await response.json()

        except aiohttp.ClientError as e:
            raise RELAIError(f"Request failed due to {type(e).__name__}: {repr(e)}") from e

    async def _get(self, *args, **kwargs) -> Any:
        """Performs a GET request."""
        return await self._request("GET", *args, **kwargs)

    async def _post(self, *args, **kwargs) -> Any:
        """Performs a POST request."""
        return await self._request("POST", *args, **kwargs)

    async def _put(self, *args, **kwargs) -> Any:
        """Performs a PUT request."""
        return await self._request("PUT", *args, **kwargs)

    async def is_connected(self) -> bool:
        """Checks if the client can connect to the RELAI API."""

        try:
            await self._get("health/")
        except RELAIError:
            return False
        else:
            return True

    async def get_benchmark_keywords(self) -> list[str]:
        """
        Retrieves the list of benchmark keywords from the RELAI API.

        Returns:
            list[str]: A list of benchmark keywords.
        """
        response = await self._get("api/v1/benchmark/keywords/")
        return response

    async def get_benchmark(self, benchmark_id: str) -> list[dict[str, Any]]:
        """
        Retrieves the benchmark data for a given benchmark ID.

        Args:
            benchmark_id (str): The ID of the benchmark to retrieve.

        Returns:
            list[dict[str, Any]]: Samples from the benchmark.

        Raises:
            RELAIError: If the request fails or the benchmark is not found.
        """
        details_url = f"api/v1/benchmarks/all/{benchmark_id}/"

        metadata = await self._get(details_url)

        samples_data = await self._get(
            f"api/v1/benchmarks/all/{benchmark_id}/samples/", params={"no_pagination": "true"}
        )

        samples = []
        for s in samples_data:
            split = s.get("split", "All").lower().capitalize()
            sample = {
                "sample_uuid": s["sample_uuid"],
                "split": split,
                **metadata.get("hyperparameters", {}),  # Fields like `style_rubrics`, `format_rubrics`
                **s["column_values"],
            }
            samples.append(sample)
        return samples

    async def get_evaluator_response(
        self, evaluator_name: str, benchmark_id: str, sample_id: str, **kwargs: Any
    ) -> Any:
        """
        Calls a RELAI evaluator with the specified name and parameters.

        Args:
            evaluator_name (str): The name of the evaluator to call.
            **kwargs: Additional parameters to pass to the evaluator.

        Returns:
            Any: The response from the evaluator.

        Raises:
            RELAIError: If the request fails or the evaluator is not found.
        """
        payload = {"evaluator_type": evaluator_name, "benchmark_id": benchmark_id, "sample_id": sample_id, **kwargs}
        response = await self._post("api/v1/enterprise/evaluator/", json=payload)
        return response["evaluation_result"]["response"]

    async def _execute_maestro_task(self, url: str, data: dict, timeout: int = 1800) -> dict[str, Any]:
        """Executes a Maestro task by POSTing data and polling for the result."""
        response = await self._post(url, json=data)
        task_id = response["result"]["task_id"]
        return await self.poll_maestro_task(task_id, timeout=timeout)

    async def poll_maestro_task(self, task_id: str, timeout: int = 1800) -> Any:
        """
        Poll the status of a Maestro until it is complete or there is an error.
        If and when the task is complete, return the output.

        Args:
            task_id (str): The ID of the Maestro task to poll.
            timeout (int): The maximum time to wait for the task to complete (in seconds).

        Returns:
            Any: The output of the Maestro task if it is complete.

        Raises:
            RELAIError: If the polling request fails.
        """
        payload = {"task_id": task_id}
        sleep_interval = 0.25  # in seconds
        start = time.time()

        while True:
            elapsed = time.time() - start
            if elapsed > timeout:
                raise RELAIError(f"Maestro task timed out after {timeout:.2f} seconds")
            response = await self._post("api/v1/maestro/poll-task/", json=payload)
            status = response["result"]["status"]
            if status == "PENDING":
                await asyncio.sleep(sleep_interval)
                sleep_interval = min(sleep_interval * 2, 10)  # Exponential backoff, max 10 seconds
                continue
            elif status == "SUCCESS":
                return response["result"]
            else:
                raise RELAIError(f"Maestro task failed: {status}")

    async def optimize_structure(self, data: dict) -> str:
        """
        Optimizes the structure of an agent using the RELAI platform.

        Args:
            data (dict): The data required for structure optimization.

        Returns:
            str: The suggestion for optimizing the agent's structure.

        Raises:
            RELAIError: If the optimization request fails.
        """
        response = await self._execute_maestro_task("api/v1/maestro/optimize-structure/", data)
        return response["response"]["suggestion"]

    async def propose_values(self, data: dict) -> tuple[str, dict[str, str]]:
        """
        Proposes new values for the agent's parameters.

        Args:
            data (dict): The data required for proposing new values.

        Returns:
            tuple[str, dict[str, str]]: A tuple containing the analysis of agent implementation
                and the proposed parameter values.

        Raises:
            RELAIError: If the proposal request fails.
        """
        response = await self._execute_maestro_task("api/v1/maestro/propose-values/", data)
        analysis = response["response"]["analysis"]
        proposed_values = response["response"]["proposed_values"]
        return analysis, proposed_values

    async def review_values(self, data: dict) -> dict[str, Any]:
        """
        Reviews proposed values for the agent's parameters.

        Args:
            data (dict): The data required for reviewing proposed values.

        Returns:
            dict[str, Any]: The review of the proposed values.

        Raises::
            RELAIError: If the review request fails.
        """
        response = await self._execute_maestro_task("api/v1/maestro/review-values/", data)
        return response["response"]["review"]

    async def select_version(self, data: dict) -> int:
        """
        Selects the version to evaluate or apply

        Args:
            data (dict): The data to use for selecting the version.

        Returns:
            int: The ID of the selected version.

        Raises:
            RELAIError: If the selection request fails.
        """
        response = await self._execute_maestro_task("api/v1/maestro/select-version/", data)
        return response["response"]["version"]

    async def upload_trace(self, data: dict[str, Any]) -> str:
        """
        Uploads OpenInference execution logs of a run to the RELAI platform.

        Args:
            data (dict[str, Any]): A dictionary containing execution log data to upload.

        Returns:
            str: The id for the uploaded execution logs.

        Raises:
            RELAIError: If the upload request fails.
        """
        response = await self._post("/api/v1/traces/runs/", json=data)
        return response["trace_uuid"]

    async def upload_run(
        self,
        group_id: str,
        trace_id: str,
        agent_inputs: dict[str, Any] | None = None,
        agent_outputs: dict[str, Any] | None = None,
        extras: dict[str, Any] | None = None,
        serialized_simulation_config: dict[str, Any] | None = None,
    ) -> None:
        agent_inputs = agent_inputs or {}
        agent_outputs = agent_outputs or {}
        extras = extras or {}
        serialized_simulation_config = serialized_simulation_config or {}
        await self._post(
            "api/v1/agents/agent-runs/",
            json={
                "group_id": group_id,
                "trace": trace_id,
                "agent_inputs": agent_inputs,
                "agent_outputs": agent_outputs,
                "extras": extras,
                "serialized_simulation_config": serialized_simulation_config,
            },
        )

    async def upload_critico_log(
        self,
        trace_id: str,
        evaluator_logs: list[dict[str, Any]],
        aggregate_score: float,
        aggregate_feedback: str,
    ) -> None:
        """
        Submits evaluation data to the RELAI platform.

        Args:
            evaluation_data (dict): The evaluation data to submit.

        Returns:
            Any: The response from the submission.

        Raises:
            RELAIError: If the submission fails.
        """
        await self._post(
            "api/v1/agents/critico-logs/",
            json={
                "trace": trace_id,
                "evaluator_logs": evaluator_logs,
                "aggregate_score": aggregate_score,
                "aggregate_feedback": aggregate_feedback,
            },
        )

    async def update_config_opt_visual(self, config_viz: ConfigOptVizSchema, uuid: str | None = None) -> str:
        """
        Updates the configuration optimization visualization data on the RELAI platform.

        Args:
            config_viz (ConfigOptVizSchema): The configuration optimization visualization data to update.
            uuid (str, optional): The UUID of the configuration optimization visualization to update. If not provided, a new visualization will be created.

        Returns:
            str: The uuid of the updated or created configuration optimization visualization.

        Raises:
            RELAIError: If the update request fails.
        """

        if uuid is None:
            response = await self._post("api/v1/maestro/config-optimizer/", json=config_viz.model_dump())
            return response["config_opt_viz_util_uuid"]
        else:
            await self._put(f"api/v1/maestro/config-optimizer/{uuid}/", json=config_viz.model_dump())
            return uuid

    async def update_graph_opt_visual(self, graph_vis: GraphOptVizSchema, uuid: str | None = None) -> str:
        """
        Updates the graph optimization visualization data on the RELAI platform.

        Args:
            graph_viz (GraphOptVizSchema): The graph optimization visualization data to update.
            uuid (str, optional): The UUID of the graph optimization visualization to update. If not provided, a new visualization will be created.

        Returns:
            str: The uuid of the updated or created graph optimization visualization.

        Raises:
            RELAIError: If the update request fails.
        """

        if uuid is None:
            response = await self._post("api/v1/maestro/graph-optimizer/", json=graph_vis.model_dump())
            return response["graph_opt_viz_util_uuid"]
        else:
            await self._put(f"api/v1/maestro/graph-optimizer/{uuid}/", json=graph_vis.model_dump())
            return uuid

    async def get_persona_set(self, persona_set_id: str) -> list[str]:
        """
        Retrieves the details of a persona set by its ID.

        Args:
            persona_set_id (str): The ID of the persona set to retrieve.

        Returns:
            list[str]: A list of system prompts in the persona set.
        """
        response = await self._get(f"api/v1/persona/persona-sets/{persona_set_id}/")
        return response["system_prompts"]

    async def get_custom_evaluator(self, evaluator_id: str) -> dict[str, Any]:
        """
        Retrieves the details of a custom evaluator by its ID.

        Args:
            evaluator_id (str): The ID of the evaluator to retrieve.

        Returns:
            dict[str, Any]: A dictionary containing the name and parameters of the evaluator.
        """
        response = await self._get(f"api/v1/benchmarks/functions/{evaluator_id}/")
        return {"name": response["name"], "parameters": response["parameters"]}


_default_client = RELAI()


def get_default_client() -> RELAI:
    return _default_client


def set_default_client(client: RELAI) -> None:
    global _default_client
    _default_client = client
