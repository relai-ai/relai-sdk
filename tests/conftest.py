import csv
import json
import os
import tempfile
from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio

from relai import AsyncRELAI
from relai.data import AgentLog, RELAISample, SimulationTape


@pytest.fixture(scope="function")
def set_env_vars():
    """
    Fixture to set environment variables for RELAI API key and URL.
    """
    os.environ["RELAI_API_KEY"] = "relai-org-test-key"
    os.environ["RELAI_API_URL"] = "https://test-api.relai.ai"

    yield

    # Cleanup after tests
    del os.environ["RELAI_API_KEY"]
    del os.environ["RELAI_API_URL"]


@pytest_asyncio.fixture(scope="function")
async def relai_client(set_env_vars) -> AsyncGenerator[AsyncRELAI, None]:
    """
    Fixture to provide a RELAI client instance.
    """
    async with AsyncRELAI() as client:
        yield client


@pytest.fixture(scope="module")
def summarization_sample() -> RELAISample:
    """
    Fixture to provide a sample for a summarization benchmark.
    """
    return RELAISample(
        benchmark_id="benchmark-123",
        id="sample-123",
        agent_inputs={
            "source": (
                """The Sun is the star at the centre of the Solar System. It is a massive, nearly perfect sphere of """
                """hot plasma, heated to incandescence by nuclear fusion reactions in its core, radiating the """
                """energy from its surface mainly as visible light and infrared radiation with 10% at ultraviolet """
                """energies. It is by far the most important source of energy for life on Earth. The Sun has been an """
                """object of veneration in many cultures. It has been a central subject for astronomical research """
                """since antiquity."""
            )
        },
        extras={
            "key_facts": {
                "The Sun is the star at the centre of the Solar System.": 10,
                "It is a massive, nearly perfect sphere of hot plasma.": 6,
                "It is heated to incandescence by nuclear fusion reactions in its core.": 8,
                "It radiates energy from its surface mainly as visible light and infrared radiation, with 10% at ultraviolet energies.": 7,
                "It is by far the most important source of energy for life on Earth.": 10,
                "The Sun has been an object of veneration in many cultures.": 5,
                "It has been a central subject for astronomical research since antiquity.": 6,
            },
            "style_rubrics": {
                "Clarity": 5,
                "Conciseness": 4,
                "Engagement": 3,
            },
            "format_rubrics": {
                "Summary should be a list": 5,
            },
        },
    )


@pytest.fixture(scope="module")
def summarization_agent_response(summarization_sample: RELAISample) -> AgentLog:
    """
    Fixture to provide a summarization agent response.
    """
    return AgentLog(
        simulation_tape=SimulationTape(summarization_sample),
        agent_outputs={
            "summary": (
                """The Sun is the star at the center of the Solar System, a massive sphere of hot plasma heated by """
                """nuclear fusion in its core. It radiates energy mainly as visible light and infrared radiation, """
                """with some ultraviolet energy. The Sun is crucial for life on Earth and has been revered in many """
                """cultures, serving as a key subject in astronomy."""
            )
        },
    )


@pytest.fixture(scope="module")
def question_answering_sample() -> RELAISample:
    """
    Fixture to provide a sample for a question answering benchmark.
    """
    return RELAISample(
        benchmark_id="benchmark-123",
        id="sample-123",
        agent_inputs={
            "question": "How many planets are in the Solar System?",
        },
        extras={
            "std_answer": "There are eight planets in the Solar System.",
            "rubrics": {
                "Mention that there are exactly eight planets.": 10,
            },
        },
    )


@pytest.fixture(scope="module")
def question_answering_agent_response(question_answering_sample: RELAISample) -> AgentLog:
    """
    Fixture to provide a question answering agent response.
    """
    return AgentLog(
        simulation_tape=SimulationTape(question_answering_sample),
        agent_outputs={
            "answer": "There are eight planets in the Solar System.",
        },
    )


@pytest.fixture(scope="module")
def sample_csv_file():
    """
    Fixture to create a temporary CSV file with sample data for testing.
    """
    path = None

    def _sample_csv_file(sample: RELAISample):
        nonlocal path
        # Create a temporary CSV file with sample data
        _, path = tempfile.mkstemp(suffix=".csv", text=True)
        with open(path, "w", encoding="utf-8") as f:
            fieldnames = list(sample.agent_inputs.keys()) + list(sample.extras.keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            # Write the sample data
            # Note: Ensure that the agent_inputs and eval_inputs are serialized with json.dumps if they are not strings.

            for _ in range(3):  # 3 duplicates of the same sample
                writer.writerow(
                    {
                        **{k: json.dumps(v) if not isinstance(v, str) else v for k, v in sample.agent_inputs.items()},
                        **{k: json.dumps(v) if not isinstance(v, str) else v for k, v in sample.extras.items()},
                    }
                )
        return path

    yield _sample_csv_file
    # Delete the temporary file
    if os.path.exists(path):  # type: ignore
        os.remove(path)  # type: ignore
