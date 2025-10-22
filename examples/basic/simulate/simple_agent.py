# ------------------------------------------------------------
# Prereqs:
#   export GEMINI_API_KEY="AI..."          # if your agent/tool uses Gemini
#   pip install relai                  # relai
#
# Here we demonstrate with a simple weather bot agent:
# How to run agents in a simulated environment.

import asyncio

from google import genai
from google.genai import types

from relai import (
    AgentOutputs,
    AsyncRELAI,
    AsyncSimulator,
    SimulationTape,
    random_env_generator,
)
from relai.mocker import MockTool, Persona
from relai.simulator import simulated

AGENT_NAME = "Weather Bot"
MODEL = "gemini-2.5-flash"  # swap as needed

# ============================================================================
# STEP 1 — Decorate inputs/tools that will be simulated
# ============================================================================


@simulated
async def get_user_query() -> str:
    """Get user's query about the live weather."""
    # In a real agent, this function might get input from a chat interface.
    # Since we are simulating this function, we return a fixed query.
    return "What's the weather like in Washington DC?"


@simulated
def get_current_weather(location: str) -> str:
    """Returns the current weather.

    Args:
      location: The city and state, e.g. San Francisco, CA
    """
    # In a real implementation, this function would query an external weather API
    # Since we are simulating this tool, we return a fixed weather response.
    return "Sunny"


# ============================================================================
# STEP 2 — Your agent core
# ============================================================================


async def weatherbot(question: str) -> dict[str, str]:
    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=question,
        config=types.GenerateContentConfig(tools=[get_current_weather]),
    )

    return {"response": response.text}  # type: ignore


# ============================================================================
# STEP 3 — Wrap agent for simulation traces
# ============================================================================


async def agent_fn(tape: SimulationTape) -> AgentOutputs:
    question = await get_user_query()
    tape.agent_inputs["question"] = question  # trace inputs for later auditing
    return await weatherbot(question)


# ============================================================================
# STEP 4 — Simulate
# ============================================================================


async def main() -> None:
    # 4.1 — Set up your simulation environment
    # Bind Personas/MockTools to fully-qualified function names
    env_generator = random_env_generator(
        config_set={
            "__main__.get_user_query": [Persona(user_persona="A polite and curious user.")],
            "__main__.get_current_weather": [MockTool(model="gemini/gemini-2.5-flash")],
        }
    )

    async with AsyncRELAI() as client:
        # 4.2 — SIMULATE
        simulator = AsyncSimulator(agent_fn=agent_fn, env_generator=env_generator, client=client)
        agent_logs = await simulator.run(num_runs=1)
        print(agent_logs)


if __name__ == "__main__":
    asyncio.run(main())
