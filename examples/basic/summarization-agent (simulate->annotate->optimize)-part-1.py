# ------------------------------------------------------------
# Prereqs:
#   export RELAI_API_KEY="relai-..."        # your RELAI API key
#   export OPENAI_API_KEY="sk-..."          # if your agent/tool uses OpenAI
#   pip install relai                   # relai
#   pip install openinference-instrumentation-openai-agents  # optional automatic tracing for openai agents SDK
# 
# Here we demonstrate with a simple summarization agent:
# 1. How to run agents in a simulated environment and collect simulation traces/runs.
# 2. How to annotate the simulation runs on RELAI platform (platform.relai.ai) and create an Annotation Benchmark
# 3. (next on `summarization-agent (simulate->annotate->optimize)-part-2.py`) How to optimize the agent over an annotation benchmark.

import asyncio

from agents import Agent, Runner
from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor

from relai import AsyncRELAI, simulated
from relai.data import SimulationTape
from relai.logger import tracer_provider
from relai.maestro import params, register_param
from relai.mocker.persona import Persona
from relai.simulator import AsyncSimulator, random_env_generator

# ---- Observability (optional but recommended) -------------------------------
OpenAIAgentsInstrumentor().instrument(tracer_provider=tracer_provider)


# ============================================================================
# STEP 1 — Decorate inputs/tools that will be simulated
# ============================================================================


@simulated
async def get_user_input():
    msg = input("User: ")
    return msg


# ============================================================================
# STEP 2 — Your agent core
# (additional) To optimize in STEP 5.4, use `register_param` to define tunable
# parameters and `params` to access them in your agent.
# ============================================================================

register_param(
    "model",
    type="model",
    init_value="gpt-4.1-mini",
    desc="LLM model for the agent",
    allowed=["gpt-4o-mini", "gpt-4.1-mini", "gpt-5-mini"],
)
register_param("prompt", type="prompt", init_value="Summarize the given text.", desc="system prompt for the agent")


async def summarization_agent(msg: str):
    agent = Agent(name="Summarization Agent", instructions=params.prompt, model=params.model)
    response = await Runner.run(agent, msg)
    return response.final_output


# ============================================================================
# STEP 3 — Wrap agent for simulation traces
# ============================================================================


async def agent_fn(tape: SimulationTape):
    input = await get_user_input()
    print("User:", input)  # Debug print
    tape.agent_inputs["user_text"] = input  # trace inputs for later auditing
    response = await summarization_agent(input)
    return {"response": response}


# ============================================================================
# STEP 4 — Simulate
# ============================================================================


async def main():
    # 4.1 — Set up your simulation environment
    # Bind Personas/MockTools to fully-qualified function names
    env_generator = random_env_generator(
        {
            "__main__.get_user_input": [
                Persona(user_persona="You have a piece of news to summarize. Include that as part of your message."),
                Persona(
                    user_persona="You have a piece of article to summarize. Include that as part of your message."
                ),
            ]
        }
        # Alternatively, set up a Persona Set through RELAI platform (platform.relai.ai) and use the code below:
        # {"__main__.get_user_input": PersonaSet(persona_set_id="your_persona_set_id_here")}
    )

    # 4.2 — SIMULATE
    async with AsyncRELAI() as client:
        simulator = AsyncSimulator(
            client=client,
            agent_fn=agent_fn,
            env_generator=env_generator,
            log_runs=True,
        )

        agent_logs = await simulator.run(num_runs=4)
        print(agent_logs)

    # 4.3 — ANNOTATE
    # Go to RELAI platform (platform.relai.ai) under ->Results->Runs,
    # click on individual runs to:
    # 1. view and provide feedback to the simulation runs you just executed. 
    # 2. create an Annotation Benchmark from these runs for future optimization
    #    with the "Add to Benchmark" button at the bottom. (IMPORTANT: Make sure
    #    only runs corresponding to the current task are included in the benchmark.

    # 4.4 — OPTIMIZE -> See `2.(annotate->optimize)summarization-agent.py` for next steps


asyncio.run(main())
