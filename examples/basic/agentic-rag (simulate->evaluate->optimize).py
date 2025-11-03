# ------------------------------------------------------------
# Prereqs:
#   export RELAI_API_KEY="relai-..."        # your RELAI API key
#   export OPENAI_API_KEY="sk-..."          # if your agent/tool uses OpenAI
#   pip install relai                   # relai
#   pip install openinference-instrumentation-openai-agents  # optional tracing
#
# Here we demonstrate with a simple agentic RAG (Retrieval-Augmented Generation) agent:
# 1. How to run agents in a simulated environment and collect simulation traces/runs.
# 2. How to evaluate the agent's performance with custom evaluators.
# 3. How to optimize the agent based on the simulation and evaluation.

import asyncio
import re

from agents import Agent, Runner, function_tool
from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor

from relai import (
    AgentLog,
    AgentOutputs,
    AsyncRELAI,
    AsyncSimulator,
    EvaluatorLog,
    SimulationTape,
    random_env_generator,
)
from relai.critico import Critico
from relai.critico.evaluate import Evaluator
from relai.logger import tracer_provider
from relai.maestro import Maestro, params, register_param
from relai.mocker import MockTool, Persona
from relai.simulator import simulated

# ---- Observability (optional but recommended) -------------------------------
OpenAIAgentsInstrumentor().instrument(tracer_provider=tracer_provider)

AGENT_NAME = "Stock Chatbot"
MODEL = "gpt-5-mini"  # swap as needed

# ============================================================================
# STEP 1 — Decorate inputs/tools that will be simulated
# ============================================================================


@simulated
async def get_user_query() -> str:
    """Get user's query about stock prices."""
    # In a real agent, this function might get input from a chat interface.
    # Since we are simulating this function, we return a fixed query.
    return "What is the current price of AAPL stock?"


@function_tool
@simulated
async def retriever(query: str) -> list[str]:
    """
    A retriever tool that returns relevant financial data for a given query about stock prices.

    Args:
        query (str): A question about stock prices.

    Returns:
        list[str]: A list of relevant financial data.
    """
    # In a real implementation, this function would query a financial database or API.
    # Since we are simulating this tool, we return an empty list.
    return []


# ============================================================================
# STEP 2 — Your agent core
# (additional) To optimize in STEP 5.4, use `register_param` to define tunable
# parameters and `params` to access them in your agent.
# ============================================================================

register_param(
    "prompt",
    type="prompt",
    init_value="You are a helpful assistant for stock price questions.",
    desc="system prompt for the agent",
)


async def stock_price_chatbot(question: str) -> dict[str, str]:
    agent = Agent(
        name=AGENT_NAME,
        instructions=params.prompt,  # access registered parameter
        model=MODEL,
        tools=[retriever],
    )
    result = await Runner.run(agent, question)
    return {"answer": result.final_output}


# ============================================================================
# STEP 3 — Wrap agent for simulation traces
# ============================================================================


async def agent_fn(tape: SimulationTape) -> AgentOutputs:
    question = await get_user_query()
    tape.agent_inputs["question"] = question  # trace inputs for later auditing
    return await stock_price_chatbot(question)


# ============================================================================
# STEP 4 — Define evaluators (Critico)
# ============================================================================


class PriceFormatEvaluator(Evaluator):
    """An illustrative evaluator that checks for correct price formats in the agent's answer."""

    def __init__(self) -> None:
        super().__init__(name="PriceFormatEvaluator", required_fields=["answer"])

    async def compute_evaluator_result(self, agent_log: AgentLog) -> EvaluatorLog:
        bad_pattern = r"\$(?!\d{1,3}(?:,\d{3})+|\d+\.\d{2}\b)\S+"
        bad_prices = re.findall(bad_pattern, agent_log.agent_outputs["answer"])
        score = 0.0 if bad_prices else 1.0
        feedback = (
            ("Incorrect price formats found: " + ", ".join(bad_prices)) if bad_prices else "Price formats look good."
        )
        return EvaluatorLog(evaluator_id=self.uid, name=self.name, outputs={"score": score, "feedback": feedback})


# (You can add built-in RELAI platform evaluators here as well.)


# ============================================================================
# STEP 5 — Orchestrate: simulate → evaluate →  optimize
# ============================================================================


async def main() -> None:
    # 5.1 — Set up your simulation environment
    # Bind Personas/MockTools to fully-qualified function names
    env_generator = random_env_generator(
        config_set={
            "__main__.get_user_query": [Persona(user_persona="A polite and curious user.")],
            "__main__.retriever": [MockTool(model=MODEL, context="The length of the list returned should be at most 3.")],
        }
    )

    async with AsyncRELAI() as client:
        # 5.2 — SIMULATE
        simulator = AsyncSimulator(agent_fn=agent_fn, env_generator=env_generator, client=client)
        agent_logs = await simulator.run(num_runs=1)

        # 5.3 — EVALUATE
        critico = Critico(client=client)
        critico.add_evaluators({PriceFormatEvaluator(): 1.0})
        critico_logs = await critico.evaluate(agent_logs)

        # Publish evaluation report to the RELAI platform
        await critico.report(critico_logs)

        # 5.4 — OPTIMIZE with Maestro
        maestro = Maestro(client=client, agent_fn=agent_fn, log_to_platform=True, name=AGENT_NAME)
        maestro.add_setup(simulator=simulator, critico=critico)
        # one can use multiple simulator+critico setups with different weights by calling `add_setup` multiple times
        # maestro.add_setup(simulator=simulator, critico=critico, weight = 1)
        # maestro.add_setup(simulator=another_simulator, critico=another_critico, weight = 0.5)

        # 5.4.1 — Optimize agent configurations (the parameters registered earlier in STEP 2)
        # params.load("saved_config.json")  # load previous params if available
        await maestro.optimize_config(
            total_rollouts=80,  # Total number of rollouts to use for optimization.
            batch_size=4,  # Base batch size to use for individual optimization steps. Defaults to 4.
            explore_radius=3,  # A positive integer controlling the aggressiveness of exploration during optimization.
            explore_factor=0.5,  # A float between 0 to 1 controlling the exploration-exploitation trade-off.
            verbose=True,  # If True, additional information will be printed during the optimization step.
        )
        params.save("saved_config.json")  # save optimized params for future usage

        # 5.4.2 — Optimize agent structure (changes that cannot be achieved by setting parameters alone)
        await maestro.optimize_structure(
            total_rollouts=5,  # Total number of rollouts to use for optimization.
            code_paths=[
                "agentic-rag (simulate->evaluate->optimize).py"
            ],  # A list of paths corresponding to code implementations of the agent.
            verbose=True,  # If True, additional information will be printed during the optimization step.
        )


if __name__ == "__main__":
    asyncio.run(main())
