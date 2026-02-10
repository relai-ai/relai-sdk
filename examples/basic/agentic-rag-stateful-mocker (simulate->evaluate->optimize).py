# ------------------------------------------------------------
# Prereqs:
#   export RELAI_API_KEY="relai-..."        # your RELAI API key
#   export OPENAI_API_KEY="sk-..."          # if your agent/tool uses OpenAI
#   pip install relai                   # relai
#   pip install openinference-instrumentation-openai-agents  # optional tracing
#
# This example demonstrates a StatefulMocker that grounds tool outputs
# in a simulated backing store (e.g., a document database), with optional
# schema validation for both state snapshots and tool output.

import asyncio
import re
from typing import Any

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
from relai.mocker import Persona, StatefulMocker
from relai.simulator import set_simulation_state_field, simulated

# ---- Observability (optional but recommended) -------------------------------
OpenAIAgentsInstrumentor().instrument(tracer_provider=tracer_provider)

AGENT_NAME = "Stateful RAG Chatbot"
MODEL = "gpt-5-mini"  # swap as needed

# ==========================================================================
# STEP 1 — Decorate inputs/tools that will be simulated
# ==========================================================================


@simulated
async def get_user_query() -> str:
    """Get user's query about stock prices."""
    return "What is the current price of AAPL stock?"


@function_tool
@simulated
async def retriever(query: str) -> dict[str, Any]:
    """
    A retriever tool that returns relevant financial data for a given query about stock prices.

    Args:
        query (str): A question about stock prices.

    Returns:
        dict[str, dict]: The retrieved docs plus the last query.
    """
    return {}


# ==========================================================================
# STEP 2 — Your agent core
# ==========================================================================

register_param(
    "prompt",
    type="prompt",
    init_value="You are a helpful assistant for stock price questions. Use the retriever tool to get relevant financial data.",
    desc="system prompt for the agent",
)


async def stock_price_chatbot(question: str) -> dict[str, str]:
    print("Agent received question:", question)

    agent = Agent(
        name=AGENT_NAME,
        instructions=params.prompt,
        model=MODEL,
        tools=[retriever],
    )
    result = await Runner.run(agent, question)
    print("Agent final output:", result.final_output)
    return {"answer": result.final_output}


# ==========================================================================
# STEP 3 — Wrap agent for simulation traces
# ==========================================================================


async def agent_fn(tape: SimulationTape) -> AgentOutputs:
    # Seed a simulated backing store for the retriever.
    # This is the internal tool state that the StatefulMocker will read/update.
    set_simulation_state_field(
        "doc_store",
        [
            {
                "id": "doc-1",
                "title": "AAPL price update",
                "content": "AAPL is trading around $190.21 as of the latest market snapshot.",
                "tags": ["aapl", "price", "market"],
            },
            {
                "id": "doc-2",
                "title": "AAPL earnings summary",
                "content": "Apple reported quarterly earnings with revenue growth driven by services.",
                "tags": ["aapl", "earnings"],
            },
            {
                "id": "doc-3",
                "title": "Macro news",
                "content": "Market volatility increased after the CPI print.",
                "tags": ["macro", "volatility"],
            },
        ],
    )
    set_simulation_state_field("last_query", "")
    set_simulation_state_field("retrieval_stats", {"calls": 0, "cache_hits": 0})

    question = await get_user_query()
    tape.agent_inputs["question"] = question
    return await stock_price_chatbot(question)


# ==========================================================================
# STEP 4 — Define evaluators (Critico)
# ==========================================================================


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


# ==========================================================================
# STEP 5 — Orchestrate: simulate → evaluate → optimize
# ==========================================================================


async def main() -> None:
    # Optional JSON schema validation for state snapshots and tool outputs.
    # These are optional; if provided, StatefulMocker validates and retries on failures.
    state_schema = {
        "type": "object",
        "properties": {
            "doc_store": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "title": {"type": "string"},
                        "content": {"type": "string"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["id", "title", "content", "tags"],
                },
            },
            "last_query": {"type": "string"},
            "retrieval_stats": {
                "type": "object",
                "properties": {
                    "calls": {"type": "integer"},
                    "cache_hits": {"type": "integer"},
                },
                "required": ["calls", "cache_hits"],
            },
        },
        "additionalProperties": False,
        "required": ["doc_store", "last_query", "retrieval_stats"],
    }

    output_schema = {
        "type": "object",
        "properties": {
            "docs": {"type": "array", "items": {"type": "string"}},
            "last_query": {"type": "string"},
        },
        "additionalProperties": False,
        "required": ["docs", "last_query"],
    }

    env_generator = random_env_generator(
        config_set={
            "__main__.get_user_query": [
                Persona(
                    user_persona="A polite and curious user.",
                    starting_message="What is the current price of AAPL stock?",
                )
            ],
            "__main__.retriever": [
                StatefulMocker(
                    model=MODEL,
                    # Context is visible to both output and update agents in StatefulMocker.
                    context=(
                        "Select up to 3 docs from doc_store that best match the query. "
                        "Return a dict with keys: "
                        "{'docs': [<content string for each selected doc>], "
                        "'last_query': <string>}. "
                        "Use the full content field from each selected doc in docs. "
                        "Set last_query in the output to the provided query. "
                        "State updates: set last_query to the query, increment retrieval_stats.calls by 1, "
                        "and increment retrieval_stats.cache_hits by 1 if any selected doc tags match the query. "
                        "Only update fields within state_fields."
                    ),
                    state_fields=["doc_store", "last_query", "retrieval_stats"],
                    state_schema=state_schema,  # optional
                    output_schema=output_schema,  # optional
                    max_validation_retries=2,
                )
            ],
        }
    )

    async with AsyncRELAI() as client:
        simulator = AsyncSimulator(agent_fn=agent_fn, env_generator=env_generator, client=client)
        agent_logs = await simulator.run(num_runs=1)

        critico = Critico(client=client)
        critico.add_evaluators({PriceFormatEvaluator(): 1.0})
        critico_logs = await critico.evaluate(agent_logs)
        await critico.report(critico_logs)

        maestro = Maestro(client=client, agent_fn=agent_fn, log_to_platform=True, name=AGENT_NAME)
        maestro.add_setup(simulator=simulator, critico=critico)

        await maestro.optimize_config(
            total_rollouts=32,
            batch_size=4,
            explore_radius=2,
            explore_factor=0.5,
            verbose=True,
        )
        params.save("saved_config.json")

        await maestro.optimize_structure(
            total_rollouts=3,
            code_paths=[
                "agentic-rag-stateful-mocker (simulate->evaluate->optimize).py",
            ],
            verbose=True,
        )


if __name__ == "__main__":
    asyncio.run(main())
