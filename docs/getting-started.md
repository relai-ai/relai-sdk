<h2>Installation</h2>

You can install the RELAI SDK with using your favorite Python package manager (requires Python 3.9+):

```bash
pip install relai
# or
uv add relai
```

<h2>Setting the RELAI API key</h2>
A RELAI API key is necessary to use features from the RELAI platform. You can get a RELAI API key from your RELAI enterprise [dashboard](https://platform.relai.ai/settings/access/api-keys). After you copy the key, assign to `RELAI_API_KEY` environment variable:

```bash
export RELAI_API_KEY="relai-..."
```

<h2>Building Reliable AI Agents with RELAI SDK</h2>

<h3>Step 1 — Decorate inputs/tools that will be simulated</h3>

```python
from relai.mocker import Persona, MockTool
from relai.simulator import simulated
from agents import function_tool

AGENT_NAME = "Stock Chatbot"
MODEL = "gpt-5-mini"


# Decorate functions to be mocked in the simulation
@simulated
async def get_user_query() -> str:
    """Get user's query about stock prices."""
    return "What is the current price of AAPL stock?"


@function_tool
@simulated
async def retriever(query: str) -> list[str]:
    """
    A retriever tool that returns relevant financial data for a given query about stock prices.
    """
    return []
```

<h3>Step 2 — Register params to be optimized and define your agent</h3>

```python
from agents import Agent, Runner
from relai.maestro import params, register_param

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
```

<h3>Step 3 — Wrap agent for simulation traces</h3>

```python
from relai import AgentOutputs, SimulationTape

async def agent_fn(tape: SimulationTape) -> AgentOutputs:
    question = await get_user_query()
    tape.agent_inputs["question"] = question  # trace inputs for later auditing
    return await stock_price_chatbot(question)
```

<h3>Step 4 - Define evaluators</h3>

```python
import re
from relai import AgentLog, EvaluatorLog
from relai.critico.evaluate import Evaluator

class PriceFormatEvaluator(Evaluator):
    """Checks for correct price formats ($… with exactly two decimals)."""

    def __init__(self) -> None:
        super().__init__(name="PriceFormatEvaluator", required_fields=["answer"])

    async def compute_evaluator_result(self, agent_log: AgentLog) -> EvaluatorLog:
        # flag $-prices that are NOT like $1,234.56 or $1234.56
        bad_pattern = r"\$(?!\d{1,3}(?:,\d{3})+|\d+\.\d{2}\b)\S+"
        bad_prices = re.findall(bad_pattern, agent_log.agent_outputs["answer"])
        score = 0.0 if bad_prices else 1.0
        feedback = (
            ("Incorrect price formats found: " + ", ".join(bad_prices))
            if bad_prices else
            "Price formats look good."
        )
        return EvaluatorLog(
            evaluator_id=self.uid,
            name=self.name,
            outputs={"score": score, "feedback": feedback},
        )
```

<h3>Step 5 - Orchestrate: simulate → evaluate → optimize</h3>

```python
import asyncio

from relai import AsyncRELAI, AsyncSimulator, random_env_generator
from relai.critico import Critico
from relai.maestro import Maestro, params
from relai.mocker import Persona, MockTool  # (already imported in Step 1 if single file)

async def main() -> None:
    # 5.1 — Set up your simulation environment
    env_generator = random_env_generator(
        config_set={
            "__main__.get_user_query": [Persona(user_persona="A polite and curious user.")],
            "__main__.retriever": [MockTool(model=MODEL)],
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

        # 5.4.1 — Optimize agent configurations
        # params.load("saved_config.json")  # load previous params if available
        await maestro.optimize_config(
            total_rollouts=50,
            batch_size=2,
            explore_radius=5,
            explore_factor=0.5,
            verbose=True,
        )
        params.save("saved_config.json")  # save optimized params for future usage

        # 5.4.2 — Optimize agent structure
        await maestro.optimize_structure(
            total_rollouts=10,
            code_paths=["agentic-rag.py"],
            verbose=True,
        )

if __name__ == "__main__":
    asyncio.run(main())
```