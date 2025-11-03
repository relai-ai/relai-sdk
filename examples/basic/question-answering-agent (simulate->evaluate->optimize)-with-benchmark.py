# ------------------------------------------------------------
# Prereqs:
#   export RELAI_API_KEY="relai-..."        # your RELAI API key
#   export GEMINI_API_KEY="AI..."          # if your agent/tool uses Gemini
#   pip install relai                  # relai
#
# Here we demonstrate with a question answering agent:
# 1. How to run agents in a simulated environment based on a csv benchmark and collect simulation traces/runs.
# 2. How to evaluate the agent's performance with custom evaluators that utilize the csv benchmark.
# 3. How to optimize the agent based on the simulation and evaluation.

import asyncio

from google import genai

from relai import (
    AgentLog,
    AgentOutputs,
    AsyncRELAI,
    AsyncSimulator,
    EvaluatorLog,
    SimulationTape,
)
from relai.benchmark import CSVBenchmark
from relai.critico import Critico
from relai.critico.evaluate import Evaluator
from relai.maestro import Maestro, params, register_param

# ============================================================================
# STEP 1 — Load your csv data into a benchmark
# ============================================================================


benchmark = CSVBenchmark(
    csv_file="./assets/sample_questions.csv",
    agent_input_columns=["Question"],
    extra_columns=["Gold answer"],
)


# ============================================================================
# STEP 2 — Your agent core
# (additional) To optimize in STEP 5.4, use `register_param` to define tunable
# parameters and `params` to access them in your agent.
# ============================================================================

register_param(
    "prompt",
    type="prompt",
    init_value="{question}",
    desc="prompt template for the agent",
)

register_param(
    "model",
    type="model",
    init_value="gemini-2.5-flash",
    desc="LLM model for the agent",
    allowed=["gemini-2.5-flash"],  # add more models as needed
)


async def question_answering_agent(question: str) -> dict[str, str]:
    client = genai.Client()
    response = client.models.generate_content(
        model=params.model,  # access registered parameter
        contents=params.prompt.format(question=question),  # access registered parameter
    )
    return response.text  # type: ignore


# ============================================================================
# STEP 3 — Wrap agent for simulation traces
# ============================================================================


async def agent_fn(tape: SimulationTape) -> AgentOutputs:
    question = tape.agent_inputs[
        "Question"
    ]  # read the question from the tape, which originates from the benchmark samples

    # It is good practice to catch exceptions in agent function
    # especially if the agent might raise errors with different configs
    try:
        return {"answer": await question_answering_agent(question)}
    except Exception as e:
        return {"answer": f"Error: {str(e)}"}


# ============================================================================
# STEP 4 — Define evaluators (Critico)
# ============================================================================


class GoldAnswerEvaluator(Evaluator):
    """An illustrative evaluator that checks for the presence of the gold answer in the agent's output."""

    def __init__(self) -> None:
        super().__init__(name="GoldAnswerEvaluator", required_fields=["answer", "Question", "Gold answer"])

    async def compute_evaluator_result(self, agent_log: AgentLog) -> EvaluatorLog:
        gold_answer = str(agent_log.simulation_tape.extras["Gold answer"])
        agent_answer = agent_log.agent_outputs["answer"]
        if gold_answer in agent_answer:
            score = 1.0
            feedback = f"The agent's answer contains the gold answer: {gold_answer}."
        else:
            score = 0.0
            feedback = f"The agent's answer does NOT contain the gold answer: {gold_answer}."
        return EvaluatorLog(evaluator_id=self.uid, name=self.name, outputs={"score": score, "feedback": feedback})


# (You can add built-in RELAI platform evaluators here as well.)


# ============================================================================
# STEP 5 — Orchestrate: simulate → evaluate →  optimize
# ============================================================================


async def main() -> None:
    # 5.1 — Set up your simulation environment
    async with AsyncRELAI() as client:
        # 5.2 — SIMULATE
        simulator = AsyncSimulator(
            agent_fn=agent_fn,
            client=client,
            benchmark=benchmark,  # IMPORTANT: use the csv benchmark for simulation
            log_runs=True,
        )
        agent_logs = await simulator.run(num_runs=1)
        print(agent_logs)

        # 5.3 — EVALUATE
        critico = Critico(client=client)
        critico.add_evaluators({GoldAnswerEvaluator(): 1.0})
        critico_logs = await critico.evaluate(agent_logs)

        # Publish evaluation report to the RELAI platform
        await critico.report(critico_logs)

        # 5.4 — OPTIMIZE with Maestro
        maestro = Maestro(
            client=client, agent_fn=agent_fn, log_to_platform=True, name="Question Answering Agent Example"
        )
        maestro.add_setup(simulator=simulator, critico=critico)
        # one can use multiple simulator+critico setups with different weights by calling `add_setup` multiple times
        # maestro.add_setup(simulator=simulator, critico=critico, weight = 1)
        # maestro.add_setup(simulator=another_simulator, critico=another_critico, weight = 0.5)

        # 5.4.1 — Optimize agent configurations (the parameters registered earlier in STEP 2)
        # params.load("saved_config.json")  # load previous params if available
        await maestro.optimize_config(
            total_rollouts=20,  # Total number of rollouts to use for optimization.
            batch_size=2,  # Base batch size to use for individual optimization steps. Defaults to 4.
            explore_radius=1,  # A positive integer controlling the aggressiveness of exploration during optimization.
            explore_factor=0.5,  # A float between 0 to 1 controlling the exploration-exploitation trade-off.
            verbose=True,  # If True, additional information will be printed during the optimization step.
        )
        params.save("saved_config.json")  # save optimized params for future usage

        # 5.4.2 — Optimize agent structure (changes that cannot be achieved by setting parameters alone)
        await maestro.optimize_structure(
            total_rollouts=10,  # Total number of rollouts to use for optimization.
            code_paths=[
                "question-answering-agent (simulate->evaluate->optimize)-with-benchmark.py"
            ],  # A list of paths corresponding to code implementations of the agent.
            verbose=True,  # If True, additional information will be printed during the optimization step.
        )


if __name__ == "__main__":
    asyncio.run(main())
