# ------------------------------------------------------------
# continues from `summarization-agent (simulate->annotate->optimize)-part-1.py`

# Here we demonstrate with a simple summarization agent:
# 1. (previously) How to run agents in a simulated environment and collect simulation traces/runs.
# 2. (previously) How to annotate the simulation runs on RELAI platform (platform.relai.ai) and create an Annotation Benchmark
# 3. How to optimize the agent over an annotation benchmark.

import asyncio

from agents import Agent, Runner
from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor

from relai import AsyncRELAI, simulated
from relai.benchmark import RELAIAnnotationBenchmark
from relai.critico import Critico
from relai.critico.evaluate import RELAIAnnotationEvaluator
from relai.data import RELAISample, SimulationTape
from relai.logger import tracer_provider
from relai.maestro import Maestro, params, register_param
from relai.mocker.persona import Persona
from relai.simulator import AsyncSimulator

OpenAIAgentsInstrumentor().instrument(tracer_provider=tracer_provider)


@simulated
async def get_user_input():
    msg = input("User: ")
    return msg


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


async def agent_fn(tape: SimulationTape):
    input = await get_user_input()
    print("User:", input)  # Debug print
    tape.agent_inputs["user_text"] = input  # trace inputs for later auditing
    response = await summarization_agent(input)
    return {"response": response}


# ============================================================================
# STEP 5 — Simulate->Evalaute->Optimize with annotation benchmark
# ============================================================================
# 5.1 — Load your annotation benchmark created in STEP 4.3 of
# `1.(simulate->annotate)summarization-agent.py`
benchmark = RELAIAnnotationBenchmark(
    benchmark_id="benchmark ID for your annotation benchmark"
)  # replace with your benchmark ID
for sample in benchmark.samples:
    print(sample)  # inspect loaded benchmark samples


# 5.2 — Set up a environment generator that customizes simulation configs based
# on the benchmark samples
def custom_env_generator(sample: RELAISample | None = None):
    if not sample:
        return {}
    return {
        "__main__.get_user_input": Persona(
            user_persona=sample.extras["simulation_config"]["__main__.get_user_input"]["user_persona"],
            starting_message=sample.agent_inputs["all_inputs"][
                "user_text"
            ],  # provide the original user input recorded
        )
    }


async def main():
    async with AsyncRELAI() as client:
        # 5.3 — Set up the simulator with the annotation benchmark and custom env generator
        simulator = AsyncSimulator(
            client=client,
            agent_fn=agent_fn,
            env_generator=custom_env_generator,  # use custom env generator
            benchmark=benchmark,  # use the annotation benchmark for simulation
            log_runs=True,
        )

        # 5.4 — Set up Critico with RELAIAnnotationEvaluator for automatic evaluation of annotation benchmarks
        critico = Critico(client=client)
        critico.add_evaluators(evaluators={RELAIAnnotationEvaluator(client=client): 1})

        # 5.5 — OPTIMIZE with Maestro
        maestro = Maestro(client=client, agent_fn=agent_fn, log_to_platform=True, name="Summarization Agent")
        maestro.add_setup(simulator=simulator, critico=critico)
        # one can use multiple simulator+critico setups with different weights by calling `add_setup` multiple times
        # maestro.add_setup(simulator=simulator, critico=critico, weight = 1)
        # maestro.add_setup(simulator=another_simulator, critico=another_critico, weight = 0.5)

        # 5.5.1 — Optimize agent configurations (the parameters registered earlier in STEP 2)
        # params.load("saved_config.json")  # load previous params if available
        await maestro.optimize_config(
            total_rollouts=10,  # Total number of rollouts to use for optimization.
            batch_size=2,  # Base batch size to use for individual optimization steps. Defaults to 4.
            explore_radius=1,  # A positive integer controlling the aggressiveness of exploration during optimization.
            explore_factor=0.5,  # A float between 0 to 1 controlling the exploration-exploitation trade-off.
            verbose=False,  # If True, additional information will be printed during the optimization step.
        )
        params.save("saved_config.json")  # save optimized params for future usage

        # 5.5.2 — Optimize agent structure (changes that cannot be achieved by setting parameters alone)
        await maestro.optimize_structure(
            total_rollouts=10,  # Total number of rollouts to use for optimization.
            code_paths=[
                "summarization-agent (simulate->annotate->optimize)-part-2.py"
            ],  # A list of paths corresponding to code implementations of the agent.
            verbose=False,  # If True, additional information will be printed during the optimization step.
        )


asyncio.run(main())
