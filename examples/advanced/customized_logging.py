# ------------------------------------------------------------
# Prereqs:
#   export OPENAI_API_KEY="sk-..."          # if your agent/tool uses OpenAI
#   pip install relai                       # relai
#   pip install langchain                   # langchain
#   pip install langchain-openai            # langchain-openai

import asyncio
from collections.abc import Callable

from langchain.agents import create_agent

from relai import AgentLog, AsyncRELAI, EvaluatorLog, simulated
from relai.critico import Critico
from relai.critico.evaluate import Evaluator
from relai.data import SimulationTape
from relai.maestro import Maestro, params, register_param
from relai.mocker.persona import Persona
from relai.simulator import AsyncSimulator, random_env_generator
from relai.utils import log_model


@simulated
async def get_user_input(agent_response: str | None = None):
    msg = input("User: ")
    return msg


register_param(
    "model",
    type="model",
    init_value="gpt-4.1-mini",
    desc="LLM model for the agent",
    allowed=["gpt-4o-mini", "gpt-4.1-mini", "gpt-5-mini"],
)
register_param("prompt", type="prompt", init_value="You are a helpful assistant.", desc="system prompt for the agent")


async def chat_agent(messages: list[dict]):
    agent = create_agent(
        model=params.model,
        system_prompt=params.prompt,
    )
    response = agent.invoke({"messages": messages})  # type: ignore
    # If also using openinference instrumentor for automatic tracing,
    # use `no_trace` to avoid double tracing with customized logging as follows:
    # with no_trace():
    #     response = agent.invoke(messages)

    response = response["messages"][-1].content

    # ============================================================================
    # CUSTOMIZED LOGGING for MORE EFFICIENT OPTIMIZATION
    # ============================================================================
    log_model(
        name=params.model,
        # customized logging for more efficient optimization later
        input={
            "messages": ["..."] + messages[-1:],  # customized: logging only the last user message to reduce trace size
            "prompt": "<prompt>",  # customized: logging the name of the registered param instead of actual prompt content
        },
        # naive logging below can result in redundant traces due to repetitive content
        # input={
        #     "messages": messages,
        #     "prompt": params.prompt,
        # },
        output=response,
    )
    return response


async def agent_fn(tape: SimulationTape):
    input = await get_user_input()
    messages = [{"role": "user", "content": input}]
    response = ""
    while "[GOOD]" not in input and "[BAD]" not in input:
        print("User:", input)  # Debug print
        tape.agent_inputs["user_text"] = input  # trace inputs for later auditing
        response = await chat_agent(messages)
        input = await get_user_input(response)
        print("Agent:", response)  # Debug print
        messages.extend([{"role": "assistant", "content": response}, {"role": "user", "content": input}])
    tape.add_record("conversation", messages)  # record full trajectory in tape for evaluation
    return {"response": response}


class ConversationEvaluator(Evaluator):
    """
    A custom evaluator for a conversation.
    """

    def __init__(
        self,
        transform: Callable | None = None,
    ):
        """
        Initialize the custom sentiment evaluator.

        Args:
            transform: Optional function to transform agent outputs
        """
        super().__init__(
            name="conversation-evaluator",
            # Specify required fields from the benchmark and agent response
            required_fields=["conversation"],
            transform=transform,
        )

    async def compute_evaluator_result(self, agent_log: AgentLog) -> EvaluatorLog:
        """
        Evaluate the agent's behavior based on the user feedback at the end of a conversation.

        Args:
            agent_log (AgentLog): The response from the AI agent, containing the original sample
                and agent outputs.

        Returns:
            EvaluatorLog: Evaluator log with score and feedback
        """
        # Extract required fields from different sources
        conversation = agent_log.simulation_tape.extras["conversation"]
        final_user_message = conversation[-1]["content"]

        if "[GOOD]" in final_user_message:
            score = 1.0
            feedback = "The agent was helpful."
        elif "[BAD]" in final_user_message:
            score = 0.0
            feedback = "The agent could do better."
        else:
            raise ValueError("Final user message should contain either [GOOD] or [BAD].")

        return EvaluatorLog(
            evaluator_id=self.uid,
            name=self.name,
            outputs={"score": score, "feedback": feedback},
        )


async def main():
    env_generator = random_env_generator(
        {
            "__main__.get_user_input": [
                Persona(
                    user_persona=(
                        "You have a single random question to ask an agent. Do not ask follow up questions. "
                        "When you think you get your answer and the conversation is over (or that it has been more than 3 turns), "
                        "output [GOOD] if the agent was helpful, [BAD] if the agent could do better."
                    ),
                ),
                Persona(
                    user_persona=(
                        "You want to know what date today is by talking to an agent. "
                        "When you think you get your answer and the conversation is over (or that it has been more than 3 turns),, "
                        "output [GOOD] if the agent was helpful, [BAD] if the agent could do better."
                    )
                ),
            ]
        }
        # Alternatively, set up a Persona Set through RELAI platform (platform.relai.ai) and use the code below:
        # {"__main__.get_user_input": PersonaSet(persona_set_id="your_persona_set_id_here")}
    )

    async with AsyncRELAI() as client:
        simulator = AsyncSimulator(
            client=client,
            agent_fn=agent_fn,
            env_generator=env_generator,
            log_runs=True,
        )

        agent_logs = await simulator.run(num_runs=1)
        print(agent_logs)

        critico = Critico(client=client)
        critico.add_evaluators(evaluators={ConversationEvaluator(): 1})

        # OPTIMIZE with Maestro
        maestro = Maestro(client=client, agent_fn=agent_fn, log_to_platform=True, name="Summarization Agent")
        maestro.add_setup(simulator=simulator, critico=critico)

        # Optimize agent configurations (the parameters registered previously)
        # params.load("saved_config.json")  # load previous params if available
        await maestro.optimize_config(
            total_rollouts=10,  # Total number of rollouts to use for optimization.
            batch_size=1,  # Base batch size to use for individual optimization steps. Defaults to 4.
            explore_radius=1,  # A positive integer controlling the aggressiveness of exploration during optimization.
            explore_factor=0.5,  # A float between 0 to 1 controlling the exploration-exploitation trade-off.
            verbose=True,  # If True, related information will be printed during the optimization step.
        )
        params.save("saved_config.json")  # save optimized params for future usage

        # Optimize agent structure (changes that cannot be achieved by setting parameters alone)
        await maestro.optimize_structure(
            total_rollouts=10,  # Total number of rollouts to use for optimization.
            code_paths=[
                "customized_logging.py"
            ],  # A list of paths corresponding to code implementations of the agent.
            verbose=True,  # If True, related information will be printed during the optimization step.
        )


asyncio.run(main())
