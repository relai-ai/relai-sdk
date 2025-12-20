import asyncio
from collections.abc import Callable, Sequence
from functools import cached_property
from typing import Any, ClassVar, cast, overload
from uuid import uuid4

from agents import Agent, Runner, SQLiteSession, Tool, function_tool
from agents.extensions.models.litellm_model import LitellmModel

from .._client import get_default_client
from ..utils import log_persona, no_trace
from .base_mocker import BaseMocker


class Persona(BaseMocker):
    """
    A mocker class for creating AI personas with specific behaviors and tools.
    A persona can be used to mimic a particular role by defining a system prompt and optionally equipping it with
    tools.

    Attributes:
        user_persona (str): The description of the persona's characteristics and behavior.
        intent (str | None): The intent or goal of the persona during interactions.
        starting_message (str | None): An optional initial message that the persona will use to start interactions.
        tools (Optional[list[Tool]]): A list of tools that the persona can use.
    """

    prompt_template: ClassVar[str] = (
        """You are an assistant helping with testing another AI agent. Your job is to """
        """impersonate a user and interact with the agent based on the following """
        """persona description. The intent of the user is also provided below. Only respond """
        """with what a user would say, without any extraneous phrases like "Here's my prompt".\n"""
        """<intent>{intent}</intent>\n"""
        """<persona>{persona}</persona>\n"""
    )

    additional_prompt_trajectory: ClassVar[str] = (
        """Additionally, you should repeat the exact behavior of {identifier_in_trajectory} """
        """in the following trajectory provided as long as it seems appropriate in the current context. """
        """The goal is to try recreating the same situation recorded in the trajectory.\n"""
        """<trajectory>\n{trajectory}\n</trajectory>\n"""
    )

    def __init__(
        self,
        user_persona: str,
        intent: str | None = None,
        starting_message: str | None = None,
        trajectory: str | None = None,
        identifier_in_trajectory: str = "user",
        tools: list[Callable[..., Any]] | None = None,
        model: str | LitellmModel = "gpt-5-mini",
    ):
        """
        Initializes the Persona with a description, intent, optional starting message, tools, and model.

        Args:
            user_persona (str): The description of the persona's characteristics and behavior.
            intent (str | None): The intent or goal of the persona during interactions.
            starting_message (str | None): An optional initial message that the persona will use to start interactions.
            trajectory (str | None): If provided, the persona will try to mimic the behavior of a certain entity
                (defaults to "user") recorded in the trajectory. The entity to mimic can be configured through
                `identifier_in_trajectory`.
            identifier_in_trajectory (str): A string to identify the entity to mimic when a trajectory is provided.
                Defaults to "user".
            tools (Optional[list[Tool]]): A list of tools that the persona can use.
            model (str | LitellmModel): The AI model to use for simulating the persona's behavior.
                This can be a string identifier for OpenAI models (e.g. gpt-5-mini) or a LitellmModel
                (from agents.extensions.models.litellm_model import LitellmModel). For a full list of
                models supported in LiteLLM, see https://docs.litellm.ai/docs/providers
        """
        super().__init__()
        self.name = f"persona-{uuid4().hex}"
        self.user_persona = user_persona
        self.intent = intent
        self.starting_message = starting_message
        self.trajectory = trajectory
        self.identifier_in_trajectory = identifier_in_trajectory
        tools = tools or []
        self.tools: list[Tool] = [function_tool(tool) for tool in tools]
        self.model = model
        self._session = SQLiteSession(self.name)
        self._start = True

    @cached_property
    def agent(self) -> Agent:
        instructions = self.prompt_template.format(intent=self.intent or self._func_doc, persona=self.user_persona)
        if self.trajectory is not None:
            instructions += self.additional_prompt_trajectory.format(
                identifier_in_trajectory=self.identifier_in_trajectory, trajectory=self.trajectory
            )

        return Agent(
            name=self.name,
            tools=self.tools,
            instructions=instructions,
            model=self.model,
            output_type=self.output_type,
        )

    async def _add_starting_message(self, agent_input: str) -> None:
        await self._session.add_items(
            [
                {"role": "user", "content": agent_input},
                {"role": "assistant", "content": cast(str, self.starting_message)},
            ]
        )

    def _run(self, *args, **kwargs):
        agent_input = str(
            {
                "args": args,
                "kwargs": kwargs,
            }
        )

        if self._start and self.starting_message is not None:
            asyncio.run(self._add_starting_message(agent_input))
            self._start = False
            output = self.starting_message
        else:
            with no_trace():
                result = Runner.run_sync(
                    self.agent,
                    agent_input,
                    session=self._session,
                )
            output = result.final_output

        log_persona(
            f"RELAI Persona ({self.name})",
            self.model if isinstance(self.model, str) else self.model.model,
            agent_input,
            output,
            note=f"Persona id: {self.name}",
        )
        return output

    async def _arun(self, *args, **kwargs):
        agent_input = str(
            {
                "args": args,
                "kwargs": kwargs,
            }
        )

        if self._start and self.starting_message is not None:
            await self._add_starting_message(agent_input)
            self._start = False
            output = self.starting_message
        else:
            with no_trace():
                result = await Runner.run(
                    self.agent,
                    agent_input,
                    session=self._session,
                )
            output = result.final_output

        log_persona(
            f"RELAI Persona ({self.name})",
            self.model if isinstance(self.model, str) else self.model.model,
            agent_input,
            output,
            note=f"Persona id: {self.name}",
        )
        return output

    def serialize(self) -> dict[str, str]:
        return {
            "user_persona": self.user_persona,
            "intent": self.intent or "",
            "starting_message": self.starting_message or "",
        }


class PersonaSet(Sequence[Persona]):
    """
    A collection of Persona instances loaded from a persona set on the RELAI platform.
    """

    def __init__(self, persona_set_id: str, **persona_kwargs: Any) -> None:
        """
        Initializes the PersonaSet with the given persona set ID.

        Args:
            persona_set_id (str): The ID of the persona set on the RELAI platform.
            **persona_kwargs: Keyword arguments that are forwarded to each Persona created from the set.
        """
        self.persona_set_id = persona_set_id
        self._user_personas = None
        self._personas = None
        self._persona_kwargs = persona_kwargs

    def user_personas(self) -> list[str]:
        if self._user_personas is None:
            # Lazy load the personas from the platform
            client = get_default_client()
            self._user_personas = client.get_persona_set(self.persona_set_id)
        return self._user_personas

    def personas(self) -> list[Persona]:
        if self._personas is None:
            self._personas = [
                Persona(user_persona=persona, **self._persona_kwargs) for persona in self.user_personas()
            ]
        return self._personas

    @overload
    def __getitem__(self, index: int) -> Persona: ...

    @overload
    def __getitem__(self, index: slice) -> list[Persona]: ...

    def __getitem__(self, index: int | slice) -> Persona | list[Persona]:
        return self.personas()[index]

    def __len__(self) -> int:
        return len(self.personas())
