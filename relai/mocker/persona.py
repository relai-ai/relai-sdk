import asyncio
from collections.abc import Callable, Sequence
from functools import cached_property
from typing import Any, ClassVar, cast, overload
from uuid import uuid4

from agents import Agent, Runner, SQLiteSession, Tool, function_tool

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

    def __init__(
        self,
        user_persona: str,
        intent: str | None = None,
        starting_message: str | None = None,
        tools: list[Callable[..., Any]] | None = None,
        model: str | None = "gpt-5-mini",
    ):
        """
        Initializes the Persona with a description, intent, optional starting message, tools, and model.

        Args:
            user_persona (str): The description of the persona's characteristics and behavior.
            intent (str | None): The intent or goal of the persona during interactions.
            starting_message (str | None): An optional initial message that the persona will use to start interactions.
            tools (Optional[list[Tool]]): A list of tools that the persona can use.
            model (str | None): The AI model to use for simulating the persona's behavior.
        """
        super().__init__()
        self.name = f"persona-{uuid4().hex}"
        self.user_persona = user_persona
        self.intent = intent
        self.starting_message = starting_message
        tools = tools or []
        self.tools: list[Tool] = [function_tool(tool) for tool in tools]
        self.model = model
        self._session = SQLiteSession(self.name)
        self._start = True

    @cached_property
    def agent(self) -> Agent:
        return Agent(
            name=self.name,
            tools=self.tools,
            instructions=self.prompt_template.format(intent=self.intent or self._func_doc, persona=self.user_persona),
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

        log_persona(f"RELAI Persona ({self.name})", self.model, agent_input, output, note=f"Persona id: {self.name}")
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

        log_persona(f"RELAI Persona ({self.name})", self.model, agent_input, output, note=f"Persona id: {self.name}")
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
