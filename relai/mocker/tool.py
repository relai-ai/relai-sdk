from functools import cached_property
from typing import ClassVar
from uuid import uuid4

from agents import Agent, Runner, SQLiteSession

from ..utils import no_trace
from .base_mocker import BaseMocker


class MockTool(BaseMocker):
    """A mocker class for simulating the behavior of a tool used by an AI agent."""

    prompt_template: ClassVar[str] = (
        """You are an assistant helping with testing another AI agent. Your job is to """
        """mock the output for a tool based on the description of the tool (provided below). Only respond """
        """with what the tool might output, without any extraneous phrases like "Here's the output".\n"""
        """<description>{description}</description>\n"""
    )

    def __init__(
        self,
        model: str | None = "gpt-5-mini",
    ):
        """
        Initializes the MockTool with an optional model specification.

        Args:
            model (str | None): The AI model to use for simulating the tool's behavior.
        """
        super().__init__()
        self.name = f"mock-tool-{uuid4().hex}"
        self.model = model
        self._session = SQLiteSession(self.name)

    @cached_property
    def agent(self) -> Agent:
        return Agent(
            name=self.name,
            instructions=self.prompt_template.format(description=self._func_doc),
            model=self.model,
            output_type=self.output_type,
        )

    def _run(self, *args, **kwargs):
        agent_input = str(
            {
                "args": args,
                "kwargs": kwargs,
            }
        )
        with no_trace():
            result = Runner.run_sync(
                self.agent,
                agent_input,
                session=self._session,
            )
        output = result.final_output
        return output

    async def _arun(self, *args, **kwargs):
        agent_input = str(
            {
                "args": args,
                "kwargs": kwargs,
            }
        )
        with no_trace():
            result = await Runner.run(
                self.agent,
                agent_input,
                session=self._session,
            )
        output = result.final_output
        return output
