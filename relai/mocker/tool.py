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
        """mock the output for a tool based on the description of the tool (provided below) """
        """and the additional context (if any). Only respond """
        """with what the tool might output, without any extraneous phrases like "Here's the output".\n"""
        """<description>{description}</description>\n"""
        """<additional context>{context}</additional context>\n"""
    )

    def __init__(
        self,
        model: str | None = "gpt-5-mini",
        context: str | None = None,
    ):
        """
        Initializes the MockTool with an optional model specification.

        Args:
            model (str | None): The AI model to use for simulating the tool's behavior.
            context (str | None): Additional context to guide behavior of the mock tool.
        """
        super().__init__()
        self.name = f"mock-tool-{uuid4().hex}"
        self.model = model
        self.context = context
        self._session = SQLiteSession(self.name)

    @cached_property
    def agent(self) -> Agent:
        return Agent(
            name=self.name,
            instructions=self.prompt_template.format(description=self._func_doc, context=self.context or ""),
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

    def serialize(self) -> dict[str, str]:
        return {
            "model": self.model or "None",
            "context": self.context or "None",
        }
