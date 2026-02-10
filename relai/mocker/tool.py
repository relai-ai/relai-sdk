from functools import cached_property
from typing import Any, ClassVar
from uuid import uuid4

from agents import Agent, AgentOutputSchema, Runner, SQLiteSession
from agents.extensions.models.litellm_model import LitellmModel

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
        model: str | LitellmModel = "gpt-5-mini",
        context: str | None = None,
    ):
        """
        Initializes the MockTool with an optional model specification.

        Args:
            model (str | LitellmModel): The AI model to use for simulating the tool's behavior.
                This can be a string identifier for OpenAI models (e.g. gpt-5-mini) or a LitellmModel
                (from agents.extensions.models.litellm_model import LitellmModel). For a full list of
                models supported in LiteLLM, see https://docs.litellm.ai/docs/providers
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
            output_type=AgentOutputSchema(self.output_type, strict_json_schema=False)
            if self.output_type is not None
            else None,
        )

    def _run(self, simulation_state: dict[str, Any], *args, **kwargs):
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

    async def _arun(self, simulation_state: dict[str, Any], *args, **kwargs):
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
            "model": self.model if isinstance(self.model, str) else self.model.model,
            "context": self.context or "None",
        }
