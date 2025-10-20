import inspect
import json
from abc import ABC, abstractmethod
from dataclasses import replace
from functools import cached_property
from itertools import chain
from typing import Any, Callable, ClassVar, Literal, Optional

from agents import Agent, Runner
from pydantic import BaseModel, create_model

from relai import AsyncRELAI

from .._client import get_default_client
from ..data import AgentLog, EvaluatorLog


class Evaluator(ABC):
    """
    Abstract base class for defining and implementing evaluators for a benchmark.

    Evaluators are responsible for assessing an AI agent's response to a specific
    benchmark sample. They can define required input fields from the `AgentLog`
    necessary for their evaluation logic and may incorporate customizable hyperparameters
    to tune their behavior.

    Subclasses must implement the `compute_evaluator_result` method to define
    their specific evaluation logic, which produces an `EvaluatorResponse` object.

    Attributes:
        name (str): The name of the evaluator, used for identification.
        required_fields (list[str]): A list of field names (keys) that must be present in either `agent_inputs`
            (of the sample), `eval_inputs` (of the sample), or `agent_outputs` (of the agent log).
        transform (Callable): An optional callable to transform (pre-process) the `agent_outputs` of the agent
            response for the evaluator. Defaults to None.
        hyperparameters (dict[str, Any]): A dictionary of arbitrary keyword arguments passed during initialization,
            allowing for custom configuration of the evaluator's behavior.
    """

    def __init__(
        self,
        name: str,
        required_fields: Optional[list[str]] = None,
        transform: Optional[Callable] = None,
        **hyperparameters: Any,
    ):
        """
        Args:
            name (str): The display name of the evaluator, used to identify the evaluator in evaluation results.
            required_fields (list[str]): A list of field names (keys) that must be present in either `agent_inputs`
                (of the sample), `eval_inputs` (of the sample), or `agent_outputs` (of the agent log).
            transform (Callable): An optional callable to transform (pre-process) the `agent_outputs` of the agent
                response for the evaluator. Defaults to None.
            hyperparameters (dict[str, Any]): A dictionary of arbitrary keyword arguments passed during
                initialization, allowing for custom configuration of the evaluator's behavior.
        """
        self.name = name
        self.required_fields = required_fields or []
        self.hyperparameters = hyperparameters
        self.transform = transform if transform is not None else lambda x: x

    def _check_fields(self, agent_log: AgentLog) -> None:
        """
        Validates the presence of all `required_fields` within the provided `AgentLog`.

        Args:
            agent_log (AgentLog): The response from the AI agent, containing the original sample
                and agent outputs.

        Raises:
            ValueError: If any of the `required_fields` are missing from `AgentLog`, indicating insufficient
                data for the evaluation.
        """
        for field in self.required_fields:
            if field not in agent_log.simulation_tape.data and field not in agent_log.agent_outputs:
                raise ValueError(f"Required field '{field}' is missing.")

    @abstractmethod
    async def compute_evaluator_result(self, agent_log: AgentLog) -> EvaluatorLog:
        """
        Abstract method: Computes the evaluation result for an agent log.

        Concrete subclasses must implement this method to define their unique evaluation logic. This method should
        process the `AgentLog` by accessing its `sample` and `agent_outputs` to derive the evaluation outcome,
        which is then encapsulated in an `EvaluatorResponse` object.

        Args:
            agent_log (AgentLog): The comprehensive response from the AI agent, including the original
                sample and agent's outputs.

        Returns:
            EvaluatorResponse: An instance of `EvaluatorResponse` containing the evaluation outcome, typically
                including a `score` and/or `feedback`, along with `evaluator_name`, `evaluator_configuration`,
                and the original `agent_log`.

        Raises:
            NotImplementedError: If a concrete subclass does not override this method.
        """
        raise NotImplementedError

    async def __call__(self, agent_log: AgentLog) -> EvaluatorLog:
        """
        Executes the evaluation process for a given AI agent log.

        Args:
            agent_log (AgentLog): The response from the AI agent to be evaluated.

        Returns:
            EvaluatorResponse: The structured result of the evaluation, including any computed `score` and `feedback`,
                as defined by the concrete evaluator.

        Raises:
            TypeError: If `agent_log` is not an instance of `AgentLog` or `agent_outputs` (after transform) in agent_log is not a dict.
            ValueError: If any `required_fields` are missing from the `agent_log`.
        """
        transformed_agent_log = replace(agent_log, agent_outputs=self.transform(agent_log.agent_outputs))
        self._check_fields(transformed_agent_log)
        if inspect.iscoroutinefunction(self.compute_evaluator_result):
            return await self.compute_evaluator_result(transformed_agent_log)
        else:
            return self.compute_evaluator_result(transformed_agent_log)  # pyright: ignore[reportReturnType]

    @cached_property
    def uid(self) -> str:
        """
        Generates a unique identifier for this specific evaluator instance. The UID is constructed from the
        evaluator's class name combined with a JSON-serialized representation of its hyperparameters.

        Returns:
            str: A unique identifier for the evaluator.
        """
        return f"{self.name}_{json.dumps(self.hyperparameters, sort_keys=True, separators=(',', ':'))}"

    def __hash__(self) -> int:
        """
        Computes the hash value for the evaluator based on its unique identifier (`uid`).

        Returns:
            int: The hash value of the evaluator's unique identifier.
        """
        return hash(self.uid)


class RELAIEvaluator(Evaluator):
    """
    Base class for all RELAI evaluators that use the RELAI API to evaluate responses on a benchmark.

    Attributes:
        name (str): The name of the specific evaluator to be invoked on the RELAI platform.
    """

    def __init__(
        self,
        client: AsyncRELAI,
        relai_evaluator_name: str,
        name: str,
        required_fields: Optional[list[str]] = None,
        transform: Optional[Callable] = None,
        **hyperparameters: Any,
    ):
        """
        Initializes a new RELAIEvaluator instance.

        Args:
            client (AsyncRELAI): An instance of the AsyncRELAI client to interact with the RELAI platform.
            relai_evaluator_name (str): The name of the RELAI evaluator to be used for evaluation.
            name (str): The display name of the evaluator, used to identify the evaluator in evaluation results.
            required_fields (list[str], optional): A list of field names that must be present in the `AgentLog`
                (across agent inputs, eval inputs, or agent outputs). Defaults to an empty list.
            transform (Callable): An optional callable to transform (pre-process) the `agent_outputs` of the agent
                response for the evaluator. Defaults to None.
            **hyperparameters (Any): Arbitrary keyword arguments passed to the base `Evaluator` class and also
                forwarded to the RELAI evaluator.
        """
        self._relai_evaluator_name = relai_evaluator_name
        super().__init__(name=name, required_fields=required_fields, transform=transform, **hyperparameters)
        self._client = client

    async def _run_evaluator_on_relai(self, agent_log: AgentLog) -> Any:
        """
        Run the RELAI evaluator on the agent log.

        Args:
            agent_log (AgentLog): The response from the AI agent.

        Returns:
            Any: The response from the RELAI evaluator.
        """
        return await self._client.get_evaluator_response(
            evaluator_name=self._relai_evaluator_name,
            benchmark_id=agent_log.simulation_tape.benchmark_id,
            sample_id=agent_log.simulation_tape.sample_id,
            **agent_log.simulation_tape.data,
            **agent_log.agent_outputs,
            **self.hyperparameters,
        )

    async def compute_evaluator_result(self, agent_log: AgentLog) -> EvaluatorLog:
        """
        Computes the structured evaluation result by invoking a RELAI evaluator.

        Args:
            agent_log (AgentLog): The response from the AI agent.

        Returns:
            EvaluatorResponse: A structured evaluation result containing the evaluator's unique ID, the original
                agent log, and an optional `score` and `feedback` computed by the RELAI evaluator.
        """

        response = await self._run_evaluator_on_relai(agent_log)
        result = EvaluatorLog(
            evaluator_id=self.uid,
            name=self.name,  # Use pretty name for readability on the platform
            config=self.hyperparameters,
            outputs={
                "score": response["score"],
                "feedback": response["feedback"],
            },
        )
        return result


class RELAILengthEvaluator(RELAIEvaluator):
    """
    Evaluator to assess the length of generated text (e.g., summaries) using a RELAI evaluator. Supports evaluating
    length in as measured by number of characters, words, or sentences, or based on the compression ratio.

    **Required fields**:

        - `source`: The original text or document from which the summary is derived.
        - `summary`: The generated summary to be evaluated.
    """

    def __init__(
        self,
        client: AsyncRELAI,
        measure: Literal["characters", "words", "sentences"] = "words",
        use_ratio: bool = False,
        acceptable_range: Optional[tuple[int, int]] = None,
        target_ratio: Optional[float] = None,
        slope: float = 1.0,
        temperature: float = 1.0,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            client (AsyncRELAI): An instance of the AsyncRELAI client to interact with the RELAI platform.
            measure (str, optional): The unit for length calculation; one of:
                - 'characters': count every character,
                - 'words': split on whitespace,
                - 'sentences': split on sentence-ending punctuation (., !, ?).
                Defaults to 'words'.
            use_ratio (bool, optional): If True, ignore `acceptable_range` and instead evaluate the length based on
                the compression ratio: `1 - (summary_length / source_length)` relative to `target_ratio`. Defaults to
                False.
            acceptable_range (tuple[int, int], optional): A two-element tuple `(min_len, max_len)` specifying
                the inclusive bounds for the length of `summary` under the chosen `measure`. Required if
                `use_ratio` is False. Ignored if `use_ratio` is True. Defaults to None.
            target_ratio (float, optional): The desired summary-to-source length ratio (between 0 and 1).
                Required if `use_ratio` is True. Defaults to None.
            slope (float, optional): A factor in [0, 1] controlling the penalty slope for summaries shorter
                than the lower bound. A slope of 1.0 yields a linear ramp from 0 at zero length to 1.0 at
                `min_len`. Defaults to 1.0.
            temperature (float, optional): A positive scaling factor that smooths the exponential penalty
                for summaries exceeding the upper bound. Higher values make the penalty curve flatter.
                Defaults to 1.0.
            transform (Callable): An optional callable to transform (pre-process) the `agent_outputs` of the agent
                response for the evaluator. Defaults to None.

        Raises:
            ValueError: If any of the parameters are invalid.
        """
        if measure not in ["characters", "words", "sentences"]:
            raise ValueError(f"Invalid measure: {measure}. Must be one of 'characters', 'words', or 'sentences'.")
        if not use_ratio and acceptable_range is None:
            raise ValueError("acceptable_range must be provided when use_ratio is False.")
        if use_ratio and target_ratio is None:
            raise ValueError("target_ratio must be provided when use_ratio is True.")
        hyperparameters = {
            "measure": measure,
            "use_ratio": use_ratio,
            "acceptable_range": acceptable_range,
            "compression_ratio": target_ratio,
            "slope": slope,
            "temperature": temperature,
        }
        super().__init__(
            client=client,
            relai_evaluator_name="relai-length-evaluator",
            name="Length Evaluator",
            required_fields=["source", "summary"],
            transform=transform,
            **hyperparameters,
        )


class RELAIContentEvaluator(RELAIEvaluator):
    """
    Evaluator for assessing the factual content of a generated summary against provided key facts, using a RELAI
    evaluator.

    **Required fields**:

        - `key_facts`: A dictionary of key facts with their associated weights, which the summary should cover.
        - `summary`: The generated summary to be evaluated.
    """

    def __init__(
        self,
        client: AsyncRELAI,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            client (AsyncRELAI): An instance of the AsyncRELAI client to interact with the RELAI platform.
            transform (Callable): An optional callable to transform (pre-process) the `agent_outputs` of the agent
                response for the evaluator. Defaults to None.
        """
        super().__init__(
            client=client,
            relai_evaluator_name="relai-content-evaluator",
            name="Content Evaluator",
            required_fields=["key_facts", "summary"],
            transform=transform,
        )


class RELAIHallucinationEvaluator(RELAIEvaluator):
    """
    Evaluator for detecting factual inconsistencies or "hallucinations" in generated text (e.g., summaries) relative
    to a source document, using a RELAI evaluator.

    **Required fields**:

        - `source`: The original text or document from which the summary is derived.
        - `summary`: The generated summary to be evaluated.
    """

    def __init__(
        self,
        client: AsyncRELAI,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            client (AsyncRELAI): An instance of the AsyncRELAI client to interact with the RELAI platform.
            transform (Callable): An optional callable to transform (pre-process) the `agent_outputs` of the agent
                response for the evaluator. Defaults to None.
        """
        super().__init__(
            client=client,
            relai_evaluator_name="relai-hallucination-evaluator",
            name="Hallucination Evaluator",
            required_fields=["source", "summary"],
            transform=transform,
        )


class RELAIStyleEvaluator(RELAIEvaluator):
    """
    Evaluator for assessing the stylistic adherence of a generated summary based on provided rubrics,
    using a RELAI evaluator.

    **Required fields**:

        - `style_rubrics`: A dictionary of style rubrics with their associated weights,
            which the summary should adhere to.
        - `summary`: The generated summary to be evaluated.
    """

    def __init__(
        self,
        client: AsyncRELAI,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            transform (Callable): An optional callable to transform (pre-process) the `agent_outputs` of the agent
                response for the evaluator. Defaults to None.
        """
        super().__init__(
            client=client,
            relai_evaluator_name="relai-style-format-evaluator",
            name="Style Evaluator",
            required_fields=["style_rubrics", "summary"],
            transform=transform,
        )

    async def _run_evaluator_on_relai(self, agent_log: AgentLog) -> Any:
        """
        Run the RELAI evaluator on the agent log.

        Args:
            agent_log (AgentLog): The response from the AI agent.

        Returns:
            Any: The response from the RELAI evaluator.
        """
        return await self._client.get_evaluator_response(
            evaluator_name=self._relai_evaluator_name,
            benchmark_id=agent_log.simulation_tape.benchmark_id,
            sample_id=agent_log.simulation_tape.sample_id,
            summary=agent_log.agent_outputs["summary"],
            style_rubrics=agent_log.simulation_tape.extras["style_rubrics"],
            format_rubrics={"Any format": 1.0},  # TODO: pass empty dict when backend is patched
        )

    async def compute_evaluator_result(self, agent_log: AgentLog) -> EvaluatorLog:  # pyright: ignore[reportIncompatibleMethodOverride]
        """
        Compute the evaluation result based on the agent log using the RELAI evaluator.

        Args:
            agent_log (AgentLog): The response from the AI agent.

        Returns:
            Result: The evaluation result containing score and feedback.
        """

        response = await self._run_evaluator_on_relai(agent_log)
        feedback: str = response["feedback"]
        feedback = feedback.replace(
            "The adherence to style and format rubrics is as follows", "The adherence to style rubrics is as follows"
        )
        result = EvaluatorLog(
            evaluator_id=self.uid,
            name=self.name,  # Use pretty name for readability on the platform
            outputs={
                "score": response["style_score"],
                "feedback": feedback,
            },
            config=self.hyperparameters,
        )
        return result


class RELAIFormatEvaluator(RELAIEvaluator):
    """
    Evaluator for assessing the formatting adherence of a generated summary based on provided rubrics,
    using a RELAI evaluator.

    **Required fields**:

        - `format_rubrics`: A dictionary of format rubrics with their associated weights,
            which the summary should adhere to.
        - `summary`: The generated summary to be evaluated.
    """

    def __init__(
        self,
        client: AsyncRELAI,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            transform (Callable): An optional callable to transform (pre-process) the `agent_outputs` of the agent
                response for the evaluator. Defaults to None.
        """
        super().__init__(
            client=client,
            relai_evaluator_name="relai-style-format-evaluator",
            name="Format Evaluator",
            required_fields=["format_rubrics", "summary"],
            transform=transform,
        )

    async def _run_evaluator_on_relai(self, agent_log: AgentLog) -> Any:
        """
        Run the RELAI evaluator on the agent log.

        Args:
            agent_log (AgentLog): The response from the AI agent.

        Returns:
            Any: The response from the RELAI evaluator.
        """
        return await self._client.get_evaluator_response(
            evaluator_name=self._relai_evaluator_name,
            benchmark_id=agent_log.simulation_tape.benchmark_id,
            sample_id=agent_log.simulation_tape.sample_id,
            summary=agent_log.agent_outputs["summary"],
            format_rubrics=agent_log.simulation_tape.extras["format_rubrics"],
            style_rubrics={"Any style": 1.0},  # TODO: pass empty dict when backend is patched
        )

    async def compute_evaluator_result(self, agent_log: AgentLog) -> EvaluatorLog:  # pyright: ignore[reportIncompatibleMethodOverride]
        """
        Compute the evaluation result based on the agent log using the RELAI evaluator.

        Args:
            agent_log (AgentLog): The response from the AI agent.

        Returns:
            Result: The evaluation result containing score and feedback.
        """

        response = await self._run_evaluator_on_relai(agent_log)
        feedback: str = response["feedback"]
        feedback = feedback.replace(
            "The adherence to style and format rubrics is as follows", "The adherence to format rubrics is as follows"
        )
        result = EvaluatorLog(
            evaluator_id=self.uid,
            name=self.name,  # Use pretty name for readability on the platform
            config=self.hyperparameters,
            outputs={
                "score": response["format_score"],
                "feedback": feedback,
            },
        )
        return result


class RELAIRubricBasedEvaluator(RELAIEvaluator):
    """
    Evaluator for performing a detailed, rubric-driven assessment of an AI agent's
    response to a query using an LLM-based evaluator on the RELAI platform.

    **Required fields**:

        - `question`: The question or prompt that the AI agent was asked to respond to.
        - `answer`: The AI agent's generated response to the question.
        - `rubrics`: A dictionary of evaluation criteria with their associated weights,
            which the answer should satisfy.
        - `std_answer`: The standard or expected answer against which the AI agent's response is evaluated.
    """

    def __init__(
        self,
        client: AsyncRELAI,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            client (AsyncRELAI): An instance of the AsyncRELAI client to interact with the RELAI platform.
            transform (Callable): An optional callable to transform (pre-process) the `agent_outputs` of the agent
                response for the evaluator. Defaults to None.
        """
        super().__init__(
            client=client,
            relai_evaluator_name="llm-evaluator-agent",
            name="Rubric Based Evaluator",
            required_fields=["question", "answer", "rubrics", "std_answer"],
            transform=transform,
        )

    async def _run_evaluator_on_relai(self, agent_log: AgentLog) -> Any:
        """
        Run the RELAI rubric based evaluator on the agent log.

        Args:
            agent_log (AgentLog): The response from the AI agent.

        Returns:
            Any: The response from the RELAI evaluator.
        """
        criteria = {
            "criteria": [
                {"description": k, "points": v}
                for k, v in agent_log.simulation_tape.extras["rubrics"].items()  # type: ignore
            ]
        }
        serialized_rubrics = json.dumps(criteria)
        response = await self._client.get_evaluator_response(
            evaluator_name=self._relai_evaluator_name,
            benchmark_id=agent_log.simulation_tape.benchmark_id,
            sample_id=agent_log.simulation_tape.sample_id,
            questions=[agent_log.simulation_tape.agent_inputs["question"]],
            answers=[agent_log.agent_outputs["answer"]],
            rubrics=[serialized_rubrics],
            std_answers=[agent_log.simulation_tape.extras["std_answer"]],
        )
        return response

    async def compute_evaluator_result(self, agent_log: AgentLog) -> EvaluatorLog:
        """
        Compute the evaluation result based on the agent log using the RELAI evaluator.

        Args:
            agent_log (AgentLog): The response from the AI agent.

        Returns:
            Result: The evaluation result containing score and feedback.
        """

        response = await self._run_evaluator_on_relai(agent_log)
        satisfied_criteria = response.get("satisfied_criteria", [[]])[0]
        rubrics = agent_log.simulation_tape.extras["rubrics"]
        score = 0.0
        total = 0.0
        did_miss_criteria = len(satisfied_criteria) < len(rubrics)
        if did_miss_criteria:
            feedback = "The response does not satisfy the following criteria:\n"
        else:
            feedback = "The response satisfies all criteria."
        for idx, (criterion, weight) in enumerate(rubrics.items()):
            total += weight
            if idx in satisfied_criteria:
                score += weight
            else:
                feedback += f"\t- {criterion}\n"
        score = score / total if total > 0 else 0.0
        result = EvaluatorLog(
            evaluator_id=self.uid,
            name=self.name,  # Use pretty name for readability on the platform
            config=self.hyperparameters,
            outputs={
                "score": score,
                "feedback": feedback.strip(),
            },
        )
        return result


class RELAIAnnotationEvaluator(RELAIEvaluator):
    """
    Evaluator for assessing agent logs based on past human preference annotations
    provided through the RELAI platform.

    **Required fields**:

        - `all_inputs`: The full set of inputs originally supplied to the agent.
        - `previous_outputs`: Prior agent output(s) shown to the human annotator.
        - `desired_outputs`: Human-preferred or target outputs provided by the annotator.
        - `feedback`: Free-text human feedback or rationale provided by the annotator.
        - `liked`: A flag indicating whether the annotator liked the agent's output.
    """

    def __init__(
        self,
        client: AsyncRELAI,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            client (AsyncRELAI): An instance of the AsyncRELAI client to interact with the RELAI platform.
            transform (Callable): An optional callable to transform (pre-process) the `agent_outputs` of the agent
                response for the evaluator. Defaults to None.
        """
        super().__init__(
            client=client,
            relai_evaluator_name="annotation-guided-evaluator",
            name="Annotation Guided Evaluator",
            required_fields=["all_inputs", "previous_outputs", "desired_outputs", "feedback", "liked"],
            transform=transform,
        )

    async def _run_evaluator_on_relai(self, agent_log: AgentLog) -> Any:
        """
        Run the RELAI evaluator on the agent log.

        Args:
            agent_log (AgentLog): The response from the AI agent.

        Returns:
            Any: The response from the RELAI evaluator.
        """

        def to_str(value: Any) -> str:
            if value is None:
                return "not provided"
            return str(value)

        return await self._client.get_evaluator_response(
            evaluator_name=self._relai_evaluator_name,
            benchmark_id=agent_log.simulation_tape.benchmark_id,
            sample_id=agent_log.simulation_tape.sample_id,
            output=to_str(agent_log.agent_outputs),
            input_=to_str(agent_log.simulation_tape.agent_inputs["all_inputs"]),
            previous_output=to_str(agent_log.simulation_tape.extras["previous_outputs"]),
            desired_output=to_str(agent_log.simulation_tape.extras["desired_outputs"]),
            feedback=to_str(agent_log.simulation_tape.extras["feedback"]),
            liked=agent_log.simulation_tape.extras["liked"],
            **self.hyperparameters,
        )


class RELAICustomEvaluator(Evaluator):
    """
    Evaluator for assessing agent logs based on the custom evaluator prompt, input and output formats
    defined on the RELAI platform.

    **Required fields**:

        - Any fields specified in the custom evaluator's input format (on the platform).
    """

    prompt_template: ClassVar[str] = (
        """You are an expert evaluator. Follow the below instructions for evaluation:\n"""
        """<instructions>{instructions}</instructions>\n"""
        """<output_fields>{output_fields}</output_fields>\n"""
        """Evaluate the inputs and output an evaluation of it. The evaluation must have all the fields in """
        """<output_fields> and in the specified format."""
    )

    def __init__(
        self,
        evaluator_id: str,
        model_name: str = "gpt-5-mini",
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            evaluator_id (str): The unique identifier of the custom evaluator defined on the RELAI platform.
            model_name (str, optional): The name of the model to use for the evaluator. Defaults to "gpt-5-mini".
            transform (Callable): An optional callable to transform (pre-process) the `agent_outputs` of the agent
                response for the evaluator. Defaults to None.
        """
        client = get_default_client()
        evaluator_details = client.get_custom_evaluator(evaluator_id)
        evaluator_prompt = evaluator_details["parameters"]["prompt"]
        input_fields = list(evaluator_details["parameters"]["input_format"].keys())
        output_fields = evaluator_details["parameters"]["output_format"]
        super().__init__(name=evaluator_details["name"], required_fields=input_fields, transform=transform)
        self.evaluator_id = evaluator_id
        self._agent = Agent(
            name=self.name,
            model=model_name,
            instructions=self.prompt_template.format(instructions=evaluator_prompt, output_fields=output_fields),
            output_type=RELAICustomEvaluator.create_output_model(output_fields),
        )

    @classmethod
    def create_output_model(cls, output_fields: dict[str, Any]) -> type[BaseModel]:
        fields = {}
        for field_name, field_type in output_fields.items():
            if field_type == "string":
                fields[field_name] = str
            elif field_type == "number":
                fields[field_name] = float
            elif field_type == "boolean":
                fields[field_name] = bool
            elif field_type == "string[]":
                fields[field_name] = list[str]
            elif field_type == "number[]":
                fields[field_name] = list[float]
            else:
                fields[field_name] = Any
        OutputModel = create_model("OutputModel", **fields)
        return OutputModel

    async def compute_evaluator_result(self, agent_log: AgentLog) -> EvaluatorLog:
        evaluator_inputs = {}
        for name, value in chain(agent_log.simulation_tape.data.items(), agent_log.agent_outputs.items()):
            if name in self.required_fields:
                evaluator_inputs[name] = value
        result = await Runner.run(
            self._agent,
            str(evaluator_inputs),
        )
        result = result.final_output
        return EvaluatorLog(
            evaluator_id=self.uid,
            name=self.name,
            config=self.hyperparameters,
            outputs={
                "score": result.score,
                "feedback": result.feedback,
            },
        )
