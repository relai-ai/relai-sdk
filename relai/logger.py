import json
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from enum import Enum
from typing import Any, Iterator, Mapping, Optional, Union, cast

from openinference.instrumentation import TracerProvider
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes, ToolCallAttributes
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.util.types import AttributeValue

from .exporter import InMemorySpanExporter
from .flags import get_current_tracking

exporter = InMemorySpanExporter()
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
"""
Tracer provider to be used for openinference instrumentation.
"""


def flatten(mapping: Mapping[str, Any]) -> Iterator[tuple[str, AttributeValue]]:
    for key, value in mapping.items():
        if value is None:
            continue
        if isinstance(value, Mapping):
            for sub_key, sub_value in flatten(value):
                yield f"{key}.{sub_key}", sub_value
        elif isinstance(value, list) and any(isinstance(item, Mapping) for item in value):
            for index, sub_mapping in enumerate(value):
                if isinstance(sub_mapping, Mapping):
                    for sub_key, sub_value in flatten(sub_mapping):
                        yield f"{key}.{index}.{sub_key}", sub_value
                else:
                    yield f"{key}.{index}", sub_mapping
        else:
            if isinstance(value, Enum):
                value = value.value
            yield key, value


def unflatten_attributes(flat: dict[str, Any], prefix: str = "") -> Union[dict[str, Any], list[Any], Any]:
    """
    Convert flattened dot-notation keys into nested dict/list structures.
    Numeric path elements become list indices.
    """
    root: dict[Any, Any] = {}

    for full_key, value in flat.items():
        if prefix:
            if not full_key.startswith(prefix + "."):
                continue
            key = full_key[len(prefix) + 1 :]
        else:
            key = full_key

        parts = key.split(".")
        node = root
        for part in parts[:-1]:
            is_index = part.isdigit()
            idx = int(part) if is_index else part

            if is_index:
                node.setdefault("_list", {})
                node = node["_list"].setdefault(idx, {})
            else:
                node = node.setdefault(part, {})

        leaf = parts[-1]
        if leaf.isdigit():
            idx = int(leaf)
            node.setdefault("_list", {})[idx] = value
        else:
            node[leaf] = value

    def finalize(x: Any) -> Any:
        if isinstance(x, dict):
            if "_list" in x:
                # Turn the stored integer keys into a list
                items = x["_list"]
                return [finalize(items[i]) for i in sorted(items)]
            return {k: finalize(v) for k, v in x.items()}
        return x

    return finalize(root)


def convert_traces(spans: tuple[ReadableSpan, ...], filter_tag: str) -> list[dict]:
    """
    Convert openinference traces to Maestro traces.
    """

    maestro_traces = []

    related_spans = [span for span in spans if filter_tag in span.attributes["tag.tags"]]  # type: ignore

    for span in related_spans:
        attributes = cast(dict[str, Any], span.attributes)
        if attributes["openinference.span.kind"] == "LLM":
            if attributes.get("relai.kind", "llm") == "llm":
                maestro_traces.append(
                    {
                        "type": "model_calling",
                        "model_name": attributes["llm.model_name"],
                        "model_input": unflatten_attributes(attributes, "llm.input_messages"),
                        "model_output": unflatten_attributes(attributes, "llm.output_messages"),
                        "note": attributes.get("relai.note"),
                    }
                )
            elif attributes.get("relai.kind", "llm") == "router":
                maestro_traces.append(
                    {
                        "type": "router",
                        "router_name": attributes["llm.model_name"],
                        "router_input": unflatten_attributes(attributes, "llm.input_messages"),
                        "router_output": unflatten_attributes(attributes, "llm.output_messages"),
                        "note": attributes.get("relai.note"),
                    }
                )
        elif attributes["openinference.span.kind"] == "EMBEDDING":
            "TODO: revisit logging options; further testing required"
        elif attributes["openinference.span.kind"] == "CHAIN":
            "TODO: revisit logging options; further testing required"
        elif attributes["openinference.span.kind"] == "RETRIEVER":
            "TODO: revisit logging options; further testing required"
        elif attributes["openinference.span.kind"] == "RERANKER":
            "TODO: revisit logging options; further testing required"
        elif attributes["openinference.span.kind"] == "TOOL":
            "TODO: tool_output is not automatically instrumented"
            maestro_traces.append(
                {
                    "type": "tool_calling",
                    "tool_name": attributes.get("tool_call.function.name"),
                    "tool_input": attributes.get("tool_call.function.arguments"),
                    "tool_output": attributes.get("relai.tool_call.output"),
                    "note": attributes.get("relai.note"),
                }
            )
        elif attributes["openinference.span.kind"] == "AGENT":
            "TODO: revisit logging options; further testing required"

    return maestro_traces


class Logger:
    """
    Logger to gathering execution traces of agents.

    Logs are only collected when `relai_tracking_flag` is enabled (disabled by default and
    can be controlled with `relai.flags.tracking_on` and `relai.flags.tracking_off`).
    """

    def __init__(self, logger_id: str = "default"):
        """
        Initializes the logger with empty logs and tracking structures.
        """
        self._spans = defaultdict(list)
        self._component_log = {}
        self._component_active_params = defaultdict(set)
        self._param_order_store = {}
        self._param_order_counter_store = 0
        self._logger_id = logger_id

    def clear(self):
        """
        Clears all logs and resets tracking state.
        """
        self._spans = defaultdict(list)
        self._component_log = {}
        self._component_active_params = defaultdict(set)
        self._param_order_store = {}
        self._param_order_counter_store = 0
        exporter.clear()

    def log_model(self, name: str, input: Any, output: Any, note: Optional[str] = None):
        """
        Logs a model call event.

        Args:
            name (str): Name of the model.
            input (Any): Input to the model.
            output (Any): Output from the model.
            note (Optional[str]): Optional annotation.
        """
        if get_current_tracking():
            span_id = span_id_var.get()
            self._spans[span_id].append(
                {
                    "type": "model_calling",
                    "model_name": name,
                    "model_input": input,
                    "model_output": output,
                    "note": note,
                }
            )

        tracer = tracer_provider.get_tracer(__name__)
        with tracer.start_as_current_span(name) as span:
            span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.LLM.value)
            if isinstance(input, Mapping):
                for attribute, value in flatten(input):
                    span.set_attribute(
                        SpanAttributes.LLM_INPUT_MESSAGES + "." + attribute, value
                    )  # List of messages sent to the LLM in a chat API request, [{"message.role": "user", "message.content": "hello"}]
            else:
                span.set_attribute(SpanAttributes.LLM_INPUT_MESSAGES, input)
            if isinstance(output, Mapping):
                for attribute, value in flatten(output):
                    span.set_attribute(
                        SpanAttributes.LLM_OUTPUT_MESSAGES + "." + attribute, value
                    )  # Messages received from a chat API, [{"message.role": "user", "message.content": "hello"}]
            else:
                span.set_attribute(SpanAttributes.LLM_OUTPUT_MESSAGES, output)
            span.set_attribute(SpanAttributes.LLM_MODEL_NAME, name)  # The name of the language model being utilized
            if note is not None:
                span.set_attribute("relai.note", note)  # The note provided by user

    def log_persona(self, name: str, model_name: str, input: Any, output: Any, note: Optional[str] = None):
        """
        Logs a persona activity.

        Args:
            name (str): Name of the persona.
            model_name (str): Name of the model.
            input (Any): Input to the persona.
            output (Any): Output from the persona.
            note (Optional[str]): Optional annotation.
        """
        if get_current_tracking():
            span_id = span_id_var.get()
            self._spans[span_id].append(
                {
                    "type": "persona",
                    "model_name": model_name,
                    "model_input": input,
                    "model_output": output,
                    "note": note,
                }
            )

        tracer = tracer_provider.get_tracer(__name__)
        with tracer.start_as_current_span(name) as span:
            span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.LLM.value)
            if isinstance(input, Mapping):
                for attribute, value in flatten(input):
                    span.set_attribute(
                        SpanAttributes.LLM_INPUT_MESSAGES + "." + attribute, value
                    )  # List of messages sent to the LLM in a chat API request, [{"message.role": "user", "message.content": "hello"}]
            else:
                span.set_attribute(SpanAttributes.LLM_INPUT_MESSAGES, input)
            if isinstance(output, Mapping):
                for attribute, value in flatten(output):
                    span.set_attribute(
                        SpanAttributes.LLM_OUTPUT_MESSAGES + "." + attribute, value
                    )  # Messages received from a chat API, [{"message.role": "user", "message.content": "hello"}]
            else:
                span.set_attribute(SpanAttributes.LLM_OUTPUT_MESSAGES, output)
            span.set_attribute(
                SpanAttributes.LLM_MODEL_NAME, model_name
            )  # The name of the language model being utilized
            if note is not None:
                span.set_attribute("relai.note", note)  # The note provided by user

    def log_tool(self, name: str, input: Any, output: Any, note: Optional[str] = None):
        """
        Logs a tool call event.

        Args:
            name (str): Name of the tool.
            input (Any): Input to the tool.
            output (Any): Output from the tool.
            note (Optional[str]): Optional annotation.
        """
        if get_current_tracking():
            span_id = span_id_var.get()
            self._spans[span_id].append(
                {
                    "type": "tool_calling",
                    "tool_name": name,
                    "tool_input": input,
                    "tool_output": output,
                    "note": note,
                }
            )

        tracer = tracer_provider.get_tracer(__name__)
        with tracer.start_as_current_span(name) as span:
            span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.TOOL.value)
            span.set_attribute(ToolCallAttributes.TOOL_CALL_FUNCTION_NAME, name)  # The name of the tool being utilized

            if isinstance(input, Mapping):
                for attribute, value in flatten(input):
                    span.set_attribute(
                        ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON + "." + attribute, value
                    )  # The arguments for the function being invoked by a tool call
            else:
                span.set_attribute(ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON, input)

            if isinstance(output, Mapping):
                for attribute, value in flatten(output):
                    span.set_attribute("relai.tool_call.output" + "." + attribute, value)  # The output of a tool call
            else:
                span.set_attribute("relai.tool_call.output", output)

            if note is not None:
                span.set_attribute("relai.note", note)  # The note provided by user

    def log_router(self, name: str, input: Any, output: Any, note: Optional[str] = None):
        """
        Logs a router event.

        Args:
            name (str): Name of the router.
            input (Any): Input to the router.
            output (Any): Output from the router.
            note (Optional[str]): Optional annotation.
        """
        if get_current_tracking():
            span_id = span_id_var.get()
            self._spans[span_id].append(
                {"type": "router", "router_name": name, "router_input": input, "router_output": output, "note": note}
            )

        # Treating router as a special LLM call in openinference
        tracer = tracer_provider.get_tracer(__name__)
        with tracer.start_as_current_span(name) as span:
            span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.LLM.value)

            if isinstance(input, Mapping):
                for attribute, value in flatten(input):
                    span.set_attribute(
                        SpanAttributes.LLM_INPUT_MESSAGES + "." + attribute, value
                    )  # List of messages sent to the LLM in a chat API request, [{"message.role": "user", "message.content": "hello"}]
            else:
                span.set_attribute(SpanAttributes.LLM_INPUT_MESSAGES, input)

            if isinstance(output, Mapping):
                for attribute, value in flatten(output):
                    span.set_attribute(
                        SpanAttributes.LLM_OUTPUT_MESSAGES + "." + attribute, value
                    )  # Messages received from a chat API, [{"message.role": "user", "message.content": "hello"}]
            else:
                span.set_attribute(SpanAttributes.LLM_OUTPUT_MESSAGES, output)

            span.set_attribute(SpanAttributes.LLM_MODEL_NAME, name)  # The name of the language model being utilized
            if note is not None:
                span.set_attribute("relai.note", note)  # The note provided by user
            span.set_attribute("relai.kind", "router")

    def _param_usage(self, param_name: str):
        """
        Records the usage of a parameter during the current span and update
        its ordering index.

        Args:
            param_name (str): Name of the parameter.
        """
        if get_current_tracking():
            span_id = span_id_var.get()
            self._component_active_params[span_id].add(param_name)
            if param_name not in self._param_order_store:
                self._param_order_store[param_name] = self._param_order_counter_store
                self._param_order_counter_store += 1

    def _param_order(self, param_name: str) -> int:
        """
        Returns the ordering index of a parameter.

        Args:
            param_name (str): Parameter name.

        Returns:
            int: Order index of the parameter, or the current counter if not found.
        """
        if param_name not in self._param_order_store:
            return self._param_order_counter_store

        return self._param_order_store[param_name]

    def _component_probe(self, uid: str, input: Any, output: Any):
        """
        Stores input and output logs for a component instance identified by uid.

        Args:
            uid (str): Unique identifier for the component instance.
            input (Any): Input data passed to the component.
            output (Any): Output data returned from the component.
        """
        if get_current_tracking():
            if uid not in self._component_log:
                self._component_log[uid] = []

            self._component_log[uid].append({"input": input, "output": output})

    def component_log(self) -> dict[str, list]:
        """
        Returns all recorded component logs.

        Returns:
            dict[str, list]: Component logs keyed by uid.
        """
        return self._component_log

    def serialize(self):
        """
        Serializes the entire log to a JSON formatted string.

        Returns:
            str: JSON string of all logged events.
        """

        return json.dumps(
            convert_traces(exporter.get_finished_spans(), filter_tag=f"relai.logger.{self._logger_id}"),
            indent=2,
        )

    def to_openinference(self) -> dict[str, list[Any]]:
        """
        Converts the entire log to a list of openinference trace dicts.

        Returns:
            list[dict]: List of all logged events as dicts in openinference format.
        """

        spans = exporter.get_finished_spans()
        return {
            "spans": [
                span.to_json()
                for span in spans
                if f"relai.logger.{self._logger_id}" in span.attributes["tag.tags"]  # type: ignore
            ]
        }


logger = Logger()  # for root context
span_id_var = ContextVar("span_id", default="root")
logger_var: ContextVar[Logger] = ContextVar("logger", default=logger)


@contextmanager
def set_current_logger(logger: Logger):
    """
    A context manager to set the logger for the enclosed context.
    """
    token = logger_var.set(logger)
    try:
        yield
    finally:
        logger_var.reset(token)


def get_current_logger() -> Logger:
    """
    Returns the logger instance for the current execution context.
    """
    return logger_var.get()
