import ast
import csv
import json
import random
from abc import ABC
from typing import Any, Callable, Iterator, Literal, Optional

from ._client import get_default_client
from .data import RELAISample


class Benchmark(ABC):
    """
    Abstract base class for defining and managing benchmarks.

    This class provides a foundational structure for benchmarks, enabling the
    download, and iteration of samples. It ensures that all concrete
    benchmark implementations have a unique identifier and a collection of samples
    to be used as inputs for AI agents and evaluators.

    Attributes:
        benchmark_id (str): A **unique identifier** for this specific benchmark.
        samples (list[RELAISample]): A list of `RELAISample` objects contained within this benchmark.
            Defaults to an empty list if not provided.
    """

    def __init__(self, benchmark_id: str, samples: Optional[list[RELAISample]] = None):
        """
        Args:
            benchmark_id (str): The unique identifier for the benchmark.
            samples (list[RELAISample], optional): A list of `RELAISample` objects to include
                in the benchmark. Defaults to an empty list.
        """
        self.benchmark_id = benchmark_id
        self.samples: list[RELAISample] = samples or []

    def __iter__(self) -> Iterator[RELAISample]:
        """
        Enables iteration over the samples within the benchmark as follows:

        ```python
        for sample in benchmark:
            # Process each sample
            pass
        ```

        Yields:
            RELAISample: Each `RELAISample` object contained in the benchmark.
        """
        for sample in self.samples:
            yield sample

    def __len__(self) -> int:
        """
        Returns the number of samples currently in the benchmark.

        Returns:
            int: The total count of `Sample` objects.
        """
        return len(self.samples)

    def sample(self, n: int = 1) -> list[RELAISample]:
        """
        Returns `n` random samples from the benchmark, with replacement.

        If `n` is greater than the total number of samples, samples may be
        repeated in the returned list.

        Args:
            n (int): The number of random samples to retrieve. Must be a positive integer.
                Defaults to 1.

        Returns:
            list[RELAISample]: A list containing `n` randomly selected `RELAISample` objects.

        Raises:
            ValueError: If `n` is less than or equal to 0.
        """
        if n <= 0:
            raise ValueError("Number of samples must be greater than 0.")
        return random.choices(self.samples, k=n)


class RELAIBenchmark(Benchmark):
    """
    A concrete implementation of `Benchmark` that downloads samples from the RELAI platform.

    Attributes:
        benchmark_id (str): The unique identifier (ID) of the RELAI benchmark to be loaded from the platform.
            You can find the benchmark ID in the metadata of the benchmark.
        samples (list[RELAISample]): A list of `RELAISample` objects contained within this benchmark.
    """

    def __init__(
        self,
        benchmark_id: str,
        field_name_mapping: Optional[dict[str, str]] = None,
        field_value_transform: Optional[dict[str, Callable]] = None,
        agent_input_fields: Optional[list[str]] = None,
        extra_fields: Optional[list[str]] = None,
    ):
        """
        Args:
            benchmark_id (str): The unique identifier for the RELAI benchmark.
                This ID is used to fetch the benchmark data from the RELAI platform.
            field_name_mapping (dict[str, str], optional): A mapping from field names
                returned by the RELAI API to standardized field names expected by the evaluators.
                If a field name is not present in this mapping, it is used as-is.
                Defaults to an empty dictionary.
            field_value_transform (dict[str, Callable], optional): A mapping from field names
                to transformation functions that convert field values from the RELAI API
                into the desired format. If a field name is not present in this mapping,
                the identity function is used (i.e., no transformation).
                Defaults to an empty dictionary.
            agent_input_fields (list[str], optional): A list of field names to extract from each
                sample for the `agent_inputs` dictionary. These fields are provided to the AI agent.
                Defaults to an empty list.
            extra_fields (list[str], optional): A list of field names to extract from each
                sample for the `extras` dictionary. These fields are also provided to the evaluators.
                Defaults to an empty list.
        """
        self._benchmark_id = benchmark_id
        self._field_name_mapping = field_name_mapping or {}
        self._field_value_transform = field_value_transform or {}

        self._agent_input_fields = agent_input_fields or []
        self._extra_fields = extra_fields or []
        super().__init__(benchmark_id=benchmark_id, samples=self.fetch_samples())

    def fetch_samples(self) -> list[RELAISample]:
        """
        Downloads samples from the RELAI platform and populates the `samples` attribute.

        This method fetches the benchmark data using the RELAI client and processes
        each sample to create `Sample` objects. The `samples` attribute is then
        updated with the newly fetched samples.
        """

        def strict_field_mapping(field: str) -> str:
            """
            Maps a field name returned by the RELAI API to a standardized name expected by the evaluators.

            Args:
                field (str): The field name as returned by the RELAI API.

            Returns:
                str: The standardized field name as defined in the `field_mapping` dictionary.
            """
            if field in self._field_name_mapping:
                return self._field_name_mapping[field]
            else:
                return field

        def strict_field_value_transform(field: str) -> Callable:
            """
            Returns a transformation function for a field value, if defined in `field_value_transform`.
            If no transformation is defined, it returns the identity function.

            Args:
                field (str): The field name for which the transformation is defined.

            Returns:
                Callable: A function that takes a value and applies the transformation.
            """
            if field in self._field_value_transform:
                return self._field_value_transform[field]
            return lambda x: x

        client = get_default_client()
        raw_samples = client.get_benchmark(self._benchmark_id)
        samples = []
        for raw_sample in raw_samples:
            try:
                agent_inputs = {
                    strict_field_mapping(field): strict_field_value_transform(field)(raw_sample[field])
                    for field in self._agent_input_fields
                }
            except KeyError as e:
                missing_key = e.args[0]
                raise ValueError(
                    f"""Expected agent input field `{missing_key}` is missing. """
                    f"""Make sure that the benchmark ID corresponds to a {self.__class__.__name__}."""
                )
            try:
                extras = {
                    strict_field_mapping(field): strict_field_value_transform(field)(raw_sample[field])
                    for field in self._extra_fields
                }
            except KeyError as e:
                missing_key = e.args[0]
                raise ValueError(
                    f"""Expected evaluation input field `{missing_key}` is missing. """
                    f"""Make sure that the benchmark ID corresponds to a {self.__class__.__name__}."""
                )
            sample = RELAISample(
                benchmark_id=self._benchmark_id,
                id=raw_sample["sample_uuid"],
                split=raw_sample["split"],
                agent_inputs=agent_inputs,
                extras=extras,
            )
            samples.append(sample)
        return samples


class RELAIQuestionAnsweringBenchmark(RELAIBenchmark):
    """
    A concrete implementation of `RELAIBenchmark` for question-answering tasks.
    All samples in this benchmark have the following fields:

    - `agent_inputs`:
        - `question`: The question to be answered by the AI agent.
    - `extras`:
        - `rubrics`:  A dictionary of rubrics for evaluating the answer.
        - `std_answer`: The standard answer to the question.
    }

    """

    def __init__(
        self,
        benchmark_id: str,
    ):
        """
        Args:
            benchmark_id (str): The unique identifier for the RELAI question-answering benchmark.
                This ID is used to fetch the benchmark data from the RELAI platform.
        """

        def rubric_transform(r: dict) -> dict:
            rubrics = {}
            for criterion in r["criteria"]:
                rubrics[criterion["description"]] = criterion["points"]
            return rubrics

        super().__init__(
            benchmark_id=benchmark_id,
            field_value_transform={"rubrics": rubric_transform},
            agent_input_fields=["question"],
            extra_fields=["rubrics", "std_answer"],
        )


class RELAISummarizationBenchmark(RELAIBenchmark):
    """
    A concrete implementation of `RELAIBenchmark` for summarization tasks. All samples in this benchmark have the following fields:

    - `agent_inputs`:
        - `source`: The text to be summarized.
    - `extras`:
        - `key_facts`: A list of key facts extracted from the source.
        - `style_rubrics`: A dictionary of rubrics for evaluating the style of the summary.
        - `format_rubrics`: A dictionary of rubrics for evaluating the format of the summary.
    """

    def __init__(
        self,
        benchmark_id: str,
    ):
        """
        Args:
            benchmark_id (str): The unique identifier for the RELAI summarization benchmark.
                This ID is used to fetch the benchmark data from the RELAI platform.
        """
        super().__init__(
            benchmark_id=benchmark_id,
            field_name_mapping={"source_text": "source", "source_keyfacts": "key_facts"},
            agent_input_fields=["source_text"],
            extra_fields=["source_keyfacts", "style_rubrics", "format_rubrics"],
        )


class RELAIAnnotationBenchmark(RELAIBenchmark):
    """
    A concrete implementation of `RELAIBenchmark` for benchmarks created from user annotations.
    All samples in this benchmark have the following fields:

    - `agent_inputs`:
        - The input(s) provided to the agent being evaluated.
    - `extras`:
        - `previous_outputs`: The previous outputs produced by the agent.
        - `desired_outputs`: The desired outputs as specified by the user.
        - `feedback`: The user feedback provided for the previous outputs.
        - `liked`: A boolean indicating whether the user liked the previous outputs.
    """

    def __init__(
        self,
        benchmark_id: str,
    ):
        """
        Args:
            client (AsyncRELAI): An instance of the AsyncRELAI client to interact with the RELAI platform.
            benchmark_id (str): The unique identifier for the RELAI summarization benchmark.
                This ID is used to fetch the benchmark data from the RELAI platform.
        """

        def nested_field_transform(value: str) -> Any:
            """
            Attempts to parse a string value as a nested field. If parsing fails, returns the original string.
            Args:
                value (str): The string value to be parsed.
            Returns:
                Any: The parsed JSON object, or the original string if parsing fails.
            """
            if value is None:
                return None
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                try:
                    return ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    return value

        def liked_transform(value: bool) -> Literal["like", "dislike"]:
            if value is None:
                return "not provided"
            return "like" if value else "dislike"

        super().__init__(
            benchmark_id=benchmark_id,
            field_name_mapping={
                "agent_outputs": "previous_outputs",
                "serialized_simulation_config": "simulation_config",
                "agent_inputs": "all_inputs",
            },
            agent_input_fields=["agent_inputs"],
            extra_fields=["agent_outputs", "desired_outputs", "feedback", "liked", "serialized_simulation_config"],
            field_value_transform={
                "agent_outputs": nested_field_transform,
                "desired_outputs": nested_field_transform,
                "serialized_simulation_config": nested_field_transform,
                "agent_inputs": nested_field_transform,
                "liked": liked_transform,
            },
        )


class CSVBenchmark(Benchmark):
    """
    A concrete implementation of `Benchmark` that loads samples from a CSV file.

    Attributes:
        benchmark_id (str): The unique identifier (ID) of the benchmark to be loaded from the CSV file.
            Defaults to the CSV file name.
        samples (list[Sample]): A list of `Sample` objects contained within this benchmark.
    """

    def __init__(
        self,
        csv_file: str,
        agent_input_columns: Optional[list[str]] = None,
        extra_columns: Optional[list[str]] = None,
        benchmark_id: Optional[str] = None,
    ):
        """
        Args:
            csv_file (str): The path to the CSV file containing benchmark samples.
            agent_input_columns (list[str], optional): A list of column names in the CSV file
                that should be used as inputs for the AI agent. Defaults to an empty list.
            extra_columns (list[str], optional): A list of column names in the CSV file
                that could be used as inputs for evaluators. Defaults to an empty list.
            benchmark_id (str, optional): A unique identifier for the benchmark.
                If not provided, it defaults to the name of the CSV file.
        """
        benchmark_id = benchmark_id or csv_file
        agent_input_columns = agent_input_columns or []
        extra_columns = extra_columns or []
        samples = []
        with open(csv_file, encoding="utf-8-sig") as csvfile:
            reader = csv.DictReader(csvfile)
            for idx, row in enumerate(reader):
                sample = RELAISample(
                    benchmark_id=benchmark_id,
                    id=str(idx),
                    agent_inputs={col: self._deserialize_str(row[col]) for col in agent_input_columns},
                    extras={col: self._deserialize_str(row[col]) for col in extra_columns},
                )
                samples.append(sample)
        super().__init__(benchmark_id=benchmark_id, samples=samples)

    def _deserialize_str(self, value: str) -> Any:
        """
        Attempts to deserialize a string value into its original type.
        This is useful for converting string representations of complex types back to their original form.
        """
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            # If deserialization fails, return the original string
            return value
