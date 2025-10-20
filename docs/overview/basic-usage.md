<h2>Accessing RELAI Benchmarks</h2>

RELAI benchmarks comprise pre-defined collections of `Sample` objects, serving as standardized test sets for evaluating AI agent capabilities. Each benchmark is assigned a unique identifier which can be obtained from the metadata of the benchmark on the RELAI platform.
We currently support the following benchmarks:

| RELAI Benchmark | Agent Input Fields | Evaluator Input Fields |
| ----------------| ----------------| ----------------|
| [RELAIQuestionAnsweringBenchmark](../benchmark.md#relai.critico.benchmark.RELAIQuestionAnsweringBenchmark) |  `question` | `rubrics`, `std_answer` |
| [RELAISummarizationBenchmark](../benchmark.md#relai.critico.benchmark.RELAISummarizationBenchmark) | `source` | `key_facts`, `style_rubrics`, `format_rubrics` |

To load a benchmark, instantiate the corresponding `RELAIBenchmark` class using the benchmark ID (You can find the benchmark ID in the 'Additional Details' section of a benchmark). This action initiates a connection to the RELAI platform and retrieves all associated samples.
For example, to create a `RELAIQuestionAnsweringBenchmark`:

```python
from relai.critico.benchmark import RELAIQuestionAnsweringBenchmark

# <benchmark_id> is available under "Additional Details".
# Make sure that <benchmark_id> corresponds to a question-answering benchmark. 
benchmark = RELAIQuestionAnsweringBenchmark("<benchmark_id>")

# All RELAIBenchmark instances are iterable. Inspection of a sample:
for i, sample in enumerate(benchmark):
    print(f"Sample ID: {sample.sample_id}")
    print(f"Agent Inputs (for AI agent processing): {sample.agent_inputs}")
    print(f"Evaluator Inputs (for evaluating AI agent outputs): {sample.eval_inputs}")
    break # Displaying only the first sample for brevity
```

A `RELAIBenchmark` object functions as an iterable collection of `Sample` objects. Each `Sample` is a Pydantic model that encapsulates the requisite data for both agent processing and subsequent evaluation.

<h2>AI Agent Response Generation</h2>

`Sample.agent_inputs` contains all the inputs need by an agent. 

```python
def custom_ai_agent(source: str **kwargs) -> dict:
    """
    A placeholder AI agent
    """
    return {"summary": f"A concise summary of the provided text: '{source[:50]}...'"}

raw_agent_output = agent(**sample.agent_inputs)  # `agent_inputs` must contain the key `source`
```

To evaluate the response subsequently using an `Evaluator`, encapsulate the raw agent output in `AgentResponse`.

```python
from relai.critico.benchmark import AgentResponse

agent_response = AgentResponse(
    sample=sample,
    agent_outputs=raw_agent_output
)
```

<h2>AI Agent Evaluation</h2>

We provide a diverse set of evaluators for automated quality assessment. Each evaluator is specialized for a distinct evaluation criterion (e.g., length, content fidelity, stylistic adherence). Many evaluators support configurable hyperparameters to fine-tune their assessment behavior. For instance, `RELAILengthEvaluator` permits specification of the measurement unit (sentences, words, characters) and an acceptable_range. To evaluate an agent response, first instantiate the evaluator:

```python
from relai.critico.evaluate import RELAILengthEvaluator

length_evaluator = RELAILengthEvaluator(
    measure="sentences",
    acceptable_range=(10, 15),
)
```

Evaluator instances are callable objects. An `AgentResponse` object can be directly passed to an evaluator instance to initiate the assessment. The evaluator will then execute its defined logic and return an `EvaluatorResponse`. Each evaluator expects a list of input fields. Each of these fields must be present in either `agent_inputs` (of the sample), `eval_inputs` (of the sample), or `agent_outputs` (of the agent response).

```python
evaluator_response = length_evaluator(agent_response)
print(f"Score: {evaluator_response.score}")
print(f"Feedback: {evaluator_response.feedback}")
```

We currently support the following evaluators:

| RELAI Evaluator | Required Fields | 
| ----------------| ----------------|
| [RELAILengthEvaluator](../evaluator.md#relai.critico.evaluate.RELAILengthEvaluator) |  `source`, `summary`|
| [RELAIContentEvaluator](../evaluator.md#relai.critico.evaluate.RELAIContentEvaluator) | `key_facts`, `summary` |
| [RELAIHallucinationEvaluator](../evaluator.md#relai.critico.evaluate.RELAIHallucinationEvaluator) | `source`, `summary` |
| [RELAIStyleEvaluator](../evaluator.md#relai.critico.evaluate.RELAIStyleEvaluator) | `style_rubrics`, `summary` |
| [RELAIFormatEvaluator](../evaluator.md#relai.critico.evaluate.RELAIFormatEvaluator) | `format_rubrics`, `summary` |
| [RELAIRubricBasedEvaluator](../evaluator.md#relai.critico.evaluate.RELAIRubricBasedEvaluator)  | `question`, `answer`, `rubrics`, `std_answer` |

You can also define custom evaluators by directly inheriting from the [Evaluator](../evaluator.md#relai.critico.evaluate.Evaluator) class and overriding the `compute_evaluator_result` method. Example:

```python
from typing import Optional, Callable
from relai.critico.benchmark import AgentResponse
from relai.critico.evaluate import Evaluator, EvaluatorResponse

class CustomMCQEvaluator(Evaluator):
    """
    An evaluator for multiple choice questions. For a correct answer, assigns a score equal to the number of
    options in the question (presumably an indicator of difficulty). Assigns 0 for an incorrect answer.
    """
    def __init__(self, transform: Optional[Callable] = None, num_options: int = 4)
        super().__init__(
            name="custom-evaluator",
            required_fields=["chosen_answer", "correct_answer"],  # Evaluator expects a `chosen_answer` field in `agent_response` and a `correct_answer` in `eval_inputs`
            transform=transform
            num_options=4
        )
    
    def compute_evaluator_result(self, agent_response: AgentResponse) -> EvaluatorResponse:
        chosen_answer = agent_response.agent_outputs["chosen_answer"]
        correct_answer = agent_response.sample.eval_inputs["eval_inputs"]
        if chosen_answer == correct_answer:
            score = self.hyperparameters["num_options"]
            feedback = "Answer is correct"
        else:
            score = 0
            feedback = f"Answer is incorrect. The correct answer is {correct_answer}"
        return EvaluatorResponse(
            evaluator_id=self.uid,
            evaluator_name=self.name,
            evaluator_configuration=self.hyperparameters,
            agent_response=agent_response,
            score=score,
            feedback=feedback
        )
```

