from typing import Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field


class ParamSchema(BaseModel):
    """
    Schema for an agent parameter.
    """

    type: str  # Type of the parameter
    current_value: Union[float, int, str]  # Current value of the parameter
    description: str  # Description of the parameter
    allowed_values: Optional[List[Union[float, int, str]]] = Field(default=None)  # Allowed values for the parameter


class RunSchema(BaseModel):
    """
    Schema for execution logs and evaluation results from a single run of an agent.
    """

    log: str  # Events in the Maestro execution log (a serialized json str)
    trace_id: str  # used to uniquely identify the openinference execution log
    input: str  # Input to the agent (serialized)
    output: str  # Output of the agent
    eval_score: float  # Evaluation score
    eval_feedback: str  # Evaluation feedback


class ConfigSchema(BaseModel):
    """
    Schema for a specific agent configuration and the corresponding statistics.
    """

    params: Dict[str, ParamSchema]  # Parameters. The keys are parameter names.
    validation_score: float  # Current average score on validation set
    validation_count: int  # Number of validation runs
    validation_runs: List[RunSchema]  # Execution logs & evaluation results (up to 100 runs)
    utc_timestamp: str  # Timestamp in UTC for when the configuration is saved (iso format)


class ConfigOptVizSchema(BaseModel):
    """
    Schema to be used for visualizing optimizer progress and results.
    """

    name: str
    configs: List[ConfigSchema]  # Different configurations saved during optimization and the corresponding statistics
    current_config: int  # Index of the current configuration
    validation_score_over_version: List[
        Tuple[int, float, int]
    ]  # Validation scores for different configs; Format: (config id, validation score, validation count)
    validation_score_over_time: List[
        Tuple[str, float, int]
    ]  # Validation scores over time; Format: (timestamp in UTC with iso format, validation score, validation count)


class GraphOptVizSchema(BaseModel):
    """
    Schema to be used for visualizing optimizer progress and results.
    """

    name: str
    proposal: str  # Proposed graph changes to the agent
    runs: List[RunSchema]  # Execution logs & evaluation results of the original agent, which motivate the proposal
