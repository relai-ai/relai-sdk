import asyncio
import os
from collections import defaultdict
from dataclasses import asdict

from relai import AsyncRELAI

from .._exceptions import RELAIError
from ..data import AgentLog, CriticoLog
from .evaluate import Evaluator


class Critico:
    """Critico orchestrates evaluation of an AI agent using a configurable set of evaluators."""

    def __init__(self, client: AsyncRELAI):
        """
        Args:
            client (AsyncRELAI): An instance of the AsyncRELAI client to interact with the RELAI platform.
        """
        self._evaluators: dict[str, dict[Evaluator, float]] = defaultdict(dict)
        self._client = client

    def reweight_evaluator(
        self,
        evaluator_group: str,
        evaluator: Evaluator,
        weight: float,
    ):
        """
        Adjusts the weight of a specific evaluator in a given evaluator group.

        Evaluator weights influence their contribution to the aggregate score during the `evaluate()` operation.

        Args:
            evaluator_group (str): The name of the evaluator group.
            evaluator (Evaluator): The `Evaluator` object whose weight is to be adjusted.
            weight (float): The new weight (a positive float) for the evaluator. Higher weights give more prominence
                to its results in aggregation.

        Raises:
            RELAIError: If the `evaluator_group` is not found in Critico, or if the `evaluator` is not associated with
                the specified `evaluator_group`.
            ValueError: If the specified `weight` is not a positive float.
        """
        if evaluator_group not in self._evaluators:
            raise RELAIError(f"Evaluator group {evaluator_group} not found in Critico.")
        if evaluator not in self._evaluators[evaluator_group]:
            raise RELAIError(f"Evaluator {evaluator} not found for evaluator group {evaluator_group}.")
        if weight <= 0:
            raise ValueError("Weight must be a positive float.")
        self._evaluators[evaluator_group][evaluator] = weight

    def add_evaluators(self, evaluators: dict[Evaluator, float], evaluator_group: str = "default") -> None:
        """
        Adds a new evaluator to Critico to a specific evaluator group.

        Args:
            evaluator_group (str): The name of the evaluator group.
            evaluators (dict[Evaluator, float], optional): A dictionary where keys are `Evaluator` objects and values
                are their corresponding weights for this evaluator group. If `None`, no evaluators are associated with
                this evaluator group initially. Defaults to `None`.
        """
        self._evaluators[evaluator_group] = evaluators

    async def _evaluate_agent_log(self, agent_log: AgentLog) -> CriticoLog:
        """
        Evaluates a single AI agent log using the evaluators in its evaluator group. Critico identifies the
        associated evaluator group and runs all evaluators configured for that group. It then aggregates the
        individual evaluator results (scores and feedback) into a single `CriticoFeedback` object.

        Args:
            agent_log (AgentLog): The `AgentLog` object to be evaluated.

        Returns:
            CriticoFeedback: A `CriticoFeedback` object summarizing the evaluation outcome for the `AgentLog`.

        Raises:
            KeyError: If the `evaluator_group` within the `AgentLog` is not found
                      among Critico's managed evaluator groups.

        """

        try:
            evaluators = self._evaluators[agent_log.simulation_tape.evaluator_group]
        except KeyError:
            raise RELAIError(
                f"""Evaluator group '{agent_log.simulation_tape.evaluator_group}' from response not found in """
                """Critico's added evaluators."""
            )
        # Handle case where a group has no evaluators assigned
        if not evaluators:
            return CriticoLog(
                agent_log=agent_log,
                aggregate_feedback="No evaluators configured for this benchmark.",
            )

        evaluator_logs = []
        aggregate_score = 0.0
        aggregate_feedback = ""

        for evaluator, weight in evaluators.items():
            evaluator_log = await evaluator(agent_log)
            evaluator_logs.append(evaluator_log)
            aggregate_score += evaluator_log.outputs["score"] * weight
            aggregate_feedback += f"{evaluator_log.outputs['feedback']}\n"
        total_weight = sum(evaluators.values())
        if total_weight == 0:
            aggregate_score = 0.0
        else:
            aggregate_score /= total_weight
        return CriticoLog(
            agent_log=agent_log,
            evaluator_logs=evaluator_logs,
            aggregate_score=aggregate_score,
            aggregate_feedback=aggregate_feedback.strip(),
            trace_id=agent_log.trace_id,
        )

    async def evaluate(self, agent_logs: list[AgentLog]) -> list[CriticoLog]:
        """
        Evaluates a list of AI agent logs against their corresponding evaluator groups using the configured evaluators.

        For each `AgentLog`, Critico identifies the associated evaluator group and runs all evaluators configured for
        that group. It then aggregates the individual evaluator results (scores and feedback) into a single
        `CriticoFeedback` object.

        Args:
            agent_logs (list[AgentLog]): A list of `AgentLog` objects to be evaluated. Each log must
                contain a `simulation_tape.evaluator_group` that corresponds to an evaluator group added to Critico.

        Returns:
            list[CriticoFeedback]: A list of `CriticoFeedback` objects, where each object summarizes the evaluation
                outcome for one `AgentLog`.

        Raises:
            KeyError: If a `evaluator_group` within an `AgentLog` is not found
                      among Critico's managed evaluator groups.
        """
        RELAI_EVAL_BATCH_SIZE = int(os.getenv("RELAI_EVAL_BATCH_SIZE", 10))
        critico_logs = []
        batches = [agent_logs[i : i + RELAI_EVAL_BATCH_SIZE] for i in range(0, len(agent_logs), RELAI_EVAL_BATCH_SIZE)]
        for batch in batches:
            tasks = [self._evaluate_agent_log(agent_log) for agent_log in batch]
            batch_critico_logs = await asyncio.gather(*tasks)
            critico_logs.extend(batch_critico_logs)
        return critico_logs

    async def report(self, critico_logs: list[CriticoLog]) -> None:
        """
        Submits the critico logs (as multiple `CriticoLog` objects) to the RELAI platform.

        Args:
            critico_logs (list[CriticoLog]): A list of `CriticoLog` objects
                containing the evaluation results for each `AgentLog`.

        Raises:
            RELAIError: If any `CriticoLog` does not have a valid `trace_id`.
        """
        for critico_log in critico_logs:
            if isinstance(critico_log.trace_id, str):
                await self._client.upload_critico_log(
                    trace_id=critico_log.trace_id,
                    evaluator_logs=[asdict(evaluator_log) for evaluator_log in critico_log.evaluator_logs],
                    aggregate_score=critico_log.aggregate_score,
                    aggregate_feedback=critico_log.aggregate_feedback,
                )
            else:
                raise RELAIError(
                    (
                        """Cannot report CriticoLog without a trace_id. """
                        """Please ensure that logging is enabled in the simulator."""
                    )
                )
