To define a custom evaluator, write a class basing `relai.critico.evaluate.Evaluator` and override `compute_evaluator_result`
to describe your custom evaluation logic.

```python
from collections.abc import Callable
from typing import override

from relai import AgentLog, EvaluatorLog
from relai.critico.evaluate import Evaluator


class CustomSentimentEvaluator(Evaluator):
    """
    A custom evaluator for sentiment analysis tasks.

    This evaluator compares the agent's predicted sentiment against the ground truth
    and provides detailed scoring based on prediction accuracy and confidence.
    """

    def __init__(
        self,
        transform: Callable | None = None,
        correct_score: float = 1.0,
        incorrect_score: float = 0.0,
        partial_credit: bool = True,
    ):
        """
        Initialize the custom sentiment evaluator.

        Args:
            transform: Optional function to transform agent outputs
            correct_score: Score to assign for correct predictions
            incorrect_score: Score to assign for incorrect predictions
            partial_credit: Whether to give partial credit for neutral predictions
        """
        super().__init__(
            name="custom-sentiment-evaluator",
            # Specify required fields from the benchmark and agent response
            required_fields=["text", "predicted_sentiment", "true_sentiment"],
            transform=transform,
            # Store configuration as hyperparameters
            correct_score=correct_score,
            incorrect_score=incorrect_score,
            partial_credit=partial_credit,
        )

    @override
    async def compute_evaluator_result(self, agent_log: AgentLog) -> EvaluatorLog:
        """
        Evaluate the agent's sentiment prediction against ground truth.

        Args:
            agent_log (AgentLog): The response from the AI agent, containing the original sample
                and agent outputs.

        Returns:
            EvaluatorLog: Evaluator log with score and feedback
        """
        # Extract required fields from different sources
        text = agent_log.simulation_tape.agent_inputs["text"]
        predicted_sentiment = agent_log.agent_outputs["predicted_sentiment"]
        true_sentiment = agent_log.simulation_tape.extras["true_sentiment"]

        # Evaluate prediction accuracy
        if predicted_sentiment.lower() == true_sentiment.lower():
            score = self.hyperparameters["correct_score"]
            feedback = f"Correct! Predicted '{predicted_sentiment}' matches true sentiment '{true_sentiment}'"
        elif (
            self.hyperparameters["partial_credit"]
            and predicted_sentiment.lower() == "neutral"
            and true_sentiment.lower() in ["positive", "negative"]
        ):
            # Give partial credit for neutral predictions on polar sentiments
            score = self.hyperparameters["correct_score"] * 0.5
            feedback = f"Partial credit: Predicted neutral for {true_sentiment} sentiment"
        else:
            score = self.hyperparameters["incorrect_score"]
            feedback = f"Incorrect: Predicted '{predicted_sentiment}' but true sentiment is '{true_sentiment}'"

        # Add text length as additional context
        text_length = len(text.split())
        feedback += f" (Text length: {text_length} words)"

        return EvaluatorLog(
            evaluator_id=self.uid,
            name=self.name,
            outputs={"score": score, "feedback": feedback},
        )
```