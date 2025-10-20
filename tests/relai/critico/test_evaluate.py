import pytest
from pytest_mock import MockerFixture

from relai.critico.evaluate import (
    EvaluatorResponse,
    RELAIContentEvaluator,
    RELAIFormatEvaluator,
    RELAIHallucinationEvaluator,
    RELAILengthEvaluator,
    RELAIRubricBasedEvaluator,
    RELAIStyleEvaluator,
)


@pytest.mark.unit
def test_evaluator_response_extra_fields(question_answering_agent_response):
    """
    Test the EvaluatorResponse class with extra fields.
    """
    response = EvaluatorResponse(
        evaluator_id="test-evaluator",
        evaluator_name="Test Evaluator",
        evaluator_configuration={"param1": "value", "param2": 1},
        agent_response=question_answering_agent_response,
        score=0.9,
        feedback="Good job!",
        extra_field1="This is some extra information.",  # pyright: ignore[reportCallIssue]
        extra_field2={"key": "value"},  # pyright: ignore[reportCallIssue]
    )

    assert response.extra_field1 == "This is some extra information."  # pyright: ignore[reportAttributeAccessIssue]
    assert response.extra_field2 == {"key": "value"}  # pyright: ignore[reportAttributeAccessIssue]
    assert response.evaluator_outputs == {
        "score": 0.9,
        "feedback": "Good job!",
        "extra_field1": "This is some extra information.",
        "extra_field2": {"key": "value"},
    }


@pytest.mark.unit
@pytest.mark.asyncio
async def test_relai_length_evaluator(relai_client, summarization_agent_response, mocker: MockerFixture):
    """
    Test the RELAI length evaluator.
    """
    mocker.patch(
        "relai._client.AsyncRELAI._request",
        new=mocker.AsyncMock(
            return_value={
                "evaluation_result": {
                    "response": {
                        "score": 0.75,
                        "feedback": "Summary is a tad too long.",
                    },
                },
            }
        ),
    )
    # Initialize the RELAI length evaluator
    evaluator = RELAILengthEvaluator(client=relai_client, measure="words", acceptable_range=(10, 50))

    # Compute the evaluation result
    result = await evaluator(summarization_agent_response)

    # Assert the result
    assert isinstance(result, EvaluatorResponse)
    assert result.score == 0.75
    assert result.feedback == "Summary is a tad too long."


@pytest.mark.unit
@pytest.mark.asyncio
async def test_relai_content_evaluator(relai_client, summarization_agent_response, mocker: MockerFixture):
    """
    Test the RELAI content evaluator.
    """
    mocker.patch(
        "relai._client.AsyncRELAI._request",
        new=mocker.AsyncMock(
            return_value={
                "evaluation_result": {
                    "response": {
                        "score": 0.81,
                        "feedback": "The summary captures most key facts but misses a few details.",
                    },
                },
            }
        ),
    )
    # Initialize the RELAI content evaluator
    evaluator = RELAIContentEvaluator(client=relai_client)

    # Compute the evaluation result
    result = await evaluator(summarization_agent_response)

    # Assert the result
    assert isinstance(result, EvaluatorResponse)
    assert result.score == 0.81
    assert result.feedback == "The summary captures most key facts but misses a few details."


@pytest.mark.unit
@pytest.mark.asyncio
async def test_relai_hallucination_evaluator(relai_client, summarization_agent_response, mocker: MockerFixture):
    """
    Test the RELAI hallucination evaluator.
    """
    mocker.patch(
        "relai._client.AsyncRELAI._request",
        new=mocker.AsyncMock(
            return_value={
                "evaluation_result": {
                    "response": {
                        "hallucinations": {},
                        "score": 1.0,
                        "feedback": "The summary does not contain any hallucinations.",
                    },
                },
            }
        ),
    )
    # Initialize the RELAI hallucination evaluator
    evaluator = RELAIHallucinationEvaluator(client=relai_client)

    # Compute the evaluation result
    result = await evaluator(summarization_agent_response)

    # Assert the result
    assert isinstance(result, EvaluatorResponse)
    assert result.score == 1.0
    assert result.feedback == "The summary does not contain any hallucinations."


@pytest.mark.unit
@pytest.mark.asyncio
async def test_relai_style_evaluator(relai_client, summarization_agent_response, mocker: MockerFixture):
    """
    Test the RELAI style evaluator.
    """
    mocker.patch(
        "relai._client.AsyncRELAI._request",
        new=mocker.AsyncMock(
            return_value={
                "evaluation_result": {
                    "response": {
                        "style_score": 1.0,
                        "format_score": 1.0,
                        "feedback": (
                            """The adherence to style and format rubrics is as follows:\n"""
                            """- Perfect adherence for Clarity\n"""
                            """- Perfect adherence for Conciseness\n"""
                            """- Perfect adherence for Engagement\n"""
                            """- Perfect adherence for Any format"""
                        ),
                    },
                },
            }
        ),
    )
    # Initialize the RELAI style evaluator
    evaluator = RELAIStyleEvaluator(client=relai_client)

    # Compute the evaluation result
    result = await evaluator(summarization_agent_response)

    # Assert the result
    assert isinstance(result, EvaluatorResponse)
    assert result.score == 1.0
    assert result.feedback == (
        """The adherence to style rubrics is as follows:\n"""
        """- Perfect adherence for Clarity\n"""
        """- Perfect adherence for Conciseness\n"""
        """- Perfect adherence for Engagement\n"""
        """- Perfect adherence for Any format"""
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_relai_format_evaluator(relai_client, summarization_agent_response, mocker: MockerFixture):
    """
    Test the RELAI format evaluator.
    """
    mocker.patch(
        "relai._client.AsyncRELAI._request",
        new=mocker.AsyncMock(
            return_value={
                "evaluation_result": {
                    "response": {
                        "style_score": 1.0,
                        "format_score": 0.0,
                        "feedback": (
                            """The adherence to style and format rubrics is as follows:\n"""
                            """- Perfect adherence for Any style\n"""
                            """- No adherence to list format"""
                        ),
                    },
                },
            }
        ),
    )
    # Initialize the RELAI format evaluator
    evaluator = RELAIFormatEvaluator(client=relai_client)

    # Compute the evaluation result
    result = await evaluator(summarization_agent_response)

    # Assert the result
    assert isinstance(result, EvaluatorResponse)
    assert result.score == 0.0
    assert result.feedback == (
        """The adherence to format rubrics is as follows:\n"""
        """- Perfect adherence for Any style\n"""
        """- No adherence to list format"""
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_relai_rubric_based_evaluator(relai_client, question_answering_agent_response, mocker: MockerFixture):
    """
    Test the RELAI rubric-based evaluator.
    """
    mocker.patch(
        "relai._client.AsyncRELAI._request",
        new=mocker.AsyncMock(
            return_value={
                "evaluation_result": {
                    "response": {"satisfied_criteria": [[0]]},
                },
            }
        ),
    )
    # Initialize the RELAI rubric-based evaluator
    evaluator = RELAIRubricBasedEvaluator(client=relai_client)

    # Compute the evaluation result
    result = await evaluator(question_answering_agent_response)

    # Assert the result
    assert isinstance(result, EvaluatorResponse)
    assert result.score == 1.0
    assert result.feedback == "The response satisfies all criteria."
