import pytest
from pytest_mock import MockerFixture

from relai.benchmark import (
    Benchmark,
    CSVBenchmark,
    RELAIQuestionAnsweringBenchmark,
    RELAISummarizationBenchmark,
)
from relai.data import RELAISample


@pytest.mark.unit
def test_benchmark_creation(summarization_sample: RELAISample):
    """
    Test the creation of a benchmark sample.
    """
    sample = summarization_sample
    benchmark = Benchmark(benchmark_id=sample.benchmark_id, samples=[sample])
    assert len(benchmark) == 1


@pytest.mark.unit
def test_benchmark_samples_generator(summarization_sample: RELAISample):
    """
    Test the creation of a benchmark sample.
    """
    sample = summarization_sample
    benchmark = Benchmark(benchmark_id=sample.benchmark_id, samples=[sample])
    it = iter(benchmark)
    assert next(it) == sample
    with pytest.raises(StopIteration):
        next(it)


@pytest.mark.unit
def test_benchmark_sampling(summarization_sample: RELAISample):
    """
    Test the sampling of a benchmark.
    """
    sample = summarization_sample
    benchmark = Benchmark(benchmark_id=sample.benchmark_id, samples=[sample])
    samples = benchmark.sample(3)
    assert len(samples) == 3
    assert all(s == sample for s in samples)


@pytest.mark.unit
def test_csv_benchmark(sample_csv_file, question_answering_sample):
    """
    Test the CSVBenchmark class with a sample CSV file.
    """
    benchmark = CSVBenchmark(
        csv_file=sample_csv_file(question_answering_sample),
        agent_input_columns=["question"],
        extra_columns=["std_answer", "rubrics"],
        benchmark_id="tmp-csv-benchmark",
    )

    assert benchmark.benchmark_id == "tmp-csv-benchmark"
    assert len(benchmark.samples) == 3
    sample = benchmark.samples[0]
    assert sample.agent_inputs["question"] == "How many planets are in the Solar System?"
    assert sample.extras["std_answer"] == "There are eight planets in the Solar System."
    assert sample.extras["rubrics"] == {
        "Mention that there are exactly eight planets.": 10,
    }


@pytest.mark.unit
def test_relai_question_answering_benchmark(relai_client, mocker: MockerFixture):
    """
    Test the RELAIQuestionAnsweringBenchmark class with a sample.
    """
    mocker.patch(
        "relai._client.RELAI._request",
        new=mocker.Mock(
            side_effect=[
                # Mock the benchmark metadata response
                {
                    "benchmark_uuid": "relai-qa-benchmark",
                    "title": "RELAI Question Answering Benchmark",
                    "description": "A benchmark for evaluating question answering agents.",
                    "hyperparameters": {},
                },
                # Mock the benchmark samples response
                [
                    {
                        "sample_uuid": "1",
                        "benchmark": "relai-qa-benchmark",
                        "column_values": {
                            "rubrics": {
                                "criteria": [
                                    {
                                        "points": 3,
                                        "description": "Mention that there are exactly eight planets.",
                                    },
                                ]
                            },
                            "question": "How many planets are in the Solar System?",
                            "std_answer": "There are eight planets in the Solar System.",
                        },
                        "split": "TEST",
                    }
                ],
            ]
        ),
    )
    benchmark = RELAIQuestionAnsweringBenchmark(
        benchmark_id="relai-qa-benchmark",
    )

    assert benchmark.benchmark_id == "relai-qa-benchmark"
    assert len(benchmark.samples) == 1
    sample = benchmark.samples[0]
    assert sample.agent_inputs["question"] == "How many planets are in the Solar System?"
    assert sample.extras["std_answer"] == "There are eight planets in the Solar System."
    assert sample.extras["rubrics"] == {
        "Mention that there are exactly eight planets.": 3,
    }


@pytest.mark.unit
def test_relai_summarization_benchmark(relai_client, mocker: MockerFixture):
    """
    Test the RELAISummarizationBenchmark class with a sample.
    """
    mocker.patch(
        "relai._client.RELAI._request",
        new=mocker.Mock(
            side_effect=[
                # Mock the benchmark metadata response
                {
                    "benchmark_uuid": "relai-summarization-benchmark",
                    "title": "RELAI Summarization Benchmark",
                    "description": "A benchmark for evaluating summarization agents.",
                    "hyperparameters": {
                        "style_rubrics": {
                            "Summary should be comic in tone": 5,
                        },
                        "format_rubrics": {
                            "Summary should be a list": 5,
                        },
                    },
                },
                # Mock the benchmark samples response
                [
                    {
                        "sample_uuid": "1",
                        "benchmark": "relai-summarization-benchmark",
                        "column_values": {
                            "source_text": "The Solar System is composed of eight planets.",
                            "source_keyfacts": "The Solar System has eight planets.",
                        },
                        "split": "TEST",
                    }
                ],
            ]
        ),
    )
    benchmark = RELAISummarizationBenchmark(
        benchmark_id="relai-summarization-benchmark",
    )

    assert benchmark.benchmark_id == "relai-summarization-benchmark"
    assert len(benchmark.samples) == 1
    sample = benchmark.samples[0]
    assert sample.agent_inputs["source"] == "The Solar System is composed of eight planets."
    assert sample.extras["key_facts"] == "The Solar System has eight planets."
    assert sample.extras["style_rubrics"] == {
        "Summary should be comic in tone": 5,
    }
    assert sample.extras["format_rubrics"] == {
        "Summary should be a list": 5,
    }
