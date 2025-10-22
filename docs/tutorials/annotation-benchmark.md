<h1>Agent Annotation Benchmark</h1>

Annotation benchmarks are benchmarks created by annotating (providing feedback to) runs of agents. 
They can be used directly in agent optimization (configs, structure). 
For a detailed example of how to run agents in a simulated environment and 
how to use annotation benchmarks in agent optimization, see [summarization-agent (simulate→annotate→optimize)-part-1.py](/examples/basic/summarization-agent%20(simulate-%3Eannotate-%3Eoptimize)-part-1.py) and [summarization-agent (simulate→annotate→optimize)-part-2.py](/examples/basic/summarization-agent%20(simulate-%3Eannotate-%3Eoptimize)-part-2.py).

<h2>Create Annotation Benchmark</h2>

1. To create an annotation benchmark, first go to RELAI platform and find [Run](https://platform.relai.ai/results/run) under Results.

    <img src="../assets/tutorials/annotation-benchmark/1.png" alt="RELAI platform->Results->Run"/>

2. Click on individual runs to inspect any agent you executed in a simulated environment.

    <img src="../assets/tutorials/annotation-benchmark/2.png" alt="Inspect agent runs."/>

3. Annotate the runs with the `Like/Dislike`, `Desired Output`, `Feedback` fields and save your changes.

    <img src="../assets/tutorials/annotation-benchmark/3.png" alt="Annotate agent runs."/>

4. Use the "Add to Benchmark" button at the bottom to add the annotated run as a sample to the benchmark of your choice. 
(Use the `Create a new annoatation benchmark` function if you have not created any benchmark yet)

    <img src="../assets/tutorials/annotation-benchmark/4.png" alt="Add the annotated run to a benchmark."/>

5. Continue to annotate and add other runs to the benchmark. The benchmark is already ready-to-use with its benchmark id. See [summarization-agent (simulate→annotate→optimize)-part-1.py](/examples/basic/summarization-agent%20(simulate-%3Eannotate-%3Eoptimize)-part-1.py) and [summarization-agent (simulate→annotate→optimize)-part-2.py](/examples/basic/summarization-agent%20(simulate-%3Eannotate-%3Eoptimize)-part-2.py) for how to use annotation benchmarks in 
agent optimization.