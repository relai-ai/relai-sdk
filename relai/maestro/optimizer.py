import asyncio
import copy
import json
import os
from datetime import datetime, timezone
from typing import Any, Awaitable, Optional
from uuid import uuid4

from tqdm.auto import tqdm

from relai import AsyncRELAI
from relai.critico.critico import Critico, CriticoLog
from relai.simulator import AgentLog, AsyncAgent, AsyncSimulator
from relai.utils import no_trace

from ..schema.visual import ConfigOptVizSchema, ConfigSchema, GraphOptVizSchema, ParamSchema, RunSchema
from .graph import param_graph
from .params import params
from .utils import ProportionalSampler, extract_code, get_full_func_name


class Maestro:
    """
    Maestro automatically optimizes an AI agent to maximize its Critico score, navigating the space of
    configurations to intelligently improve performance on the chosen criteria.
    """

    def __init__(
        self,
        client: AsyncRELAI,
        agent_fn: AsyncAgent,
        goal: Optional[str] = None,
        max_memory: int = 20,
        name: str = "No Name",
        log_to_platform: bool = True,
    ):
        """
        Args:
            client (AsyncRELAI): An instance of the AsyncRELAI client to interact with the RELAI platform.
            agent_fn (AsyncAgent): The agent function to be optimized.
            goal (str, optional): Optional description of the goal of optimization. If None, increasing evaluation score
                will be considered as the only goal. Defaults to None.
            max_memory (int, optional): Control the maximum number of previous optimization history visible at each
                optimization step. Defaults to 20.
            name (str, optional): Name of the configuration optimization visualization on RELAI platform.
                Defaults to "No Name".
            log_to_platform (bool): Whether to log optimization progress and results on RELAI platform.
                Defaults to True.
        """
        self.setups: list[dict] = []
        self.agent_fn: AsyncAgent = agent_fn
        self.log: list[dict] = []
        self.max_memory: int = max_memory
        self._client: AsyncRELAI = client
        self.goal: str = goal if goal is not None else "Higher scores"
        self.log_to_platform: bool = log_to_platform
        self.config_opt_viz_id: str | None = None
        self.name: str = name

        self.versions: list[dict] = [
            {
                "idx": 0,
                "params": copy.deepcopy(params),
                "log": copy.deepcopy(self.log),
                "visits": 0,
                "average_score": 0.0,
                "utc_timestamp": datetime.now(timezone.utc).isoformat(),
                "validation_runs": [],
                "validation_count": 0,
            }
        ]
        self.current_version: int = 0
        self.total_visits: int = 0
        self.inf: float = 1e208
        self.score_min: float = self.inf
        self.score_max: float = -self.inf

    def add_setup(self, simulator: AsyncSimulator, critico: Critico, weight: float = 1):
        """
        Add a new setup consisting of a simulator and a critico to Maestro.

        Args:
            simulator (AsyncSimulator): An AsyncSimulator to run the agent in the new setup.
            critico (Critico): A Critico with evaluators for the new setup.
            weight (float): A positive float representing the weight of this setup in comparson to others.
                Defaults to 1.
        """

        if not isinstance(simulator, AsyncSimulator):
            raise ValueError("Currently only AsyncSimulator is supported in Maestro.")
        if weight <= 0:
            raise ValueError(f"`weight` must be a positive float, got {weight}.")

        self.setups.append({"simulator": simulator, "critico": critico, "weight": weight})

    def _serialize_past_proposals(self) -> str:
        """
        Serialize the previous optimization history of the current version.

        Returns:
            str: A string representing the previous optimization history of the current version.
        """
        return json.dumps(self.log[-self.max_memory :], indent=2)

    async def _select(self, explore: bool = False):
        """
        Select the best performing or the most promising version (of the agent to optimize)
        according to past evidences and set it as the current version.

        Args:
            explore (bool): If False, setting the best performing version as current;
                If True, setting the most promising version as current. Defaults to False.
        """
        selected_version = await self._client.select_version(
            {
                "versions": [{"average_score": v["average_score"], "visits": v["visits"]} for v in self.versions],
                "explore": explore,
                "min_score": self.score_min,
                "max_score": self.score_max,
                "total_visits": self.total_visits,
            }
        )
        self.current_version = selected_version
        params.sync(self.versions[selected_version]["params"])
        self.log = copy.deepcopy(self.versions[selected_version]["log"])

    def _update_score(self, score: float):
        """
        Adding an evaluation score of the current version to evidence for version selection.

        Args:
            score (float): The evaluation score of the current version to be added as evidence
            for version selection.
        """
        self.total_visits += 1
        self.versions[self.current_version]["average_score"] = (
            self.versions[self.current_version]["average_score"] * self.versions[self.current_version]["visits"] + score
        ) / (self.versions[self.current_version]["visits"] + 1.0)
        self.versions[self.current_version]["visits"] += 1

        self.score_min = min(self.score_min, score)
        self.score_max = max(self.score_max, score)

    def _serialize_agent_outputs(self, agent_outputs: Any) -> str:
        """
        Serialize agent outputs.

        Args:
            agent_outputs (Any): The agent outputs to serialize.

        Returns:
            str: A string representing the agent outputs.
        """
        if isinstance(agent_outputs, dict):
            ret = ""
            for key, value in agent_outputs.items():
                ret += f"{key}:\n {value}\n"
            return ret
        else:
            return str(agent_outputs)

    async def _evaluate(
        self, awaitables: list[Awaitable], criticos: list[Critico], verbose: bool = False, print_flag: str = ""
    ) -> tuple[list[dict[str, Any]], list[AgentLog]]:
        """
        Run and evaluate the current version of the agent through a set of awaitables.

        Args:
            awaitables (list[Awaitable]): A list of awaitables, each representing a run of the agent
            criticos (list[Critico]): A list of Critico objects, each corresponding to an awaitable
            verbose (bool): If True, additional information will be printed during evaluation.
                Defaults to False.
            print_flag (str): A string to be put next to the printed info when `verbose` is True.
                Used to distinguish printed info from different types of evaluations.

        Returns:
            list[dict]: A list of dictionary. Each dictionary contains evaluation info corresponding to
                one `Sample` object and has the following keys: "input", "log", "output", "eval_score",
                "eval_feedback".
        """
        RELAI_EVAL_BATCH_SIZE = int(os.getenv("RELAI_EVAL_BATCH_SIZE", 10))

        param_graph.clear()
        test_cases = []

        async def _evaluate_awaitable(awaitable: Awaitable, critico: Critico) -> tuple[dict[str, Any], AgentLog]:
            agent_log: AgentLog = (await awaitable)[0]
            with no_trace():
                eval_result: CriticoLog = (await critico.evaluate([agent_log]))[0]

            return {
                "input": str(agent_log.simulation_tape.agent_inputs),
                "log": agent_log.simulation_tape.extras["relai_log"],
                "trace_id": agent_log.trace_id,
                "output": self._serialize_agent_outputs(agent_log.agent_outputs),
                "eval_score": eval_result.aggregate_score,
                "eval_feedback": eval_result.aggregate_feedback,
            }, agent_log

        batches_awaitable = [
            awaitables[i : i + RELAI_EVAL_BATCH_SIZE] for i in range(0, len(awaitables), RELAI_EVAL_BATCH_SIZE)
        ]
        batches_critico = [
            criticos[i : i + RELAI_EVAL_BATCH_SIZE] for i in range(0, len(criticos), RELAI_EVAL_BATCH_SIZE)
        ]
        test_cases = []
        agent_logs = []
        for batch_awaitable, batch_critico in zip(batches_awaitable, batches_critico):
            tasks = [_evaluate_awaitable(sample, critico) for sample, critico in zip(batch_awaitable, batch_critico)]
            batch_test_cases = await asyncio.gather(*tasks)
            test_cases.extend([test_case for test_case, _ in batch_test_cases])
            agent_logs.extend([agent_log for _, agent_log in batch_test_cases])

        if verbose:
            for test_case in test_cases:
                print("=================agent excution result===================")
                print(f"- input:\n{test_case['input']}\n")
                print(f"- log{print_flag}:\n{test_case['log']}\n")
                print(f"- output{print_flag}:\n{test_case['output']}\n")
                print(f"- eval score{print_flag}:\n{test_case['eval_score']}\n")
                print(f"- eval feedback{print_flag}:\n{test_case['eval_feedback']}\n")
                print("=========================================================\n\n")

        return test_cases, agent_logs

    async def _iterate(
        self,
        batch_size: int,
        sampler: ProportionalSampler,
        verbose: bool = False,
        group_id: str | None = None,
        pbar: tqdm | None = None,
    ) -> bool:
        """
        An iterate step will propose changes to the current version of the agent and
        conduct a preliminary examination of the proposed changes.

        It returns True if the proposed changes pass the preliminary examination and
        False otherwise.

        Args:
            batch_size (int): 2 * `batch_size` samples will be obtained from the main Critico agent,
                i.e. `critico`, where `batch_size` of them will be used to propose changes and the other
                `batch_size` of them will be used for preliminary examinations.
            sampler (ProportionalSampler): Sampler to use for selecting setups.
            verbose (bool): If True, additional information will be printed during the iterate step.
                Defaults to False.
            group_id (str, optional): An optional group ID to associate all runs together. If not provided,
                a new UUID will be generated.
            pbar (tqdm, optional): A progress bar to display the progress of the iteration. Defaults to None.

        Returns:
            bool: True if the proposed changes pass the preliminary examination and False otherwise.

        Raises:
            ValueError: If no setup (simulator, critico) has been added to Maestro.
        """
        if len(self.setups) == 0:
            raise ValueError(
                "No setup (simulator, critico) has been added to Maestro. Please add at least one setup before optimization."
            )

        group_id = uuid4().hex if group_id is None else group_id

        setups = sampler.sample(batch_size * 2)
        awaitables = []
        criticos = []
        for setup in setups:
            simulator = setup["simulator"]
            critico = setup["critico"]
            awaitables.append(simulator.run(num_runs=1, group_id=group_id))
            criticos.append(critico)

        test_cases, agent_logs = await self._evaluate(awaitables=awaitables, criticos=criticos, verbose=verbose)

        if pbar is not None:
            pbar.update(len(test_cases))

        analysis, proposed_values = await self._client.propose_values(
            {
                "params": params.export(),
                "serialized_past_proposals": self._serialize_past_proposals(),
                "test_cases": test_cases[:batch_size],
                "goal": self.goal,
                "param_graph": param_graph.export(),
            }
        )

        changes = []
        for param, value in proposed_values.items():
            changes.append({"param": param, "previous value": params.__getattr__(param), "new value": value})
            if verbose:
                print("=" * 60)
                print("- proposed param change:", param)
                print("")
                print("- previous value:\n\n", params.__getattr__(param))
                print("")
                print("- new value:\n\n", value)
                print("=" * 60)

        self.log.append({"proposal id": len(self.log), "proposed changes": changes})

        # Decide if the change should be kept

        for name, proposed_value in proposed_values.items():
            params.update(name, proposed_value)

        new_awaitables = []
        new_criticos = []
        for test_case, agent_log, setup in zip(test_cases, agent_logs, setups):
            simulator = setup["simulator"]
            critico = setup["critico"]
            new_awaitables.append(simulator.rerun([agent_log.simulation_tape], group_id=group_id))
            new_criticos.append(critico)

        test_cases_updated, _ = await self._evaluate(
            awaitables=new_awaitables, criticos=new_criticos, verbose=verbose, print_flag=" (changed)"
        )

        if pbar is not None:
            pbar.update(len(test_cases_updated))

        for sample_id in range(0, batch_size * 2):
            test_cases_updated[sample_id]["previous_log"] = test_cases[sample_id]["log"]
            test_cases_updated[sample_id]["previous_output"] = test_cases[sample_id]["output"]
            test_cases_updated[sample_id]["previous_eval_score"] = test_cases[sample_id]["eval_score"]
            test_cases_updated[sample_id]["previous_eval_feedback"] = test_cases[sample_id]["eval_feedback"]

        for change in changes:
            params.update(change["param"], change["previous value"])

        previous_score = 0
        new_score = 0
        for test_case in test_cases_updated[batch_size:]:
            previous_score += test_case["previous_eval_score"]
            new_score += test_case["eval_score"]
        previous_score = float(previous_score) / batch_size
        new_score = float(new_score) / batch_size

        review_decision = await self._client.review_values(
            {
                "params": params.export(),
                "serialized_past_proposals": self._serialize_past_proposals(),
                "proposal": changes,
                "test_cases": test_cases_updated[:batch_size],
                "holdout_test_cases": test_cases_updated[batch_size:],
                "previous_score": previous_score,
                "new_score": new_score,
                "goal": self.goal,
                "analysis": analysis,
            }
        )

        if review_decision["accepted"]:
            self.log[-1]["status"] = "ACCEPTED"
            self.log[-1]["review comment"] = review_decision["full comment"]
            for change in changes:
                params.update(change["param"], change["new value"])

        else:
            self.log[-1]["status"] = "REJECTED"
            self.log[-1]["review comment"] = review_decision["full comment"]

        if verbose:
            print("previous avg score: ", previous_score)
            print("new avg score: ", new_score)
            print("accepted: ", review_decision["accepted"])
            print("review comment:\n", review_decision["full comment"])
            print("-" * 60 + "\n\n")

        return review_decision["accepted"]

    async def optimize_config(
        self,
        total_rollouts: int,
        batch_size: int = 8,
        explore_radius: int = 5,
        explore_factor: float = 0.5,
        verbose: bool = False,
    ):
        """
        Optimize the configs (parameters) of the agent.

        Args:
            total_rollouts (int): Total number of rollouts to use for optimization.
            batch_size (int): Base batch size to use for individual optimization steps. Defaults to 8.
            explore_radius (int): A positive integer controlling the aggressiveness of exploration during optimization.
                A larger `explore_radius` encourages the optimizer to make more substantial changes between successive configurations.
                Defaults to 5.
            explore_factor (float): A float between 0 to 1 controlling the exploration-exploitation trade-off.
                A higher `explore_factor` allocates more rollouts to discover new configs,
                while a lower value allocates more rollouts to ensure the discovered configs are thoroughly evaluated.
                Defaults to 0.5.
            verbose (bool): If True, related information will be printed during the optimization step.
                Defaults to False.

        Raises:
            ValueError: If the input parameters are not valid.
        """
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"`batch_size` must be a positive integer, got {batch_size}.")

        if not isinstance(explore_radius, int) or explore_radius <= 0:
            raise ValueError(f"`explore_radius` must be a positive integer, got {explore_radius}.")

        if explore_factor <= 0 or explore_factor >= 1:
            raise ValueError(f"`explore_factor` must be a float between 0 and 1, got {explore_factor}.")

        group_size = (batch_size + 1) // 2
        # total_rollouts = (iterate_steps * group_size * 4 + select_steps * group_size) * num_rounds
        # explore_factor = (iterate_steps * group_size * 4) / (iterate_steps * group_size * 4 + select_steps * group_size)
        iterate_steps: int = explore_radius
        select_steps: int = int(explore_radius * 4 * (1 - explore_factor) / explore_factor)
        num_rounds: int = int(total_rollouts / (iterate_steps * group_size * 4 + select_steps * group_size))
        total_rollouts = num_rounds * (iterate_steps * group_size * 4 + select_steps * group_size)

        print("optimize_config settings:")
        print("  total_rollouts: ", total_rollouts)
        print("  (adjusted) batch_size: ", group_size * 2)
        print("  explore_radius: ", explore_radius)
        print("  explore_factor: ", explore_factor)
        print("-" * 60)
        print("  iterate_steps: ", iterate_steps)
        print("  select_steps: ", select_steps)
        print("  num_rounds: ", num_rounds)
        print("=" * 80 + "\n\n")

        if num_rounds == 0:
            raise ValueError(
                f"`total_rollouts` is too small for the given `batch_size` {batch_size}, `explore_radius` {explore_radius}, and `explore_factor` {explore_factor}. "
                f"Please increase `total_rollouts` to at least {iterate_steps * group_size * 4 + select_steps * group_size}."
            )

        sampler = ProportionalSampler(
            elements=self.setups,
            weights=[setup["weight"] for setup in self.setups],
        )
        group_id = "Maestro-Config-" + uuid4().hex
        pbar = tqdm(total=total_rollouts, desc="Total rollouts consumed for config optimization")

        for round in range(num_rounds):
            print("\n\n" + "=" * 30 + f" Round {round + 1}/{num_rounds} begins" + "=" * 30)
            print("Total versions accepted: ", len(self.versions))
            print("Rebase to version: ", self.current_version)
            print(
                "Score for the current base version: %s based on %s rollouts"
                % (
                    self.versions[self.current_version]["average_score"],
                    self.versions[self.current_version]["visits"] * group_size,
                )
            )
            print("\n\n")

            new_version = False
            for _ in range(iterate_steps):
                changes_accepted = await self._iterate(
                    batch_size=group_size, verbose=verbose, sampler=sampler, group_id=group_id, pbar=pbar
                )
                if changes_accepted:
                    new_version = True

            if new_version:
                self.current_version = len(self.versions)
                self.versions.append(
                    {
                        "idx": self.current_version,
                        "params": copy.deepcopy(params),
                        "log": copy.deepcopy(self.log[-self.max_memory :]),
                        "visits": 0,
                        "average_score": 0.0,
                        "utc_timestamp": datetime.now(timezone.utc).isoformat(),
                        "validation_runs": [],
                        "validation_count": 0,
                    }
                )

            # Evaluate existing versions
            for _ in range(select_steps):
                await self._select(explore=True)

                setups = sampler.sample(group_size)
                awaitables = []
                criticos = []
                for setup in setups:
                    simulator = setup["simulator"]
                    critico = setup["critico"]
                    awaitables.append(simulator.run(num_runs=1, group_id=group_id))
                    criticos.append(critico)

                test_cases_validation, _ = await self._evaluate(
                    awaitables=awaitables, criticos=criticos, verbose=verbose, print_flag="(validation)"
                )

                if pbar is not None:
                    pbar.update(len(test_cases_validation))

                validation_score = 0.0
                for test_case in test_cases_validation:
                    validation_score += test_case["eval_score"]

                    self.versions[self.current_version]["validation_runs"].append(
                        RunSchema(
                            log=test_case["log"],
                            trace_id=test_case["trace_id"],
                            input=test_case["input"],
                            output=test_case["output"],
                            eval_score=test_case["eval_score"],
                            eval_feedback=test_case["eval_feedback"],
                        )
                    )

                    # Kept up to 100 runs per config for browsing
                    self.versions[self.current_version]["validation_runs"] = self.versions[self.current_version][
                        "validation_runs"
                    ][-100:]

                    self.versions[self.current_version]["validation_count"] += 1

                validation_score /= len(test_cases_validation)
                self._update_score(validation_score)

            # Switch to the current version with highest score
            await self._select(explore=False)

            print("\n\n" + "=" * 30 + f" Round {round + 1}/{num_rounds} finishes" + "=" * 30)
            print("Total versions accepted: ", len(self.versions))
            print("Best version index: ", self.current_version)
            print(
                "Score for the best version: %s based on %s rollouts"
                % (
                    self.versions[self.current_version]["average_score"],
                    self.versions[self.current_version]["visits"] * group_size,
                )
            )

            print(
                "All versions: ",
                {
                    i: {"score": self.versions[i]["average_score"], "rollouts evaluated": self.versions[i]["visits"] * group_size}
                    for i in range(len(self.versions))
                },
            )
            print("--------------------")

            async def sync_to_platform():
                payload = ConfigOptVizSchema(
                    name=self.name,
                    configs=[
                        ConfigSchema(
                            params={
                                key: ParamSchema(**config["params"]._params[key]) for key in config["params"].all()
                            },
                            validation_score=config["average_score"],
                            validation_count=config["validation_count"],
                            validation_runs=config["validation_runs"],
                            utc_timestamp=config["utc_timestamp"],
                        )
                        for config in self.versions
                    ],
                    current_config=self.current_version,
                    validation_score_over_version=[
                        (idx, config["average_score"], config["validation_count"])
                        for idx, config in enumerate(self.versions)
                    ],
                    validation_score_over_time=[
                        (config["utc_timestamp"], config["average_score"], config["validation_count"])
                        for config in self.versions
                    ],
                )

                self.config_opt_viz_id = await self._client.update_config_opt_visual(
                    config_viz=payload, uuid=self.config_opt_viz_id
                )

                return self.config_opt_viz_id

            if self.log_to_platform:
                await sync_to_platform()
                print(
                    f"Results of round {round + 1}/{num_rounds} uploaded to RELAI platform, visualization id: {self.config_opt_viz_id}"
                )

    async def optimize_structure(
        self,
        total_rollouts: int,
        description: Optional[str] = None,
        code_paths: Optional[list[str]] = None,
        verbose: bool = False,
    ) -> str:
        """
        Propose structural changes (i.e. changes that cannot be achieved by setting parameters alone) to
        improve the agent.

        Args:
            total_rollouts (int): Total number of rollouts to use for optimization.
                Generally, a moderate number of rollouts (e.g. 10-20) is required and recommended.
                For agents with longer execution traces: Try reducing the number of rollouts if an error is raised.
            description (str, optional): Text description of the current structure/workflow/... of the agent.
            code_paths (list[str], optional): A list of paths corresponding to code files containing
                the implementation of the agent.
            verbose (bool): If True, additional information will be printed during the optimization.
                Defaults to False.

        Returns:
            str: Suggestion for structural changes to the agent.
        """

        print("optimize_structure settings:")
        print("  total_rollouts: ", total_rollouts)
        print("=" * 80 + "\n\n")

        if code_paths is not None:
            code = extract_code(code_paths=code_paths)
        else:
            code = None

        sampler = ProportionalSampler(
            elements=self.setups,
            weights=[setup["weight"] for setup in self.setups],
        )
        group_id = "Maestro-Struct-" + uuid4().hex

        print("=" * 80)
        print("Running the agent to collect traces...\n\n")

        setups = sampler.sample(total_rollouts)
        awaitables = []
        criticos = []
        for setup in setups:
            simulator = setup["simulator"]
            critico = setup["critico"]
            awaitables.append(simulator.run(num_runs=1, group_id=group_id))
            criticos.append(critico)

        test_cases, _ = await self._evaluate(awaitables=awaitables, criticos=criticos, verbose=verbose)

        print("=" * 80)
        print("Optimizing structure...\n\n")
        suggestion = await self._client.optimize_structure(
            {
                "agent_name": get_full_func_name(self.agent_fn),
                "agent_code": code,
                "structure_description": description,
                "params": params.export(),
                "serialized_past_proposals": self._serialize_past_proposals(),
                "test_cases": test_cases,
                "goal": self.goal,
                "max_correction_rounds": 1,
            }
        )

        async def sync_to_platform():
            payload = GraphOptVizSchema(
                name=self.name,
                proposal=suggestion,
                runs=[
                    RunSchema(
                        log=test_case["log"],
                        trace_id=test_case["trace_id"],
                        input=test_case["input"],
                        output=test_case["output"],
                        eval_score=test_case["eval_score"],
                        eval_feedback=test_case["eval_feedback"],
                    )
                    for test_case in test_cases
                ],
            )

            return await self._client.update_graph_opt_visual(payload)

        print("=" * 40 + "suggestion" + "=" * 40)
        print(suggestion)
        print("=" * 90 + "\n\n")

        if self.log_to_platform:
            uid = await sync_to_platform()
            print(f"Results uploaded to RELAI platform, visualization id: {uid}")

        return suggestion
