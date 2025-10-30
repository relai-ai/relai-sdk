import functools
import inspect
import random
import sys
from collections.abc import Callable, Coroutine, Generator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, ParamSpec, Protocol, TypeAlias, TypeVar, cast, get_type_hints
from uuid import uuid4

from relai.benchmark import Benchmark
from relai.mocker.base_mocker import BaseMocker

from ._client import RELAI, AsyncRELAI
from .data import AgentLog, AgentOutputs, RELAISample, Serializable, SimulationConfigT, SimulationTape
from .flags import tracking_off, tracking_on
from .utils import create_logging_span, get_current_logger

P = ParamSpec("P")
T = TypeVar("T")

SyncAgent: TypeAlias = Callable[[SimulationTape], AgentOutputs]
AsyncAgent: TypeAlias = Callable[[SimulationTape], Coroutine[None, None, AgentOutputs]]
Samplable: TypeAlias = Sequence[BaseMocker]


_simulation_mode: ContextVar[bool] = ContextVar("simulation_mode", default=False)
_simulation_config: ContextVar[SimulationConfigT] = ContextVar("simulation_config", default={})
_mocked_funcs: set[str] = set()


@contextmanager
def _simulate(config: SimulationConfigT):
    """
    A context manager to enter simulation mode with the given configuration.
    """
    simulation_mode_token = _simulation_mode.set(True)
    simulation_config_token = _simulation_config.set(config or {})
    yield
    _simulation_mode.reset(simulation_mode_token)
    _simulation_config.reset(simulation_config_token)


def get_current_simulation_mode() -> bool:
    """
    Retrieves the current simulation mode.

    Returns:
        bool: True if in simulation mode, False otherwise.
    """
    return _simulation_mode.get()


def get_current_simulation_config() -> SimulationConfigT:
    """
    Retrieves the current simulation configuration.

    Returns:
        dict[str, BaseMocker]: The current simulation configuration.
    """
    return _simulation_config.get()


def simulated(
    func: Callable[..., Coroutine[Any, Any, Any]] | Callable[P, T],
) -> Callable[..., Coroutine[Any, Any, Any]] | Callable[P, T]:
    """
    Decorator to mark a function to be simulated using a mocker in simulation mode. All such functions must have a
    corresponding mocker set in the simulation configuration. Supports both synchronous and asynchronous functions.
    """
    func_name = f"{func.__module__}.{func.__qualname__}"
    _mocked_funcs.add(func_name)

    func_doc = inspect.getdoc(func)
    mod_globals = sys.modules[func.__module__].__dict__
    try:
        type_hints = get_type_hints(func, globalns=mod_globals)
    except Exception:
        type_hints = dict(func.__annotations__)
    output_type = cast(type, type_hints.get("return", None))

    @functools.wraps(func)
    def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        if not get_current_simulation_mode():
            return cast(Callable[P, T], func)(*args, **kwargs)
        else:
            try:
                mocker = get_current_simulation_config()[func_name]
            except KeyError as e:
                raise RuntimeError(
                    f"No mocker found for function {func_name}. Make sure to set the mocker in the simulation config."
                ) from e
            if mocker.func_doc is None:
                mocker.func_doc = func_doc
            if mocker.output_type is None:
                mocker.output_type = output_type
            return mocker._run(*args, **kwargs)

    @functools.wraps(func)
    async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        if not get_current_simulation_mode():
            return await cast(Callable[P, Coroutine[Any, Any, T]], func)(*args, **kwargs)
        else:
            try:
                mocker = get_current_simulation_config()[func_name]
            except KeyError as e:
                raise RuntimeError(
                    f"No mocker found for function {func_name}. Make sure to set the mocker in the simulation config."
                ) from e
            if mocker.func_doc is None:
                mocker.func_doc = func_doc
            if mocker.output_type is None:
                mocker.output_type = output_type
            return await mocker._arun(*args, **kwargs)

    if inspect.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


class EnvGenerator(Protocol):
    """
    Protocol for functions that generate simulation configurations. These functions take an optional RELAISample
    and return a SimulationConfigT which is a dictionary mapping qualified function names of functions decorated with
    `@simulated` to their respective mocker instances.
    """

    def __call__(self, sample: RELAISample | None = None) -> SimulationConfigT: ...


def _default_env_generator(sample: RELAISample | None = None) -> SimulationConfigT:
    return {}


def random_env_generator(config_set: dict[str, Samplable]) -> EnvGenerator:
    """
    An environment generator that uniformly samples a mocker for each simulated function from the provided set of
    mockers.

    Args:
        config_set (dict[str, Sequence[BaseMocker]]): A mapping from qualified function names to a sequence of possible
            mockers for that function.
    """

    def sampler(sample: RELAISample | None = None) -> SimulationConfigT:
        return {k: random.choice(v) for k, v in config_set.items()}

    return sampler


class BaseSimulator:
    def __init__(self, env_generator: EnvGenerator | None = None, benchmark: Benchmark | None = None) -> None:
        if env_generator is None:
            self.env_generator = _default_env_generator
        else:
            self.env_generator = env_generator
        self.benchmark = benchmark

    def tape_and_config_generator(
        self, num_runs: int
    ) -> Generator[tuple[SimulationTape, SimulationConfigT], None, None]:
        if self.benchmark is None:
            for _ in range(num_runs):
                config = self.env_generator()
                tape = SimulationTape()
                tape.simulation_config = config
                yield tape, config
        else:
            samples = self.benchmark.sample(num_runs)
            for sample in samples:
                config = self.env_generator(sample)
                tape = SimulationTape(sample=sample)
                tape.simulation_config = config
                yield tape, config


class SyncSimulator(BaseSimulator):
    """A simulator for synchronous agent functions."""

    def __init__(
        self,
        agent_fn: SyncAgent,
        env_generator: EnvGenerator | None = None,
        benchmark: Benchmark | None = None,
        log_runs: bool = True,
        client: RELAI | None = None,
    ) -> None:
        """
        Args:
            agent_fn (SyncAgent): The synchronous agent function to be simulated.
            env_generator (EnvGenerator | None): An optional environment generator function. If not provided, a default
                generator that returns an empty configuration will be used. The default generator can only be used with
                simulations that don't require any mockers.
            benchmark (Benchmark | None): An optional benchmark to source simulation samples from.
            log_runs (bool): Whether to log the runs to the RELAI platform. Defaults to True.
            client (RELAI | None): A synchronous RELAI client for logging. Must be provided if log_runs is True.
        """
        super().__init__(env_generator, benchmark)
        self.agent_fn = agent_fn
        self.log_runs = log_runs
        if log_runs and client is None:
            raise ValueError("client must be provided if log_runs is True")
        self.client = client

    def run(self, num_runs: int, group_id: str | None = None) -> list[AgentLog]:
        """
        Run the simulator for a specified number of times.

        Args:
            num_runs (int): The number of simulation runs to execute.
            group_id (str, optional): An optional group ID to associate all runs together. If not provided,
                a new UUID will be generated.
        """
        agent_logs: list[AgentLog] = []
        group_id = ("Simulate-" + uuid4().hex) if group_id is None else group_id
        tracking_on()
        for tape, config in self.tape_and_config_generator(num_runs):
            with _simulate(config), create_logging_span(tape.id):
                agent_outputs = self.agent_fn(tape)
                if not isinstance(agent_outputs, dict):
                    raise TypeError("agent_fn must return an instance of AgentOutputs")
                agent_log = AgentLog(simulation_tape=tape, agent_outputs=agent_outputs)
                if self.log_runs and self.client is not None:
                    trace_id = self.client.upload_trace(data=get_current_logger().to_openinference())
                    agent_log.trace_id = trace_id
                    self.client.upload_run(
                        group_id=group_id,
                        trace_id=trace_id,
                        agent_inputs=tape.agent_inputs,
                        agent_outputs=agent_outputs,
                        extras=tape.extras,
                        serialized_simulation_config={
                            k: v.serialize() for k, v in tape.simulation_config.items() if isinstance(v, Serializable)
                        },
                    )
                agent_logs.append(agent_log)
                tape.add_record("relai_log", get_current_logger().serialize())
        tracking_off()
        return agent_logs

    def rerun(self, simulation_tapes: list[SimulationTape], group_id: str | None = None) -> list[AgentLog]:
        """
        Rerun the simulator for a list of simulation tapes.

        Args:
            simulation_tapes (list[SimulationTape]): The list of simulation tapes to rerun. This allows for re-executing
                the agent in an environment identical to a previous run and is useful for debugging and optimization.
            group_id (str, optional): An optional group ID to associate all runs together. If not provided,
                a new UUID will be generated.
        """
        agent_logs: list[AgentLog] = []
        group_id = ("Simulate-" + uuid4().hex) if group_id is None else group_id
        tracking_on()
        for tape in simulation_tapes:
            new_tape = tape.copy()
            with _simulate(new_tape.simulation_config), create_logging_span(new_tape.id):
                agent_outputs = self.agent_fn(new_tape)
                agent_log = AgentLog(simulation_tape=new_tape, agent_outputs=agent_outputs)
                agent_logs.append(agent_log)
                if self.log_runs and self.client is not None:
                    trace_id = self.client.upload_trace(data=get_current_logger().to_openinference())
                    self.client.upload_run(
                        group_id=group_id,
                        trace_id=trace_id,
                        agent_inputs=new_tape.agent_inputs,
                        agent_outputs=agent_outputs,
                        extras=new_tape.extras,
                        serialized_simulation_config={
                            k: v.serialize()
                            for k, v in new_tape.simulation_config.items()
                            if isinstance(v, Serializable)
                        },
                    )
                new_tape.add_record("relai_log", get_current_logger().serialize())
        tracking_off()
        return agent_logs


class AsyncSimulator(BaseSimulator):
    """A simulator for asynchronous agent functions."""

    def __init__(
        self,
        agent_fn: AsyncAgent,
        env_generator: EnvGenerator | None = None,
        benchmark: Benchmark | None = None,
        log_runs: bool = True,
        client: AsyncRELAI | None = None,
    ) -> None:
        """
        Args:
            agent_fn (AsyncAgent): The asynchronous agent function to be simulated.
            env_generator (EnvGenerator | None): An optional environment generator function. If not provided, a default
                generator that returns an empty configuration will be used. The default generator can only be used with
                simulations that don't require any mockers.
            benchmark (Benchmark | None): An optional benchmark to source simulation samples from.
            log_runs (bool): Whether to log the runs to the RELAI platform. Defaults to True.
            client (RELAI | None): A asynchronous RELAI client for logging. Must be provided if log_runs is True.
        """
        super().__init__(env_generator, benchmark)
        self.agent_fn = agent_fn
        self.log_runs = log_runs
        if log_runs and client is None:
            raise ValueError("client must be provided if log_runs is True")
        self.client = client

    async def run(self, num_runs: int, group_id: str | None = None) -> list[AgentLog]:
        """Run the simulator for a specified number of times.

        Args:
            num_runs (int): The number of simulation runs to execute.
            group_id (str, optional): An optional group ID to associate all runs together. If not provided,
                a new UUID will be generated.
        """
        agent_logs: list[AgentLog] = []
        group_id = ("Simulate-" + uuid4().hex) if group_id is None else group_id
        tracking_on()
        for tape, config in self.tape_and_config_generator(num_runs):
            with _simulate(config), create_logging_span(tape.id):
                agent_outputs = await self.agent_fn(tape)
                agent_log = AgentLog(simulation_tape=tape, agent_outputs=agent_outputs)
                if self.log_runs and self.client is not None:
                    trace_id = await self.client.upload_trace(data=get_current_logger().to_openinference())
                    agent_log.trace_id = trace_id
                    await self.client.upload_run(
                        group_id=group_id,
                        trace_id=trace_id,
                        agent_inputs=tape.agent_inputs,
                        agent_outputs=agent_outputs,
                        extras=tape.extras,
                        serialized_simulation_config={
                            k: v.serialize() for k, v in tape.simulation_config.items() if isinstance(v, Serializable)
                        },
                    )
                agent_logs.append(agent_log)
                tape.add_record("relai_log", get_current_logger().serialize())
        tracking_off()
        return agent_logs

    async def rerun(self, simulation_tapes: list[SimulationTape], group_id: str | None = None) -> list[AgentLog]:
        """
        Rerun the simulator for a list of simulation tapes.

        Args:
            simulation_tapes (list[SimulationTape]): The list of simulation tapes to rerun. This allows for re-executing
                the agent in an environment identical to a previous run and is useful for debugging and optimization.
            group_id (str, optional): An optional group ID to associate all runs together. If not provided,
                a new UUID will be generated.
        """
        agent_logs: list[AgentLog] = []
        group_id = ("Simulate-" + uuid4().hex) if group_id is None else group_id
        tracking_on()
        for tape in simulation_tapes:
            new_tape = tape.copy()
            with _simulate(new_tape.simulation_config), create_logging_span(new_tape.id):
                agent_outputs = await self.agent_fn(new_tape)
                agent_log = AgentLog(simulation_tape=new_tape, agent_outputs=agent_outputs)
                agent_logs.append(agent_log)
                if self.log_runs and self.client is not None:
                    trace_id = await self.client.upload_trace(data=get_current_logger().to_openinference())
                    await self.client.upload_run(
                        group_id=group_id,
                        trace_id=trace_id,
                        agent_inputs=new_tape.agent_inputs,
                        agent_outputs=agent_outputs,
                        extras=new_tape.extras,
                        serialized_simulation_config={
                            k: v.serialize()
                            for k, v in new_tape.simulation_config.items()
                            if isinstance(v, Serializable)
                        },
                    )
                new_tape.add_record("relai_log", get_current_logger().serialize())
        tracking_off()
        return agent_logs
