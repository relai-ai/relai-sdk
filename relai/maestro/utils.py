import inspect
import os
import random
import time
import uuid
from functools import wraps
from typing import Any, Callable, List, Optional

from ..flags import get_current_tracking
from ..logger import Logger, get_current_logger, span_id_var
from .graph import param_graph


def component(name: str, uid: Optional[str] = None, note: Optional[str] = None):
    """
    Decorator to trace a component's execution.

    Args:
        name (str): Name of the component.
        uid (Optional[str]): Unique identifier for the component instance.
        note (Optional[str]): Optional annotation.

    Returns:
        Callable: Decorated function with logging around its execution.
    """

    def decorator(func):
        def pre_call() -> tuple[Logger, Any, float]:
            parent_span_id = span_id_var.get()
            span_id = str(uuid.uuid4())
            logger = get_current_logger()
            logger._spans[parent_span_id].append({"type": "child span", "span_id": span_id})
            token = span_id_var.set(span_id)

            if get_current_tracking():
                logger._spans[span_id].append({"type": "begin of span", "span_name": name, "uid": uid, "note": note})

            start = time.perf_counter()
            return logger, token, start

        def post_call(logger: Logger, token: Any, start: float, args: Any, kwargs: Any, ret: Any):
            end = time.perf_counter()

            if get_current_tracking():
                elapsed_time = end - start
                span_id = span_id_var.get()
                logger._spans[span_id].append(
                    {
                        "type": "end of span",
                        "span name": name,
                        "uid": uid,
                        "note": note,
                        "time spent in span": elapsed_time,
                        "parameters used": list(logger._component_active_params[span_id]),
                    }
                )

                for param1 in logger._component_active_params[span_id]:
                    for param2 in logger._component_active_params[span_id]:
                        param_graph.add_edge(param1, param2)
                logger._component_active_params.pop(span_id, None)

                if uid is not None:
                    logger._component_probe(uid=uid, input=(args, kwargs), output=ret)

            span_id_var.reset(token)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger, token, start = pre_call()
            ret = await func(*args, **kwargs)
            post_call(logger, token, start, args, kwargs, ret)
            return ret

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger, token, start = pre_call()
            ret = func(*args, **kwargs)
            post_call(logger, token, start, args, kwargs, ret)
            return ret

        return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper

    return decorator


def get_full_func_name(func: Callable) -> str:
    """
    Get the full name of a function, method, or callable object.
    For callable instances, returns their class name.

    Args:
        func (Callable): A function.

    Returns:
        str: A single string contains the full name.
    """
    if inspect.ismethod(func):
        module = func.__func__.__module__
        qualname = func.__func__.__qualname__
    elif inspect.isfunction(func):
        module = func.__module__
        qualname = func.__qualname__
    elif callable(func):
        # Callable object -> just use its class
        cls = func.__class__
        module = cls.__module__
        qualname = cls.__qualname__
    else:
        raise TypeError(f"Object {func!r} is not callable")

    return f"{module}.{qualname}"


def extract_code(code_paths: List[str], encoding: str = "utf-8") -> str:
    """
    Given a list of file paths, reads files and formats them by
    wrapping each file's content with a header, then concatenates all
    source code into a single string.

    Args:
        code_paths: List of file paths to source files.
        encoding: File encoding to use when reading files. Defaults to 'utf-8'.

    Returns:
        str: A single string containing all files, each prefixed
        by a filename header.
    """
    formatted_sources = []

    for path in code_paths:
        # Process all files
        if not os.path.isfile(path):
            continue

        with open(path, "r", encoding=encoding) as f:
            src = f.read()

        # Format with filename header
        header = f"----<{os.path.basename(path)}>----"
        formatted_sources.append(f"{header}\n{src.strip()}")

    # Join all sources with spacing for readability
    return "\n\n".join(formatted_sources)


class ProportionalSampler:
    """
    Generates samples from a list of elements so that, over time, their
    frequencies approximate a target probability distribution.

    The sampler keeps internal counts of how often each element has been
    drawn and always selects the element whose observed frequency lags most
    behind its expected frequency. Random tie-breaking ensures variety.
    """

    def __init__(self, elements: list, weights: list[float]):
        """
        Initializes the sampler with a list of elements and associated weights.

        Args:
            elements (list): The items to sample from.
            weights (list[float]): Non-negative weights for each element.
                Must have the same length as `elements`.

        Raises:
            ValueError: If the lengths differ or any weight is negative.
        """
        if len(elements) != len(weights):
            raise ValueError("Elements and weights must have the same length.")
        if any(w < 0 for w in weights):
            raise ValueError("Weights must be non-negative.")

        total_weight = sum(weights)
        if total_weight == 0:
            raise ValueError("Sum of weights must be greater than zero.")

        self.elements = elements
        self.probs = [w / total_weight for w in weights]
        self.counts = [0] * len(elements)
        self.total = 0

    def _next(self):
        """
        Generates the next sample in the sequence.

        Returns:
            Any: The next sampled element, chosen so that its long-term
            frequency approaches the target distribution.
        """
        if self.total == 0:
            idx = random.choices(range(len(self.elements)), weights=self.probs, k=1)[0]
        else:
            expected = [p * self.total for p in self.probs]
            deficits = [e - c for e, c in zip(expected, self.counts)]
            max_deficit = max(deficits)
            candidates = [i for i, d in enumerate(deficits) if abs(d - max_deficit) < 1e-12]
            idx = random.choice(candidates)

        self.counts[idx] += 1
        self.total += 1
        return self.elements[idx]

    def sample(self, num_samples: int) -> list:
        """
        Generates a list of samples using the balancing algorithm.

        Args:
            num_samples (int): Number of samples to generate.

        Returns:
            list: A list of sampled elements.
        """
        return [self._next() for _ in range(num_samples)]
