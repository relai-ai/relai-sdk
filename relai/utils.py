import uuid
from contextlib import contextmanager

from openinference.instrumentation import suppress_tracing, using_attributes

from .flags import RELAITrackingFlag, relai_tracking_flag_var
from .logger import Logger, get_current_logger, logger_var


@contextmanager
def no_trace():
    """
    Disable OpenInference tracing for a code block.
    """
    with suppress_tracing():
        yield


@contextmanager
def create_logging_span(logger_id: str | None = None):
    """Context manager to create a new span for logging."""
    if logger_id is None:
        logger_id = uuid.uuid4().hex
    logger = Logger(logger_id=logger_id)
    logger_token = logger_var.set(logger)
    flag = RELAITrackingFlag(False)
    flag_token = relai_tracking_flag_var.set(flag)
    try:
        with using_attributes(tags=[f"relai.logger.{logger_id}"]):
            yield
    finally:
        logger_var.reset(logger_token)
        relai_tracking_flag_var.reset(flag_token)


def log_model(*args, **kwargs):
    """
    Logs a model call event.

    Args:
        name (str): Name of the model.
        input (Any): Input to the model.
        output (Any): Output from the model.
        note (Optional[str]): Optional annotation.
    """
    logger = get_current_logger()
    logger.log_model(*args, **kwargs)


def log_tool(*args, **kwargs):
    """
    Logs a tool call event.

    Args:
        name (str): Name of the tool.
        input (Any): Input to the tool.
        output (Any): Output from the tool.
        note (Optional[str]): Optional annotation.
    """
    logger = get_current_logger()
    logger.log_tool(*args, **kwargs)


def log_persona(*args, **kwargs):
    """
    Logs a persona activity.

    Args:
        name (str): Name of the persona.
        model_name (str): Name of the model.
        input (Any): Input to the persona.
        output (Any): Output from the persona.
        note (Optional[str]): Optional annotation.
    """
    logger = get_current_logger()
    logger.log_persona(*args, **kwargs)


def log_router(*args, **kwargs):
    """
    Logs a router event.

    Args:
        name (str): Name of the router.
        input (Any): Input to the router.
        output (Any): Output from the router.
        note (Optional[str]): Optional annotation.
    """
    logger = get_current_logger()
    logger.log_router(*args, **kwargs)
