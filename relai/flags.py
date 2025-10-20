from contextlib import contextmanager
from contextvars import ContextVar


class RELAITrackingFlag:
    """
    A simple mutable boolean-like flag object.

    Useful for controlling state across different parts of a system, such as toggling
    a global tracking mode.
    """

    def __init__(self, init: bool):
        """
        Initializes the flag with a given value.

        Args:
            init (bool): Initial value of the flag.
        """
        self._value = init

    def __bool__(self) -> bool:
        """
        Returns the boolean value of the flag.

        Returns:
            bool: The current value of the flag.
        """
        return self._value

    def set(self, value: bool):
        """
        Sets the flag to a new value.

        Args:
            value (bool): The new value to assign to the flag.
        """
        self._value = value


relai_tracking_flag = RELAITrackingFlag(False)  # for root context
relai_tracking_flag_var: ContextVar[RELAITrackingFlag] = ContextVar("relai_tracking_flag", default=relai_tracking_flag)


@contextmanager
def set_current_tracking(flag: RELAITrackingFlag):
    """
    A context manager to set the tracking flag for the enclosed context.
    """
    token = relai_tracking_flag_var.set(flag)
    try:
        yield
    finally:
        relai_tracking_flag_var.reset(token)


def get_current_tracking() -> RELAITrackingFlag:
    """
    Returns the tracking flag instance for the current execution context.
    """
    return relai_tracking_flag_var.get()


def tracking_on():
    """
    Enables tracking by setting the global `relai_tracking_flag` to True.
    """
    relai_tracking_flag = get_current_tracking()
    relai_tracking_flag.set(True)


def tracking_off():
    """
    Disables tracking by setting the global `relai_tracking_flag` to False.
    """
    relai_tracking_flag = get_current_tracking()
    relai_tracking_flag.set(False)
