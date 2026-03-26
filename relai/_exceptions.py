class RELAIError(Exception):
    """Base class for all RELAI exceptions."""

    pass


class ContextLengthExceededError(RELAIError):
    """Raised when a request or Maestro task exceeds the model context limit."""

    pass
