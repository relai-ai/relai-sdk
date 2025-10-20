from abc import ABC, abstractmethod
from typing import Any


class BaseMocker(ABC):
    def __init__(self) -> None:
        super().__init__()
        self._func_doc: str | None = None
        self._output_type: type | None = None

    @property
    def func_doc(self) -> str | None:
        return self._func_doc

    @func_doc.setter
    def func_doc(self, value: str | None) -> None:
        if self._func_doc is not None:
            raise ValueError("func_doc is already set")
        self._func_doc = value

    @property
    def output_type(self) -> type | None:
        return self._output_type

    @output_type.setter
    def output_type(self, value: type) -> None:
        if self._output_type is not None:
            raise ValueError("output_type is already set")
        self._output_type = value

    @abstractmethod
    def _run(self, *args: Any, **kwargs: Any) -> Any: ...

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        return self._run(*args, **kwargs)
