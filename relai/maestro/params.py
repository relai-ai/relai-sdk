import copy
import json
from typing import Any, Optional

from ..flags import get_current_tracking
from ..utils import get_current_logger


class RegisteredParameters:
    """
    A manager for optimizable parameters.
    """

    def __init__(self):
        """
        Initializes an empty parameter registry.
        """
        self._params = {}

    def __getattr__(self, name: str) -> Any:
        """
        Access a parameter's current value by name.

        Args:
            name (str): Name of the parameter.

        Returns:
            Any: The current value of the parameter.

        Raises:
            Exception: If the parameter is not registered.
        """
        if name in self._params:
            if get_current_tracking():
                logger = get_current_logger()
                logger._param_usage(param_name=name)
            return self._params[name]["current_value"]
        raise Exception(f"'{name}' is not a registered parameter.")

    def __deepcopy__(self, memo: dict) -> "RegisteredParameters":
        """
        Returns a deep copy of this parameter registry.

        Args:
            memo (dict): Internal memoization dict used during deepcopy.

        Returns:
            RegisteredParameters: A deep copy of the current instance.
        """
        ret = RegisteredParameters()
        ret._params = copy.deepcopy(self._params, memo)
        memo[id(self)] = ret
        return ret

    def sync(self, params: dict):
        """
        Replaces the current parameter registry with a deep copy of another one.

        Args:
            params (RegisteredParameters): Source parameter registry to copy from.
        """
        self._params = copy.deepcopy(params._params)  # type: ignore

    def type(self, name: str) -> str:
        """
        Returns the type of the given parameter.

        Args:
            name (str): Parameter name.

        Returns:
            str: Type of the parameter.

        Raises:
            Exception: If the parameter is not registered.
        """
        if name not in self._params:
            raise Exception(f"'{name}' is not a registered parameter.")

        return self._params[name]["type"]

    def type_order(self, name: str) -> int:
        """
        Returns a numeric ordering for the parameter type, used for sorting.

        Args:
            name (str): Parameter name.

        Returns:
            int: Ordering index for the parameter type.

        Raises:
            Exception: If the parameter is not registered.
        """
        if name not in self._params:
            raise Exception(f"'{name}' is not a registered parameter.")

        return {
            "model": 0,
            "prompt": 1,
            "string": 2,
            "integer": 2,
            "float": 2,
        }[self._params[name]["type"]]

    def register(self, name: str, type: str, init_value: Any, desc: str, allowed: Optional[list[Any]] = None):
        """
        Registers a new (optimizable) parameter.

        Args:
            name (str): Parameter name.
            type (str): Parameter type. Must be one of "prompt", "model", "string", "integer", or "float".
            init_value (Any): Initial value of the parameter.
            desc (str): Description of the parameter.
            allowed (Optional[list[Any]]): Optional list of allowed values.

        Raises:
            Exception: If the parameter is already registered.
            ValueError: If the specified `type` of the parameter is not recognized.
        """
        if name in self._params:
            raise Exception("Detecting an attempt to register duplicate parameters.")

        if type not in ["prompt", "model", "string", "integer", "float"]:
            raise ValueError(f"Unrecognized parameter type: {type}")

        self._params[name] = {
            "type": type,
            "current_value": init_value,
            "description": desc,
        }

        if allowed is not None:
            self._params[name]["allowed"] = allowed

    def update(self, name: str, value: Any):
        """
        Updates the value of a registered parameter.

        Args:
            name (str): Parameter name.
            value (Any): New value to assign.
        """
        if self._params[name]["type"] in ["prompt", "model", "string"]:
            if "allowed" not in self._params[name] or value in self._params[name]["allowed"]:
                self._params[name]["current_value"] = value
        elif self._params[name]["type"] == "integer":
            if "allowed" not in self._params[name] or int(value) in self._params[name]["allowed"]:
                self._params[name]["current_value"] = int(value)
        elif self._params[name]["type"] == "float":
            if "allowed" not in self._params[name] or float(value) in self._params[name]["allowed"]:
                self._params[name]["current_value"] = float(value)

    def contain(self, name: str) -> bool:
        """
        Checks if a parameter is registered.

        Args:
            name (str): Parameter name.

        Returns:
            bool: True if the parameter exists, False otherwise.
        """
        return name in self._params

    def all(self) -> list[str]:
        """
        Returns all registered parameter names.

        Returns:
            list[str]: List of parameter names.
        """
        return [name for name in self._params]

    def serialize(self) -> str:
        """
        Serializes all parameters into a JSON string.

        Returns:
            str: JSON-formatted string representing all parameters.
        """
        return json.dumps(self._params, indent=2)

    def serialize_conditional(
        self, relevant_params: set[str], fixed_params: set[str], proposed_values: dict[str, Any]
    ):
        """
        Serializes a filtered set of parameters with conditional visibility.

        Args:
            relevant_params (set[str]): Parameters to include in the output.
            fixed_params (set[str]): Parameters whose values are fixed.
            proposed_values (dict[str, Any]): Proposed new values for non-fixed parameters.

        Returns:
            str: JSON string of conditionally serialized parameters.
        """
        visible_params = {}
        for name in self._params.keys():
            if name not in relevant_params:
                continue

            visible_params[name] = copy.deepcopy(self._params[name])
            visible_params[name]["last_value"] = visible_params[name]["current_value"]

            if name in fixed_params:
                visible_params[name]["current_value"] = "<same as last_value>"
            else:
                if name in proposed_values:
                    visible_params[name]["current_value"] = proposed_values[name]
                else:
                    visible_params[name]["current_value"] = "<to be decided>"

        return json.dumps(visible_params, indent=2)

    def save(self, path: str):
        """
        Saves all parameters to a JSON file.

        Args:
            path (str): File path to save the parameters.
        """
        with open(path, "w") as f:
            json.dump(self._params, f, indent=2)

    def load(self, path: str):
        """
        Loads parameters from a JSON file.

        Args:
            path (str): File path to load the parameters from.
        """
        with open(path, "r") as f:
            self._params = json.load(f)

    def export(self) -> dict[str, Any]:
        """
        Exports all parameters in the format expected by agent backend.

        Returns:
            dict[str, Any]: Dictionary representation of all parameters.
        """
        exported_params = {}
        for name, param in self._params.items():
            exported_params[name] = {
                "name": name,
                "type": param["type"],
                "current_value": param["current_value"],
                "description": param["description"],
            }
            if "allowed" in param:
                exported_params[name]["allowed_values"] = param["allowed"]
            logger = get_current_logger()
            exported_params[name]["execution_order"] = logger._param_order(name)
        return exported_params


params = RegisteredParameters()


def register_param(name: str, type: str, init_value: Any, desc: str, allowed: Optional[list[Any]] = None):
    """
    Register an optmizable parameter in the global `params` registry.

    Args:
        name (str): Parameter name.
        type (str): Parameter type.  Must be one of "prompt", "model", "string", "integer", or "float".
        init_value (Any): Initial value of the parameter.
        desc (str): Description of the parameter.
        allowed (Optional[list[Any]]): Optional list of allowed values.
    """
    params.register(name=name, type=type, init_value=init_value, desc=desc, allowed=allowed)
