import inspect
from abc import ABC, abstractmethod
from typing import Any, Dict


class BenchmarkStateVector(ABC):

    @classmethod
    def call_signature(cls):
        """
        Returns:
            List[Tuple[str, type]]: A list of (argument_name, argument_type) tuples for the __call__ method.
        """
        sig = inspect.signature(cls.__call__)
        result = []
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            annotation = (
                param.annotation if param.annotation is not inspect._empty else object
            )
            result.append((name, annotation))
        return result

    @abstractmethod
    def __call__(self, *args, **kwds): ...


class BenchmarkStateVectorWithParameters(BenchmarkStateVector):

    @classmethod
    @abstractmethod
    def sample_parameters(cls, *args, **kwargs) -> Dict[str, Any]: ...
