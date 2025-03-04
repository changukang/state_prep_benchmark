from abc import ABC, abstractmethod


class BenchmarkStateVector(ABC):
    pass

    @abstractmethod
    def __call__(self, *args, **kwds): ...
