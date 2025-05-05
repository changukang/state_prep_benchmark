from abc import ABC, abstractmethod


class BenchmarkStateVector(ABC):

    @abstractmethod
    def __call__(self, *args, **kwds): ...
