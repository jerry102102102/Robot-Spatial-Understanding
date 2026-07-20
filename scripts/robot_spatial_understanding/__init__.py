"""Public SDK for simulation-grounded robot result understanding."""

from .adapters import SimulatorAdapter
from .benchmark import BenchmarkSuite
from .counterfactual import CounterfactualAssurance
from .predicates import PredicateEngine
from .report import AssuranceReport
from .simulation import SimulationRun
from .task import TaskSpec

__all__ = [
    "AssuranceReport",
    "BenchmarkSuite",
    "CounterfactualAssurance",
    "PredicateEngine",
    "SimulationRun",
    "SimulatorAdapter",
    "TaskSpec",
]

__version__ = "0.2.0"
