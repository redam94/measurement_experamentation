"""
Core domain models for measurement experiment design.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DesignSpec:
    """Structured experimental design specification."""

    method_key: str
    method_name: str
    score: float
    explanation: str = ""

    # Design details
    primary_objective: str = ""
    kpi: str = ""
    treatment_assignment: str = ""
    control_definition: str = ""
    randomization_unit: str = ""
    pre_period_weeks: int | None = None
    test_duration_weeks: int | None = None
    num_units: str = ""
    statistical_approach: str = ""
    primary_model: str = ""
    minimum_detectable_effect: str = ""

    # Implementation checklist
    implementation_steps: list[str] = field(default_factory=list)
    data_requirements: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    pros: list[str] = field(default_factory=list)
    cons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


class ExperimentMethod(ABC):
    """
    Abstract base class for an experimental measurement method.
    Subclasses implement:
      - score(facts) → float [0, 100]
      - generate_spec(facts) → DesignSpec
      - generate_scaffold(facts) → str  (PyMC code)
    """

    key: str = ""              # unique snake_case identifier
    name: str = ""             # human-readable name
    short_description: str = ""

    @abstractmethod
    def score(self, facts: dict) -> float:
        """Return a suitability score 0–100 for these elicited facts."""
        ...

    @abstractmethod
    def generate_spec(self, facts: dict, explanation: str = "") -> DesignSpec:
        """Return a populated DesignSpec for this method."""
        ...

    @abstractmethod
    def generate_scaffold(self, facts: dict) -> str:
        """Return a PyMC code scaffold string tailored to these facts."""
        ...


def clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))
