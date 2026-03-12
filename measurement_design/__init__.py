"""
measurement_design — Core domain library for measurement experiment design.

Pure domain logic with no framework dependencies (no FastAPI, LangChain,
LangGraph, or Streamlit). Provides experimental methods, scoring, simulation,
output generation, and domain knowledge.
"""
from .types import (
    ElicitedFacts,
    Phase,
    SetupParams,
    SetupPhase,
    PowerResults,
    MDEResults,
    SyntheticDataResult,
    ValidationResult,
    RedFlag,
    RedFlagSeverity,
    ELICITATION_TOPICS,
    SETUP_TOPICS,
    MAX_FOLLOWUP_ROUNDS,
)
from .models import DesignSpec, ExperimentMethod, clamp

__all__ = [
    # Types
    "ElicitedFacts",
    "Phase",
    "SetupParams",
    "SetupPhase",
    "PowerResults",
    "MDEResults",
    "SyntheticDataResult",
    "ValidationResult",
    "RedFlag",
    "RedFlagSeverity",
    "ELICITATION_TOPICS",
    "SETUP_TOPICS",
    "MAX_FOLLOWUP_ROUNDS",
    # Models
    "DesignSpec",
    "ExperimentMethod",
    "clamp",
]
