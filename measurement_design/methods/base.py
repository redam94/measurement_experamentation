"""
Re-export base classes from measurement_design.models for backward compatibility.

Method implementations use `from .base import ExperimentMethod, DesignSpec, clamp`.
"""
from measurement_design.models import ExperimentMethod, DesignSpec, clamp

__all__ = ["ExperimentMethod", "DesignSpec", "clamp"]
