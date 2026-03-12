"""
Domain validation — feasibility checks, red flag detection, and parameter defaults.
"""
from .feasibility import (
    unit_label_for_method,
    check_params_sufficient,
    detect_red_flags,
    run_interim_power,
    apply_defaults,
    build_assumptions_summary,
    identify_design_problems,
    run_validation,
)

__all__ = [
    "unit_label_for_method",
    "check_params_sufficient",
    "detect_red_flags",
    "run_interim_power",
    "apply_defaults",
    "build_assumptions_summary",
    "identify_design_problems",
    "run_validation",
]
