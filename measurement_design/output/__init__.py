"""
Output package — report, spec, and scaffold generation.
"""
from .report import generate_report
from .spec import generate_spec_json, generate_spec_yaml
from .scaffold import generate_combined_scaffold, get_top_scaffold

__all__ = [
    "generate_report",
    "generate_spec_json",
    "generate_spec_yaml",
    "generate_combined_scaffold",
    "get_top_scaffold",
]
