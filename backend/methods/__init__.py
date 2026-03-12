"""
Experimental method registry.

Re-exports from measurement_design.methods for backward compatibility.
"""
from measurement_design.methods import (
    ABTest,
    DifferenceInDifferences,
    DoubleDML,
    GeoLiftTest,
    SyntheticControl,
    MatchedMarketTest,
    ALL_METHODS,
    METHOD_MAP,
)

__all__ = [
    "ABTest",
    "DifferenceInDifferences",
    "DoubleDML",
    "GeoLiftTest",
    "SyntheticControl",
    "MatchedMarketTest",
    "ALL_METHODS",
    "METHOD_MAP",
]
