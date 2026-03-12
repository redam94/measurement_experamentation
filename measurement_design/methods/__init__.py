"""
Experimental method registry.
"""
from .ab_test import ABTest
from .did import DifferenceInDifferences
from .ddml import DoubleDML
from .geo_lift import GeoLiftTest
from .synthetic_control import SyntheticControl
from .matched_market import MatchedMarketTest

ALL_METHODS: list = [
    ABTest(),
    DifferenceInDifferences(),
    DoubleDML(),
    GeoLiftTest(),
    SyntheticControl(),
    MatchedMarketTest(),
]

METHOD_MAP: dict = {m.key: m for m in ALL_METHODS}

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
