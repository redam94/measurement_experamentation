"""
Simulation package — power analysis, Monte Carlo MDE, and synthetic data generation.
"""
from .power import compute_power
from .mde import compute_mde
from .synthetic import generate_synthetic_data

__all__ = ["compute_power", "compute_mde", "generate_synthetic_data"]
