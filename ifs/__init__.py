"""Integral field spectroscopy of slsim lenses."""

from .cube import Cube, chromatic_microlensing, simulate
from .instrument import LazuliIFS

__all__ = ["Cube", "LazuliIFS", "chromatic_microlensing", "simulate"]
