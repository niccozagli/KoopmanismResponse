"""
config.py

This file contains the settings class for the dynamical system
"""

from typing import Dict, Literal, Optional, Union

from pydantic_settings import BaseSettings


class OneDimensionalMapSettings(BaseSettings):
    alpha: float = 3
    gamma: float = 0.4
    Delta: float = 0.08

    M: int = 10**5
    transient: int = 500


class EdmdFourierSettings(BaseSettings):
    max_wave_vector: int = 5
    flight_time: int = 1
    dimension: int = 2
