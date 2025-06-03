"""
config.py

This file contains the settings class for the dynamical system
"""

from typing import Dict, Literal, Optional, Union

import numpy as np
from pydantic_settings import BaseSettings


class OneDimensionalMapSettings(BaseSettings):
    alpha: float = 3
    gamma: float = 0.4
    Delta: float = 0.08

    M: int = 10**5
    transient: int = 500


class ArnoldMapSettings(BaseSettings):
    mu_abs: float = 0.88
    alpha: float = -2.4
    sigma: float = 0.01

    M: int = 10**5
    transient: int = 500


class EdmdFourierSettings(BaseSettings):
    max_wave_vector: int = 5
    flight_time: int = 1
    dimension: int = 2
