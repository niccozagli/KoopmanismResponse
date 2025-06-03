"""
load_config.py

Holds the functions to load the settings specified in `config.py`.
"""

from functools import lru_cache

from KoopmanismResponse.config import (
    ArnoldMapSettings,
    EdmdFourierSettings,
    OneDimensionalMapSettings,
)


@lru_cache
def get_one_dimensional_model_settings() -> OneDimensionalMapSettings:
    """
    Loads the settings for the One dimensional Map.
    """
    return OneDimensionalMapSettings()


@lru_cache
def get_Arnold_model_settings() -> ArnoldMapSettings:
    """
    Loads the settings for the generalised Arnold Map.
    """
    return ArnoldMapSettings()


@lru_cache
def get_edmd_Fourier_settings() -> EdmdFourierSettings:
    """
    Loads the settings for the Edmd settings.
    """
    return EdmdFourierSettings()
