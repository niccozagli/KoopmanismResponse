"""
load_config.py

Holds the functions to load the settings specified in `config.py`.
"""

from functools import lru_cache

from KoopmanismResponse.config import OneDimensionalMapSettings


@lru_cache
def one_dimensional_model_settings() -> OneDimensionalMapSettings:
    """
    Loads the settings for the One dimensional Map.
    """
    return OneDimensionalMapSettings()


# @lru_cache
# def get_edmd_settings() -> EDMDSettings:
#     """
#     Loads the settings for the Dynamical Systems.

#     :return: Dynamical System settings object.
#     """
#     return EDMDSettings()
