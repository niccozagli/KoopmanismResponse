"""
Models.py 

This file contains the definition of the dynamical models used in the project

"""

from typing import List, Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from KoopmanismResponse.config import ArnoldMapSettings, OneDimensionalMapSettings
from KoopmanismResponse.utils.load_config import (
    get_Arnold_model_settings,
    get_one_dimensional_model_settings,
)

ONE_DIM_MODEL_SETTINGS = get_one_dimensional_model_settings()
ARNOLD_MAP_SETTINGS = get_Arnold_model_settings()


class one_dim_map:
    def __init__(
        self,
        model_settings_handler: OneDimensionalMapSettings = ONE_DIM_MODEL_SETTINGS,
    ):
        self.alpha: float = model_settings_handler.alpha
        self.gamma: float = model_settings_handler.gamma
        self.Delta: float = model_settings_handler.Delta

        self.M: int = model_settings_handler.M
        self.transient: int = model_settings_handler.transient

        self.x0: Optional[float] = None
        self.trajectory: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def set_random_initial_condition(self):
        self.x0 = np.random.uniform(0, 2 * np.pi)

    def _drift(self, t, x):
        alpha = self.alpha
        gamma = self.gamma
        Delta = self.Delta

        drift = alpha * x - gamma * np.sin(6 * x) + Delta * np.cos(3 * x)

        return np.mod(drift, 2 * np.pi)

    def integrate(self, show_progress: bool = True):
        if self.x0 is None:
            raise ValueError(
                "Initial condition `x0` has not been set. Call `set_random_initial_condition()` first."
            )

        xold = self.x0
        tsave = np.arange(0, self.M)
        x = np.zeros(len(tsave))

        for idx, t in enumerate(tqdm(tsave, disable=not show_progress)):
            xnew = self._drift(t=t, x=xold)
            xold = xnew.copy()
            x[idx] = xnew

        t = tsave[self.transient :]
        x = x[self.transient :]

        self.trajectory = (t, x)
        return t, x


class Arnold_map:
    def __init__(
        self,
        model_settings_handler: ArnoldMapSettings = ARNOLD_MAP_SETTINGS,
    ):
        self.mu_abs: float = ARNOLD_MAP_SETTINGS.mu_abs
        self.alpha: float = ARNOLD_MAP_SETTINGS.alpha
        self.sigma: float = ARNOLD_MAP_SETTINGS.sigma

        self.M: int = ARNOLD_MAP_SETTINGS.M
        self.transient: int = ARNOLD_MAP_SETTINGS.transient

        self.Y0: Optional[np.ndarray] = None
        self.trajectory: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def _diffusion(self, t, Y):
        diffusion = self.sigma * np.eye(2)
        return diffusion

    def _drift(self, t, Y):
        s = np.sum(Y)
        mu_abs = self.mu_abs
        alpha = self.alpha

        A = np.array([[2, 1], [1, 1]])
        num = mu_abs * np.sin(2 * np.pi * s - alpha)
        den = 1 - mu_abs * np.cos(2 * np.pi * s - alpha)
        zeta = np.atan(num / den)

        return A @ Y + 1 / np.pi * zeta * np.ones_like(Y)

    def set_random_initial_condition(self):
        self.Y0 = np.random.uniform(0, 1, size=2)

    def integrate(
        self, show_progress: bool = True, rng: Optional[np.random.Generator] = None
    ):
        if self.Y0 is None:
            raise ValueError(
                "Initial condition `Y0` has not been set. Call `set_random_initial_condition()` first."
            )

        t = np.arange(0, self.M)

        Y = np.zeros((len(t), 2))

        Yold = self.Y0

        if rng is None:
            rng = np.random.default_rng()

        for idx, tt in enumerate(tqdm(t, disable=not show_progress)):

            f = self._drift(t=tt, Y=Yold)
            g = self._diffusion(t=tt, Y=Yold)
            noise = rng.normal(0, 1, size=2)

            Ynew = np.mod(f + g @ noise, 1)
            Yold = Ynew.copy()

            Y[tt, :] = Ynew

        tsave = t[self.transient :]
        Y = Y[self.transient :, :]

        self.trajectory = (tsave, Y)
        return tsave, Y
