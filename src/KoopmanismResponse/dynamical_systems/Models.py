"""
Models.py 

This file contains the definition of the dynamical models used in the project

"""

from typing import List, Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from KoopmanismResponse.config import OneDimensionalMapSettings
from KoopmanismResponse.utils.load_config import get_one_dimensional_model_settings

ONE_DIM_MODEL_SETTINGS = get_one_dimensional_model_settings()


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

    # def _diffusion(self, t, Y):
    #     diffusion = self.noise * np.eye(3)
    #     return diffusion

    # def integrate_EM(self, show_progress: bool = True, rng: np.random.Generator = None):
    #     t0, tf = self.t_span
    #     n_steps = int((tf - t0) / self.dt)
    #     ts = np.linspace(t0, tf, n_steps + 1)

    #     tsave = ts[: -1 : self.tau]
    #     ysave = np.zeros((len(tsave), 3))

    #     yold = self.y0

    #     if rng is None:
    #         rng = np.random.default_rng()

    #     index = 0
    #     for i in tqdm(range(n_steps), disable=not show_progress):
    #         t = ts[i]
    #         f = self._drift(t=t, Y=yold)
    #         g = self._diffusion(t=t, Y=yold)
    #         dW = rng.normal(0, np.sqrt(self.dt), size=3)

    #         ynew = yold + f * self.dt + g @ dW

    #         if np.mod(i, self.tau) == 0:
    #             ysave[index, :] = ynew
    #             index += 1

    #         yold = ynew.copy()

    #     ind_transient = np.where(tsave >= self.transient)[0][0]
    #     tsave = tsave[ind_transient:]
    #     ysave = ysave[ind_transient:, :]

    #     self.trajectory = (tsave, ysave)
    #     return tsave, ysave

    # def plot3d_trajectory(self, figsize=(8, 6), color="royalblue", alpha=0.8, lw=0.7):
    #     if self.trajectory is None:
    #         raise ValueError("No trajectory found. Run integrate() first.")

    #     ts, ys = self.trajectory

    #     fig = plt.figure(figsize=figsize)
    #     ax = fig.add_subplot(111, projection="3d")
    #     ax.plot(ys[:, 0], ys[:, 1], ys[:, 2], color=color, lw=lw, alpha=alpha)

    #     ax.set_xlabel("x")
    #     ax.set_ylabel("y")
    #     ax.set_zlabel("z")
    #     ax.set_title("Lorenz Attractor")

    #     plt.tight_layout()
    #     plt.show()
