from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import product
from typing import DefaultDict, Dict, List, Tuple, cast

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from KoopmanismResponse.config import EdmdFourierSettings
from KoopmanismResponse.utils.load_config import get_edmd_Fourier_settings

# from LorenzEDMD.utils.data_processing import (
#     get_spectral_properties,
#     find_index
# )
# from LorenzEDMD.utils.load_config import get_edmd_settings

EDMD_FOURIER_SETTINGS = get_edmd_Fourier_settings()


def fourier_indices(K_max: int, dim: int = 2) -> List[Tuple[int, ...]]:
    return list(product(range(-K_max, K_max + 1), repeat=dim))


######### EDMD CLASS ##########


class BaseEDMD(ABC):
    def __init__(self, flight_time: int = 1):
        self.flight_time = flight_time
        self.G = None
        self.A = None
        self.K = None

    @abstractmethod
    def evaluate_dictionary_batch(
        self, data: NDArray[np.float64]
    ) -> NDArray[np.complex128]:
        pass

    def _create_edmd_snapshots(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.flight_time < 1:
            raise ValueError("Flight time must be >= 1.")
        N = data.shape[0]
        if self.flight_time >= N:
            raise ValueError(
                f"Flight time = {self.flight_time} is too large for data length {N}."
            )
        return data[: -self.flight_time], data[self.flight_time :]

    def perform_edmd(self, data: np.ndarray, batch_size: int = 10_000) -> np.ndarray:

        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        X, Y = self._create_edmd_snapshots(data)
        N = X.shape[0]
        n_features = self.evaluate_dictionary_batch(X[:1]).shape[1]

        G = np.zeros((n_features, n_features), dtype=np.complex128)
        A = np.zeros((n_features, n_features), dtype=np.complex128)

        for start in tqdm(range(0, N, batch_size)):
            end = min(start + batch_size, N)
            X_batch = X[start:end]
            Y_batch = Y[start:end]

            Phi_X = self.evaluate_dictionary_batch(X_batch)
            Phi_Y = self.evaluate_dictionary_batch(Y_batch)

            G += Phi_X.T.conj() @ Phi_X
            A += Phi_X.T.conj() @ Phi_Y

        L = N  # total number of snapshot pairs
        G /= L
        A /= L

        self.G = G
        self.A = A
        self.K = np.linalg.solve(G, A)
        return self.K


# ---------------------- Fourier EDMD ----------------------


class Edmd_Fourier(BaseEDMD):
    def __init__(
        self, edmd_settings_handler: EdmdFourierSettings = EDMD_FOURIER_SETTINGS
    ):
        super().__init__(edmd_settings_handler.flight_time)
        self.max_wave_vector = edmd_settings_handler.max_wave_vector
        self.dimension = edmd_settings_handler.dimension
        self.box_length = edmd_settings_handler.box_length

    def _set_indices(self):
        self.indices = fourier_indices(K_max=self.max_wave_vector, dim=self.dimension)

    def evaluate_dictionary_batch(self, data: np.ndarray) -> np.ndarray:
        """
        Evaluate the Fourier dictionary using exp(2pi / L i * k · x) in fully vectorised way.
        Works for 1D or higher-dimensional data.
        """
        if data.ndim == 1:
            data = data[:, np.newaxis]  # Ensure shape (T, d=1)

        T, d = data.shape
        L = self.box_length
        dimension = self.dimension
        N = len(self.indices)
        Psi = np.empty((T, N), dtype=np.complex128)

        # Stack wavevectors into an array of shape (N, d)
        K = np.array(self.indices)  # shape (N, d)

        # Compute dot products
        dot_products = data @ K.T

        Psi = np.exp(2j * np.pi / L * dot_products) / (
            L ** (dimension / 2)
        )  # shape (T, N)
        return Psi

    def evaluate_koopman_eigenfunctions_batch(
        self, data: np.ndarray, eigenvectors: np.ndarray
    ) -> np.ndarray:
        Psi_X = self.evaluate_dictionary_batch(data)
        return Psi_X @ eigenvectors
