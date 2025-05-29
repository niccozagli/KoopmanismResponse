from typing import List, Tuple, cast

import numpy as np
import statsmodels.api as sm
from numpy.typing import NDArray
from scipy.linalg import eig


def get_spectral_properties(K: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns the sorted (decreasing orders in terms of absolute value) of eigenvalues
    and eigenvectors of the Koopman matrix
    """

    eig_result = cast(
        tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128]],
        eig(K, left=True, right=True),
    )

    eigenvalues, right_eigenvectors, left_eigenvectors = eig_result

    # Sort indices by decreasing magnitude of eigenvalues
    sorted_indices = np.argsort(np.abs(eigenvalues))[::-1]

    # Apply sorting
    eigenvalues = eigenvalues[sorted_indices]
    right_eigenvectors = right_eigenvectors[:, sorted_indices]
    left_eigenvectors = left_eigenvectors[:, sorted_indices]

    diag = np.diag(left_eigenvectors.T.conj() @ right_eigenvectors)
    scale_factors = 1.0 / np.sqrt(diag)
    right_eigenvectors_normalised = right_eigenvectors * scale_factors[np.newaxis, :]
    left_eigenvectors_normalised = (
        left_eigenvectors * scale_factors[np.newaxis, :].conj()
    )
    return eigenvalues, right_eigenvectors_normalised, left_eigenvectors_normalised


def check_if_complex(obs: np.ndarray):
    return np.iscomplex(obs).any()


# def get_acf(
#     obs: np.ndarray,
#     Dt: float,
#     nlags: int = 1500,
# ):
#     is_complex = check_if_complex(obs)
#     if is_complex:
#         obs_real, obs_imag = np.real(obs), np.imag(obs)
#         cf_real = sm.tsa.acf(obs_real, nlags=nlags) * np.var(obs_real)
#         cf_imag = sm.tsa.acf(obs_imag, nlags=nlags) * np.var(obs_imag)
#         cf = cf_real + cf_imag
#     else:
#         cf = sm.tsa.acf(obs, nlags=nlags) * np.var(obs)

#     lags = np.linspace(0, nlags * Dt, nlags + 1)
#     return lags, cf


def Koopman_correlation_function(t, M, alpha1, alpha2, eigenvalues, to_include=None):
    if to_include is None:
        to_include = len(eigenvalues)

    alpha1 = alpha1[1 : to_include + 1]
    alpha2 = alpha2[1 : to_include + 1]
    eigenvalues = eigenvalues[1 : to_include + 1]
    M = M[1 : to_include + 1, 1 : to_include + 1]

    return (alpha1 * np.exp(t * eigenvalues)) @ M @ np.conj(alpha2)
