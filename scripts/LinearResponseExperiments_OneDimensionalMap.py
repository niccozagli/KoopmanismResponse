import gc
import os
import pickle

import numpy as np
import psutil
from joblib import Parallel, delayed
from tqdm import tqdm

from KoopmanismResponse.dynamical_systems.Models import one_dim_map
from KoopmanismResponse.utils.paths import get_data_folder_path


def print_memory(label=""):
    process = psutil.Process(os.getpid())
    mem_MB = process.memory_info().rss / 1024**2
    print(f"[{label}] Memory usage: {mem_MB:.2f} MB")


def get_observables(trajectory: np.ndarray):
    x = trajectory
    observables = (
        np.cos(x),
        np.cos(2 * x),
        np.cos(3 * x),
        np.cos(4 * x),
        np.cos(5 * x),
        np.sin(x),
        np.sin(2 * x),
        np.sin(3 * x),
        np.sin(4 * x),
        np.sin(5 * x),
        1 / (2 + np.sin(2 * x)),
        np.cos(np.atan(3 * np.sin(x))) / np.sin(np.atan(3)),
        np.atan(20 * np.sin(2 * x)) / np.atan(20),
        (1 / 2 + 1 / 2 * np.sin(2 * x)) / (2 + np.cos(10 * x)),
        (x - np.pi) ** 2,
    )
    return np.column_stack(observables)


def uniform_pert(point: np.ndarray):
    return np.array([1])


def single_response_uniform_pertubation(xi, eps, avg_obs, base_map):
    try:
        perturbed_map = one_dim_map()
        perturbed_map.M = base_map.M
        perturbed_map.transient = base_map.transient

        perturbed_map.x0 = xi + eps * uniform_pert(xi)
        _, resp_p = perturbed_map.integrate(show_progress=False)

        perturbed_map.x0 = xi - eps * uniform_pert(xi)
        _, resp_m = perturbed_map.integrate(show_progress=False)

        obs_p = get_observables(resp_p) - avg_obs  # shape (T, N_Observables)
        obs_m = get_observables(resp_m) - avg_obs

        return obs_p, obs_m
    except Exception as e:
        print(f"Error in single_response: {e}")
        return None


def main():
    # Unperturbed system
    unperturbed_map = one_dim_map()
    unperturbed_map.M = int(5 * 10**5)
    unperturbed_map.set_random_initial_condition()

    t, X = unperturbed_map.integrate()
    avg_obs = get_observables(X).mean(axis=0)

    # Perturbation experiments settings
    perturbed_map = one_dim_map()
    perturbed_map.M = 30
    perturbed_map.transient = 0

    amplitudes = [0.03, 0.04, 0.08]
    RESP_P = []
    RESP_M = []

    n_chunks = 10
    chunk_size = int(np.ceil(X.shape[0] / n_chunks))

    for eps in amplitudes:
        print_memory(f"Start eps={eps}")
        resp_p_acc = 0
        resp_m_acc = 0
        count = 0

        for start in range(0, X.shape[0], chunk_size):
            end = min(start + chunk_size, X.shape[0])
            print_memory(f"  Chunk {start}-{end} before run")

            chunk_X = X[start:end]

            results = Parallel(
                n_jobs=-1, batch_size=10 # pyright: ignore[reportArgumentType]
            )(  
                delayed(single_response_uniform_pertubation)(
                    chunk_X[i], eps, avg_obs, perturbed_map
                )
                for i in range(chunk_X.shape[0])
            )

            print_memory(f"  Chunk {start}-{end} after run")

            results = [r for r in results if r is not None]
            if results:
                resp_p_chunk = np.mean([r[0] for r in results], axis=0)
                resp_m_chunk = np.mean([r[1] for r in results], axis=0)
                resp_p_acc += resp_p_chunk
                resp_m_acc += resp_m_chunk
                count += 1

            del results
            gc.collect()

        resp_p_all = resp_p_acc / count
        resp_m_all = resp_m_acc / count

        RESP_P.append(resp_p_all)
        RESP_M.append(resp_m_all)

    unperturbed_map.trajectory = None
    perturbed_map.trajectory = None
    dictionary = {
        "Positive Response": RESP_P,
        "Negative Response": RESP_M,
        "Amplitudes": amplitudes,
        "Response Settings": perturbed_map,
        "Unperturbed Settings": unperturbed_map,
    }
    data_path = get_data_folder_path()
    f_name = "response_one_dimensional_map.pkl"

    with open(data_path / f_name, "wb") as f:
        pickle.dump(dictionary, f)


if __name__ == "__main__":
    main()
