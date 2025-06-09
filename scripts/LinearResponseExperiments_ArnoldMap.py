import gc
import os
import pickle

import numpy as np
import psutil
from joblib import Parallel, delayed
from tqdm import tqdm

from KoopmanismResponse.dynamical_systems.Models import Arnold_map
from KoopmanismResponse.utils.data_processing import get_observables_response_ArnoldMap
from KoopmanismResponse.utils.paths import get_data_folder_path


def print_memory(label=""):
    process = psutil.Process(os.getpid())
    mem_MB = process.memory_info().rss / 1024**2
    print(f"[{label}] Memory usage: {mem_MB:.2f} MB")


def sinusoidal_perturbation(point):
    x, y = point
    amplitude = np.sin(2 * np.pi * (x - 2 * y))
    return amplitude * np.ones_like(point)


def coefficient_perturbation(point):
    x, y = point
    return x * np.array([1, 0])


def single_response_experiment(xi, eps, avg_obs, base_map, perturbation, seed=None):
    try:
        if seed is None:
            seed = np.random.SeedSequence().generate_state(1)[0]
        ss = np.random.SeedSequence(seed)
        rng_p, rng_m = [np.random.default_rng(s) for s in ss.spawn(2)]

        perturbed_map = Arnold_map()
        perturbed_map.M = base_map.M
        perturbed_map.transient = base_map.transient

        perturbed_map.Y0 = xi + eps * perturbation(xi)
        _, resp_p = perturbed_map.integrate(rng=rng_p, show_progress=False)

        perturbed_map.Y0 = xi - eps * perturbation(xi)
        _, resp_m = perturbed_map.integrate(rng=rng_m, show_progress=False)

        obs_p = get_observables_response_ArnoldMap(resp_p) - avg_obs
        obs_m = get_observables_response_ArnoldMap(resp_m) - avg_obs

        return obs_p, obs_m
    except Exception as e:
        print(f"Error in single_response: {e}")
        return None


def main():
    # Unperturbed system
    unperturbed_map = Arnold_map()
    unperturbed_map.M = int(10**6)
    unperturbed_map.set_random_initial_condition()

    t, X = unperturbed_map.integrate()
    avg_obs = get_observables_response_ArnoldMap(X).mean(axis=0)

    # Perturbation experiments settings
    perturbed_map = Arnold_map()
    perturbed_map.M = 50
    perturbed_map.transient = 0
    # amplitudes = [0.01, 0.02, 0.03]  # Coefficient perturbation
    amplitudes = [0.02, 0.04, 0.05]  # Sinusoidal perturbation
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
                n_jobs=-1, batch_size=10  # pyright: ignore[reportArgumentType]
            )(
                delayed(single_response_experiment)(
                    chunk_X[i],
                    eps,
                    avg_obs,
                    perturbed_map,
                    perturbation=sinusoidal_perturbation,
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
    f_name = "response_two_dimensional_map_sinusoidal_perturbation.pkl"

    with open(data_path / f_name, "wb") as f:
        pickle.dump(dictionary, f)


if __name__ == "__main__":
    main()
