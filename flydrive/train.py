"""Training: evolve the fly's free biophysical parameters to keep the lane.

Spiking networks with hard thresholds give no usable gradient, and the
connectome fixes the wiring anyway, so we use a derivative-free method. The
cross-entropy method (a simple, robust cousin of CMA-ES) samples parameter
vectors from a Gaussian, keeps the best, and refits the Gaussian to them.

Each candidate is scored on several road seeds so it cannot overfit one track.
"""
from __future__ import annotations

import json
import os
import time
from multiprocessing import Pool

import numpy as np

from .controllers import FlyController, fitness, rollout
from .snn import DEFAULT, PARAM_HIGH, PARAM_LOW, PARAM_NAMES, clip_params


def evaluate_params(args) -> float:
    params, seeds, duration, speed, curvature, obstacles, noise = args
    scores = []
    for sd in seeds:
        ctrl = FlyController(params=params, noise=noise, seed=sd)
        m, _ = rollout(ctrl, seed=sd, duration=duration, speed_kmh=speed,
                       curvature=curvature, n_obstacles=obstacles)
        scores.append(fitness(m, duration))
    return float(np.mean(scores))


def train(generations: int = 25, population: int = 32, elite_frac: float = 0.25,
          seeds_per_eval: int = 3, duration: float = 25.0, speed: float = 18.0,
          curvature: float = 1.0, obstacles: int = 2, noise: float = 0.05,
          sigma0: float = 0.35, workers: int | None = None,
          out: str = "artifacts", rng_seed: int = 0, verbose: bool = True):
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(rng_seed)
    scale = PARAM_HIGH - PARAM_LOW
    mean = DEFAULT.copy()
    sigma = sigma0 * scale
    n_elite = max(2, int(population * elite_frac))
    history = []
    best_ever, best_ever_score = DEFAULT.copy(), -np.inf
    workers = workers or min(os.cpu_count() or 1, 8)

    pool = Pool(workers) if workers > 1 else None
    try:
        for gen in range(generations):
            t0 = time.time()
            # a fresh set of road seeds each generation keeps it honest
            seeds = [int(rng.integers(0, 10_000)) for _ in range(seeds_per_eval)]
            samples = clip_params(rng.normal(mean, sigma, (population, len(mean))))
            jobs = [(s, seeds, duration, speed, curvature, obstacles, noise)
                    for s in samples]
            scores = np.array(pool.map(evaluate_params, jobs) if pool
                              else [evaluate_params(j) for j in jobs])

            order = np.argsort(scores)[::-1]
            elites = samples[order[:n_elite]]
            mean = elites.mean(axis=0)
            sigma = np.maximum(elites.std(axis=0), 0.02 * scale)

            if scores[order[0]] > best_ever_score:
                best_ever_score = float(scores[order[0]])
                best_ever = samples[order[0]].copy()
            record = {"generation": gen, "best": float(scores[order[0]]),
                      "mean": float(scores.mean()),
                      "elite_mean": float(scores[order[:n_elite]].mean()),
                      "seconds": round(time.time() - t0, 2)}
            history.append(record)
            if verbose:
                print(f"gen {gen:>3}  best {record['best']:+.3f}  "
                      f"elite {record['elite_mean']:+.3f}  "
                      f"pop {record['mean']:+.3f}  ({record['seconds']}s)")
    finally:
        if pool:
            pool.close()
            pool.join()

    # Pick between the distribution mean and the best single candidate, judged
    # on roads neither of them was trained on.
    holdout = [90_001 + i for i in range(6)]
    cand = {"distribution mean": mean, "best of run": best_ever}
    scored = {k: evaluate_params((v, holdout, duration, speed, curvature,
                                  obstacles, noise)) for k, v in cand.items()}
    winner = max(scored, key=scored.get)
    mean = cand[winner]
    final_score = scored[winner]
    if verbose:
        print("\nholdout: " + "  ".join(f"{k} {v:+.3f}" for k, v in scored.items())
              + f"  -> keeping {winner}")
    result = {
        "params": {k: float(v) for k, v in zip(PARAM_NAMES, mean)},
        "params_vector": mean.tolist(),
        "holdout_fitness": final_score,
        "history": history,
        "config": {"generations": generations, "population": population,
                   "seeds_per_eval": seeds_per_eval, "duration": duration,
                   "speed_kmh": speed, "curvature": curvature,
                   "obstacles": obstacles, "noise": noise},
    }
    path = os.path.join(out, "trained_params.json")
    with open(path, "w") as fh:
        json.dump(result, fh, indent=2)
    if verbose:
        print(f"\nholdout fitness {final_score:+.3f}  ->  {path}")
    return result


def load_params(path: str = "artifacts/trained_params.json") -> np.ndarray:
    if not os.path.exists(path):
        return DEFAULT.copy()
    with open(path) as fh:
        return np.array(json.load(fh)["params_vector"], dtype=float)
