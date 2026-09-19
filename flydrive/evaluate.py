"""Evaluation: trained fly vs untrained fly vs PID, plus ablations."""
from __future__ import annotations

import json
import os

import numpy as np

from .controllers import FlyController, PIDController, fitness, rollout
from .snn import Ablation, DEFAULT

ABLATIONS = {
    "intact": Ablation(),
    "H2 left silenced": Ablation(h2_left=True),
    "H2 right silenced": Ablation(h2_right=True),
    "DNa02 right silenced": Ablation(dn_right=True),
    "no motion (EMD off)": Ablation(no_motion=True),
    "no position (edges off)": Ablation(no_position=True),
}


def run_condition(label: str, controller_factory, seeds, duration, **kw) -> dict:
    rows = []
    for sd in seeds:
        ctrl = controller_factory(sd)
        m, _ = rollout(ctrl, seed=sd, duration=duration, **kw)
        m["fitness"] = fitness(m, duration)
        m.pop("log", None)
        rows.append(m)
    agg = {k: float(np.mean([r[k] for r in rows]))
           for k in ("mean_abs_dev", "max_abs_dev", "rms_dev", "lane_exits",
                     "distance_m", "duration_s", "steer_effort", "fitness")}
    agg["crash_rate"] = float(np.mean([r["crashed"] for r in rows]))
    agg["condition"] = label
    agg["n_seeds"] = len(seeds)
    return agg


def evaluate(params=None, seeds=range(5000, 5008), duration: float = 30.0,
             out: str = "artifacts", **kw) -> dict:
    params = DEFAULT if params is None else np.asarray(params, float)
    os.makedirs(out, exist_ok=True)
    seeds = list(seeds)
    results = {"controllers": [], "ablations": []}

    results["controllers"].append(run_condition(
        "Fly SNN (trained)", lambda sd: FlyController(params=params, seed=sd),
        seeds, duration, **kw))
    results["controllers"].append(run_condition(
        "Fly SNN (untrained defaults)",
        lambda sd: FlyController(params=DEFAULT, seed=sd), seeds, duration, **kw))
    results["controllers"].append(run_condition(
        "PID baseline", lambda sd: PIDController(), seeds, duration, **kw))

    for label, abl in ABLATIONS.items():
        results["ablations"].append(run_condition(
            label, lambda sd, a=abl: FlyController(params=params, ablation=a, seed=sd),
            seeds, duration, **kw))

    with open(os.path.join(out, "evaluation.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    return results


def table(rows: list[dict]) -> str:
    head = f"{'condition':<30}{'mean|dev|':>11}{'max|dev|':>10}{'exits':>7}{'crash':>8}{'fitness':>10}"
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(f"{r['condition']:<30}{r['mean_abs_dev']:>10.3f}m"
                     f"{r['max_abs_dev']:>9.3f}m{r['lane_exits']:>7.2f}"
                     f"{r['crash_rate']:>8.0%}{r['fitness']:>10.3f}")
    return "\n".join(lines)
