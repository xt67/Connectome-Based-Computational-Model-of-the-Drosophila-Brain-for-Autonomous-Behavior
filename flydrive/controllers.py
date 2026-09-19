"""Controllers and the rollout loop shared by training and evaluation."""
from __future__ import annotations

import numpy as np

from .connectome import Connectome, load as load_connectome
from .env import DrivingEnv
from .snn import Ablation, DEFAULT, FlyBrain, PARAM_NAMES
from .vision import EMDArray, Retina


class FlyController:
    """Connectome SNN driving the car through its own eyes."""

    name = "fly"

    def __init__(self, params: np.ndarray | None = None,
                 conn: Connectome | None = None,
                 ablation: Ablation | None = None,
                 noise: float = 0.05, seed: int = 0):
        self.conn = conn or load_connectome()
        self.params = DEFAULT if params is None else np.asarray(params, float)
        self.brain = FlyBrain(self.conn, self.params, ablation=ablation,
                              noise=noise, rng=np.random.default_rng(seed))
        self.retina = Retina()
        self.emd = EMDArray()
        self.motor_tau = float(self.brain.p["motor_tau"])

    def reset(self) -> None:
        self.brain.reset()
        self.emd.reset()

    def __call__(self, env: DrivingEnv) -> float:
        img = self.retina.sample(env)
        resp = self.emd.step(img, env.dt)
        m_l, m_r = EMDArray.pool(resp)
        e_l, e_r = env.edge_signals()
        return self.brain.step(m_l, m_r, e_l, e_r, env.dt)


class PIDController:
    """Conventional baseline with full access to the true state."""

    name = "pid"

    def __init__(self, kp: float = 1.15, kd: float = 2.4):
        self.kp, self.kd = kp, kd
        self.motor_tau = 0.12

    def reset(self) -> None:
        pass

    def __call__(self, env: DrivingEnv) -> float:
        off = env.lat - env.center_at(env.s + 8.0)
        return float(np.clip(-(self.kp * off + self.kd * env.heading), -1, 1))


def rollout(controller, seed: int = 0, duration: float = 40.0,
            speed_kmh: float = 18.0, curvature: float = 1.0,
            n_obstacles: int = 2, record: bool = False) -> tuple[dict, dict | None]:
    env = DrivingEnv(seed=seed, speed_kmh=speed_kmh, curvature=curvature,
                     n_obstacles=n_obstacles, duration=duration)
    controller.reset()
    trace = {"t": [], "h2_l": [], "h2_r": [], "dn_l": [], "dn_r": [],
             "spikes": []} if record else None
    while not env.done:
        cmd = controller(env)
        env.step(cmd, getattr(controller, "motor_tau", 0.12))
        if record and isinstance(controller, FlyController):
            r = controller.brain.rates()
            trace["t"].append(env.t)
            for k in ("h2_l", "h2_r", "dn_l", "dn_r"):
                trace[k].append(r[k])
            trace["spikes"].append(
                {k: np.flatnonzero(v).tolist()
                 for k, v in controller.brain.spikes.items()})
    m = env.metrics()
    m["log"] = env.log
    return m, trace


def fitness(metrics: dict, duration: float) -> float:
    """Higher is better. Lane keeping first, smoothness second."""
    f = -metrics["rms_dev"]
    f -= 0.6 * metrics["lane_exits"]
    f -= 0.02 * metrics["steer_effort"]
    f += 0.5 * (metrics["duration_s"] / duration)     # reward surviving
    if metrics["crashed"]:
        f -= 2.0
    return float(f)


def describe(params: np.ndarray) -> str:
    return "\n".join(f"  {n:<12} {v:8.4f}" for n, v in zip(PARAM_NAMES, params))
