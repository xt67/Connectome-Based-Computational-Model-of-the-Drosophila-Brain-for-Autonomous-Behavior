"""The virtual road and the car the fly is learning to drive."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Obstacle:
    d: float
    side: int
    ped: bool
    v: float


@dataclass
class Road:
    seed: int = 0
    curvature: float = 1.0
    _a: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        # three sinusoidal components: amplitude, wavelength, phase
        self._a = np.stack([
            rng.uniform(0.8, 3.0, 3) * self.curvature,
            rng.uniform(18.0, 70.0, 3),
            rng.uniform(0, 2 * np.pi, 3),
        ])

    def center(self, s: float | np.ndarray):
        amp, lam, ph = self._a
        return sum(amp[i] * np.sin(s / lam[i] + ph[i]) for i in range(3))


class DrivingEnv:
    """Kinematic bicycle-ish car on a curving single lane."""

    lane_half = 3.5          # metres from centreline to lane edge
    dt = 0.02

    def __init__(self, seed: int = 0, speed_kmh: float = 18.0,
                 curvature: float = 1.0, n_obstacles: int = 2,
                 duration: float = 60.0):
        self.seed = seed
        self.road = Road(seed, curvature)
        self.target_v = speed_kmh / 3.6
        self.n_obstacles = n_obstacles
        self.duration = duration
        self.reset()

    # -- road -----------------------------------------------------------
    def center_at(self, s):
        return self.road.center(s)

    # -- lifecycle -------------------------------------------------------
    def reset(self) -> None:
        rng = np.random.default_rng(self.seed + 991)
        self.rng = rng
        self.t = 0.0
        self.s = 0.0
        self.lat = self.center_at(0.0) + rng.uniform(-0.6, 0.6)
        self.heading = rng.uniform(-0.05, 0.05)
        self.v = self.target_v
        self.steer = 0.0
        self.brake = 0.0
        self.gas = 0.0
        self.obstacles = [
            Obstacle(d=18 + i * 17 + rng.uniform(0, 8),
                     side=int(rng.choice([-1, 1])),
                     ped=bool(rng.random() < 0.35),
                     v=rng.uniform(2, 5))
            for i in range(self.n_obstacles)
        ]
        self.log = {"t": [], "dev": [], "s": [], "lat": [], "center": [],
                    "steer": [], "speed": []}
        self.crashed = False

    @property
    def deviation(self) -> float:
        return self.lat - self.center_at(self.s)

    @property
    def done(self) -> bool:
        return self.crashed or self.t >= self.duration or abs(self.deviation) > 6.0

    # -- dynamics --------------------------------------------------------
    def step(self, steer_cmd: float, motor_tau: float = 0.12) -> None:
        dt = self.dt
        self.steer += min(dt / motor_tau, 1.0) * (steer_cmd - self.steer)

        risk_d = np.inf
        for o in self.obstacles:
            o.d -= (self.v - o.v) * dt
            if o.d < -8 or o.d > 150:
                o.d = 60 + self.rng.uniform(0, 50)
                o.side = int(self.rng.choice([-1, 1]))
            gap = abs(self.center_at(self.s + o.d) + o.side * 1.7 - self.lat)
            if gap < 1.6:
                risk_d = min(risk_d, o.d)
                if o.d < 1.2 and self.v > o.v + 1.0:
                    self.crashed = True

        target_brake = 1.0 if risk_d < 10 else 0.35 if risk_d < 22 else 0.0
        self.brake += min(dt * 3, 1.0) * (target_brake - self.brake)
        self.gas = float(np.clip(0.55 * (1 - self.brake), 0, 1))

        # full braking brings the car to a crawl behind the obstacle
        v_target = self.target_v * (1 - 0.95 * self.brake)
        self.v += min(dt * 1.2, 1.0) * (v_target - self.v)
        self.s += self.v * dt
        self.heading = float(np.clip(
            self.heading + self.steer * dt * np.clip(self.v / 6, 0.2, 2.2), -0.8, 0.8))
        self.lat += self.heading * self.v * dt
        self.t += dt

        lg = self.log
        lg["t"].append(self.t)
        lg["dev"].append(self.deviation)
        lg["s"].append(self.s)
        lg["lat"].append(self.lat)
        lg["center"].append(self.center_at(self.s))
        lg["steer"].append(self.steer)
        lg["speed"].append(self.v * 3.6)

    # -- what the eyes are given ----------------------------------------
    def edge_signals(self) -> tuple[float, float]:
        """Proximity of the left and right lane edge, lookahead-weighted."""
        off = self.lat - self.center_at(self.s + 8.0)
        left = self.lane_half + off
        right = self.lane_half - off
        return 1.0 / max(left, 0.4), 1.0 / max(right, 0.4)

    # -- metrics ---------------------------------------------------------
    def metrics(self) -> dict:
        dev = np.abs(np.array(self.log["dev"])) if self.log["dev"] else np.array([9.0])
        exits = int(np.sum((dev[1:] > self.lane_half) & (dev[:-1] <= self.lane_half)))
        return {
            "mean_abs_dev": float(dev.mean()),
            "max_abs_dev": float(dev.max()),
            "rms_dev": float(np.sqrt((dev ** 2).mean())),
            "lane_exits": exits,
            "distance_m": float(self.s),
            "duration_s": float(self.t),
            "crashed": bool(self.crashed),
            "steer_effort": float(np.abs(np.diff(self.log["steer"])).sum())
            if len(self.log["steer"]) > 1 else 0.0,
        }
