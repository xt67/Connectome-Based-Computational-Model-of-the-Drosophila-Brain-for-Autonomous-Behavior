"""Connectome-constrained spiking network: T4/T5 input -> H2 -> DNa02.

Every synapse *sign and relative strength* comes from the connectome. Training
never rewires the brain; it only tunes a handful of free biophysical
parameters that the connectome does not specify: input gain, membrane time
constants, thresholds, tonic drive and the motor read-out gain. That is the
honest version of "training a fly to drive".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .connectome import Connectome

# Free parameters, in the order used by the optimiser.
PARAM_NAMES = [
    "visual_gain",    # T4/T5 pooled motion -> H2 drive
    "edge_gain",      # tonic positional component of the visual drive
    "h2_tau",         # H2 membrane time constant (s)
    "h2_thresh",      # H2 spike threshold
    "h2_bias",        # H2 tonic drive (LPTCs have a resting rate)
    "dn_tau",         # DNa02 membrane time constant (s)
    "dn_thresh",
    "dn_bias",
    "motor_gain",     # DNa02 rate difference -> steering command
    "motor_tau",      # steering smoothing (s)
]
PARAM_LOW = np.array([0.05, 0.05, 0.005, 0.3, -0.5, 0.005, 0.3, -0.5, 0.05, 0.02])
PARAM_HIGH = np.array([6.00, 6.00, 0.120, 2.0, 1.50, 0.120, 2.0, 1.50, 6.00, 0.40])
DEFAULT = np.array([1.0, 1.0, 0.030, 1.0, 0.40, 0.040, 1.0, 0.35, 1.0, 0.12])


@dataclass
class Ablation:
    h2_left: bool = False
    h2_right: bool = False
    dn_left: bool = False
    dn_right: bool = False
    no_motion: bool = False     # remove EMD input, keep positional edges
    no_position: bool = False   # remove positional edges, keep EMD


class FlyBrain:
    """Leaky integrate-and-fire populations wired by the connectome."""

    def __init__(self, conn: Connectome, params: np.ndarray | None = None,
                 n_per_pop: int = 12, ablation: Ablation | None = None,
                 noise: float = 0.05, rng: np.random.Generator | None = None):
        self.c = conn
        self.p = dict(zip(PARAM_NAMES, DEFAULT if params is None else params))
        self.n = n_per_pop
        self.abl = ablation or Ablation()
        self.noise = noise
        self.rng = rng or np.random.default_rng(0)
        self.reset()

    def reset(self) -> None:
        z = lambda: np.zeros(self.n)
        self.v = {k: z() for k in ("h2_l", "h2_r", "dn_l", "dn_r")}
        self.rate = {k: 0.0 for k in self.v}
        self.spikes = {k: np.zeros(self.n, dtype=bool) for k in self.v}

    # -- one population update ------------------------------------------
    def _lif(self, key: str, drive: float, tau: float, thresh: float,
             dt: float) -> float:
        v = self.v[key]
        v += dt / tau * (-v + drive + self.rng.normal(0, self.noise, self.n))
        fired = v > thresh
        v[fired] = 0.0
        self.spikes[key] = fired
        inst = fired.mean() / dt
        self.rate[key] += min(dt / 0.05, 1.0) * (inst - self.rate[key])
        return self.rate[key]

    # -- full step -------------------------------------------------------
    def step(self, motion_l: float, motion_r: float,
             edge_l: float, edge_r: float, dt: float) -> float:
        """Inputs are left/right hemifield motion energy and edge proximity.

        Returns a steering command in [-1, 1] (positive = right).
        """
        p = self.p
        mg = 0.0 if self.abl.no_motion else p["visual_gain"]
        eg = 0.0 if self.abl.no_position else p["edge_gain"]

        t4_l = mg * motion_l + eg * edge_l
        t4_r = mg * motion_r + eg * edge_r

        # contralateral projection, straight from the connectome
        drive_h2_l = (self.c.w("T4T5_R", "H2_L") * t4_r
                      + self.c.w("H2_R", "H2_L") * self.rate["h2_r"] / 20.0
                      + p["h2_bias"])
        drive_h2_r = (self.c.w("T4T5_L", "H2_R") * t4_l
                      + self.c.w("H2_L", "H2_R") * self.rate["h2_l"] / 20.0
                      + p["h2_bias"])
        if self.abl.h2_left:
            drive_h2_l = -5.0
        if self.abl.h2_right:
            drive_h2_r = -5.0

        r_h2_l = self._lif("h2_l", drive_h2_l, p["h2_tau"], p["h2_thresh"], dt)
        r_h2_r = self._lif("h2_r", drive_h2_r, p["h2_tau"], p["h2_thresh"], dt)

        drive_dn_l = (self.c.w("H2_L", "DNa02_L") * r_h2_l / 20.0
                      + self.c.w("H2_R", "DNa02_L") * r_h2_r / 20.0
                      + p["dn_bias"])
        drive_dn_r = (self.c.w("H2_R", "DNa02_R") * r_h2_r / 20.0
                      + self.c.w("H2_L", "DNa02_R") * r_h2_l / 20.0
                      + p["dn_bias"])
        if self.abl.dn_left:
            drive_dn_l = -5.0
        if self.abl.dn_right:
            drive_dn_r = -5.0

        r_dn_l = self._lif("dn_l", drive_dn_l, p["dn_tau"], p["dn_thresh"], dt)
        r_dn_r = self._lif("dn_r", drive_dn_r, p["dn_tau"], p["dn_thresh"], dt)

        # In walking flies, turn direction follows the DNa02 left/right
        # difference. Same read-out here.
        cmd = p["motor_gain"] * (r_dn_r - r_dn_l) / 20.0
        return float(np.clip(cmd, -1.0, 1.0))

    def rates(self) -> dict[str, float]:
        return dict(self.rate)


def clip_params(x: np.ndarray) -> np.ndarray:
    return np.clip(x, PARAM_LOW, PARAM_HIGH)
