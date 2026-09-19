"""Vision front end: a 1-D retina and a Hassenstein-Reichardt motion detector.

The fly does not see pixels the way a camera does. It samples the world with a
ring of ommatidia and computes local motion with correlation-type elementary
motion detectors (EMDs), the T4/T5 circuit. We reproduce that at low
resolution:

1. Render the road into a 1-D luminance profile across azimuth.
2. Low-pass one channel, correlate it with its undelayed neighbour, subtract
   the mirror-image correlation. That gives a direction-selective signal per
   ommatidium pair.
3. Pool the left and right visual hemifields, which is what an LPTC like H2
   integrates over.
"""
from __future__ import annotations

import numpy as np

N_OMMATIDIA = 32
FOV_DEG = 140.0


class Retina:
    """1-D luminance sampler over the frontal field of view."""

    def __init__(self, n: int = N_OMMATIDIA, fov_deg: float = FOV_DEG):
        self.n = n
        self.az = np.linspace(-np.radians(fov_deg) / 2,
                              np.radians(fov_deg) / 2, n)
        # Gaussian acceptance angle blur, as in real ommatidia
        self.blur = np.exp(-0.5 * (np.subtract.outer(self.az, self.az)
                                   / np.radians(5.0)) ** 2)
        self.blur /= self.blur.sum(axis=1, keepdims=True)

    def sample(self, env) -> np.ndarray:
        """Return luminance in [0, 1] per ommatidium for the current state."""
        img = np.zeros(self.n)
        for d in (4.0, 8.0, 14.0, 22.0, 34.0):
            centre = env.center_at(env.s + d) - env.lat
            half = env.lane_half
            # azimuth of the two road edges at this lookahead distance
            a_left = np.arctan2(centre - half, d) - env.heading
            a_right = np.arctan2(centre + half, d) - env.heading
            inside = (self.az > a_left) & (self.az < a_right)
            img += inside * (1.0 / d)          # nearer contours weigh more
            # the edges themselves are the high-contrast features
            for a in (a_left, a_right):
                img += 0.6 * np.exp(-0.5 * ((self.az - a) / np.radians(3)) ** 2) / d

        for o in env.obstacles:
            if not (2.0 < o.d < 60.0):
                continue
            x = env.center_at(env.s + o.d) + o.side * 1.7 - env.lat
            a = np.arctan2(x, o.d) - env.heading
            width = np.arctan2(0.5 if o.ped else 1.0, o.d)
            img -= 1.2 * np.exp(-0.5 * ((self.az - a) / max(width, 1e-3)) ** 2) / o.d

        img = self.blur @ img
        span = np.ptp(img)
        return (img - img.min()) / span if span > 1e-9 else np.zeros(self.n)


class EMDArray:
    """Correlation-type motion detectors with a first-order delay line."""

    def __init__(self, n: int = N_OMMATIDIA, tau: float = 0.035):
        self.tau = tau
        self.delayed = np.zeros(n)
        self.prev = np.zeros(n)
        self.n = n

    def reset(self) -> None:
        self.delayed[:] = 0.0
        self.prev[:] = 0.0

    def step(self, img: np.ndarray, dt: float) -> np.ndarray:
        """Return per-pair direction-selective response (+ = front-to-back)."""
        alpha = dt / (self.tau + dt)
        self.delayed += alpha * (self.prev - self.delayed)
        # Reichardt: delayed(i) * undelayed(i+1) - undelayed(i) * delayed(i+1)
        resp = (self.delayed[:-1] * img[1:]) - (img[:-1] * self.delayed[1:])
        self.prev = img.copy()
        return resp

    @staticmethod
    def pool(resp: np.ndarray) -> tuple[float, float]:
        """Pool into left and right hemifield motion energy.

        The right hemifield is mirrored so that both values are expressed in
        back-to-front terms for their own eye, which is the direction H2
        prefers. A yaw rotation then drives the two pools in *opposite*
        directions, which is exactly the rotation signal the fly uses to
        stabilise its course; pure forward translation drives them together.
        """
        half = len(resp) // 2
        return float(resp[:half].sum()), float(-resp[half:].sum())
