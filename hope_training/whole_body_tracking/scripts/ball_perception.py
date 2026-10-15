"""Camera / motion-capture model for the MuJoCo evaluator (``mujoco_eval_onnx.py --perception``).

The deploy planner never sees the true ball. A tracking system samples the ball position at a fixed
rate, each sample is noisy, some are lost, and each arrives ``latency_s`` after it was taken. The
velocity is not measured: like HITTER, it is fitted (a quadratic per axis over the last
``window_s``) and the spin is unknown unless ``spin="true"``. :class:`BallTracker` turns the true ball
path (``PingPongRealPhysicsScene.ball_track``, one point per physics sub-step) into exactly that
estimate, so the planner and the policy can be scored with a given camera system.

The presets are assumptions for comparison, not measurements of a particular product (see
``docs/g1_forehand/research/perception.md``).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class CameraModel:
    name: str = "oracle"
    rate_hz: float = 0.0        # 0 = oracle: the planner reads the true ball state every tick
    latency_s: float = 0.0      # capture -> estimate available to the planner
    noise_m: float = 0.0        # per-axis Gaussian std of one triangulated 3D position
    dropout: float = 0.0        # probability that a sample is lost (occlusion, missed detection)
    window_s: float = 0.086     # quadratic fit window (HITTER: 31 samples at 360 Hz)
    min_samples: int = 5
    spin: str = "zero"          # "true": the planner knows the spin; "zero": it assumes none;
                                # "read": an estimate with a per-ball error, e.g. read off the opponent's racket
    spin_rel_err: float = 0.0   # "read": 1-sigma relative error of the spin magnitude
    spin_ang_err_deg: float = 0.0  # "read": 1-sigma error of the spin axis (degrees)

    @property
    def oracle(self) -> bool:
        return self.rate_hz <= 0.0


PRESETS = {
    # the current evaluator: true position, velocity and spin every tick
    "oracle": CameraModel("oracle", spin="true"),
    # true position and velocity, but the spin is unknown (no tracking system measures it directly)
    "oracle_nospin": CameraModel("oracle_nospin", spin="zero"),
    # HITTER-like OptiTrack: 360 Hz, mm accuracy, a few ms of latency
    "mocap_360": CameraModel("mocap_360", rate_hz=360, latency_s=0.005, noise_m=0.001, dropout=0.02),
    "mocap_240": CameraModel("mocap_240", rate_hz=240, latency_s=0.008, noise_m=0.0015, dropout=0.02),
    # two synced global-shutter machine-vision cameras (DeepMind 2024: 125 fps stereo pair)
    "stereo_125": CameraModel("stereo_125", rate_hz=125, latency_s=0.015, noise_m=0.003, dropout=0.05),
    # phones: rolling shutter, no hardware sync, video streamed off the phone
    "iphone_240": CameraModel("iphone_240", rate_hz=240, latency_s=0.060, noise_m=0.008, dropout=0.10),
    "iphone_120": CameraModel("iphone_120", rate_hz=120, latency_s=0.060, noise_m=0.008, dropout=0.10),
    "iphone_60": CameraModel("iphone_60", rate_hz=60, latency_s=0.080, noise_m=0.010, dropout=0.10),
    "webcam_30": CameraModel("webcam_30", rate_hz=30, latency_s=0.100, noise_m=0.015, dropout=0.10),
}


def camera_from_spec(spec: str) -> CameraModel:
    """A preset name, optionally with overrides: ``mocap_360:latency_s=0.03,noise_m=0.005``."""
    name, _, overrides = spec.partition(":")
    if name not in PRESETS:
        raise ValueError(f"unknown perception preset {name!r}; choose from {sorted(PRESETS)}")
    cam = PRESETS[name]
    if overrides:
        kw = {}
        for item in overrides.split(","):
            key, _, value = item.partition("=")
            kw[key] = value if key in ("spin", "name") else (int(value) if key == "min_samples" else float(value))
        cam = replace(cam, name=spec, **kw)
    return cam


@dataclass
class BallEstimate:
    pos: np.ndarray       # at ``t``
    vel: np.ndarray
    spin: np.ndarray
    t: float              # capture time of the newest sample used
    bounced: bool         # a bounce on the robot's half has been seen in the samples


class BallTracker:
    """Samples, delays, corrupts and fits the ball path for one served ball."""

    def __init__(self, cam: CameraModel, rng: np.random.Generator, surface_z: float, net_x: float):
        self.cam = cam
        self.rng = rng
        self.surface_z = float(surface_z)
        self.net_x = float(net_x)
        self.reset(0.0)

    def reset(self, t_now: float) -> None:
        self.samples: list[tuple[float, np.ndarray]] = []
        # one spin-reading error per ball (the opponent's stroke is read once): scale and axis tilt
        self._spin_scale = max(0.0, 1.0 + self.rng.normal(0.0, self.cam.spin_rel_err))
        axis = self.rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        ang = np.deg2rad(self.rng.normal(0.0, self.cam.spin_ang_err_deg))
        k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        self._spin_rot = np.eye(3) + np.sin(ang) * k + (1 - np.cos(ang)) * k @ k
        self._last = None  # last (t, true pos) of the path, for interpolation
        period = 1.0 / self.cam.rate_hz if not self.cam.oracle else 0.0
        self._next_t = t_now + (self.rng.uniform(0.0, period) if period else 0.0)

    def ingest(self, track) -> None:
        """Add camera samples for the true path points ``track`` [(t, pos), ...] (in time order)."""
        if self.cam.oracle:
            return
        period = 1.0 / self.cam.rate_hz
        for t, p in track:
            if self._last is None:
                self._last = (t, p)
                continue
            t0, p0 = self._last
            while self._next_t <= t:
                if self._next_t >= t0:
                    f = (self._next_t - t0) / max(t - t0, 1e-12)
                    if self.rng.random() >= self.cam.dropout:
                        meas = p0 + f * (p - p0) + self.rng.normal(0.0, self.cam.noise_m, 3)
                        self.samples.append((self._next_t, meas))
                self._next_t += period
            self._last = (t, p)

    def spin_estimate(self, true_spin: np.ndarray) -> np.ndarray:
        """What the planner believes the spin is, given the true spin now."""
        if self.cam.spin == "true":
            return np.asarray(true_spin, dtype=np.float64)
        if self.cam.spin == "read":
            return self._spin_scale * (self._spin_rot @ np.asarray(true_spin, dtype=np.float64))
        return np.zeros(3)

    def _bounce_index(self, z: np.ndarray, x: np.ndarray) -> int | None:
        """Index of the lowest sample of a bounce on the robot's half, once the ball is clearly rising."""
        tol = 3.0 * self.cam.noise_m + 0.01
        i_min = int(np.argmin(z))
        if (z[i_min] < self.surface_z + 0.05 + tol and x[i_min] < self.net_x
                and i_min < len(z) - 1 and z[-1] - z[i_min] > tol):
            return i_min
        return None

    def estimate(self, t_now: float) -> BallEstimate | None:
        avail = [(t, p) for t, p in self.samples if t + self.cam.latency_s <= t_now]
        if not avail:
            return None
        ts = np.array([t for t, _ in avail])
        ps = np.array([p for _, p in avail])
        bounce = self._bounce_index(ps[:, 2], ps[:, 0])
        if bounce is not None:  # fit only the post-bounce segment
            ts, ps = ts[bounce + 1:], ps[bounce + 1:]
        n = max(self.cam.min_samples, int(round(self.cam.window_s * self.cam.rate_hz)))
        if len(ts) < self.cam.min_samples:
            return None
        ts, ps = ts[-n:], ps[-n:]
        tau = ts - ts[-1]
        pos, vel = np.empty(3), np.empty(3)
        for k in range(3):
            deg = 2 if len(ts) >= 4 else 1
            c = np.polyfit(tau, ps[:, k], deg)
            pos[k] = c[-1]
            vel[k] = c[-2]
        return BallEstimate(pos, vel, np.zeros(3), float(ts[-1]), bounce is not None)
