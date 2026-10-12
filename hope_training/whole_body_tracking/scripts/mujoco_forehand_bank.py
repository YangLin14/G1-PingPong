"""Club forehand bank questions for the MuJoCo evaluator (``mujoco_eval_onnx.py --serve-bank``).

The frame is the training env frame, which is also the MuJoCo world frame of the evaluator when the
robot stands at the origin: robot facing +x, near table edge at x = 0.5, table centred on y = 0.

* :class:`ForehandBank` reads ``g1_fh_crosscourt_apex_bank.npz`` (optionally only some target zones);
* :func:`feed_start` turns a row's feed (pre-bounce velocity/spin at the feed bounce) into the ball
  state where it crosses the far end line, integrating the venue spin model backward;
* :class:`ApexPlanner` is the deploy-style planner: from the REAL ball state it predicts the bounce on
  the robot's half and the following apex (= racket target position and time-to-strike), and once the
  ball has bounced it re-solves the racket velocity for the row's intended landing point.

Isaac-free: NumPy for the per-tick prediction, the bank tools' torch solver once per serve.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent


def _load(name: str, path: pathlib.Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bank_tools = _load("_make_g1_forehand_bank", _HERE / "make_g1_forehand_bank.py")
fs = bank_tools.fs
vb = bank_tools.vb


class ForehandBank:
    def __init__(self, path: str, zones: list[str] | None = None):
        with np.load(path, allow_pickle=False) as archive:
            self.rows = {k: np.asarray(archive[k]) for k in archive.files}
        self.zone_names = tuple(fs.TARGET_ZONES)
        receipt = pathlib.Path(path).with_suffix(".receipt.json")
        if receipt.is_file():
            import json

            names = json.loads(receipt.read_text()).get("contract", {}).get("target_zone_id")
            if names:
                self.zone_names = tuple(names)
        keep = np.ones(len(self.rows["clip"]), dtype=bool)
        if zones:
            unknown = sorted(set(zones) - set(self.zone_names))
            if unknown:
                raise ValueError(f"unknown zones {unknown}; bank zones: {list(self.zone_names)}")
            ids = [self.zone_names.index(z) for z in zones]
            keep &= np.isin(self.rows["target_zone_id"], ids)
        self.index = np.where(keep)[0]
        if len(self.index) == 0:
            raise ValueError("no bank rows for the selected zones")

    def sample(self, rng: np.random.Generator) -> dict:
        i = int(rng.choice(self.index))
        row = {k: v[i] for k, v in self.rows.items() if v.ndim >= 1 and len(v) == len(self.rows["clip"])}
        row["row"] = i
        row["zone"] = self.zone_names[int(row["target_zone_id"])]
        return row


class Flight:
    """Venue flight model a = g - k_d|v|v + k_m (omega x v), RK4, NumPy (spin constant in flight)."""

    def __init__(self, prm):
        self.prm = prm
        self.g = np.array([0.0, 0.0, -9.81])

    def accel(self, v, w):
        return self.g - self.prm.k_d * np.linalg.norm(v) * v + self.prm.k_m * np.cross(w, v)

    def step(self, p, v, w, h):
        a1 = self.accel(v, w)
        a2 = self.accel(v + 0.5 * h * a1, w)
        a3 = self.accel(v + 0.5 * h * a2, w)
        a4 = self.accel(v + h * a3, w)
        p_new = p + h * v + (h * h / 6.0) * (a1 + a2 + a3)
        v_new = v + (h / 6.0) * (a1 + 2 * a2 + 2 * a3 + a4)
        return p_new, v_new


def feed_start(row: dict, prm, scfg=None, h: float = 0.002, max_steps: int = 2000):
    """(pos, vel, spin) where the row's feed crosses the far end line (backward from the feed bounce)."""
    scfg = scfg or fs.ForehandSuccessCfg()
    fl = Flight(prm)
    w = np.asarray(row["feed_spin_pre"], dtype=np.float64)
    p = np.array([*row["feed_bounce_xy"], scfg.table_surface_z + fs.BALL_RADIUS], dtype=np.float64)
    v = np.asarray(row["feed_velocity_pre"], dtype=np.float64).copy()
    for _ in range(max_steps):
        p1, v1 = fl.step(p, v, w, -h)
        if p1[0] >= scfg.far_x:
            f = (scfg.far_x - p[0]) / max(p1[0] - p[0], 1e-12)
            return p + f * (p1 - p), v + f * (v1 - v), w
        p, v = p1, v1
    raise RuntimeError(f"bank row {row.get('row')}: feed never reaches the far end line")


class ApexPlanner:
    """Predicts the post-bounce apex of the real ball and the racket velocity for a target landing."""

    def __init__(self, prm, scfg=None, h: float = 0.004):
        self.prm = prm
        self.scfg = scfg or fs.ForehandSuccessCfg()
        self.tprm = bank_tools.table_params(prm, vb.default_venue_yaml_path())
        self.flight = Flight(prm)
        self.h = h
        self.surface = self.scfg.table_surface_z + fs.BALL_RADIUS

    def _bounce(self, v, w):
        import torch

        t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64)).unsqueeze(0)  # noqa: E731
        v_post, w_post = vb.predict_paddle_contact(t(v), t(np.zeros(3)), t([0.0, 0.0, 1.0]), t(w), self.tprm)
        return v_post[0].numpy(), w_post[0].numpy()

    def table_bounce(self, v, w):
        """Venue table bounce for the MuJoCo scene (world frame, NumPy in and out)."""
        return self._bounce(v, w)

    def predict(self, p, v, w, after_bounce: bool, max_t: float = 2.0):
        """From the ball state now: (apex_pos, apex_vel, spin_at_apex, time_to_apex), or None.

        Before the bounce on the robot's half (``after_bounce`` False) the ball is flown to the table,
        bounced with the venue table model and flown on to the apex; after it, flown to the apex
        directly. None once the ball is past its apex or never bounces on the robot's half."""
        p, v, w = (np.asarray(a, dtype=np.float64).copy() for a in (p, v, w))
        t, h = 0.0, self.h
        if not after_bounce:
            while True:
                p1, v1 = self.flight.step(p, v, w, h)
                if v1[2] < 0 and p1[2] <= self.surface and p1[0] < self.scfg.net_x:
                    f = (p[2] - self.surface) / max(p[2] - p1[2], 1e-12)
                    p, v, t = p + f * (p1 - p), v + f * (v1 - v), t + f * h
                    v, w = self._bounce(v, w)
                    break
                p, v, t = p1, v1, t + h
                if t > max_t or p[2] < 0.0:
                    return None
        if v[2] <= 0:
            return None
        while t < max_t:
            p1, v1 = self.flight.step(p, v, w, h)
            if v1[2] <= 0:
                f = v[2] / max(v[2] - v1[2], 1e-12)
                return p + f * (p1 - p), v + f * (v1 - v), w, t + f * h
            p, v, t = p1, v1, t + h
        return None

    def solve_racket(self, apex_pos, apex_vel, spin, target_xy):
        """Racket velocity that sends the ball from the apex to ``target_xy`` (bank solver, one row)."""
        import torch

        t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64)).reshape(1, -1)  # noqa: E731
        v_r = bank_tools.solve_racket(
            t(apex_pos), t(apex_vel), t(spin), t(target_xy), self.prm, self.scfg, bank_tools.BankCfg()
        )
        return v_r[0].numpy()


def default_bank_path() -> pathlib.Path:
    return _HERE.parent / "assets" / "venue_tuple" / "g1_fh_crosscourt_apex_bank.npz"
