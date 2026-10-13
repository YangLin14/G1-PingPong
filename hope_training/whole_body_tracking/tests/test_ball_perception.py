"""Camera model for the MuJoCo evaluator: sampling, latency, fit and bounce segmentation."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import ball_perception as bp  # noqa: E402

G = np.array([0.0, 0.0, -9.81])
SURFACE = 0.78


def _path(p0, v0, t0, t1, h=0.002, bounce_at=None, e=0.9):
    """Drag-free ballistic path sampled every h, optionally bouncing (vz -> -e vz) on z = SURFACE."""
    out, p, v, t = [], np.array(p0, float), np.array(v0, float), t0
    while t <= t1 + 1e-12:
        out.append((t, p.copy()))
        p, v, t = p + h * v + 0.5 * h * h * G, v + h * G, t + h
        if bounce_at and p[2] < SURFACE and v[2] < 0:
            p[2] = 2 * SURFACE - p[2]
            v[2] = -e * v[2]
    return out


def test_noise_free_fit_recovers_state_and_latency_hides_recent_samples():
    cam = bp.CameraModel("t", rate_hz=200, latency_s=0.03, noise_m=0.0, dropout=0.0)
    tr = bp.BallTracker(cam, np.random.default_rng(0), SURFACE, 1.87)
    track = _path([3.0, 0.4, 1.1], [-5.0, -0.8, 0.5], 0.0, 0.2)
    tr.reset(0.0)
    tr.ingest(track)
    est = tr.estimate(0.2)
    assert est is not None and est.t <= 0.2 - 0.03 + 1e-9
    true_p = np.array([3.0, 0.4, 1.1]) + est.t * np.array([-5.0, -0.8, 0.5]) + 0.5 * est.t ** 2 * G
    true_v = np.array([-5.0, -0.8, 0.5]) + est.t * G
    assert np.allclose(est.pos, true_p, atol=1e-4) and np.allclose(est.vel, true_v, atol=1e-3)  # 2 ms path interpolation
    assert not est.bounced


def test_bounce_is_detected_and_only_the_post_bounce_segment_is_fitted():
    cam = bp.CameraModel("t", rate_hz=360, latency_s=0.0, noise_m=0.001, dropout=0.0)
    tr = bp.BallTracker(cam, np.random.default_rng(1), SURFACE, 1.87)
    track = _path([1.6, -0.3, 0.95], [-3.0, 0.0, -1.0], 0.0, 0.25, bounce_at=True)
    tr.reset(0.0)
    tr.ingest(track)
    est = tr.estimate(0.25)
    assert est is not None and est.bounced
    assert 0.3 < est.vel[2] < 0.7  # true vz ~0.47: rising, fitted on the post-bounce samples only


def test_oracle_and_overrides():
    assert bp.PRESETS["oracle"].oracle and bp.PRESETS["oracle"].spin == "true"
    cam = bp.camera_from_spec("mocap_360:latency_s=0.05,spin=true")
    assert cam.latency_s == 0.05 and cam.spin == "true" and cam.rate_hz == 360


def test_spin_reading_error_is_fixed_per_ball_and_zero_error_is_exact():
    w = np.array([0.0, 30.0, 5.0])
    exact = bp.BallTracker(bp.camera_from_spec("mocap_360:spin=read"), np.random.default_rng(0), SURFACE, 1.87)
    assert np.allclose(exact.spin_estimate(w), w)
    noisy = bp.BallTracker(bp.camera_from_spec("mocap_360:spin=read,spin_rel_err=0.3,spin_ang_err_deg=20"),
                           np.random.default_rng(0), SURFACE, 1.87)
    first = noisy.spin_estimate(w)
    assert np.allclose(noisy.spin_estimate(w), first) and not np.allclose(first, w)
    assert np.allclose(bp.BallTracker(bp.PRESETS["mocap_360"], np.random.default_rng(0), SURFACE, 1.87).spin_estimate(w), 0)
