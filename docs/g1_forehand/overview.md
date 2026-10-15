# Overview: context, club rules, how the system works

[繁體中文](overview.zh-TW.md) · [Docs index](README.md)

## Context

| Repository | What it is |
|---|---|
| [`hitchopen/HOPE`](https://github.com/hitchopen/HOPE) | Upstream HOPE: the open humanoid table-tennis platform for the **Agibot A3** (Isaac Lab training, ball planner, MuJoCo sim-to-sim, C++ deploy runner). |
| `YangLin14/G1-PingPong` | Our fork of HOPE (this checkout, `origin`). |
| [`triton-droids/HOPE`](https://github.com/triton-droids/HOPE) branch `g1-port` | The club's G1 port by Oren Gershony: G1 URDF/MJCF with a racket, recipe v2 (A3 numbers rescaled to the G1), two derived G1 clip pairs, MuJoCo plant parity, Slurm jobs. Documented in [`UNITREE_G1.md`](../UNITREE_G1.md). |
| `triton-droids/pingpong` | An earlier, separate prototype (DLS IK + PPO residual on a G1, Isaac Lab); not used here. |

This work lives on branch **`g1-forehand-rally`**, branched from `g1-port`.

HOPE's design follows **HITTER** (UC Berkeley, [arXiv 2508.21043](https://arxiv.org/abs/2508.21043)):
a model-based planner predicts the ball and decides *where, when and how fast* the racket must be,
and a reinforcement-learned whole-body controller makes the robot get there while imitating a
human reference swing. HITTER ran exactly this on a **Unitree G1** (96 % hit rate, 92 % returns,
rallies up to 106 shots), using references made by filming a human, lifting the video with
**GVHMR** and retargeting with **GMR**.

---

## Task specification (club rules)

All coordinates are the **env frame, robot's view**: robot root at the origin facing **+x** (toward
the table), **+y = robot's left**, z up. Near table edge at x = 0.5 m, table surface at z = 0.76 m,
table centred on y = 0, net at x = 1.87 m, far end at x = 3.24 m.

A forehand return is a **success** only if all hold:

1. **Timing.** The ball is struck at the **apex** after bouncing on the robot's side: incoming
   vertical speed at contact |vz| ≤ 0.3 m/s (about ±30 ms around the top).
2. **Contact.** Racket centre within 9.5 cm of the ball, moving into it faster than 0.3 m/s (the
   training capture gate).
3. **Legal.** The ball clears the net (centre above net top + ball radius) and first lands on the
   opponent's half.
4. **Zone.** That first bounce is inside the target zone.

Diagonal geometry (both directions use the same diagonal):

* **Feed**: hit from the **opponent's right** (the robot's +y half, crossing the far end line 5–45 cm
  above the table), clears the net and bounces in the feed zone on the **robot's right** (−y).
* **Return**: the robot hits from **its right** to the **opponent's right**, which is the robot's
  **left** half of the far table (+y).

| Zone | Depth (fraction of a half, from the net) | Lateral (fraction of half width, +y) | Env-frame box |
|---|---|---|---|
| **Feed origin** (far end line) | at x = 3.24 | +0.25 … +0.95 (opponent's right) | y 0.191–0.724, z 0.81–1.21 |
| **Feed** (incoming bounce, robot's half) | middle third | −0.75 … −0.25 (robot's right) | x 0.957–1.413, y −0.572…−0.191 |
| `opp_right_mid`: open court | middle third | +0.25 … +0.75 | x 2.327–2.783, y 0.191–0.572 |
| `center_line_deep`: beside the centre line (opponent's right side of it), at the end line | 0.78 … 0.98 | +0.03 … +0.23 | x 2.939–3.213, y 0.023–0.175 |
| `opp_right_deep`: middle of the right half, at the end line | 0.78 … 0.98 | +0.25 … +0.75 | x 2.939–3.213, y 0.191–0.572 |
| `opp_right_corner`: end line × side line | 0.78 … 0.98 | 0.78 … 0.98 | x 2.939–3.213, y 0.595–0.747 |

The three edge zones are **lined up along the end line**: they share the back edge band (the last 2–22 % of
the half) and go centre line → middle of the right half → corner. The centre-line zone sits just on the
opponent's-right side of the line: its inner edge is 2.3 cm off the line (more than a ball radius),
so a landing in it never touches the line. A side-line zone at middle depth
(`opp_right_side`) is kept, commented out, in `TARGET_ZONES` for later.
The zones do not overlap. The training bank draws the four targets equally (25 % each). The
policy does not need to be told the target, because the bank solves the racket velocity that sends
the ball to it, and the policy tracks that velocity.

---

## How the system works

```text
                 QUESTION (one per swing)                              POLICY (50 Hz)
  bank row: station_xy, racket contact point,      ──►   104-D obs: proprioception, racket target
  racket velocity (= face normal), incoming ball         pos/vel, time-to-strike, base target ...
  velocity/spin at contact, intended landing             ──► 29-D action ──► joint PD targets (G1)
                                                                    │
                                                 Isaac Lab / PhysX  ▼  (200 Hz physics)
  REWARD (per step)                                       robot moves; reference clock runs
   * track the reference swing clip (BeyondMimic)
   * racket position / velocity / normal at the exact strike frame
   * VIRTUAL BALL at the strike frame: achieved racket state ──► spin contact model ──►
     Magnus/drag flight ──► net clearance + landing vs the row's intended landing point
   * stability / regularization terms
```

* **Reference clip.** A 50 Hz swing (ready → strike → recover) in the G1 Isaac joint order with 14
  tracked bodies; the strike frame is marked in the sidecar YAML. The policy is rewarded for following
  it, which gives a human-like motion prior (BeyondMimic tracking inherited from HOPE).
* **Question.** HOPE's `RacketTargetCommand` gives the policy a racket target at a known
  time-to-strike. In `fixed_balanced_bank_v1` mode the target comes from a precomputed bank.
  `HOPEPingPongUnitreeG1_FH_XC` sets the bank share to 100 %, so every question is a physically
  consistent apex strike that returns cross-court.
* **Virtual ball.** The ball is not simulated as a PhysX body in training. At the exact strike frame
  the venue-fitted model ([`configs/ball_physics_venue.yaml`](../../configs/ball_physics_venue.yaml),
  [`virtual_ball.py`](../../hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/virtual_ball.py))
  computes the return from the **achieved** racket state:
  - flight: `a = g − k_d|v|v + k_m(ω × v)` (k_d = 0.1261 m⁻¹, k_m = 0.00444);
  - racket and table contact: an impulse model with tangential grip (spin generation) and
    speed-dependent normal restitution `e(u_n) = 0.759·exp(−0.0441|u_n|)`;
  - fitted on real venue data, valid to about 7 m/s and 15 rev/s.
* **Deploy reference clock (D17).** Every 20 ms tick the runner tells the policy how long until the
  strike. `countdown` (original): copy the planner's `time_to_strike`, then subtract 20 ms in the same
  tick, and keep subtracting 20 ms per tick until the next message. `strike_time` (opt-in): remember the
  planner's absolute deadline and show `deadline − now`. Example, a message sent at 10.000 s saying
  "0.500 s left" (deadline 10.500 s), no message delay:

  | wall time | true time left | `countdown` shows | `strike_time` shows |
  |---|---:|---:|---:|
  | 10.000 (message arrives) | 0.500 | 0.480 | 0.500 |
  | 10.020 | 0.480 | 0.460 | 0.480 |
  | 10.040 | 0.460 | 0.440 | 0.460 |

  `countdown` is always one tick short, which made the policy swing 20 ms early. If the message took
  35 ms to arrive, `countdown` would show 0.480 at 10.035 (true 0.465, 15 ms long) while `strike_time`
  shows 0.465: it is right for any delay, as long as planner and runner share a synchronized clock.
* **Changing the racket.** The policy is trained to put the racket *centre point* (0.21, 0.015, 0.01 m
  from the right wrist) at a target position with a target velocity and face normal; it does not know
  about rubber. So: a new rubber or blade of the same shape, size and similar mass needs **no
  retraining**, only a new paddle model for the planner (fit restitution and grip from recorded hits)
  and new timing/aim offsets. A different mount position, handle length or clearly different mass
  changes where the racket centre is and how the arm moves: update `G1_MOUNT_OFFSET` and the generated
  URDF/MJCF (`unitree/scripts/build_g1_pingpang_models.py`), then **fine-tune** from the current
  checkpoint (warm start, a few thousand iterations) rather than train from scratch.
* **Same physics everywhere.** The bank generator, the success metric, the training reward and the
  visualizer all call this one module, so they cannot disagree.

