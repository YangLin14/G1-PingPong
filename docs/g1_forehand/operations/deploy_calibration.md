# Deploy calibration: what to measure on the real robot, and how

[繁體中文](deploy_calibration.zh-TW.md) · [Docs index](../README.md)

Some numbers cannot come from simulation: they depend on the real racket, motors, cameras and
computers. Each is a single value in a config file (or will be), measured with a short ball-feeding
session. Calibrate in the order below; each step assumes the previous ones are done. Re-calibrate after
changing the policy checkpoint, the racket or rubber, the cameras or the planner computer.

### Current values

| what | where | value now | source | status |
|---|---|---|---|---|
| Reference clock | runtime YAML `lifecycle.clock` | `strike_time` | D17 | in the G1 config |
| Allowed message age | `lifecycle.max_message_age_s` | 0.1 s | D17 | in the G1 config |
| **Timing offset** | `lifecycle.tts_offset_s` | **+0.04 s** | MuJoCo, `model_15700` | in the G1 config; **re-measure** |
| **Aim offset** (along the table, sideways) | planner; today only `mujoco_eval_onnx.py --aim-offset` | **(+0.14, −0.03) m** with the training paddle model; (+0.19, +0.04) m with MuJoCo's contact (old grip `model_15700`: +0.40/−0.10 and +0.63/−0.06) | MuJoCo, fist `model_32299` | **not yet in the deploy planner**; re-measure with the real racket |
| Racket centre (mount offset) | `G1_FIST_MOUNT_OFFSET` / `G1_FIST_MOUNT_QUAT` (fist grip, D20), generated fist URDF/MJCF | blade origin (0.083, 0.0226, 0.14) m from the right wrist, racket perpendicular to the forearm (old grip: `G1_MOUNT_OFFSET` (0.21, 0.015, 0.01) m) | CAD estimate | measure the real 3D-printed fist holder |
| Paddle model (restitution, grip) | `configs/ball_physics_venue.yaml` `paddle` | venue fit (another table, another racket) | HOPE | refit with our racket |
| Table bounce / ball flight | `configs/ball_physics_venue.yaml` | venue fit | HOPE | refit on our table |

The two compensation values are the ones you asked about: **timing** (`tts_offset_s`) and **aim**
(DX, DY). The rest are inputs that decide how large those two will be.

### 0. Before any ball: frames and clocks

1. **Clocks.** If the planner and the robot runner are on different computers, sync them (chrony/PTP,
   error well under 5 ms). Check: the runner's `clock_rejections` stays 0 and the message age
   `now − (strike_time − time_to_strike)` is small and positive (a few ms to a few tens of ms).
2. **Frames.** The planner's world frame (table) and the robot's base pose must agree: put the robot at
   its station, compare the tracked base position with the expected one, and fix the
   `mocap_to_base_link` transform until they match within ~1 cm.
3. **Racket centre.** Measure where the blade centre really is relative to the wrist link. If it differs
   from `G1_MOUNT_OFFSET` by more than ~1 cm, update the model and fine-tune the policy ([how the system works](../overview.md),
   "Changing the racket").

### 1. Racket and table model (planner physics)

Record 50–100 hits with the real racket (a person swinging is fine) and 50 table bounces with the
tracking system. Fit the paddle restitution `e(u_n)` and grip, and the table bounce, the same way the
venue model was fitted. This makes the planner's predicted landing match reality, so the aim offset
in step 3 stays small. Skipping it is possible; the aim offset then has to absorb the difference
(in MuJoCo, the wrong racket model cost an extra 23 cm of offset).

### 2. Timing offset (`tts_offset_s`)

Goal: the racket meets the ball at the top of its bounce.

1. Feed 30–40 balls into one zone (ball machine; same speed and spin).
2. For each hit, read the ball's vertical speed just before contact from the tracking data
   (positive = still rising = early).
3. New offset = old offset + median(vz) / 9.81. Example: median vz +0.45 m/s means about 46 ms early, so
   add ~0.046 s.
4. Repeat until |median vz| < 0.1 m/s and at least 80 % of hits are within ±0.3 m/s ("apex OK").

In simulation: `mujoco_eval_onnx.py --tts-offset-s X --trace-out t.json` and read `apex_vz_at_contact`
per ball from the JSON. Calibrate timing **before** aim: timing changes where the ball goes.

### 3. Aim offset (DX along the table, DY sideways)

Goal: the average landing point equals the aim point.

1. With timing done, feed 40+ balls per target zone and record each first bounce (x, y).
2. DX = −median(landing x − aim x), DY = −median(landing y − aim y). Example from MuJoCo: balls landed
   35 cm short and 10 cm wide, so the planner aims 0.40 m deeper and 0.10 m inward.
3. Feed again with the offset and check the per-zone in-zone rate. If one zone is still off by more
   than ~10 cm (in MuJoCo the corner needed more depth than the middle), use a per-zone offset.
4. Watch the side line: the corner zone ends 1.5 cm inside it, so a sideways bias sends corner balls out.

In simulation: `mujoco_eval_onnx.py --aim-offset DX DY`. **Missing piece:** the deploy planner has no
aim-offset parameter yet; add it (the planner solves the racket velocity for target + offset) before
the first hardware session.

### 4. Check

Run the club evaluation per zone (contact → over the net → legal → in zone, plus apex OK) on 40+
balls per zone and record it in the status report. In MuJoCo the result after calibration was 35 % in
zone with 0.9 s notice and 34 % at rally pace (status log 2026-10-14).

