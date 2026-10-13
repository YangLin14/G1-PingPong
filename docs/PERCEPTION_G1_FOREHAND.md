# Perception for the G1 forehand: what the robot needs to see the ball

[繁體中文版](PERCEPTION_G1_FOREHAND.zh-TW.md) · Design: [`TRITON_G1_FOREHAND.md`](TRITON_G1_FOREHAND.md) ·
Training status: [`TRAINING_STATUS_G1_FOREHAND.md`](TRAINING_STATUS_G1_FOREHAND.md)

On the real robot, the planner has to know where the ball is, how fast it moves and (ideally) how it
spins, early enough for the swing. This document collects what we found about the sensing hardware
(why someone quoted "10 cameras for $40k", whether iPhones can do it) and a simulation experiment
that measures how much each property of a camera system matters for **our** policy.

---

## 1. Short answer

* The $40k figure is a motion-capture (mocap) system: 10 × OptiTrack PrimeX 22 at $3,999 each. It is
  what HITTER used (9 OptiTrack cameras) and what HOPE's deploy code expects.
* A camera system has to deliver the ball position **fast** (low latency), **often** (high frame
  rate), **precisely** (mm–cm), from several synchronized views. Mocap is the easy, proven way.
* It is not the only way. Two synced 125 fps machine-vision cameras were enough for DeepMind's 2024
  table-tennis robot. iPhones film at 120–240 fps but have rolling shutters, no hardware sync and tens
  of ms of latency, so they suit offline measurement better than live play.
* The simulation (section 6) says the biggest gap is not the camera at all but the **spin**. Without
  spin, legal returns drop from 61 % to 10 % even with perfect tracking. With the spin known, even
  iPhone-level tracking keeps 54 %. A rough spin estimate (±50 % magnitude, ±40° axis), for example read
  off the opponent's racket, already gets back most of it (48–49 %).

---

## 2. What the reference systems use

* **HITTER** (UC Berkeley, Unitree G1): "Nine OptiTrack cameras track the ball's position" at 360 Hz with
  millimetre accuracy. The ball is wrapped in reflective tape. The same mocap system also gives the
  robot's base position and heading. Velocity is not measured: it is fitted with a second-order
  polynomial over the last 31 positions (86 ms). The paper says this restricts deployment to controlled
  environments and that vision-based sensing would remove the dependency.
* **HOPE** (our upstream, Agibot A3): the deploy scripts read `/optitrack/poses` for the ball and a
  rigid body on the robot's pelvis (`a3_deploy/a3_deploy_example/scripts/pp_gate3_sim_mocap.py`).
* So "copy HITTER/HOPE onto the real G1 as is" means buying or borrowing a mocap system, for both the
  ball and the robot's position relative to the table.

## 3. Why mocap cameras are expensive

| Property | Why table tennis needs it |
|---|---|
| High frame rate (240–360 Hz) | The feed flies at 4–8 m/s. At 360 Hz the ball moves ~2 cm between frames; at 30 fps, over 20 cm. |
| Global shutter, very short exposure | The whole image is taken at one instant, so a fast ball is not smeared or bent. |
| Infrared strobe + IR filter | Only the reflective markers show up, so detection is nearly error-free and nearly free to compute. |
| On-camera processing | The camera sends marker centroids, not video, so the latency is a few ms. |
| Hardware sync, factory-calibrated low-distortion lenses | Triangulation is millimetre-accurate, which makes the velocity fit accurate. |
| Many cameras | The robot's body, the racket and the opponent block the view. At least 2–3 cameras must see the ball at every moment; more cameras = fewer gaps. |

Prices (list, USD): PrimeX 22 $3,999, PrimeX 13 $2,499 per camera. Software licence, PoE switches,
mounting and calibration come on top (we found no prices for those). Note that reflective tape also
changes the ball's mass and surface, which changes its flight and spin slightly.

## 4. Cheaper options

| Option | What it is | Trade-off |
|---|---|---|
| Fewer / smaller mocap cameras | e.g. 6–9 × PrimeX 13, about $15k–22k for cameras | Same software and accuracy; more occlusion gaps. |
| Synced machine-vision stereo | DeepMind 2024: 2 × Ximea MQ013CG-ON, 125 fps, global shutter, untaped ball | Much cheaper hardware (quote only), but we write detection, sync and triangulation ourselves. Open-source multi-camera ball trackers exist. |
| Event cameras | Report only changing pixels, microsecond timing. A secondary report says DeepMind's 2026 "Ace" uses them. | Expensive, new software; we did not verify the Ace details. |
| Borrow | Some UCSD labs may have OptiTrack or Vicon. | Costs nothing; depends on space and schedule. |

## 5. iPhones

| | Finding |
|---|---|
| Frame rate | iPhone 16 Pro: 4K at 120 fps (main camera), 1080p at 240 fps (ultra-wide and telephoto). Comparable to the 125 fps stereo rig on paper. |
| Latency | High-rate modes are made for recording. Developers report that 120 fps leaves 8.3 ms per frame to process, that the high-rate modes are not always available for live frame access, and ~10 % dropped frames at 240 fps on an iPhone 12 Pro. Getting frames or detections off the phone adds more. We found no measured end-to-end number; we assume 60–80 ms. |
| Rolling shutter | Rows are read top to bottom. An iPhone 4s took 22 ms per frame; we found no measurement for current models. A ball moving several cm during the readout appears bent and shifted. |
| Sync | Several phones have no hardware sync; they must be aligned by network time or afterwards. Published work notes that frame misalignment causes 3D reconstruction errors. Two fixed phones have reached mm accuracy for ordinary motion capture. |
| Image processing | Auto exposure, denoising and compression blur a fast ball. Locking a short exposure needs a custom app and bright lighting. |

**Conclusion:** iPhones are good for what we already use them for (filming the reference forehand),
for recording real-robot tests, and for measuring ball flights offline to refit the ball physics on
our own table. For live play they are the weakest option we looked at.

---

## 6. Simulation experiment: how good does the perception have to be?

### Method

`mujoco_eval_onnx.py --perception <preset>` puts a camera model between the simulated ball and the
planner (`scripts/ball_perception.py`). The model:

1. samples the true ball path at `rate_hz` (random phase), adds Gaussian noise `noise_m` per axis and
   drops a fraction `dropout` of samples;
2. makes each sample available `latency_s` after it was taken;
3. estimates position and velocity like HITTER: a quadratic fit per axis over the last `window_s`
   (86 ms, at least 5 samples), then predicts forward by the sample age;
4. detects the bounce in the samples and fits only the post-bounce part afterwards;
5. gives the planner **no spin** (`spin=zero`) unless told otherwise, because no camera system measures it.

The planner then predicts the bounce and apex and solves the racket velocity from this estimate,
exactly as with the true state. `perception_sweep.py` runs every condition on the same bank balls and
rally gaps (same seeds), at real rally pace, and stands the robot up again after a fall.

Presets (assumed values for comparison, not measurements of a product):

| preset | rate | latency | noise (1σ per axis) | dropout | meant to resemble |
|---|---|---|---|---|---|
| `oracle` | true state | 0 | 0 | 0 | the old evaluator (knows the spin too) |
| `oracle_nospin` | true state | 0 | 0 | 0 | perfect tracking, spin unknown |
| `mocap_360` | 360 Hz | 5 ms | 1 mm | 2 % | HITTER's OptiTrack |
| `mocap_240` | 240 Hz | 8 ms | 1.5 mm | 2 % | a smaller mocap system |
| `stereo_125` | 125 Hz | 15 ms | 3 mm | 5 % | DeepMind 2024 stereo pair |
| `iphone_240` / `_120` / `_60` | 240 / 120 / 60 Hz | 60 / 60 / 80 ms | 8 / 8 / 10 mm | 10 % | phones streaming to a PC |
| `webcam_30` | 30 Hz | 100 ms | 15 mm | 10 % | a cheap webcam |

### Results (2026-10-13)

Policy `g1_fh_rally` `model_5900`, default bank, real rally pace, 3 seeds × 40 balls = **120 balls per
row** (one standard error is about ±4–5 percentage points). Columns: how long before the strike the
first command came; the error of the last commanded apex position and time against the truth; then
the club funnel (contact → over the net → legal → in zone) and falls. "apex OK" = struck within
±0.3 m/s of the apex.

| group | condition | first command (s before strike) | apex error (cm) | time error (ms) | contact | apex OK | over net | legal | in zone | falls |
|---|---|---|---|---|---|---|---|---|---|---|
| reference | `oracle` | 0.59 | 0.0 | 0 | 88 % | 2 % | 67 % | **61 %** | 6 % | 0 % |
| reference | `oracle_nospin` | 0.59 | 1.6 | 6 | 69 % | 3 % | 49 % | **10 %** | 0 % | 0 % |
| systems (no spin) | `mocap_360` | 0.56 | 1.9 | 7 | 72 % | 2 % | 52 % | 16 % | 2 % | 0 % |
| systems (no spin) | `mocap_240` | 0.55 | 1.9 | 6 | 65 % | 2 % | 48 % | 21 % | 0 % | 0 % |
| systems (no spin) | `stereo_125` | 0.54 | 3.6 | 11 | 77 % | 2 % | 57 % | 18 % | 2 % | 0 % |
| systems (no spin) | `iphone_240` | 0.48 | 8.2 | 21 | 82 % | 5 % | 54 % | 26 % | 5 % | 2 % |
| systems (no spin) | `iphone_120` | 0.47 | 11.6 | 32 | 88 % | 6 % | 68 % | 38 % | 5 % | 1 % |
| systems (no spin) | `iphone_60` | 0.41 | 24.2 | 58 | 86 % | 33 % | 49 % | 39 % | 7 % | 5 % |
| systems (no spin) | `webcam_30` | 0.32 | 17.8 | 28 | 74 % | 37 % | 49 % | 38 % | 3 % | 4 % |
| latency | `mocap_360`, 30 ms | 0.53 | 3.2 | 10 | 59 % | 1 % | 43 % | 15 % | 1 % | 0 % |
| latency | `mocap_360`, 60 ms | 0.51 | 4.9 | 17 | 80 % | 2 % | 63 % | 32 % | 2 % | 0 % |
| latency | `mocap_360`, 100 ms | 0.48 | 6.8 | 24 | 83 % | 5 % | 68 % | 34 % | 3 % | 1 % |
| noise | `mocap_360`, 5 mm | 0.55 | 3.5 | 12 | 79 % | 8 % | 44 % | 14 % | 2 % | 0 % |
| noise | `mocap_360`, 10 mm | 0.64 | 5.8 | 18 | 72 % | 20 % | 34 % | 19 % | 4 % | 1 % |
| noise | `mocap_360`, 20 mm | 0.50 | 9.8 | 31 | 68 % | 41 % | 39 % | 23 % | 2 % | 5 % |
| rate | `mocap_360`, 120 Hz | 0.53 | 1.9 | 7 | 61 % | 2 % | 40 % | 15 % | 0 % | 1 % |
| rate | `mocap_360`, 60 Hz | 0.50 | 3.0 | 10 | 87 % | 2 % | 67 % | 31 % | 2 % | 0 % |
| spin known | `mocap_360` | 0.57 | 0.6 | 2 | 88 % | 2 % | 65 % | **56 %** | 5 % | 0 % |
| spin known | `iphone_120` | 0.47 | 9.7 | 21 | 90 % | 3 % | 70 % | **54 %** | 7 % | 0 % |
| spin read ±10 %, ±10° | `mocap_360` | 0.57 | 0.6 | 2 | 85 % | 6 % | 67 % | 54 % | 4 % | 1 % |
| spin read ±25 %, ±20° | `mocap_360` | 0.58 | 0.6 | 2 | 87 % | 8 % | 66 % | 55 % | 6 % | 1 % |
| spin read ±50 %, ±40° | `mocap_360` | 0.57 | 0.8 | 2 | 86 % | 2 % | 58 % | **49 %** | 3 % | 0 % |
| spin read ±10 %, ±10° | `iphone_120` | 0.50 | 9.7 | 25 | 94 % | 6 % | 66 % | 48 % | 5 % | 1 % |
| spin read ±25 %, ±20° | `iphone_120` | 0.48 | 10.8 | 22 | 88 % | 4 % | 59 % | 48 % | 6 % | 1 % |
| spin read ±50 %, ±40° | `iphone_120` | 0.47 | 9.1 | 22 | 93 % | 7 % | 62 % | **48 %** | 2 % | 1 % |

### What it means

1. **Spin matters more than the camera.** Perfect position and velocity but no spin: legal returns
   fall from 61 % to 10 %. The ball's spin changes how it bounces on our side and how it leaves our
   racket, so the racket velocity solved for a spin-free ball sends the real ball into the net or long.
   With the spin known, mocap (56 %) and iPhone-level tracking (54 %) are almost the same.
2. **A rough spin estimate is enough for most of the gain.** Errors of ±50 % in size and ±40° in axis
   still give 48–49 % legal returns (vs 10–38 % with none). What matters is knowing roughly which way
   and how hard the ball spins (topspin vs backspin vs sidespin).
3. **Among spin-blind systems, the ranking is not about camera quality.** Without spin, worse cameras
   sometimes score higher (iPhone 38 % vs mocap 16 %), and 30 ms latency scores below 100 ms. Do not
   read this as "worse is better": the policy already strikes ~40 ms early (see the status report),
   and late or noisy commands happen to shift its timing, which also raises "apex OK" for the noisiest
   rows. These rows are confounded by that bias; the spin rows are not.
4. **Lead time.** The first command arrives 0.56–0.59 s before the strike with mocap, 0.47–0.48 s with
   iPhone 120–240 fps, 0.41 s at 60 fps and 0.32 s at 30 fps. The swing needs 0.4–0.8 s; below ~0.45 s
   the robot starts its swing late.
5. **Position accuracy.** The final apex prediction is within 2 cm with mocap, 4 cm with the stereo
   pair, 8–12 cm with iPhones at 120–240 fps and over 20 cm at 60 fps. The racket face is ~15 cm wide.

Limits: one policy (which still strikes early), simulated feeds from a ball launcher with ≤5 rev/s
spin (real opponents spin the ball much harder, which makes spin matter even more), no rolling-shutter
or calibration model beyond Gaussian noise, and the "spin read" rows model only the size of the error,
not how it is obtained.

### Reading the spin from the opponent's racket

Human players judge spin mainly from the opponent's stroke (racket angle and swing direction at
contact) and the rubber. For a robot this is a good fit, because the cameras that track the ball can
also track the opponent's racket (a few markers on it, or racket pose from vision). Physically, the
outgoing spin follows from the racket's velocity and face normal at contact; our venue paddle model
(`virtual_ball.predict_paddle_contact`) computes exactly that for our own racket, so the same model can
turn an observed opponent racket state into a spin estimate.

What the literature has: Tebbe et al. (2019) compared logo tracking at 380 Hz with fitting the Magnus
force to the trajectory; the trajectory fit was the most accurate and was used against a human.
SpinDOE (2023) prints dots on the ball and measures spin up to 175 rev/s at 350 fps with a 2.4°
orientation error. Event cameras track the logo without motion blur (CVPRW 2024). Using the
opponent's motion for spin is named as future work in a 2025 legged-robot paper; we found no robot
that does it yet.

Our results say the racket reading does not need to be precise (±50 %, ±40° still helps a lot). A
practical plan is to combine the two: a racket-based estimate at the opponent's hit (early, rough),
refined by the Magnus curvature of the ball's flight (later, more accurate).

---

## 7. Recommendation

1. Do not buy cameras yet. The robot is not here and the policy still strikes early.
2. Add spin to the plan: estimate it from the opponent's racket at contact plus the ball's curved flight,
   and train the policy with spin errors like the ones above.
3. For hardware, aim for: ≥120 Hz, ≤15–30 ms latency, ≤5 mm position error, synced views covering the
   whole table and the opponent's racket. Two to four synced global-shutter machine-vision cameras meet
   this on paper; borrowing a lab's mocap is the fastest way to start.
4. Use iPhones for filming reference motions, recording tests and measuring ball flights offline.

## 8. How to rerun

```bash
.venv/bin/python hope_training/whole_body_tracking/scripts/perception_sweep.py \
  --onnx artifacts/evals/g1_fh_rally_model_5900/policy.onnx --serves 40 --seeds 0 1 2 --jobs 9 \
  --out artifacts/perception/g1_fh_rally_model_5900          # ~45 min on a 10-core laptop
# one condition, any preset with overrides:
.venv/bin/python hope_training/whole_body_tracking/scripts/mujoco_eval_onnx.py --robot unitree_g1 \
  --onnx <policy.onnx> --serve-bank default --rally-pace --reset-after-fall --num-serves 40 \
  --perception "iphone_120:spin=read,spin_rel_err=0.25,spin_ang_err_deg=20" --json-out out.json
```

`--perception oracle` reproduces the old evaluator exactly (checked ball by ball).

---

Sources:
[HITTER (arXiv 2508.21043)](https://arxiv.org/html/2508.21043v1) ·
[OptiTrack PrimeX 22](https://optitrack.com/cameras/primex-22/buy) ·
[OptiTrack cameras / PrimeX 13](https://www.optitrack.com/cameras) ·
[OptiTrack hardware brochure 2025](https://optitrack.com/api/media/file/Hardware-Brochure-2025.pdf) ·
[Robotic Table Tennis case study (arXiv 2309.03315)](https://arxiv.org/pdf/2309.03315) ·
[Reliable real-time ball tracking (arXiv 1908.07332)](https://arxiv.org/pdf/1908.07332) ·
[Hackster on DeepMind's robot](https://www.hackster.io/news/this-robot-can-beat-you-at-ping-pong-c009ee2bcd96) ·
[TechCrunch on DeepMind's robot](https://techcrunch.com/2024/08/08/google-deepmind-develops-a-solidly-amateur-table-tennis-robot/) ·
[Ace report (secondary)](https://startupfortune.com/google-deepminds-ace-robot-becomes-the-first-ai-system-to-beat-elite-table-tennis-players-in-real-matches/) ·
[iPhone 16 Pro specs](https://www.apple.com/nz/iphone-16-pro/specs) ·
[Apple TN2445 frame drops](https://developer.apple.com/library/archive/technotes/tn2445/_index.html) ·
[Apple forum: 240 fps recording](https://developer.apple.com/forums/thread/60388) ·
[JUCE forum: high-rate processing](https://forum.juce.com/t/real-time-video-processing-at-high-frame-rates/47621) ·
[Rolling-shutter tutorial, Linköping](https://www.cvl.isy.liu.se/education/tutorials/rolling-shutter-tutorial/pdf/Intro30m.pdf) ·
[Synchronized phone capture (arXiv 2103.07883)](https://arxiv.org/pdf/2103.07883v1) ·
[Two-smartphone tracking (CEUR-WS)](https://ceur-ws.org/Vol-2498/short10.pdf) ·
[Spin Detection in Robotic Table Tennis (arXiv 1905.07967)](https://arxiv.org/pdf/1905.07967) ·
[SpinDOE (arXiv 2303.03879)](https://arxiv.org/abs/2303.03879v1) ·
[Event-camera spin estimation (CVPRW 2024)](https://openaccess.thecvf.com/content/CVPR2024W/CVsports/html/Gossard_Table_Tennis_Ball_Spin_Estimation_with_an_Event_Camera_CVPRW_2024_paper.html) ·
[Legged-robot table tennis (2510.08754)](https://www.emergentmind.com/papers/2510.08754)
