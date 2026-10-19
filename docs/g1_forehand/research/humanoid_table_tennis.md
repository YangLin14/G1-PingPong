# Other humanoid table-tennis systems: what is worth trying (2026-10-19)

[繁體中文](humanoid_table_tennis.zh-TW.md) · [Docs index](../README.md)

Survey of SMASH (HKU, silver at the World Humanoid Robot Games 2026), the PKU-BAAI team (gold at
the same games) and closely related work (HITTER = our base recipe, PACE), focused on balance and
stable returns. Numbers are as reported by the sources; "real" = measured on hardware.

## The systems in short

| | SMASH (HKU) | PKU-BAAI | HITTER (Berkeley) | PACE (Purdue) |
|---|---|---|---|---|
| Robot | Unitree G1 | AgiBot A3 at the games (every team used the A3); paper is a 6-DoF ABB arm | Unitree G1, 29 DoF | Booster T1 |
| Control | model-based planner -> PPO whole-body tracking of mocap clips (400 real + 5k generated), wrists free of tracking | humanoid method unpublished; arm paper: physics prediction + learned residual + imitation policy conditioned on landing region | planner gives racket target **and a base xy target**; PPO | one end-to-end RL policy (stroke + footwork) |
| Feet | lateral moves allowed | not published | **one deliberate step** per move on the real robot (vs. multi-step shuffling with velocity commands) | footwork learned |
| Balance | no explicit balance reward; pushes + friction randomization; 20/20 stable after strikes (real) | not published | not stated | terminate on fall |
| Returns | real: 18/20 with mocap, 12/20 with head camera; 59.7 % over 642 balls | arm, real: 92 % within 30 cm of target | real: 92 % returns, rallies to 106 | real: 61 % returns |

Sources: SMASH arXiv 2604.01158 (https://mmlab.hk/Smash/); SpikePingpong arXiv 2506.06690 (PKU);
WHRG 2026 reports (Xinhua, PKU News); HITTER arXiv 2508.21043; PACE arXiv 2509.21690.
None of them plants the feet; none uses ZMP/CoM constraints.

## Worth trying for us (ranked)

1. **Same post-strike command in training and deploy; one deliberate step instead of shuffling.**
   HITTER (real G1) got single steps from a base-target command. **Done today:** our deploy showed a
   wrong target after contact; `hold_last_target` removed the stepping completely in MuJoCo (3.5 -> 0
   steps per ball). Next: gate a foot-contact-change cost off only when the base target is far away
   (D22 jobs test the extremes: fixed station, planted feet).
2. **Command noise that shrinks as time-to-strike shrinks** (SMASH Eq. 14): noise on target position,
   velocity and time, large far from contact, small near it. Real G1: returns 13/20 with it vs 7/20
   without. Low effort. Our `vb_timing_jitter_s` only covers timing.
3. **Fix the short returns at the source.** (a) Re-fit restitution/drag so the planner's outgoing
   ball matches what the racket really does (HITTER fitted from 15 recorded balls); a constant short
   bias (our aim offset +0.23 m) usually means the restitution or delivered racket speed is off.
   (b) PACE-style reward on the predicted landing point and net clearance right after contact.
   (c) On hardware later: SpikePingpong-style learned residual on the racket target from logged
   (command -> landing) pairs (arm: 23 % -> 92 % within 30 cm).
4. **Adaptive tracking-std curriculum and region-adaptive sampling** (SMASH, sim only): shrink the
   racket-reward std as errors fall (without it 86 % -> 23 % success in their sim); oversample the
   zone/feed bins with the lowest success.
5. **Free the wrist from imitation; sign-agnostic face reward** (SMASH): lets the policy steer the
   face (= landing direction) without breaking the body prior.
6. **Harder randomization for hardware**: PACE lost 94 % sim -> 61 % real and blames the ankle
   mechanism and motor dynamics; the G1 ankle is also a parallel linkage.
7. **More forehand clips** (SMASH: 400 -> 5k clips, 77 % -> 86 % sim success). High effort.

Not recommended now: hard ZMP/CoM constraints (no system uses them). A strict no-step rule has no
supporting evidence and may cost push robustness; job 3 of D22 measures exactly that.
