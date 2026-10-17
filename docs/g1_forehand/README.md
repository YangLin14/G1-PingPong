# Triton Droids: Unitree G1 forehand cross-court rally — docs

[繁體中文](README.zh-TW.md)

What the Triton Droids team built on top of HOPE and the club's G1 port: a G1 that returns a
cross-court forehand into one of four target zones. Simulation only for now (the G1 has not
arrived); everything is designed so the sim-to-real path stays open.

## Current status (2026-10-17)

* **Cluster:** zixian was rebooted on 10-16 and CUDA works again. We still avoid idx 0 and 5. The balance-first runs (D21) are running: 5858 `g1_fh_stable` and 5859 `g1_fh_stable_swing`. Both start from `model_32299` and run to 40 300 (ETA 10-17 ~15:00). `g1_fh_fist` (D20) finished at 32 299.
* **Best policy: `g1_fh_fist` `model_32299`** (fist grip, MuJoCo, deploy defaults, no aim offset, 240 balls):
  hits 100 %, at the apex ~95 %, legal return ~86 %, in the target zone ~26 %. With the aim offset (+0.14, −0.03) m: **42 % in zone**, 80 % legal (held-out balls, training paddle model).
  (Previous best, old grip `g1_fh_diag` `model_15700`: 35 % in zone with the aim offset.)
* **Balance:** no falls without shoves or with shoves up to 0.5 m/s; 5 % at 1.0 m/s (old: 26 %).
  Passes our criterion, so no new balance reward (D19).
* **Robot model:** fist grip, racket perpendicular to the forearm (D20).
* Neighbouring checkpoints differ a lot (32 000: 73 % hits): always pick by the MuJoCo test.

Details: [status/log.md](status/log.md) (newest first).

## Where to find what

| Folder | File | What is in it |
|---|---|---|
| (top) | [overview.md](overview.md) | Context, the club's success rules, how the whole system works |
| | [decisions.md](decisions.md) | Every design decision (D1, D2, …) and why |
| `implementation/` | [changes.md](implementation/changes.md) | What changed in the code, file by file |
| | [question_bank.md](implementation/question_bank.md) | How the feeds and target zones are generated |
| | [pipeline.md](implementation/pipeline.md) | End-to-end pipeline: video → clip → training → evaluation → deploy |
| `operations/` | [runbook.md](operations/runbook.md) | Day-to-day commands: check training, export, open MuJoCo, start/stop jobs |
| | [running.md](operations/running.md) | Setup and full run commands (laptop and cluster) |
| | [deploy_calibration.md](operations/deploy_calibration.md) | What to measure on the real robot, and how |
| `status/` | [log.md](status/log.md) | Status log: every check, incident and finding, newest first |
| | [runs_and_results.md](status/runs_and_results.md) | All runs, training curves, per-zone success tables |
| | [how_we_track.md](status/how_we_track.md) | Which metrics we use, how they are measured, how to add an entry |
| | [results.md](status/results.md) | Early results summary (2026-10-12) |
| `research/` | [perception.md](research/perception.md) | Cameras / mocap: what is needed and the perception experiment |
| `plans/` | [next_steps.md](plans/next_steps.md) | What to do next |
| | [serve_side_backspin.md](plans/serve_side_backspin.md) | Shelved plan: backhand side-backspin serve |

Related docs outside this folder: [../UNITREE_G1.md](../UNITREE_G1.md) (the G1 port: robot model,
racket mount, plant parity) and the upstream HOPE docs in [`docs/`](..).

## Adding to the docs

* A check, incident or experiment result → a new entry at the top of [status/log.md](status/log.md);
  numbers that replace a table → [status/runs_and_results.md](status/runs_and_results.md).
* A design choice → a new row in [decisions.md](decisions.md); the code change → [implementation/changes.md](implementation/changes.md).
* Always both languages (`*.md` and `*.zh-TW.md`), plain wording, but keep the numbers and caveats.
