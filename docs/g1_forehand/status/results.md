# Results summary

[繁體中文](results.zh-TW.md) · [Docs index](../README.md)

Full tables, curves and the status log are in
[runs_and_results.md](runs_and_results.md) and [log.md](log.md). Summary:

* Environment: Isaac Sim 5.1 + Isaac Lab 0.54.2 boot and train on the 2080 Ti. The G1 smoke passes
  all steps. The Isaac/MuJoCo plant replay matches `g1-port`'s numbers (0.023 rad, 0.047 m, fall
  onset 1.34 s vs 1.40 s).
* Video pipeline: verified on GVHMR's example video (person height 1.73 m, 10 m/s peak racket
  speed); converter round trip within 0.012 rad / 1.6 mm. Our own forehand video is not recorded yet.
* Runs (4096 envs, ~8 s/iteration):

  | Run | Task | Latest (10-12 14:00) |
  |---|---|---|
  | `g1_fh_diag` (**main**) | `_FH_XC` | iteration 3814 / 20 000: racket ≤10 cm 89 %, contact 62 %, legal return 39 %, falls 7.7 % |
  | `g1_fh_rally` | `_FH_XC_RALLY` (D14), warm start at 2600 | iteration 4713 / 14 600: racket ≤10 cm 92 %, contact 64 %, legal 39 %, falls 3.6 %, rally pace 26 % of swings, jitter 21 ms |
  | `g1_fh` (baseline) | `_FH` (A3 bank) | stopped at 2196 |

* Club success in Isaac (`evaluate_forehand.py`, 8100 balls): `g1_fh_diag` `model_2300` **11.1 %**
  (24.5 % `opp_right_mid`, 6.2 % `center_line_deep`, 11.3 % `opp_right_deep`, 2.3 % corner; contact
  92 %, over the net 91 %, legal 79 %). The baseline `g1_fh` `model_2100` reaches 4.6 %.
* MuJoCo with a real ball fed from the bank: `model_2400` makes contact on 10 of 12 but strikes
  15–40 ms before the apex, and none land in the zone yet. At real rally pace (`--rally-pace`) contact
  drops from 88 % to 38 % and the robot falls on 44 % of balls. This is why `g1_fh_rally` exists.
* HOPE upstream issues found along the way: commit `00e2e96` made the Gate 3 table in
  `a3_pingpong.xml` collide with the robot (it pins the racket arm in the standalone evaluator), and
  the evaluator's default geometry does not match Gate 3 (0.30 m vs 0.50 m to the table edge).
  `g1-port` already moves the evaluator to recipe geometry.

