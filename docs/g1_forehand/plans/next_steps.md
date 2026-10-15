# Next steps

[繁體中文](next_steps.zh-TW.md) · [Docs index](../README.md)

1. Record the forehand video, convert it with `video_to_g1`, and inspect the preview.
2. Train `_FH_XC` with the human clip (`motion_file=...`; update `strike_phase_per_clip`). A shorter
   pre-strike segment also helps the lead-time problem.
3. Add the lean HITTER reward recipe and A/B it against the HOPE recipe (D12).
4. ~~Isaac evaluation with `forehand_success`, per zone~~: done (2026-10-11),
   `scripts/evaluate_forehand.py` / `eval_forehand.sh`.
5. Zone selection (D13): a `target_zone_ids` filter on the bank for training/eval/play, and the
   same choice exposed in the deploy planner. (Wider feeds: done with the faster-feed bank.)
6. Close the real-ball gap found by the MuJoCo bank evaluation. **Strike timing: found and fixed in
   simulation (D17).** Next: the ~30 ms not yet explained (Isaac-vs-MuJoCo replay), and aim (3–10 % in zone). Done so far (D14): timing jitter on
   the virtual strike, a late swing start at rally pace, faster feeds. Still open: check MuJoCo's racket
   contact against the venue paddle model (MuJoCo returns the ball lower), then compare `g1_fh_rally`
   with `g1_fh_diag` on `eval_rally` / `eval_bank` / Isaac zones once the ramps are full (~iteration 5600).
7. Spin estimation (D16): opponent-racket reading + Magnus fit of the flight; train with spin errors.
   Hardware target from the perception study: ≥120 Hz, ≤15–30 ms, ≤5 mm, synced views of the table
   and the opponent's racket.
8. **Aim precision (D18):** the remaining miss is scatter (~20 cm); needs a more precise swing in training
   (e.g. more weight on racket velocity at the strike), and the real racket's paddle model + aim offset.
   **To do:** test whether the `strike_time` clock also helps the original A3 setup (A3 policy in its own
   MuJoCo evaluation, `countdown` vs `strike_time`), and find the remaining ~40 ms (Isaac-vs-MuJoCo replay).
9. Before hardware: measure the real racket adapter, refit ball physics on our table, and write a
   G1 deploy runner (the C++ runner is A3-only).
10. Then footwork (free station; also needed for feeds slower than ~4.4 m/s), then
   backhand, then the serve plan.

11. **Balance (D19):** submit `g1_fh_bal` when zixian is back (runbook 4); compare with the push table
    (status log 2026-10-14 17:00). If hits still drop under shoves, also randomize the torso CoM more
    and add pushes during the swing only after this works.
12. **Fist adapter for the real G1 (D20):** the stock rubber hand cannot close. Design a 3D-printed fist
    holder that puts the blade centre at `G1_MOUNT_OFFSET` (0.210, 0.015, 0.010) m in the wrist frame;
    if it ends up elsewhere, update the mount and fine-tune.
