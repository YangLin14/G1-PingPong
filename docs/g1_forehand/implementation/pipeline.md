# End-to-end pipeline

[繁體中文](pipeline.zh-TW.md) · [Docs index](../README.md)

```text
 phone video ──► GVHMR (SMPL-X) ──► GMR (G1) ──► gmr_video_clip_to_g1 ──► G1 reference clip (.npz/.yaml)  [pending video]
                                                                                   │
 club rules ──► forehand_success.py ──► make_g1_forehand_bank ──► question bank ───┤
                                                                                   ▼
                                     Isaac Lab PPO (task HOPEPingPongUnitreeG1_FH_XC, SDSC, ~45 h)
                                                                                   │ warm start
                                                                                   ▼
                                     rally-timing run (task _FH_XC_RALLY: faster feeds, rally pace, jitter)
                                                                                   │ checkpoints
                                                                                   ▼
            evaluation: evaluate_forehand.py, per-zone club success (Isaac)  ·  export_onnx --robot unitree_g1
                                                                                   ▼
          MuJoCo sim-to-sim with a real ball (mujoco_eval_onnx.py: bank feeds, rally pace, club scoring)
                                                                                   ▼
                       hardware: G1 runner + mocap + planner  [not built yet; C++ runner is A3-only]
```

