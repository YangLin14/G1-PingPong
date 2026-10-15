# 端到端流程

[English](pipeline.md) · [文件目錄](../README.zh-TW.md)

```text
 手機影片 ──► GVHMR（SMPL-X）──► GMR（G1）──► gmr_video_clip_to_g1 ──► G1 參考動作 clip（.npz/.yaml）  [等影片]
                                                                                   │
 社團規則 ──► forehand_success.py ──► make_g1_forehand_bank ──► 題庫 ──────────────┤
                                                                                   ▼
                                     Isaac Lab PPO（任務 HOPEPingPongUnitreeG1_FH_XC，SDSC，約 45 小時）
                                                                                   │ 接續權重
                                                                                   ▼
                                     真實對打節奏的 run（任務 _FH_XC_RALLY：快球、對打節奏、時機誤差）
                                                                                   │ checkpoints
                                                                                   ▼
            評估：evaluate_forehand.py，按落點算社團成功率（Isaac）  ·  export_onnx --robot unitree_g1
                                                                                   ▼
          MuJoCo sim-to-sim 打真的球（mujoco_eval_onnx.py：題庫餵球、對打節奏、社團規則計分）
                                                                                   ▼
                       實機：G1 runner + 動作捕捉 + 規劃器  [尚未建置；C++ runner 只支援 A3]
```

