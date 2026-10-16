# 程式改了什麼（`g1-forehand-rally` branch）

[English](changes.md) · [文件目錄](../README.zh-TW.md)

### 訓練程式

| 檔案 | 改動 |
|---|---|
| `tasks/tracking/mdp/hope_commands.py` | 新增 `RacketTargetCommandCfg.active_clip_ids`，代表這個設定實際訓練哪幾手（空的 = 正反手都練）。只練一手時有三個效果：(1) 能力門檻只檢查那一手，否則永遠抽不到的反手會讓門檻一直打不開；(2) `fixed_balanced_bank_v1` 的兩組固定題目都分給那一手，否則會報錯 `permanent physical tuple cohort disagrees with MotionCommand clip`；(3) 題庫載入器只要求那一手有足夠的題目。沒有設定時，行為跟原本完全一樣。 |
| `cfg/task/HOPEPingPongUnitreeG1_FH.yaml` | 只練正手的 variant A 設定：`eval_clip_sequence=[0]`、`fixed_clip_env_fraction_per_clip=0`（不再有永久的反手環境）、`active_clip_ids=[0]`。還在用 A3 題庫，當作 baseline。 |
| `cfg/task/HOPEPingPongUnitreeG1_FH_XC.yaml` | 在 `_FH` 的基礎上加上斜線最高點題庫，比例 100%。**這是主要的訓練設定。** |
| `cfg/task/HOPEPingPongUnitreeG1_FH_XC_RALLY.yaml` | `_FH_XC` 加上更快來球的題庫、對打節奏（`rally_pace_prob` 0.6）、±30 ms 時機誤差，在 72,000 個環境步數（3000 個 iteration）內逐步加入。是 `g1_fh_rally` 用的設定（D14）。 |
| `tasks/tracking/mdp/commands.py` | `MotionCommand` 的對打節奏（`rally_pace_prob`、`rally_pace_ramp_steps`、`rally_pace_start_tts_range`、`rally_pace_post_strike_range`）：連續揮拍時，下一拍從 clip 的中段開始（距離擊球 0.4–0.83 秒、沒有等待），前一拍的收拍在擊球後 0.7–1.0 秒就切掉。指標 `rally_pace_swing`、`rally_pace_prob`。預設關閉。 |
| `tasks/tracking/mdp/hope_commands.py`（另外） | `vb_timing_jitter_s` / `vb_timing_jitter_ramp_steps`：擊球那一幀的虛擬球會沿著飛行軌跡往前或往後移 τ ~ U(−s, s)，所以策略要學會打稍早或稍晚到的球。指標 `virtual_timing_jitter_s`。預設關閉。 |
| `scripts/train.py`、`cfg/train.yaml` | `restore_step_counter`（D15）。 |
| `utils/forehand_success.py` | [社團規則](../overview.zh-TW.md)的成功判定：`classify_strikes`、`ForehandSuccessTally`（分母包含沒打到的球）、`TARGET_ZONES`、`FEED_ZONE`、餵球起點。統計是一層一層的漏斗（接觸 → 過網 → 落在對方半場 → 落在目標區），`ZoneSuccessReport` 再按落點分開。不需要 Isaac 也能執行。 |
| `tasks/tracking/mdp/events.py`、`scripts/train.py` | `push_by_setting_velocity_ramped`（Isaac 的推擠，範圍隨 `env.common_step_counter` 從 0 增加）；task YAML 的 `domain_rand.push_velocity_range` / `push_interval_s` / `push_ramp_steps` / `push_ramp_start_steps` 會裝上 `events.push_robot`（沒寫就不變）。測試：`tests/test_domain_rand_overrides.py`。 |
| `cfg/task/HOPEPingPongUnitreeG1_FH_XC_RALLY_BAL.yaml` | `_FH_XC_RALLY` + 推擠（D19）。 |
| `unitree/scripts/build_g1_pingpang_models.py` | 兩種球拍安裝（`Mount`）：`FOREARM`（原本的，張開的手）和 `FIST`（D20：握把橫穿拳頭、和前臂垂直）→ `unitree/URDF/g1_29dof_pingpang_fist/`，含產生的拳頭網格 `right_fist.STL`；`--check` 兩個都比對。 |
| `robots/g1.py`、`scripts/train.py` | `G1_FIST_MOUNT_OFFSET/QUAT`、`G1_RACKET_MOUNTS`；task YAML 的 `g1_racket_mount: fist` 會換掉 Isaac 資產（獨立的 `assets/unitree_g1_fist`，沒有就自動準備）和球拍目標的安裝參數。 |
| `tasks/tracking/mdp/hope_rewards.py`、`config/g1/hope_env_cfg.py` | D21：`root_height_ceiling_debt`（骨盆高於 `max_height` 的懲罰，Huber）和 `com_support_debt`（依質量加權的全身重心投影到地面，離兩個 ankle-roll link 連線超過 `margin` 的懲罰；質量在開機隨機化之後讀一次）。`add_balance_terms` 把兩個都裝到每個 G1 hitter 環境，**權重 0**（現有任務不變）；任務用 `overrides: rewards.<term>.weight` 開啟。指標 `root_height`、`com_height`、`com_support_offset`。 |
| `cfg/task/HOPEPingPongUnitreeG1_FH_FIST_STABLE(_SWING).yaml` | D21 平衡優先的設定（見決策）。 |
| `scripts/mujoco_pingpong_scene.py`、`scripts/mujoco_eval_onnx.py` | `scene.posture()`；每球的 JSON 多了 `posture`：平均骨盆高度、平均重心高度、重心離雙腳連線的平均 / 最大距離。 |
| `slurm/pip_env/train.sbatch`、`g1_smoke.sbatch` | `AVOID_GPU_IDX` 預設改成 `0,5`。用 `--gres=rtx2080ti:2` 時，job 會把要避開的 GPU 對 CUDA 隱藏（`CUDA_VISIBLE_DEVICES` = 第一張好的，PCI 順序）；沒有好的就停止。smoke 第 0 步會印出 CUDA 看到的 GPU。 |
| `retarget_a3_clip_to_g1_ik.py --mount fist` | 在握拳模型上做 IK 重定向 → `g1ikfist_hope_{forehand,backhand}`；持拍手臂只追球拍中心和拍面（擊球幀拍面誤差 8°）。 |
| `cfg/task/HOPEPingPongUnitreeG1_FH_FIST.yaml` | `_FH_XC_RALLY_BAL` + 握拳安裝 + 握拳 clip（D20）。 |
| `slurm/pip_env/g1_smoke.sbatch` | 加上 `AVOID_GPU_IDX` 保護（不會跑在 idx 5）。 |

### 工具（`hope_training/whole_body_tracking/scripts/`）

| 檔案 | 用途 |
|---|---|
| `make_g1_forehand_bank.py` | 產生題庫（[題庫](question_bank.zh-TW.md)）。`--vin-x/--vin-y/--vin-z/--spin-rev-s` 設定來球範圍。 |
| `evaluate_forehand.py`（+ `cfg/evaluate_forehand.yaml`） | 在 Isaac 裡按落點算社團成功率：確定性策略、512 個環境 × 60 秒，每一球的落點區由題目的目標落點決定。 |
| `mujoco_forehand_bank.py` | 把題庫的題目搬到 MuJoCo：遠端底線的餵球初始狀態、NumPy 版的 venue 飛行模型、閉迴路的最高點規劃器（真的反彈之後會重新解一次球拍速度）。 |
| `mujoco_eval_onnx.py`（擴充） | `--serve-bank`（用我們的題庫餵真的球）、`--zones`、`--feed-lead-s`、`--rally-pace` / `--rally-gap-s`（我們擊球後 0.7–1.0 秒對手就回下一球，球發出時才知道打哪裡）；每一球、每個落點都用社團規則計分。預設模式不變。 |
| `mujoco_pingpong_scene.py`（擴充） | 可選的 Magnus 力、畫出目標區、球的旋轉，以及 venue 的桌面反彈模型（MuJoCo 自己的反彈會讓最高點深 20–40 cm）。 |
| `check_rally_pace.py` | 在 Isaac 檢查對打節奏的時鐘和時機誤差（揮拍開始時間、兩拍間隔、跌倒）。 |
| `mujoco_eval_onnx.py`（新增） | `--racket-contact venue`（用訓練用的球拍模型取代 MuJoCo 的軟碰撞）、`--aim-offset DX DY`（D18）、`--trace-out`（逐幀紀錄，含拍面方向和球的旋轉）。 |
| `onnx_from_checkpoint.py` | 不用 Isaac 就能產生 checkpoint 的 policy.onnx：在 CPU 上把 actor 權重放進同一個任務已匯出的 ONNX 圖。驗證過跟真正的匯出完全相同。 |
| `ball_perception.py`、`perception_sweep.py` | `mujoco_eval_onnx.py --perception` 的攝影機模型（幀率、延遲、雜訊、漏幀、HITTER 式擬合、旋轉不知道／知道／有誤差的估計），以及跑遍各種攝影機系統的掃描腳本。`--reset-after-fall` 會在兩球之間把機器人扶起來。 |
| `mujoco_eval_onnx.py --push V_XY W_RPY`、`--push-window-s` | 平衡測試：每球推一次，底座速度隨機改變最多 V_XY m/s（x、y）和 W_RPY rad/s（翻滾、俯仰、偏航），時間在餵球後 0–1 秒內隨機。推擠用獨立的亂數，所以同一個 seed 的發球序列和沒推時一樣。每球紀錄有 `push`。 |
| `view_feed_bank.py` | 在 MuJoCo 裡播放題庫的每一題：從對手右邊斜對角餵球 → 反彈 → 最高點 → 回球，同時畫出餵球起點、所有區域（目前的目標會亮起來）和球拍目標；可以開互動視窗或輸出 MP4，鏡頭可選 `--camera behind|top`。 |
| `gvhmr_to_g1_headless.py` | GMR 的「GVHMR → G1」轉換，但不開視窗（叢集節點沒有螢幕）。 |
| `gmr_video_clip_to_g1.py` | 把 GMR 的輸出轉成 HOPE 的 G1 clip：裁切、30 → 50 Hz、對齊朝向和位置、重排關節順序並限制在關節範圍內、把結尾混接回開頭讓 clip 可以循環、用 MuJoCo 算出 14 個身體部位的姿態、標記擊球幀（自動取球拍速度最快的一幀，或手動指定），可選擇輸出 MP4 預覽。 |

### 資料、腳本、測試

| 路徑 | 內容 |
|---|---|
| `assets/venue_tuple/g1_fh_crosscourt_apex_bank.{npz,receipt.json}` | 5 萬題的題庫（seed 0，可以重現，每個落點各 12,500 題）和 receipt，receipt 記錄了完整的產生規格和統計數字。 |
| `assets/venue_tuple/g1_fh_crosscourt_apex_bank_wide.{npz,receipt.json}` | `_FH_XC_RALLY` 用的快球題庫（[題庫](question_bank.zh-TW.md)）。因為 `assets/venue_tuple/*` 在 gitignore 裡，所以是用 force-add 加進來的。 |
| `hope_training/slurm/pip_env/` | 不用容器的 SDSC 安裝、smoke 測試、訓練、通用執行腳本（[README](../../../hope_training/slurm/pip_env/README.md)）。`train.sbatch` 新增 `INIT_CHECKPOINT`（從別的 run 接續權重）、自動改用正常 GPU、`restore_step_counter`；另有 `status.sh`、`progress.sh`、`eval_checkpoint.sh`（MuJoCo 影片：範例球、題庫、對打節奏）、`eval_forehand.sh`（Isaac 分區評估）。用法見 [runbook](../operations/runbook.zh-TW.md)。 |
| `hope_training/slurm/video_to_g1/` | 「影片 → G1 clip」的安裝腳本和 Slurm 腳本（[README](../../../hope_training/slurm/video_to_g1/README.md)）。 |
| `tests/test_forehand_success.py` | 12 個測試：四個落點（三個沿底線排成一排、互不重疊）、餵球區和餵球起點、最高點擊球**確實可以**打進目標區（用網格搜尋證明）、往下掉的球／沒打到／太輕／打直線都判定失敗、沒打到的球有算進分母、漏斗統計、分區報告。 |
| `tests/test_make_g1_forehand_bank.py` | 4 個測試：符合載入器的格式；符合餵球區／最高點／擊球範圍／球拍速度的限制；餵球確實從對手右邊斜對角過來；每個落點題數相同，而且**每一題用它自己的落點檢查都是成功**。 |
| `tests/test_gmr_video_clip_to_g1.py` | 3 個測試：把現有的 G1 clip 偽裝成 GMR 輸出再轉回來（關節誤差 < 0.03 rad、身體位置誤差 < 1 cm、擊球幀誤差 ±1），sidecar 格式正確，結尾混接正確。 |
| `tests/test_mujoco_forehand_bank.py` | 3 個測試：MuJoCo 端的飛行模型和規劃器算出來的最高點、球拍速度跟題庫一致。 |
| `a3_deploy/.../reference/a3_deploy_onnx_ref_pingpong/{lifecycle,racket_command,ros_command_source,config,runner}.py` | 可選的 `clock: strike_time`（D17）：策略看到的距離擊球時間是 `strike_time − 現在時間`。預設 `countdown` 維持原本的行為。 |
| `tests/test_lifecycle_strike_time.py` | 5 個測試：countdown 不變、strike_time 時鐘對新訊息和晚到的訊息都準確、沒有絕對時刻時改用收到時間、沒有 `now` 時退回 countdown、ROS 第 [10] 欄。 |
| `tests/test_ball_perception.py` | 4 個測試：無雜訊的擬合和延遲、反彈切段、preset、每顆球固定的旋轉估計誤差。 |
| `docs/g1_forehand/plans/serve_side_backspin.md` | 暫緩的發球計畫（D4）。 |

