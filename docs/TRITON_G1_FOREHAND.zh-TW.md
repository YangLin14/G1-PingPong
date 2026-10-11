# Triton Droids：Unitree G1 正手斜線對打

[English](TRITON_G1_FOREHAND.md)

這份文件記錄 Triton Droids 在 HOPE 和社團 G1 移植版上做了哪些事：每個決定的理由、整套系統從頭到尾怎麼運作，以及每個部分怎麼執行。
狀態為 **2026-10-10**：G1 實機還沒到，目前只在模擬中進行，但所有設計都保留了之後上實機（sim-to-real）的路徑。

---

## 1. 背景

| Repository | 說明 |
|---|---|
| [`hitchopen/HOPE`](https://github.com/hitchopen/HOPE) | 上游 HOPE：給 **Agibot A3** 用的開源人形機器人桌球平台，包含 Isaac Lab 訓練、球路規劃器、MuJoCo sim-to-sim、C++ 部署 runner。 |
| `YangLin14/G1-PingPong` | 我們 fork 的 HOPE，也就是本 repo（remote 名稱 `origin`）。 |
| [`triton-droids/HOPE`](https://github.com/triton-droids/HOPE) 的 `g1-port` branch | 社團 Oren Gershony 做的 G1 移植版：裝了球拍的 G1 URDF/MJCF、recipe v2（把 A3 的數值按比例換算成 G1）、兩組從 A3 轉換來的 G1 動作 clip、MuJoCo 物理一致性檢查、Slurm 腳本。說明見 [`UNITREE_G1.md`](UNITREE_G1.md)。 |
| `triton-droids/pingpong` | 更早期、獨立的原型（在 G1 上用 DLS IK 加 PPO 學修正量，Isaac Lab），這裡沒有使用。 |

這次的工作都在 **`g1-forehand-rally`** branch 上，從 `g1-port` 分出。

HOPE 的架構來自 UC Berkeley 的 **HITTER**（[arXiv 2508.21043](https://arxiv.org/abs/2508.21043)）：用模型預測球路，決定球拍**在哪裡、什麼時候、多快**擊球；再用強化學習訓練全身控制器，讓機器人做到這件事，同時模仿一段人類的揮拍動作。HITTER 就是在 **Unitree G1** 上做的：擊中率 96%、成功回球 92%、最長連續對打 106 拍。它的參考動作做法是拍一段人揮拍的影片，用 **GVHMR** 重建成人體動作，再用 **GMR** 重定向到 G1。

---

## 2. 決策紀錄

| # | 決策 | 理由 |
|---|---|---|
| D1 | 以 HOPE/HITTER 為基礎，不用自己手寫的 `pingpong` 原型。 | HITTER 已經在 G1 上驗證過；HOPE 提供完整、測試過的訓練、評估、部署流程。 |
| D2 | 從 Oren 的 `g1-port` 開始。 | G1 模型、球拍安裝、關節順序、MuJoCo 物理一致性、參數換算這些問題他都已經解決了。 |
| D3 | 機器人用 Unitree G1，先做模擬，但保留上實機的路徑。 | 實機還沒到。 |
| D4 | 反手側下旋**發球**已經規劃好，但**暫緩**。 | 正手對打比較簡單，優先順序也比較高。計畫保留在 [`plans/SERVE_SIDE_BACKSPIN_PLAN.md`](plans/SERVE_SIDE_BACKSPIN_PLAN.md)。 |
| D5 | **只練正手**。 | 只練一種揮拍，學習問題比較小；反手之後再做。 |
| D6 | 用社團自己的對打規則（第 3 節）判定成功，不用 HOPE 的 `success_rate`。 | HOPE 的 `success_rate` 公式算出來的回球永遠過不了網，連官方 A3 模型也是 0%。 |
| D7 | **斜線球**：從機器人的右邊打到**對手的**右邊。餵球區暫時固定一個區域，最終要有三個落點。 | 社團規格，見第 3 節。 |
| D8 | 訓練題目全部換成 G1 專用的**最高點擊球、斜線回球**題庫。 | 繼承來的 A3 題庫是在球往下掉時擊球（擊球瞬間垂直速度中位數 −0.77 m/s）、落點分佈在整個台面寬度、擊球位置是 A3 的（前方 0.58 m、高 0.85–1.30 m），而且只佔 25% 的題目。 |
| D9 | 參考動作改用**真人影片**製作（GVHMR + GMR），跟 HITTER 一樣。 | `g1-port` 的 clip 是直接把 A3 的關節角度套到 G1 上；擊球那一幀球拍只在約 0.78 m（桌面高度），第一版 G1 訓練也有搆不到的問題（x 少 9 cm、z 少 10 cm）。 |
| D10 | 不拿 A3 的 `model_21800` 來校準指標。 | 它在我們的環境裡打不到球；改用單元測試，給定「已知會成功」和「已知會失敗」的球拍狀態來驗證指標。 |
| D11 | 在 SDSC 上用 **pip** 安裝 Isaac，不用 64 GB 的容器。 | 叢集的 home 是 BeeGFS，podman 在上面跑不起來；pip 安裝不需要 sudo。節點上的問題都寫成腳本處理（第 8 節）。 |
| D12 | 照 HITTER 寫一個精簡版獎勵（上半身模仿、球拍目標、站位、平滑項），跟 HOPE 的 53 項獎勵做 A/B 比較。 | **尚未進行**：要先有影片做出來的 clip。 |

---

## 3. 任務規格（社團規則）

所有座標都是**環境座標系、機器人視角**：機器人根部在原點、面向 **+x**（朝桌子），**+y 是機器人的左邊**，z 朝上。近端桌邊在 x = 0.5 m，桌面高 z = 0.76 m，桌子以 y = 0 為中心，網子在 x = 1.87 m，遠端桌邊在 x = 3.24 m。

一球正手回球要同時滿足以下全部條件才算**成功**：

1. **時機**：在機器人這邊反彈後的**最高點**擊球，也就是擊球瞬間來球的垂直速度 |vz| ≤ 0.3 m/s，約等於最高點前後 30 ms。
2. **擊中**：球拍中心離球 9.5 cm 以內，而且正在以超過 0.3 m/s 的速度往球的方向移動（跟訓練時的判定一樣）。
3. **合法**：過網時球心要高於網子頂端加一個球半徑，而且第一次落地在對方半場。
4. **落點**：第一次落地在目標區內。

斜線的方向：機器人從**自己的右邊**（−y）擊球，打到**對手的右邊**，也就是從機器人的角度看，遠端桌子的**左半邊**（+y）。

| 區域 | 深度（半場長度的比例，從網子算起） | 橫向（半場寬度的比例，往 +y 為正） | 環境座標範圍 |
|---|---|---|---|
| **餵球區**（來球在機器人半場的落點） | 中間三分之一 | −0.75 … −0.25（機器人右邊） | x 0.957–1.413，y −0.572…−0.191 |
| `opp_right_mid`（**目前的訓練目標**） | 中間三分之一 | +0.25 … +0.75 | x 2.327–2.783，y 0.191–0.572 |
| `center_line`（中線） | 中間三分之一 | −0.10 … +0.10 | x 2.327–2.783，y −0.076…0.076 |
| `opp_right_corner`（右半邊靠近桌角） | 2/3 … 0.95 | +0.60 … +0.95 | x 2.783–3.172，y 0.458–0.724 |

最終目標是三個落點可以切換；訓練先從 `opp_right_mid` 加上固定的餵球區開始。

---

## 4. 系統怎麼運作

```text
                 題目（每次揮拍一題）                                 策略（50 Hz）
  題庫的一列：站位、球拍擊球點、                 ──►   104 維觀測：本體感知、球拍目標
  球拍速度（= 拍面法向）、擊球瞬間來球的              位置與速度、距離擊球的時間、站位目標……
  速度與旋轉、目標落點                                ──► 29 維動作 ──► G1 關節 PD 目標
                                                                    │
                                                 Isaac Lab / PhysX  ▼  （物理 200 Hz）
  獎勵（每一步）                                          機器人動作；參考動作的時鐘同步前進
   * 追蹤參考揮拍動作（BeyondMimic）
   * 擊球瞬間的球拍位置、速度、拍面法向
   * 擊球瞬間的「虛擬球」：實際的球拍狀態 ──► 旋轉接觸模型 ──►
     Magnus + 阻力飛行 ──► 過網高度、落點跟這一題目標落點的距離
   * 穩定與平滑相關的項目
```

* **參考動作 clip**：50 Hz 的揮拍動作（準備 → 擊球 → 回位），關節照 G1 在 Isaac 裡的順序排列，並記錄 14 個身體部位的軌跡；擊球是哪一幀標在 sidecar YAML 裡。策略會因為跟著這段動作做而得到獎勵，所以動作會像人（這是 HOPE 沿用的 BeyondMimic 追蹤方法）。
* **題目**：HOPE 的 `RacketTargetCommand` 會告訴策略「再過多久要讓球拍到哪裡」。在 `fixed_balanced_bank_v1` 模式下，這個目標是從預先算好的題庫抽出來的。`HOPEPingPongUnitreeG1_FH_XC` 把題庫的比例設成 100%，所以每一題都是物理上一致的「最高點擊球、斜線回球」。
* **虛擬球**：訓練時不用 PhysX 模擬真的球。在擊球那一幀，用場地實測擬合出來的模型（[`configs/ball_physics_venue.yaml`](../configs/ball_physics_venue.yaml)、[`virtual_ball.py`](../hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/virtual_ball.py)），根據機器人**實際**的球拍狀態算出回球：
  - 飛行：`a = g − k_d|v|v + k_m(ω × v)`（k_d = 0.1261 m⁻¹、k_m = 0.00444）；
  - 球拍和桌面接觸：考慮切向「咬球」的衝量模型（旋轉就是這樣產生的），恢復係數會隨法向速度改變：`e(u_n) = 0.759·exp(−0.0441|u_n|)`；
  - 用真實場地資料擬合，適用範圍約到 7 m/s、15 轉/秒。
* **到處用同一套物理**：題庫產生器、成功判定、訓練獎勵、視覺化工具都呼叫同一個模組，所以彼此的結果不會互相矛盾。

---

## 5. 改了什麼（`g1-forehand-rally` branch）

### 訓練程式

| 檔案 | 改動 |
|---|---|
| `tasks/tracking/mdp/hope_commands.py` | 新增 `RacketTargetCommandCfg.active_clip_ids`，代表這個設定實際訓練哪幾手（空的 = 正反手都練）。只練一手時有三個效果：(1) 能力門檻只檢查那一手，否則永遠抽不到的反手會讓門檻一直打不開；(2) `fixed_balanced_bank_v1` 的兩組固定題目都分給那一手，否則會報錯 `permanent physical tuple cohort disagrees with MotionCommand clip`；(3) 題庫載入器只要求那一手有足夠的題目。沒有設定時，行為跟原本完全一樣。 |
| `cfg/task/HOPEPingPongUnitreeG1_FH.yaml` | 只練正手的 variant A 設定：`eval_clip_sequence=[0]`、`fixed_clip_env_fraction_per_clip=0`（不再有永久的反手環境）、`active_clip_ids=[0]`。還在用 A3 題庫，當作 baseline。 |
| `cfg/task/HOPEPingPongUnitreeG1_FH_XC.yaml` | 在 `_FH` 的基礎上加上斜線最高點題庫，比例 100%。**這是主要的訓練設定。** |
| `utils/forehand_success.py` | 第 3 節的成功判定：`classify_strikes`、`ForehandSuccessTally`（分母包含沒打到的球）、`TARGET_ZONES`、`FEED_ZONE`。不需要 Isaac 也能執行。 |

### 工具（`hope_training/whole_body_tracking/scripts/`）

| 檔案 | 用途 |
|---|---|
| `make_g1_forehand_bank.py` | 產生題庫（第 6 節）。 |
| `view_feed_bank.py` | 在 MuJoCo 裡播放題庫的每一題：餵球 → 反彈 → 最高點 → 回球，同時畫出各區域和球拍目標；可以開互動視窗或輸出 MP4。 |
| `gvhmr_to_g1_headless.py` | GMR 的「GVHMR → G1」轉換，但不開視窗（叢集節點沒有螢幕）。 |
| `gmr_video_clip_to_g1.py` | 把 GMR 的輸出轉成 HOPE 的 G1 clip：裁切、30 → 50 Hz、對齊朝向和位置、重排關節順序並限制在關節範圍內、把結尾混接回開頭讓 clip 可以循環、用 MuJoCo 算出 14 個身體部位的姿態、標記擊球幀（自動取球拍速度最快的一幀，或手動指定），可選擇輸出 MP4 預覽。 |

### 資料、腳本、測試

| 路徑 | 內容 |
|---|---|
| `assets/venue_tuple/g1_fh_crosscourt_apex_bank.{npz,receipt.json}` | 5 萬題的題庫（seed 0，可以重現）和 receipt，receipt 記錄了完整的產生規格和統計數字。 |
| `hope_training/slurm/pip_env/` | 不用容器的 SDSC 安裝、smoke 測試、訓練、通用執行腳本（[README](../hope_training/slurm/pip_env/README.md)）。 |
| `hope_training/slurm/video_to_g1/` | 「影片 → G1 clip」的安裝腳本和 Slurm 腳本（[README](../hope_training/slurm/video_to_g1/README.md)）。 |
| `tests/test_forehand_success.py` | 8 個測試：目標區和餵球區的幾何、最高點擊球**確實可以**打進目標區（用網格搜尋證明）、往下掉的球／沒打到／太輕／打直線都判定失敗、沒打到的球有算進分母。 |
| `tests/test_make_g1_forehand_bank.py` | 3 個測試：符合載入器的格式、符合餵球區／最高點／擊球範圍／球拍速度的限制、**每一題用成功判定檢查都是成功**。 |
| `tests/test_gmr_video_clip_to_g1.py` | 3 個測試：把現有的 G1 clip 偽裝成 GMR 輸出再轉回來（關節誤差 < 0.03 rad、身體位置誤差 < 1 cm、擊球幀誤差 ±1），sidecar 格式正確，結尾混接正確。 |
| `docs/plans/SERVE_SIDE_BACKSPIN_PLAN.md` | 暫緩的發球計畫（D4）。 |

---

## 6. 題庫

`make_g1_forehand_bank.py` 每一題分四步產生：

1. **餵球**：在餵球區裡隨機選一個反彈點，隨機選反彈前的速度（vx −5…−2.5、vy −0.4…0.4、vz −3.5…−1.5 m/s）和最多 5 轉/秒的旋轉，再用場地實測的桌面接觸模型算出反彈後的速度和旋轉。
2. **最高點**：用 RK4（每步 2 ms）模擬飛行，直到最高點（vz = 0）。只保留最高點落在 G1 正手擊球範圍內的球：站位前方 0.42–0.50 m、高度 0.80–1.10 m。站位的 y 由設定裡的正手球拍偏移（−0.38…−0.32 m）推算。
3. **回球**：在 `opp_right_mid` 裡隨機選一個目標落點。用阻尼最小平方法解出球拍速度（拍面法向跟速度方向相同，這是 V14 的規定），同時對齊「落點 x、落點 y、過網高度 − 0.25 m」三個目標，並偏向 (1.9, 0.3, 1.0) m/s 附近的解。
4. **篩選**：落點誤差 ≤ 5 cm、過網高度在 0.25 ± 0.10 m 內（且至少 3 cm）、球拍速度 ≤ 3.5 m/s，而且 `classify_strikes` 判定為成功，才會保留這一題。

已 commit 的題庫統計（5 萬題，seed 0）：

| 項目 | 最小 | 平均 | p95 | 最大 |
|---|---|---|---|---|
| 球拍速度（m/s） | 1.38 | 1.94 | 2.25 | 2.47 |
| 最高點離桌面高度（m） | 0.15 | 0.27 | 0.33 | 0.34 |
| 擊球瞬間來球速度（m/s） | 1.88 | 2.68 | 3.04 | 3.25 |
| 過網高度（m） | 0.23 | 0.30 | 0.33 | 0.35 |
| 落點誤差（m） | 0.000 | 0.015 | 0.033 | 0.050 |

重新產生（筆電 CPU 約 2.5 分鐘）：
`python3 hope_training/whole_body_tracking/scripts/make_g1_forehand_bank.py --rows 50000`。
換 seed 或改物理參數，題目內容就會改變，所以**重新產生之後要重新訓練**。

---

## 7. 端到端流程

```text
 手機影片 ──► GVHMR（SMPL-X）──► GMR（G1）──► gmr_video_clip_to_g1 ──► G1 參考動作 clip（.npz/.yaml）  [等影片]
                                                                                   │
 社團規則 ──► forehand_success.py ──► make_g1_forehand_bank ──► 題庫 ──────────────┤
                                                                                   ▼
                                     Isaac Lab PPO（任務 HOPEPingPongUnitreeG1_FH_XC，SDSC，約 45 小時）
                                                                                   │ checkpoints
                                                                                   ▼
            評估：forehand_success 指標（Isaac）  ·  export_onnx --robot unitree_g1
                                                                                   ▼
                          MuJoCo sim-to-sim（mujoco_eval_onnx.py --robot unitree_g1）
                                                                                   ▼
                       實機：G1 runner + 動作捕捉 + 規劃器  [尚未建置；C++ runner 只支援 A3]
```

---

## 8. 執行方式

### 本機（macOS）

```bash
cd G1-PingPong
python3 -m venv .venv && .venv/bin/pip install -r a3_deploy/a3_deploy_example/reference/requirements.txt torch pytest imageio imageio-ffmpeg
.venv/bin/python -m pytest -q unitree/tests hope_training/whole_body_tracking/tests/test_forehand_success.py \
  hope_training/whole_body_tracking/tests/test_make_g1_forehand_bank.py hope_training/whole_body_tracking/tests/test_gmr_video_clip_to_g1.py

# 看餵球的畫面。macOS 上開 MuJoCo 視窗要用 mjpython；因為這個路徑裡有空格，
# 要透過 venv 的 python 去執行 mjpython（它的 shebang 遇到空格會壞掉）。
.venv/bin/python .venv/bin/mjpython hope_training/whole_body_tracking/scripts/view_feed_bank.py --rows 30 --speed 0.5
.venv/bin/python hope_training/whole_body_tracking/scripts/view_feed_bank.py --rows 8 --record artifacts/feed.mp4
```

畫面中：橘色是餵球區，亮綠色是目前的訓練目標，淡綠色是另外兩個目標；黃色軌跡是球；紅色圓盤和箭頭是球拍目標（拍面和速度）。

### 叢集（SDSC Atlantis，`ssh sdsc-scc`）

* 帳號 `tritondroids`，全社團共用 **32 CPU / 96 GB / 8 GPU**。Isaac 需要 RTX 顯卡，所以要用 `--gres=rtx2080ti:1`（節點 `zixian`，11 GB，driver 610.43）。
* 檔案位置：`~/hope/{HOPE（本 repo）, isaac-venv, IsaacLab, syslibs, mocap, slurm_logs}`。
* 所有 job 都要**在 repo 根目錄**送出。

```bash
sbatch hope_training/slurm/pip_env/install_isaac.sbatch                    # 只需一次
bash   hope_training/slurm/video_to_g1/install_mocap.sh                    # 只需一次（另外要放 SMPL/SMPL-X，見它的 README）
sbatch --export=ALL,TASK=HOPEPingPongUnitreeG1_FH_XC hope_training/slurm/pip_env/g1_smoke.sbatch
sbatch --export=ALL,TASK=HOPEPingPongUnitreeG1_FH_XC,RUN_NAME=g1_fh_xc hope_training/slurm/pip_env/train.sbatch
squeue -u $USER; tail -f hope-g1-train-<jobid>.out
# 在筆電上看 TensorBoard：
ssh -L 6006:localhost:6006 sdsc-scc '~/hope/isaac-venv/bin/tensorboard --port 6006 \
  --logdir ~/hope/HOPE/hope_training/whole_body_tracking/logs/rsl_rl/unitree_g1_hitter_pingpong'
```

要看的指標：`strike_composite_success_exact`、`exact_strike_pos_success_10cm`、`virtual_legal_rate_forehand`、`virtual_landing_error_m_forehand`, `virtual_contact_rate_forehand`、`Episode_Termination/base_fell_tilt`。不要看 `success_rate`。

節點上遇到的問題（每一個都真的發生過，`install_isaac.sbatch` 和它產生的 `setup_train_env.local.sh` 都已經處理好）：

| 症狀 | 解法 |
|---|---|
| Kit 報 `No device could be created` | 設定 `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json` |
| 匯入 URDF 後卡住不動（卡在 `Neuray::LoadModule`） | 從 `mesa-libGLU` RPM 解出 `libGLU.so.1`，加進 `LD_LIBRARY_PATH` |
| `/tmp/isaaclab`、`/tmp/IsaacLab` 沒有寫入權限 | 每個使用者用自己的 `TMPDIR`，並在 Isaac Lab 改一行程式碼讓它遵守 `TMPDIR` |
| `isaaclab.sh` 需要 sudo；`flatdict` 無法 build | 改用 `pip install -e source/*`；用 `setuptools<70` 先單獨 build `flatdict` |
| GVHMR 報 `get_yaml_loader` 錯誤；節點上沒有 `ffmpeg` | 固定 `omegaconf==2.3.0`；改用 imageio-ffmpeg 附帶的靜態版 ffmpeg |
| GMR 的轉換腳本一定要開視窗 | 改用 `gvhmr_to_g1_headless.py` |
| MuJoCo 的 EGL 選到壞掉的 GPU | 用 `hope_training/slurm/pick_egl_device.py` 先挑一個能用的 |

### 錄參考影片

鏡頭固定在側前方約 45 度，畫面要拍到全身和球拍，30–60 fps，一次完整的正手從準備姿勢開始、揮完再回到準備姿勢，錄 5–10 次。之後照 [`slurm/video_to_g1/README.md`](../hope_training/slurm/video_to_g1/README.md) 的步驟轉換。SMPL/SMPL-X 的授權是個人的，**絕對不能 commit**（本機已加進 `.git/info/exclude`）。

---

## 9. 目前結果（2026-10-10）

* 環境：Isaac Sim 5.1 + Isaac Lab 0.54.2 可以在 2080 Ti 上啟動和訓練。G1 smoke 測試每一步都通過。Isaac 和 MuJoCo 的物理比對結果跟 `g1-port` 文件寫的一樣（0.023 rad、0.047 m、跌倒時間 1.34 s 對 1.40 s）。
* 影片流程：用 GVHMR 的範例影片驗證過（估計人身高 1.73 m、球拍最快 10 m/s）；轉換器往返測試誤差 0.012 rad / 1.6 mm。
* 訓練跑在 `rtx2080ti` 上，4096 個環境，兩個訓練共用一個節點時每個 iteration 約 8 秒：

  | Job | 任務設定 | 最後一次檢查時的狀態 |
  |---|---|---|
  | 4354 `g1_fh` | `_FH`（A3 題庫，baseline） | 第 602 個 iteration：擊球位置在 10 cm 以內的比例 7.8%，綜合指標 0，結束的回合有 39% 是跌倒 |
  | 4367 `g1_fh_xc` | `_FH_XC`（**主要**） | 第 240 個 iteration：還在早期，尚未開始擊球 |

* 過程中發現的 HOPE 上游問題：commit `00e2e96` 讓 `a3_pingpong.xml` 裡的 Gate 3 桌子會跟機器人碰撞，在獨立評估器裡會卡住球拍那隻手；評估器的預設幾何也跟 Gate 3 不一致（離桌邊 0.30 m，Gate 3 是 0.50 m）。`g1-port` 已經把評估器的幾何改成跟訓練一致。

---

## 10. 下一步

1. 錄正手影片，用 `video_to_g1` 轉換，檢查預覽影片。
2. 用真人動作 clip 訓練 `_FH_XC`（設定 `motion_file=...`，並更新 `strike_phase_per_clip`）。
3. 加入精簡版 HITTER 獎勵設定，跟 HOPE 的獎勵設定做 A/B 比較（D12）。
4. 寫 Isaac 評估腳本：用固定的題庫題目跑 `forehand_success`，回報成功率、最高點、擊中、過網、落點的比例和落點誤差。
5. 加入另外兩個落點（`center_line`、`opp_right_corner`），把目標落點也當成策略的輸入，並放寬餵球的範圍。
6. 上實機之前：量測實際的球拍轉接座、在我們自己的桌子上重新擬合球的物理參數、寫 G1 的部署 runner（現有的 C++ runner 只支援 A3）。
7. 接著處理平衡和步法（站位可以移動），然後是反手，最後是發球計畫。
