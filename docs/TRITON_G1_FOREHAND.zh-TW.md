# Triton Droids：Unitree G1 正手斜線對打

[English](TRITON_G1_FOREHAND.md) · 日常操作指令：[RUNBOOK_G1_FOREHAND.zh-TW.md](RUNBOOK_G1_FOREHAND.zh-TW.md)

這份文件記錄 Triton Droids 在 HOPE 和社團 G1 移植版上做了哪些事：每個決定的理由、整套系統從頭到尾怎麼運作，以及每個部分怎麼執行。每個 run 的數字和每次的狀態檢查記錄在 [`TRAINING_STATUS_G1_FOREHAND.zh-TW.md`](TRAINING_STATUS_G1_FOREHAND.zh-TW.md)；感知硬體和感知實驗在 [`PERCEPTION_G1_FOREHAND.zh-TW.md`](PERCEPTION_G1_FOREHAND.zh-TW.md)。
狀態為 **2026-10-12**：G1 實機還沒到，目前只在模擬中進行，但所有設計都保留了之後上實機（sim-to-real）的路徑。

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
| D7 | **斜對角對打**：餵球從**對手的右邊**打到機器人的右邊，機器人再從自己的右邊打回**對手的**右邊。共四個落點：一個在中間，三個沿著遠端底線排成一排。反彈的餵球區暫時固定。 | 社團規格，見第 3 節。餵球和回球走同一條斜線，比較接近真正的斜線對打。 |
| D8 | 訓練題目全部換成 G1 專用的**最高點擊球、斜線回球**題庫。 | 繼承來的 A3 題庫是在球往下掉時擊球（擊球瞬間垂直速度中位數 −0.77 m/s）、落點分佈在整個台面寬度、擊球位置是 A3 的（前方 0.58 m、高 0.85–1.30 m），而且只佔 25% 的題目。 |
| D9 | 參考動作改用**真人影片**製作（GVHMR + GMR），跟 HITTER 一樣。 | `g1-port` 的 clip 是直接把 A3 的關節角度套到 G1 上；擊球那一幀球拍只在約 0.78 m（桌面高度），第一版 G1 訓練也有搆不到的問題（x 少 9 cm、z 少 10 cm）。 |
| D10 | 不拿 A3 的 `model_21800` 來校準指標。 | 它在我們的環境裡打不到球；改用單元測試，給定「已知會成功」和「已知會失敗」的球拍狀態來驗證指標。 |
| D11 | 在 SDSC 上用 **pip** 安裝 Isaac，不用 64 GB 的容器。 | 叢集的 home 是 BeeGFS，podman 在上面跑不起來；pip 安裝不需要 sudo。節點上的問題都寫成腳本處理（第 8 節）。 |
| D12 | 照 HITTER 寫一個精簡版獎勵（上半身模仿、球拍目標、站位、平滑項），跟 HOPE 的 53 項獎勵做 A/B 比較。 | **尚未進行**：要先有影片做出來的 clip。 |
| D13 | **四個落點共用同一個策略，由規劃器決定打哪一個。** 落點是執行時可以切換的模式（可以選一個或多個），不是分開訓練的策略。 | 策略只負責追蹤球拍目標；規劃器（題庫）會把選定的落點換算成對應的球拍速度。共用一個策略可以讓不同落點共享身體技能，GPU 時間只要四分之一，而且換落點時不用切換控制器、動作不會突然改變。評估時可以按落點分別統計（`target_zone_id`），看哪個落點需要多練或微調。 |
| D14 | **用真實對打的時間節奏訓練，開一個獨立的 run（`g1_fh_rally`，task `HOPEPingPongUnitreeG1_FH_XC_RALLY`），從 `g1_fh_diag` 接續權重。** 最多 60% 的後續揮拍會晚開始（距離擊球 0.4–0.83 秒、沒有等待），前一拍的收拍在擊球後 0.7–1.0 秒就結束；虛擬球到達時間有 ±30 ms 誤差；來球更快（反彈前 4.4–8.5 m/s，原本 4.4–6.6）。三項都在 3000 個 iteration 內逐步加入。 | MuJoCo 真實對打節奏（`--rally-pace`）下，接觸率從 88% 掉到 38%，44% 的球會跌倒：訓練時有 0.83 秒預告、兩拍之間約 3 秒，真實對打只有 0.4–0.8 秒和約 1.5 秒。「晚開始」正是部署時規劃器給的時間比較短時，參考時鐘會做的事，所以不需要把動作做時間壓縮。在彈跳點固定於中間三分之一、擊球平面固定的情況下，較慢的來球物理上做不到（最高點會在機器人前面），需要往前移動的步伐才行。主要 run 繼續跑不動，作為對照。 |
| D15 | **同一個 run 被 requeue 後，依步數漸進的設定會接著走；從別的 run 接續權重時則從頭開始。** `train.sbatch` 只有在接續這個 run 自己最新的 checkpoint 時才會加上 `restore_step_counter=true`。 | 叢集停機和被搶佔都會讓 job requeue。`runner.load` 會還原權重和 optimizer，但 `env.common_step_counter` 會停在 0，所以對打節奏和時機誤差的漸進比例每次 requeue 都從頭開始。這個計數器本來就存在 checkpoint 的 exact-resume 狀態裡。 |
| D16 | **把球的旋轉當成必要的輸入，現在先不買攝影機。** 規劃用「對手擊球時的球拍」加上「球的彎曲軌跡」來估計旋轉。 | 模擬中，追蹤完美但不知道旋轉時，合法回球從 61% 掉到 10%；知道旋轉時，mocap 和 iPhone 等級分別是 56% 和 54%；粗略估計（±50%、±40°）仍有 48–49%。對我們的規劃器來說，攝影機好壞遠不如旋轉重要。細節見 [`PERCEPTION_G1_FOREHAND.zh-TW.md`](PERCEPTION_G1_FOREHAND.zh-TW.md)。 |
| D17 | **校正部署端的時鐘：策略看到的「距離擊球時間」要跟訓練時一致。** MuJoCo 裡 +60 ms 最合適；實機上要重新量。 | 球拍比時鐘早 53 ms 到達擊球點（20 ms 來自 `lifecycle.py` 在策略看到之前先減掉一個 tick，約 30 ms 還沒找到原因），所以每一球都約比最高點早 45 ms 擊球。加上 +60 ms 後，最高點擊球從 0 升到 91%，合法回球從 57% 升到 78%（`model_15400`）。其中 20 ms 已經在程式裡修好：`lifecycle.py` 的 `clock: strike_time`（2026-10-14 起是 G1 的預設；A3 維持 `countdown`）改用規劃器給的絕對擊球時刻，並有時鐘沒對好時的保護；剩下的部分是校正參數 `lifecycle.tts_offset_s`（G1：依 MuJoCo 設為 0.04 秒；上實機要重新量）。見狀態紀錄 2026-10-14。 |
| D18 | **系統性的瞄準誤差在規劃器修正（瞄準補償），不改球拍指令；補償值要用真的球拍量。** | 回球落點短 30–60 cm、而且往外偏。針對球拍速度偏差改指令反而更差（策略不會照訓練範圍外的指令做）；把規劃器的瞄準點往平均偏差的反方向移，落在目標區的比例變成三倍（MuJoCo 11.7% → 35%，對打節奏 34%）。補償值取決於球拍碰撞模型（兩種模型分別是 0.40 m 和 0.63 m）。見狀態紀錄 2026-10-14。 |

---

## 3. 任務規格（社團規則）

所有座標都是**環境座標系、機器人視角**：機器人根部在原點、面向 **+x**（朝桌子），**+y 是機器人的左邊**，z 朝上。近端桌邊在 x = 0.5 m，桌面高 z = 0.76 m，桌子以 y = 0 為中心，網子在 x = 1.87 m，遠端桌邊在 x = 3.24 m。

一球正手回球要同時滿足以下全部條件才算**成功**：

1. **時機**：在機器人這邊反彈後的**最高點**擊球，也就是擊球瞬間來球的垂直速度 |vz| ≤ 0.3 m/s，約等於最高點前後 30 ms。
2. **擊中**：球拍中心離球 9.5 cm 以內，而且正在以超過 0.3 m/s 的速度往球的方向移動（跟訓練時的判定一樣）。
3. **合法**：過網時球心要高於網子頂端加一個球半徑，而且第一次落地在對方半場。
4. **落點**：第一次落地在目標區內。

斜對角的方向（餵球和回球走同一條斜線）：

* **餵球**：從**對手的右邊**打過來（也就是機器人視角 +y 的那半邊，越過遠端底線時離桌面 5–45 cm），過網後在**機器人右邊**（−y）的餵球區反彈。
* **回球**：機器人從**自己的右邊**擊球，打到**對手的右邊**，也就是從機器人的角度看，遠端桌子的**左半邊**（+y）。

| 區域 | 深度（半場長度的比例，從網子算起） | 橫向（半場寬度的比例，往 +y 為正） | 環境座標範圍 |
|---|---|---|---|
| **餵球起點**（遠端底線） | x = 3.24 | +0.25 … +0.95（對手右邊） | y 0.191–0.724，z 0.81–1.21 |
| **餵球區**（來球在機器人半場的落點） | 中間三分之一 | −0.75 … −0.25（機器人右邊） | x 0.957–1.413，y −0.572…−0.191 |
| `opp_right_mid`：對手右半邊中間 | 中間三分之一 | +0.25 … +0.75 | x 2.327–2.783，y 0.191–0.572 |
| `center_line_deep`：中線旁邊（對手右邊那一側）、靠近底線 | 0.78 … 0.98 | +0.03 … +0.23 | x 2.939–3.213，y 0.023–0.175 |
| `opp_right_deep`：對手右半邊中間、靠近底線 | 0.78 … 0.98 | +0.25 … +0.75 | x 2.939–3.213，y 0.191–0.572 |
| `opp_right_corner`：底線和邊線交會的桌角 | 0.78 … 0.98 | 0.78 … 0.98 | x 2.939–3.213，y 0.595–0.747 |

三個靠邊的落點**沿著底線排成一排**：深度都在底線那一條邊緣帶（半場最後的 2–22%），由中線 → 右半邊中間 → 桌角。中線那一區整個在中線的對手右邊那一側：內側邊緣離中線 2.3 cm（大於一個球半徑），所以落在這一區的球不會碰到中線。「中間深度、靠近邊線」的落點（`opp_right_side`）先在 `TARGET_ZONES` 裡註解掉，留著以後用。四個區域互不重疊。訓練題庫平均分配四個落點（各 25%）。策略不需要被告知目標落點是哪一個，因為題庫已經算好要把球送到那裡的球拍速度，策略只要跟上這個速度就好。

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
* **部署端的參考時鐘（D17）**：每 20 ms 一個 tick，部署端都要告訴策略「距離擊球還有多久」。
  `countdown`（原本的）：拿規劃器給的 `time_to_strike`，同一個 tick 就先減 20 ms，之後每個 tick 再減 20 ms，直到下一則訊息。
  `strike_time`（可選）：記住規劃器給的絕對擊球時刻，顯示「擊球時刻 − 現在時間」。
  例子：10.000 秒送出一則訊息，說「還剩 0.500 秒」（擊球時刻 10.500 秒），訊息沒有延遲：

  | 牆上時間 | 真正還剩 | `countdown` 顯示 | `strike_time` 顯示 |
  |---|---:|---:|---:|
  | 10.000（訊息到達） | 0.500 | 0.480 | 0.500 |
  | 10.020 | 0.480 | 0.460 | 0.480 |
  | 10.040 | 0.460 | 0.440 | 0.460 |

  `countdown` 永遠少一個 tick，所以策略提早 20 ms 揮拍。如果訊息晚了 35 ms 才到，`countdown` 在 10.035 會顯示 0.480
  （真正是 0.465，多了 15 ms），`strike_time` 則顯示 0.465。只要規劃器和部署端的時鐘有同步，不管延遲多少它都是對的。
* **換球拍要不要重新訓練？** 策略學的是把球拍*中心點*（右手腕往外 0.21、0.015、0.01 m）在指定時間送到目標位置，帶著指定的速度和拍面方向；
  它不知道膠皮是什麼。所以：換膠皮，或換形狀、大小、重量差不多的球拍，**不用重新訓練**，只要幫規劃器換一個球拍模型
  （用錄下來的擊球擬合恢復係數和咬球參數），再重新量時機和瞄準補償。如果安裝位置、握柄長度或重量明顯不同，
  球拍中心的位置和手臂的動態都會變：要更新 `G1_MOUNT_OFFSET` 和產生的 URDF/MJCF（`unitree/scripts/build_g1_pingpang_models.py`），
  然後從目前的 checkpoint **微調**（接續權重，練幾千個 iteration），不用從頭練。
* **到處用同一套物理**：題庫產生器、成功判定、訓練獎勵、視覺化工具都呼叫同一個模組，所以彼此的結果不會互相矛盾。

---

## 5. 改了什麼（`g1-forehand-rally` branch）

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
| `utils/forehand_success.py` | 第 3 節的成功判定：`classify_strikes`、`ForehandSuccessTally`（分母包含沒打到的球）、`TARGET_ZONES`、`FEED_ZONE`、餵球起點。統計是一層一層的漏斗（接觸 → 過網 → 落在對方半場 → 落在目標區），`ZoneSuccessReport` 再按落點分開。不需要 Isaac 也能執行。 |

### 工具（`hope_training/whole_body_tracking/scripts/`）

| 檔案 | 用途 |
|---|---|
| `make_g1_forehand_bank.py` | 產生題庫（第 6 節）。`--vin-x/--vin-y/--vin-z/--spin-rev-s` 設定來球範圍。 |
| `evaluate_forehand.py`（+ `cfg/evaluate_forehand.yaml`） | 在 Isaac 裡按落點算社團成功率：確定性策略、512 個環境 × 60 秒，每一球的落點區由題目的目標落點決定。 |
| `mujoco_forehand_bank.py` | 把題庫的題目搬到 MuJoCo：遠端底線的餵球初始狀態、NumPy 版的 venue 飛行模型、閉迴路的最高點規劃器（真的反彈之後會重新解一次球拍速度）。 |
| `mujoco_eval_onnx.py`（擴充） | `--serve-bank`（用我們的題庫餵真的球）、`--zones`、`--feed-lead-s`、`--rally-pace` / `--rally-gap-s`（我們擊球後 0.7–1.0 秒對手就回下一球，球發出時才知道打哪裡）；每一球、每個落點都用社團規則計分。預設模式不變。 |
| `mujoco_pingpong_scene.py`（擴充） | 可選的 Magnus 力、畫出目標區、球的旋轉，以及 venue 的桌面反彈模型（MuJoCo 自己的反彈會讓最高點深 20–40 cm）。 |
| `check_rally_pace.py` | 在 Isaac 檢查對打節奏的時鐘和時機誤差（揮拍開始時間、兩拍間隔、跌倒）。 |
| `mujoco_eval_onnx.py`（新增） | `--racket-contact venue`（用訓練用的球拍模型取代 MuJoCo 的軟碰撞）、`--aim-offset DX DY`（D18）、`--trace-out`（逐幀紀錄，含拍面方向和球的旋轉）。 |
| `onnx_from_checkpoint.py` | 不用 Isaac 就能產生 checkpoint 的 policy.onnx：在 CPU 上把 actor 權重放進同一個任務已匯出的 ONNX 圖。驗證過跟真正的匯出完全相同。 |
| `ball_perception.py`、`perception_sweep.py` | `mujoco_eval_onnx.py --perception` 的攝影機模型（幀率、延遲、雜訊、漏幀、HITTER 式擬合、旋轉不知道／知道／有誤差的估計），以及跑遍各種攝影機系統的掃描腳本。`--reset-after-fall` 會在兩球之間把機器人扶起來。 |
| `view_feed_bank.py` | 在 MuJoCo 裡播放題庫的每一題：從對手右邊斜對角餵球 → 反彈 → 最高點 → 回球，同時畫出餵球起點、所有區域（目前的目標會亮起來）和球拍目標；可以開互動視窗或輸出 MP4，鏡頭可選 `--camera behind|top`。 |
| `gvhmr_to_g1_headless.py` | GMR 的「GVHMR → G1」轉換，但不開視窗（叢集節點沒有螢幕）。 |
| `gmr_video_clip_to_g1.py` | 把 GMR 的輸出轉成 HOPE 的 G1 clip：裁切、30 → 50 Hz、對齊朝向和位置、重排關節順序並限制在關節範圍內、把結尾混接回開頭讓 clip 可以循環、用 MuJoCo 算出 14 個身體部位的姿態、標記擊球幀（自動取球拍速度最快的一幀，或手動指定），可選擇輸出 MP4 預覽。 |

### 資料、腳本、測試

| 路徑 | 內容 |
|---|---|
| `assets/venue_tuple/g1_fh_crosscourt_apex_bank.{npz,receipt.json}` | 5 萬題的題庫（seed 0，可以重現，每個落點各 12,500 題）和 receipt，receipt 記錄了完整的產生規格和統計數字。 |
| `assets/venue_tuple/g1_fh_crosscourt_apex_bank_wide.{npz,receipt.json}` | `_FH_XC_RALLY` 用的快球題庫（第 6 節）。因為 `assets/venue_tuple/*` 在 gitignore 裡，所以是用 force-add 加進來的。 |
| `hope_training/slurm/pip_env/` | 不用容器的 SDSC 安裝、smoke 測試、訓練、通用執行腳本（[README](../hope_training/slurm/pip_env/README.md)）。`train.sbatch` 新增 `INIT_CHECKPOINT`（從別的 run 接續權重）、自動改用正常 GPU、`restore_step_counter`；另有 `status.sh`、`progress.sh`、`eval_checkpoint.sh`（MuJoCo 影片：範例球、題庫、對打節奏）、`eval_forehand.sh`（Isaac 分區評估）。用法見 [runbook](RUNBOOK_G1_FOREHAND.zh-TW.md)。 |
| `hope_training/slurm/video_to_g1/` | 「影片 → G1 clip」的安裝腳本和 Slurm 腳本（[README](../hope_training/slurm/video_to_g1/README.md)）。 |
| `tests/test_forehand_success.py` | 12 個測試：四個落點（三個沿底線排成一排、互不重疊）、餵球區和餵球起點、最高點擊球**確實可以**打進目標區（用網格搜尋證明）、往下掉的球／沒打到／太輕／打直線都判定失敗、沒打到的球有算進分母、漏斗統計、分區報告。 |
| `tests/test_make_g1_forehand_bank.py` | 4 個測試：符合載入器的格式；符合餵球區／最高點／擊球範圍／球拍速度的限制；餵球確實從對手右邊斜對角過來；每個落點題數相同，而且**每一題用它自己的落點檢查都是成功**。 |
| `tests/test_gmr_video_clip_to_g1.py` | 3 個測試：把現有的 G1 clip 偽裝成 GMR 輸出再轉回來（關節誤差 < 0.03 rad、身體位置誤差 < 1 cm、擊球幀誤差 ±1），sidecar 格式正確，結尾混接正確。 |
| `tests/test_mujoco_forehand_bank.py` | 3 個測試：MuJoCo 端的飛行模型和規劃器算出來的最高點、球拍速度跟題庫一致。 |
| `a3_deploy/.../reference/a3_deploy_onnx_ref_pingpong/{lifecycle,racket_command,ros_command_source,config,runner}.py` | 可選的 `clock: strike_time`（D17）：策略看到的距離擊球時間是 `strike_time − 現在時間`。預設 `countdown` 維持原本的行為。 |
| `tests/test_lifecycle_strike_time.py` | 5 個測試：countdown 不變、strike_time 時鐘對新訊息和晚到的訊息都準確、沒有絕對時刻時改用收到時間、沒有 `now` 時退回 countdown、ROS 第 [10] 欄。 |
| `tests/test_ball_perception.py` | 4 個測試：無雜訊的擬合和延遲、反彈切段、preset、每顆球固定的旋轉估計誤差。 |
| `docs/plans/SERVE_SIDE_BACKSPIN_PLAN.md` | 暫緩的發球計畫（D4）。 |

---

## 6. 題庫

`make_g1_forehand_bank.py` 每一題分四步產生：

1. **餵球**：在餵球區裡隨機選一個反彈點，隨機選反彈前的速度（vx −5.5…−2.5、vy −2.4…−0.3、vz −3.5…−1.5 m/s）和最多 5 轉/秒的旋轉。從反彈點**往回**模擬飛行，只保留越過遠端底線時在對手右邊（y 0.19–0.72 m、離桌面 5–45 cm）、而且過網高度至少 3 cm 的球，也就是從對手右邊斜對角餵過來的球。再用場地實測的桌面接觸模型算出反彈後的速度和旋轉。
2. **最高點**：用 RK4（每步 2 ms）模擬飛行，直到最高點（vz = 0）。只保留最高點落在 G1 正手擊球範圍內的球：站位前方 0.42–0.50 m、高度 0.80–1.10 m。站位的 y 由設定裡的正手球拍偏移（−0.38…−0.32 m）推算。
3. **回球**：先從四個落點中隨機選一個（記錄在 `target_zone_id`），再在裡面隨機選一個目標點。用阻尼最小平方法解出球拍速度（拍面法向跟速度方向相同，這是 V14 的規定），同時對齊「落點 x、落點 y、過網高度 − 0.25 m」三個目標，並輕微偏向 (1.9, 0.3, 1.0) m/s 附近的解（λ = 1e-3）。靠底線和邊線的落點需要又快又平的揮拍，偏向的力道太強會解不出來。
4. **篩選**：落點誤差 ≤ 5 cm、過網高度在 0.25 ± 0.10 m 內（且至少 3 cm）、球拍速度 ≤ 3.5 m/s，而且 `classify_strikes` 用這一題自己的落點判定為成功，才會保留。會一直產生到每個落點都達到配額為止，所以四個落點的題數一樣。

已 commit 的題庫統計（5 萬題，seed 0）：

| 項目 | 最小 | 平均 | p95 | 最大 |
|---|---|---|---|---|
| 球拍速度（m/s） | 1.15 | 2.73 | 3.38 | 3.50 |
| 最高點離桌面高度（m） | 0.13 | 0.24 | 0.32 | 0.34 |
| 擊球瞬間來球速度（m/s） | 1.99 | 3.03 | 3.51 | 3.78 |
| 過網高度（m） | 0.23 | 0.26 | 0.26 | 0.27 |
| 落點誤差（m） | 0.000 | 0.002 | 0.003 | 0.047 |

每個落點各 12,500 題，從 460 萬個候選餵球中篩出。大部分被淘汰的原因是最高點不在 G1 的擊球範圍內。靠底線和邊線的落點需要更快的揮拍，球拍速度最高會到 3.5 m/s 的上限。

重新產生（筆電 CPU 約 15 分鐘）：
`python3 hope_training/whole_body_tracking/scripts/make_g1_forehand_bank.py --rows 50000`。
換 seed 或改物理參數，題目內容就會改變，所以**重新產生之後要重新訓練**。

**快球題庫**（`g1_fh_crosscourt_apex_bank_wide`，`_FH_XC_RALLY` 使用）：步驟一樣，但反彈前速度改成
vx −7.5…−1.8、vy −3.0…−0.2、vz −4.5…−1.0 m/s，旋轉最多 8 轉/秒；從 700 萬個候選中篩出 5 萬題。
反彈前來球速度 4.4–8.5 m/s（平均 6.6），從遠端底線到最高點 0.38–0.85 秒（平均 0.51），擊球瞬間來球速度 2.0–5.1 m/s。
比約 4.4 m/s 更慢的球，沒辦法同時在中間三分之一反彈、又在固定的擊球範圍內到最高點，需要往前跨步才行。

```bash
python3 hope_training/whole_body_tracking/scripts/make_g1_forehand_bank.py --rows 50000 \
  --vin-x -7.5 -1.8 --vin-y -3.0 -0.2 --vin-z -4.5 -1.0 --spin-rev-s 8 \
  --out hope_training/whole_body_tracking/assets/venue_tuple/g1_fh_crosscourt_apex_bank_wide
```

---

## 7. 端到端流程

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

---

## 8. 執行方式

### 本機（macOS）

```bash
cd G1-PingPong
python3 -m venv .venv && .venv/bin/pip install -r a3_deploy/a3_deploy_example/reference/requirements.txt torch pytest imageio imageio-ffmpeg
.venv/bin/python -m pytest -q unitree/tests hope_training/whole_body_tracking/tests/test_forehand_success.py \
  hope_training/whole_body_tracking/tests/test_make_g1_forehand_bank.py hope_training/whole_body_tracking/tests/test_gmr_video_clip_to_g1.py \
  hope_training/whole_body_tracking/tests/test_mujoco_forehand_bank.py

# 看餵球的畫面。macOS 上開 MuJoCo 視窗要用 mjpython；因為這個路徑裡有空格，
# 要透過 venv 的 python 去執行 mjpython（它的 shebang 遇到空格會壞掉）。
.venv/bin/python .venv/bin/mjpython hope_training/whole_body_tracking/scripts/view_feed_bank.py --rows 30 --speed 0.5
.venv/bin/python .venv/bin/mjpython hope_training/whole_body_tracking/scripts/view_feed_bank.py --camera top
.venv/bin/python hope_training/whole_body_tracking/scripts/view_feed_bank.py --rows 8 --record artifacts/feed.mp4
```

畫面中：紫色是餵球起點（對手右邊、遠端底線），橘色是餵球的反彈區，淡綠色是四個落點，亮綠色是這一題的目標落點；黃色軌跡是球；紅色圓盤和箭頭是球拍目標（拍面和速度）。

### 叢集（SDSC Atlantis，`ssh sdsc-scc`）

* 帳號 `tritondroids`，全社團共用 **32 CPU / 96 GB / 8 GPU**。Isaac 需要 RTX 顯卡，所以要用 `--gres=rtx2080ti:1`（節點 `zixian`，11 GB，driver 610.43）。一個訓練 run 只用一張 GPU。
* `zixian` 的 8 張 GPU 其他帳號也能用（10-11 有一個 `scc` 的 job 占了 4 張），所以社團配額 8 張不代表有 8 張空著。QOS `normal`（上限 4 小時）的 job 可以搶佔 QOS `long`（訓練用）的 job；被搶佔時 `train.sbatch` 會 requeue，從最新的 checkpoint 接續（最多重跑 100 個 iteration）。
* 10-11 停機後，GPU `0000:61:00.0` 有一天讀不到，10-12 已經修好。`train.sbatch` 仍保留「CUDA 起不來就改用正常那張」的備援。
* 檔案位置：`~/hope/{HOPE（本 repo）, isaac-venv, IsaacLab, syslibs, mocap, slurm_logs}`。
* 所有 job 都要**在 repo 根目錄**送出。

```bash
sbatch hope_training/slurm/pip_env/install_isaac.sbatch                    # 只需一次
bash   hope_training/slurm/video_to_g1/install_mocap.sh                    # 只需一次（另外要放 SMPL/SMPL-X，見它的 README）
sbatch --export=ALL,TASK=HOPEPingPongUnitreeG1_FH_XC hope_training/slurm/pip_env/g1_smoke.sbatch
sbatch --export=ALL,TASK=HOPEPingPongUnitreeG1_FH_XC,RUN_NAME=g1_fh_diag hope_training/slurm/pip_env/train.sbatch
# 從別的 run 的 checkpoint 接續權重、開一個新的 run（TOTAL_ITERS 是總數）：
sbatch --export=ALL,TASK=HOPEPingPongUnitreeG1_FH_XC_RALLY,RUN_NAME=g1_fh_rally,TOTAL_ITERS=14600,INIT_CHECKPOINT=<.../model_2600.pt> \
  hope_training/slurm/pip_env/train.sbatch
bash hope_training/slurm/pip_env/status.sh            # 所有 run：iteration、主要指標、最新 checkpoint
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

## 9. 目前結果（2026-10-12）

完整的表格、訓練曲線和狀態紀錄在 [`TRAINING_STATUS_G1_FOREHAND.zh-TW.md`](TRAINING_STATUS_G1_FOREHAND.zh-TW.md)。摘要：

* 環境：Isaac Sim 5.1 + Isaac Lab 0.54.2 可以在 2080 Ti 上啟動和訓練。G1 smoke 測試每一步都通過。Isaac 和 MuJoCo 的物理比對結果跟 `g1-port` 文件寫的一樣（0.023 rad、0.047 m、跌倒時間 1.34 s 對 1.40 s）。
* 影片流程：用 GVHMR 的範例影片驗證過（估計人身高 1.73 m、球拍最快 10 m/s）；轉換器往返測試誤差 0.012 rad / 1.6 mm。我們自己的正手影片還沒錄。
* 訓練（4096 個環境，每個 iteration 約 8 秒）：

  | Run | 任務設定 | 最新（10-12 14:00） |
  |---|---|---|
  | `g1_fh_diag`（**主要**） | `_FH_XC` | 第 3814 / 20,000 個 iteration：球拍 ≤10 cm 89%、接觸 62%、合法回球 39%、跌倒 7.7% |
  | `g1_fh_rally` | `_FH_XC_RALLY`（D14），從 2600 接續 | 第 4713 / 14,600 個 iteration：球拍 ≤10 cm 92%、接觸 64%、合法回球 39%、跌倒 3.6%、對打節奏佔 26% 的揮拍、時機誤差 21 ms |
  | `g1_fh`（baseline） | `_FH`（A3 題庫） | 停在 2196 |

* Isaac 社團成功率（`evaluate_forehand.py`，8100 球）：`g1_fh_diag` `model_2300` **11.1%**（`opp_right_mid` 24.5%、`center_line_deep` 6.2%、`opp_right_deep` 11.3%、桌角 2.3%；接觸 92%、過網 91%、合法 79%）。baseline `g1_fh` `model_2100` 只有 4.6%。
* MuJoCo 用題庫餵真的球：`model_2400` 12 球接觸到 10 球，但比最高點早 15–40 ms 擊球，還沒有球落在目標區。用真實對打節奏（`--rally-pace`）時，接觸率從 88% 掉到 38%，44% 的球會跌倒，這就是開 `g1_fh_rally` 的原因。
* 過程中發現的 HOPE 上游問題：commit `00e2e96` 讓 `a3_pingpong.xml` 裡的 Gate 3 桌子會跟機器人碰撞，在獨立評估器裡會卡住球拍那隻手；評估器的預設幾何也跟 Gate 3 不一致（離桌邊 0.30 m，Gate 3 是 0.50 m）。`g1-port` 已經把評估器的幾何改成跟訓練一致。

---

## 10. 下一步

1. 錄正手影片，用 `video_to_g1` 轉換，檢查預覽影片。
2. 用真人動作 clip 訓練 `_FH_XC`（設定 `motion_file=...`，並更新 `strike_phase_per_clip`）。擊球前的動作短一點，也能改善預告時間不夠的問題。
3. 加入精簡版 HITTER 獎勵設定，跟 HOPE 的獎勵設定做 A/B 比較（D12）。
4. ~~寫 Isaac 評估腳本，用 `forehand_success` 分區統計~~：已完成（2026-10-11），`scripts/evaluate_forehand.py` / `eval_forehand.sh`。
5. 落點選擇（D13）：在題庫加上 `target_zone_ids` 篩選，讓訓練、評估、播放都能指定落點，部署時的規劃器也提供同樣的選擇。（放寬餵球範圍：已經用快球題庫完成。）
6. 縮小 MuJoCo 題庫評估發現的真球落差。**擊球時機：已找到原因，並在模擬中修正（D17）。** 接下來：還沒解釋的約 30 ms（Isaac 和 MuJoCo 重播比對），以及瞄準（只有 3–10% 落在目標區）。已完成（D14）：虛擬擊球的時機誤差、對打節奏下晚開始揮拍、更快的來球。還沒做：拿 MuJoCo 檢查球拍碰撞模型（MuJoCo 回球比較低）；等漸進比例加滿（約第 5600 個 iteration）後，用 `eval_rally` / `eval_bank` / Isaac 分區評估比較 `g1_fh_rally` 和 `g1_fh_diag`。
7. 旋轉估計（D16）：從對手球拍判斷，加上飛行軌跡的 Magnus 擬合；訓練時加入旋轉誤差。感知實驗得出的硬體目標：≥120 Hz、≤15–30 ms、≤5 mm，多個同步視角涵蓋球桌和對手球拍。
8. **瞄準精度（D18）：** 剩下的偏差是分散（約 20 cm），要靠訓練出更精準的揮拍（例如提高擊球瞬間球拍速度的權重），以及真實球拍的模型和瞄準補償值。
   **待辦：** 測試 `strike_time` 時鐘對原本的 A3 是否也有改善（A3 策略在它自己的 MuJoCo 評估裡比較 `countdown` 和 `strike_time`），並找出剩下約 40 ms 的原因（Isaac 和 MuJoCo 重播比對）。
9. 上實機之前：量測實際的球拍轉接座、在我們自己的桌子上重新擬合球的物理參數、寫 G1 的部署 runner（現有的 C++ runner 只支援 A3）。
10. 接著處理平衡和步法（站位可以移動，比約 4.4 m/s 更慢的球也需要），然後是反手，最後是發球計畫。
