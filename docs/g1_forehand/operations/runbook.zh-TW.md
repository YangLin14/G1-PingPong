# 操作手冊：G1 正手訓練（Triton Droids）

[文件目錄](../README.zh-TW.md)

[English](runbook.md) · 背景與設計說明：[文件目錄](../README.zh-TW.md)

日常操作用的指令，可以直接複製貼上。這裡的每個指令都在 2026-10-10 實際跑過。範例裡的 job 編號（4382、4354）是當時正在跑的訓練，請換成 `status.sh` 顯示的編號。在 shell 裡不要輸入尖括號：bash 會把 `<x>` 當成「從檔案 x 讀取輸入」。

* **筆電**：`G1-PingPong` 資料夾（macOS），指令都從 repo 根目錄開始執行。
* **叢集**：`ssh sdsc-scc`，repo 在 `~/hope/HOPE`，叢集上的指令都先 `cd ~/hope/HOPE`。

---

## 0. 目前在跑什麼

| Run 名稱 | 任務設定 | 用途 |
|---|---|---|
| `g1_fh_diag` | `HOPEPingPongUnitreeG1_FH_XC` | **主要訓練**。G1 正手：從對手右邊斜對角餵球、最高點擊球、四個落點。 |
| `g1_fh_rally` | `HOPEPingPongUnitreeG1_FH_XC_RALLY` | 同一個任務，用真實對打的節奏訓練（揮拍晚開始、收拍時間短、±30 ms 時機誤差、更快的來球），從 `g1_fh_diag` 接續（決策 D14）。Job 5560。 |
| `g1_fh` | `HOPEPingPongUnitreeG1_FH` | Baseline。一樣是 G1 正手，但用的是 HOPE 原本照 A3 幾何設計的題目（在球往下掉時擊球、落點分佈在整個台面），只拿來比較。 |

兩個都是 **G1**，都不是在訓練 A3。

---

## 1. 查看訓練狀態

```bash
ssh sdsc-scc
cd ~/hope/HOPE && bash hope_training/slurm/pip_env/status.sh          # 正在跑的 job
bash hope_training/slurm/pip_env/status.sh --all                       # 加上最近 3 天內結束的
bash hope_training/slurm/pip_env/status.sh 4382                        # 只看某一個 job
bash hope_training/slurm/pip_env/progress.sh 4382                      # 每 500 個 iteration 的指標（markdown 表格）
```

會列出你的 job、社團配額的使用量，以及每個訓練目前的 iteration、主要指標和最新的 checkpoint。指標的意思：

| 欄位 | 意思 | 正常的變化方向 |
|---|---|---|
| mean reward | 每一步的總獎勵 | 上升，會有波動 |
| racket within 10 cm at strike | 擊球那一幀，球拍有沒有到達目標點 10 cm 以內 | 最先上升（幾百到幾千個 iteration） |
| pos+vel+normal all OK at strike | 位置、速度、拍面方向三者都在容許範圍內 | 在上一項之後上升 |
| virtual ball: contact rate | 這一拍真的會打到球 | 跟上面兩項一起上升 |
| virtual ball: legal return rate | 過網而且落在對方半場 | 真正的目標，最後才上升 |
| virtual ball: landing error (m) | 落點和目標點的距離 | 下降 |
| falls (share of episode ends) | 回合因為跌倒而結束的比例 | 一開始接近 1.0，之後應該降到 0.5 以下很多 |
| seconds per iteration | 每個 iteration 花的秒數 | 單獨跑約 5.3 秒，兩個訓練共用一個節點時約 8 秒 |

一次訓練 20,000 個 iteration，大約 30–45 小時。原始 log 裡的 `success_rate` 不用看，HOPE 的那個公式算出來的球永遠過不了網。

原始 log 和 TensorBoard：

```bash
tail -f ~/hope/slurm_logs/hope-g1-fh-diag-4382.out    # 數字是 job 編號
# 在筆電上執行，然後打開 http://localhost:6006
ssh -L 6006:localhost:6006 sdsc-scc '~/hope/isaac-venv/bin/tensorboard --port 6006 \
  --logdir ~/hope/HOPE/hope_training/whole_body_tracking/logs/rsl_rl/unitree_g1_hitter_pingpong'
```

---

## 2. 看某個 checkpoint 的表現（匯出 → MuJoCo 影片）

```bash
cd ~/hope/HOPE
# 在這個 run 自己的 job 裡執行（用 run 名稱自動找到 job）：馬上開始，不用另外占配額（約 1 分鐘）
bash hope_training/slurm/pip_env/eval_checkpoint.sh --in-job auto g1_fh_diag
# 或自己指定 job 編號（從 status.sh / squeue 看；直接打數字，不要打 <...>）
bash hope_training/slurm/pip_env/eval_checkpoint.sh --in-job 4382 g1_fh_diag
# 或者另外送一個小 job（8 GB），例如訓練已經結束的時候
sbatch hope_training/slurm/pip_env/eval_checkpoint.sh g1_fh_diag
```

它會把這個 run 最新的 `model_*.pt` 匯出成 ONNX（也可以在第二個參數指定某個 checkpoint），確認 ONNX 跟 PyTorch 模型的輸出一致，然後在 MuJoCo 跑兩組，結果都存在 `~/hope/evals/<run>/<model_N>/`：

* **`eval_bank.mp4` / `eval_bank.json`（看這個）**：12 球都來自**我們的題庫**，用一顆真的球：斜對角發球、帶旋轉、在最高點擊球。
  4 個目標區畫在對面台上，這一球的目標區會亮起來。json 裡有每一球的接觸、時機（接觸時的 vz）、落點，以及分區成功率；
* `eval.mp4` / `eval.json`：評估器原本的範例來球（正反手、固定落點），不是我們的任務，只用來對照。

可調整的參數：`BANK_SERVES=24`（球數）、`ZONES="opp_right_mid opp_right_corner"`（只打某幾區）、
`FEED_LEAD_S=0`（只有在球發出時才告訴策略，也就是真機部署時的情況；預設是 0.9 秒，和訓練一樣），例如：
`BANK_SERVES=24 ZONES=opp_right_corner bash hope_training/slurm/pip_env/eval_checkpoint.sh --in-job auto g1_fh_diag`

複製到筆電上觀看：

```bash
mkdir -p artifacts/evals
scp -r sdsc-scc:hope/evals/g1_fh_diag/model_2000 artifacts/evals/g1_fh_diag_model_2000
open artifacts/evals/g1_fh_diag_model_2000/eval_bank.mp4
```

### 2b. 各目標區的社團成功率（Isaac）

```bash
cd ~/hope/HOPE
bash hope_training/slurm/pip_env/eval_forehand.sh --in-job auto g1_fh_diag       # 最新 checkpoint，約 20 分鐘
bash hope_training/slurm/pip_env/eval_forehand.sh --in-job 4382 g1_fh            # 評估別的 run，借用 4382 的 GPU
sbatch hope_training/slurm/pip_env/eval_forehand.sh g1_fh_diag                   # 自己開一個 job
```

用我們的成功定義（最高點擊球、接觸、過網、落在對方半場、落在該題的目標區）幫題庫的每一題打分數，
每個目標區印一列，再加一列總計。輸出在 `~/hope/evals/<run>/<model_N>/forehand_zones.{json,txt}`。
結果請記到 [`status/runs_and_results.zh-TW.md`](../status/runs_and_results.zh-TW.md)（各區成功率表）。

---

## 3. 在筆電上開 MuJoCo

第一次要先設定：

```bash
python3 -m venv .venv
.venv/bin/pip install -r a3_deploy/a3_deploy_example/reference/requirements.txt torch pytest imageio imageio-ffmpeg
```

macOS 上開 MuJoCo 視窗要用 `mjpython`。這個 repo 的路徑有空格（"Triton Droids"），會讓 `mjpython` 的 shebang 失效，所以**一律透過 venv 的 python 執行**：`.venv/bin/python .venv/bin/mjpython <script>`。

| 要看什麼 | 指令 |
|---|---|
| 餵球（訓練題目），俯視 | `.venv/bin/python .venv/bin/mjpython hope_training/whole_body_tracking/scripts/view_feed_bank.py --camera top` |
| 餵球，從機器人後方看 | `... view_feed_bank.py --camera behind --rows 30 --speed 0.5` |
| 餵球輸出成 MP4（不開視窗） | `.venv/bin/python hope_training/whole_body_tracking/scripts/view_feed_bank.py --rows 8 --record artifacts/feed.mp4` |
| 訓練好的 G1 策略揮拍 | `PYTHONPATH=a3_deploy/a3_deploy_example/reference .venv/bin/python .venv/bin/mjpython -m a3_deploy_onnx_ref_pingpong --robot unitree_g1 --onnx artifacts/evals/<dir>/policy.onnx --view --realtime` |
| 訓練好的 G1 策略打真的球 | `.venv/bin/python .venv/bin/mjpython hope_training/whole_body_tracking/scripts/mujoco_eval_onnx.py --robot unitree_g1 --onnx artifacts/evals/<dir>/policy.onnx --num-serves 12 --view` |

餵球畫面的顏色：紫色 = 餵球起點（對手右邊），橘色 = 餵球反彈區，淡綠色 = 四個落點，亮綠色 = 這一題的目標落點，黃色 = 球的軌跡，紅色圓盤和箭頭 = 球拍目標（拍面和速度）。

---

## 4. 開始、停止、接續訓練

都在叢集上 `cd ~/hope/HOPE` 之後執行（job 一定要從 repo 根目錄送出）。

```bash
# 開始（或接續：同一個 RUN_NAME 會從最新的 checkpoint 接著練，總數維持 20,000）
sbatch --output=$HOME/hope/slurm_logs/%x-%j.out --job-name=hope-g1-fh-diag \
  --export=ALL,TASK=HOPEPingPongUnitreeG1_FH_XC,RUN_NAME=g1_fh_diag hope_training/slurm/pip_env/train.sbatch
scancel <jobid>                    # 停止（最新的 checkpoint 會保留，每 100 個 iteration 存一次）
squeue -u $USER                    # 列出 job
```

* **從頭開始**：換一個新的 `RUN_NAME`，或先刪掉 `hope_training/whole_body_tracking/logs/rsl_rl/unitree_g1_hitter_pingpong/*_<run>`。
* 可調整的參數：`TASK`、`RUN_NAME`、`TOTAL_ITERS`（預設 20000）、`NUM_ENVS`（預設 4096）。
* job 跑在 `long` partition 並設了 `--requeue`：節點被搶走的話會自動重新排隊，並接著練。
* 正在跑的 job 用的是它開始時的程式碼。拉了新程式碼之後，要重新啟動 job 才會套用。
* 長時間訓練新的任務設定之前，先跑 smoke 測試：
  `sbatch --export=ALL,TASK=<task> hope_training/slurm/pip_env/g1_smoke.sbatch`。

---

**握拳 run（`g1_fh_fist`，D20）：** 評估時一定要用握拳模型，不然球拍位置是錯的：在 `mujoco_eval_onnx.py` 加上
`--model-xml unitree/URDF/g1_29dof_pingpang_fist/mjcf/g1_29dof_pingpang_fist.xml`。`onnx_from_checkpoint.py` 不用改（網路一樣）。

## 4b. 在 MuJoCo 做平衡測試（筆電）

```bash
.venv/bin/python hope_training/whole_body_tracking/scripts/mujoco_eval_onnx.py --robot unitree_g1 \
  --onnx artifacts/evals/<dir>/policy.onnx \
  --serve-bank hope_training/whole_body_tracking/assets/venue_tuple/g1_fh_crosscourt_apex_bank.npz \
  --eval-mode continuous --reset-after-fall --num-serves 40 --seed 0 \
  --push 0.5 0.5 --json-out /tmp/push.json        # 每球推一次，最多 0.5 m/s 和 0.5 rad/s
```
把 `fall_rate` 和社團數字跟狀態紀錄（2026-10-14 17:00）的表格比較；用 seed 0、1、2，推擠強度 0.25 / 0.5 / 0.75 / 1.0。

* **要避開的 GPU：idx 0 和 5**（`AVOID_GPU_IDX`，預設 `0,5`）。Slurm 會給編號最小的空 GPU，所以要用
  `--gres=rtx2080ti:2` 送出：job 會把壞的那張對 CUDA 隱藏，用另一張跑（log 會有 `[train] using GPU index N`）。
  用 `:1` 的話，拿到 idx 0 就會停止。
* **平衡優先的訓練（D21），等 zixian 的 CUDA 恢復後：** 先跑 smoke，再兩個都從 `g1_fh_fist` `model_32299` 開始：
  ```bash
  INIT=$(ls hope_training/whole_body_tracking/logs/rsl_rl/unitree_g1_hitter_pingpong/*_g1_fh_fist/model_32299.pt)
  sbatch --gres=rtx2080ti:2 --export=ALL,TASK=HOPEPingPongUnitreeG1_FH_FIST_STABLE_SWING hope_training/slurm/pip_env/g1_smoke.sbatch
  for v in STABLE:g1_fh_stable STABLE_SWING:g1_fh_stable_swing; do
    sbatch --gres=rtx2080ti:2 --job-name=hope-${v#*:} --output=$HOME/hope/slurm_logs/%x-%j.out \
      --export=ALL,TASK=HOPEPingPongUnitreeG1_FH_FIST_${v%%:*},RUN_NAME=${v#*:},TOTAL_ITERS=40300,INIT_CHECKPOINT=$INIT \
      hope_training/slurm/pip_env/train.sbatch
  done
  ```
  啟動 log 裡要看到 `rewards.root_height_ceiling.weight = -1.0`、`rewards.com_support.weight = -2.0`、
  `rewards.upright.weight = -3.0` 和變大的 `events.push_robot`。訓練中檢查 `root_height` 有沒有降到約 0.68 m 以下。

*（已被 `g1_fh_fist` 取代，留著參考）* **平衡訓練 `g1_fh_bal`（D19），等 zixian 恢復後：** 先試跑，再從最新的 `g1_fh_diag` checkpoint 接著訓練（`<N>` 換成實際數字）：
```bash
sbatch --export=ALL,TASK=HOPEPingPongUnitreeG1_FH_XC_RALLY_BAL hope_training/slurm/pip_env/g1_smoke.sbatch
sbatch --gres=rtx2080ti:1 --job-name=hope-g1-fh-bal --output=$HOME/hope/slurm_logs/%x-%j.out \
  --export=ALL,TASK=HOPEPingPongUnitreeG1_FH_XC_RALLY_BAL,RUN_NAME=g1_fh_bal,TOTAL_ITERS=<N+12000>,INIT_CHECKPOINT=$(ls -t hope_training/whole_body_tracking/logs/rsl_rl/unitree_g1_hitter_pingpong/*_g1_fh_diag/model_<N>.pt | head -1) \
  hope_training/slurm/pip_env/train.sbatch
```
啟動 log 裡一定要看到 `events.push_robot = ramped push ...`。

---

## 5. 改程式碼，然後更新叢集

```bash
# 筆電
.venv/bin/python -m pytest -q hope_training/whole_body_tracking/tests/test_forehand_success.py \
  hope_training/whole_body_tracking/tests/test_make_g1_forehand_bank.py \
  hope_training/whole_body_tracking/tests/test_gmr_video_clip_to_g1.py unitree/tests
git commit ... && git push origin g1-forehand-rally
# 叢集
cd ~/hope/HOPE && git pull --ff-only origin g1-forehand-rally
# 然後重新啟動受影響的 job（第 4 節）
```

---

## 6. 改落點或餵球（重新產生題庫）

1. 修改 `hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/utils/forehand_success.py` 裡的 `TARGET_ZONES`、`FEED_ZONE` 或 `FEED_ORIGIN_*`（餵球速度和落點清單在 `scripts/make_g1_forehand_bank.py` 的 `BankCfg`）。
2. 重新產生（筆電約 15 分鐘）：
   `.venv/bin/python hope_training/whole_body_tracking/scripts/make_g1_forehand_bank.py --rows 50000`
3. 用第 3 節的餵球畫面看一下，並跑測試（第 5 節）。
4. commit 題庫（`git add -f hope_training/whole_body_tracking/assets/venue_tuple/g1_fh_crosscourt_apex_bank.*`），push，叢集上 pull，然後**從頭開始**重新訓練（因為題目變了）。

---

## 7. 手機影片 → G1 參考動作 clip

錄影：鏡頭固定在側前方約 45 度，拍到全身和球拍，30–60 fps，從準備姿勢 → 正手 → 回到準備姿勢，錄 5–10 次。

```bash
scp forehand.mp4 sdsc-scc:hope/mocap/videos/
ssh sdsc-scc
cd ~/hope/HOPE && sbatch --export=ALL,VIDEO=$HOME/hope/mocap/videos/forehand.mp4,NAME=g1vid_forehand \
  hope_training/slurm/video_to_g1/video_to_g1.sbatch
# 結果：~/hope/mocap/clips/g1vid_forehand/{g1vid_forehand.npz,.yaml,.mp4}（紅框是擊球那一幀）
```

可以用 `START=`/`END=` 裁切（原始影片的幀數，30 Hz），用 `STRIKE=` 手動指定擊球幀（輸出 clip 的幀數，50 Hz），然後重新送出。詳細說明見 [`slurm/video_to_g1/README.md`](../../../hope_training/slurm/video_to_g1/README.md)。

---

## 8. 疑難排解

| 狀況 | 解法 |
|---|---|
| job 卡在 `PENDING (AssocGrpCpuLimit)` / `(AssocGrpMemLimit)` | 社團的 32 CPU / 96 GB 配額滿了（`status.sh` 可以看到誰在用）。等一下、停掉 baseline，或者少要一點資源（`--mem`、`--cpus-per-task`；訓練實際約用 8 GB RAM、5 GB GPU 記憶體）。 |
| 想快速評估，但配額滿了 | 用 `eval_checkpoint.sh --in-job auto <run 名稱>`，在這個 run 現有 job 的資源裡執行。 |
| `status.sh` 顯示 "errors in the log" | `grep -n -A30 Traceback <log>`。Isaac 遇到 Python 錯誤後常常不會自己結束，要用 `scancel` 停掉。 |
| Kit 報 `No device could be created`、匯入 URDF 後卡住、`/tmp` 沒有權限 | 節點的修正設定不見了：重新執行 `sbatch hope_training/slurm/pip_env/install_isaac.sbatch`（它會重新產生 `setup_train_env.local.sh`）。 |
| Mac 上 MuJoCo 視窗開不起來 | 改用 `.venv/bin/python .venv/bin/mjpython ...`（路徑有空格的問題）；`--record out.mp4` 不需要開視窗。 |
| 叢集上 `unable to open EGL` | 設定 `MUJOCO_EGL_DEVICE_ID=$(MUJOCO_GL=egl ~/hope/isaac-venv/bin/python hope_training/slurm/pick_egl_device.py)`。 |
| 找 job 編號 | `sbatch` 會印出編號；log 在 `~/hope/slurm_logs/<job 名稱>-<編號>.out`。 |

---

## 9. 檔案位置

| 內容 | 位置 |
|---|---|
| 訓練 log（stdout） | 叢集 `~/hope/slurm_logs/` |
| Checkpoint、TensorBoard 紀錄 | 叢集 `~/hope/HOPE/hope_training/whole_body_tracking/logs/rsl_rl/unitree_g1_hitter_pingpong/<日期>_<run>/` |
| 評估結果 | 叢集 `~/hope/evals/<run>/<model_N>/` |
| 手機影片轉出來的影片和 clip | 叢集 `~/hope/mocap/clips/<名稱>/` |
| Isaac venv、Isaac Lab、libGLU | 叢集 `~/hope/{isaac-venv, IsaacLab, syslibs}` |
| 只存在筆電上的輸出（git 忽略） | `artifacts/` |
| 任務設定 | `hope_training/whole_body_tracking/cfg/task/HOPEPingPongUnitreeG1_FH*.yaml` |
| 題庫 | `hope_training/whole_body_tracking/assets/venue_tuple/g1_fh_crosscourt_apex_bank.*` |
| 落點和成功判定 | `.../whole_body_tracking/utils/forehand_success.py` |
| 叢集 job 腳本 | `hope_training/slurm/pip_env/`、`hope_training/slurm/video_to_g1/` |


## 10. 在 Antioch 上訓練（目前用這個，2026-10-19 起）

完整指令和注意事項：[antioch_sim/README.md](../../../antioch_sim/README.md)。簡單說，在 `antioch_sim/` 裡：
`session new --idle-timeout 1d` → `./push_ignored.sh`（再用 `service cp` 傳接續用的 checkpoint）→
`service exec --no-stream -- env TASK=… RUN_NAME=… TOTAL_ITERS=… bash /workspace/project/hope_training/antioch/train.sh`
→ 看 `logs/antioch/<RUN_NAME>.log` → 用 `service cp` 把 checkpoint 拿回來 → `session release`。
釋放 session 時上面的檔案會消失。上面 Slurm 的章節留著參考。
