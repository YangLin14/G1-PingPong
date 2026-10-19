# 在 Antioch 上訓練 G1（雲端 GPU）

[文件目錄](../README.zh-TW.md)

2026-10-19 起，G1 正手的訓練從 SDSC 叢集（zixian 的 GPU 一直壞）改到 Antioch。
這份文件記錄：怎麼設定、怎麼用、測試結果，以及設定過程中遇到的每個問題和解法。

---

## 1. 結論先講

* **可以用了。** 用 `model_40299` 接著訓練 20 圈，跑得正常，D21 的平衡獎勵項目也有作用。
* **版本跟叢集完全一樣**：Isaac Sim 5.1、Isaac Lab v2.3.2、torch 2.7.0+cu128、rsl-rl 3.1.2。所以叢集上的 checkpoint 可以直接接著練。
* **GPU：** 1 張 NVIDIA RTX PRO 6000 Blackwell（96 GB），24 個 CPU 核心，214 GB 記憶體，機房在 us-central。
* **速度沒有變快（4096 個環境時）：** 每圈約 6.0 秒，跟叢集的 2080 Ti 一樣。原因在第 5 節。
  開 16384 個環境的話，每秒的樣本量是 3–4 倍，但會改變 PPO 的 batch，**還沒決定要不要用**。
* **session 釋放時，上面的檔案全部消失。** 訓練完一定要先把 checkpoint 複製回來。

---

## 2. 為什麼不用 Antioch 內建的 Isaac Lab

Antioch 內建三種引擎：`isaac-sim-6.0.1`、`isaac-lab-3.0`、`base`。

* 我們的 HOPE 程式是寫給 **Isaac Lab 2.3.2 / Isaac Sim 5.1** 的。
* Isaac Lab 3.0 有不相容的改動，例如四元數改成 XYZW 順序。HOPE 光 `hope_commands.py` 就有一萬多行，移植的工作量很大。
  而且物理引擎版本不同，舊的 checkpoint 行為也可能改變。
* 所以選 **`base` 引擎，加上我們自己的 Dockerfile**，在上面裝跟叢集一模一樣的版本。

---

## 3. 檔案在哪裡

| 檔案 | 作用 |
|---|---|
| `antioch_sim/antioch.yaml` | Antioch 專案設定：一個服務 `sim`、GPU `rtx-pro-6000`、把整個 repo 同步到 `/workspace/project` |
| `antioch_sim/Dockerfile` | 映像檔：Ubuntu 24.04 base + 系統函式庫 + Python 3.11 venv + Isaac Sim 5.1 + Isaac Lab v2.3.2 + HOPE 需要的套件 |
| `antioch_sim/cluster-constraints.txt` | 叢集 venv 的 `pip freeze`，安裝時當作版本限制，確保版本跟叢集一樣 |
| `antioch_sim/.dockerignore` | 只讓 `cluster-constraints.txt` 進入映像檔建置 |
| `antioch_sim/push_ignored.sh` | 把「已 commit 但被 git 忽略」的訓練檔案（題庫）複製到 session 上（見問題 6） |
| `antioch_sim/README.md` | 英文的快速指令清單 |
| `antioch_sim/.venv/` | 本機的 Antioch SDK 0.5.338（不 commit） |
| `hope_training/antioch/train.sh` | 訓練啟動腳本：等於拿掉 Slurm 的 `train.sbatch`，加上卡住看門狗 |

同步規則（`antioch.yaml` 的 `sync`）：
* 來源是 repo 根目錄（`..`），目標是 `/workspace/project`。
* 會跳過：`.gitignore` 裡的檔案、`.git`、虛擬環境、快取，以及 `agibot/**`、`antioch_sim/**`、`artifacts/**`、`MUJOCO_LOG.TXT`。
* 已確認 **SMPL 模型、`artifacts/`、`.git` 都沒有上傳**（在 session 上找過，0 個 SMPL 檔）。
* 在 session 上**新產生的檔案**（log、checkpoint）同步時不會被動到。

---

## 4. 怎麼用（一步一步）

所有 `antioch` 指令都要在 `antioch_sim/` 資料夾裡執行（它要找 `antioch.yaml`）。

```bash
cd antioch_sim
A=.venv/bin/antioch

# 1. 開 session（第一次會建置映像檔，之後沒改 Dockerfile 就直接用快取）
#    idle-timeout 要涵蓋整個訓練時間；閒置超過這個時間 session 會自動釋放
$A session new --idle-timeout 1d

# 2. 把被 git 忽略的題庫檔傳上去（每開一個新 session 都要做）
./push_ignored.sh
#    要接著練的話，把 checkpoint 也傳上去
$A service cp ../artifacts/evals/g1_fh_stable_swing_model_40299/model_40299.pt \
  sim:/workspace/project/hope_training/whole_body_tracking/checkpoints_in/model_40299.pt

# 3. 開始訓練（在背景跑，關掉終端機也不會停）
#    路徑一律用絕對路徑（見問題 4）
$A service exec --no-stream -- env \
  TASK=HOPEPingPongUnitreeG1_FH_FIST_STABLE_SWING RUN_NAME=my_run TOTAL_ITERS=48300 \
  INIT_CHECKPOINT=/workspace/project/hope_training/whole_body_tracking/checkpoints_in/model_40299.pt \
  bash /workspace/project/hope_training/antioch/train.sh

# 4. 看進度
$A service exec --no-stream -- tail -n 40 \
  /workspace/project/hope_training/whole_body_tracking/logs/antioch/my_run.log
$A service exec --no-stream -- bash -c \
  'ls /workspace/project/hope_training/whole_body_tracking/logs/rsl_rl/*/*_my_run | tail'

# 5. 把 checkpoint 拿回來（一定要在釋放之前！）
$A service cp sim:/workspace/project/hope_training/whole_body_tracking/logs/rsl_rl/unitree_g1_hitter_pingpong/<資料夾>/model_N.pt ./

# 6. 確認沒有殘留的訓練程式，再釋放
$A service exec --no-stream -- bash -c 'pgrep -fa scripts/train.py || echo none'
$A session release
```

`train.sh` 的參數（用 `env` 傳）：

| 參數 | 預設 | 說明 |
|---|---|---|
| `TASK` | 必填 | `cfg/task/*.yaml` 的名稱 |
| `RUN_NAME` | 必填 | run 名稱；如果已經有這個 run 的 checkpoint，會自動從最新的接著跑 |
| `TOTAL_ITERS` | 20000 | 總圈數（絕對值，不是「再跑幾圈」） |
| `NUM_ENVS` | 4096 | 跟叢集一樣，PPO 的 batch 才不會變 |
| `INIT_CHECKPOINT` | 無 | 新 run 要從別的 run 的 checkpoint 開始時用 |
| `EXTRA` | 無 | 額外傳給 `train.py` 的參數 |
| `FOREGROUND` | 無 | 設成 1 就在前景跑（測試用），否則在背景跑 |
| `HANG_GRACE_S` | 300 | 看門狗等待秒數（見問題 7） |

---

## 5. 測試結果

### 5.1 功能

* `model_40299` 在 `_FH_FIST_STABLE_SWING` 上接著跑 20 圈（40 299 → 40 319）：結束碼 0。
* value loss 0.05，平均獎勵 24.3，`root_height_ceiling` 等 D21 項目有出現在獎勵裡，數值跟叢集上同一階段差不多。

### 5.2 速度

| 設定 | 每秒步數 | 每圈時間 | 收集資料 | 學習 |
|---|---:|---:|---:|---:|
| 叢集 2080 Ti，4096 環境（job 5859） | 約 15k | 6.5 秒 | — | — |
| Antioch RTX PRO 6000，4096 環境 | **約 16k** | **6.0 秒** | 5.9 秒 | 0.2 秒 |
| Antioch RTX PRO 6000，16384 環境 | 45–61k | 6.4–8.7 秒 | 6.1–8.5 秒 | 0.26 秒 |

* GPU 強很多，但 4096 環境時速度幾乎一樣。時間幾乎都花在「收集資料」，跟 GPU 快慢無關。
* 原因：HOPE 每一個模擬步驟都有很多小的 Python 計算和小的 GPU 呼叫，每一步的固定開銷很大。
  環境數變多時，這個開銷不太會增加，所以 16384 環境時每圈時間差不多，但樣本量變成 3–4 倍。
* 16384 環境那組是在有一個卡住的程式同時佔用 GPU 的情況下量的（見問題 7），所以實際可能更快。
* **要不要改用 16384 環境還沒決定：** 樣本量變 4 倍，PPO 每次更新的 batch 也變 4 倍，學習的行為會跟之前的 run 不一樣，可能要調學習率或 mini-batch 數。

---

## 6. 遇到的問題和解法

### 問題 1：不知道要用哪個 Isaac 版本

* **狀況：** Antioch 內建的是 Isaac Sim 6 / Isaac Lab 3.0，我們的程式和 checkpoint 都是 Isaac Lab 2.3.2 / Isaac Sim 5.1。
* **解法：** 用 `antioch init --engine base` 建立專案，自己寫 Dockerfile 安裝 5.1 / 2.3.2。
  版本從叢集直接抓：在叢集上跑 `pip freeze` 存成 `cluster-constraints.txt`，安裝時用 `-c` 限制版本。

### 問題 2：套件裝到錯的 Python

* **狀況：** 第一次建置失敗：`isaacsim[all]==5.1.0.0 has no wheels with a matching Python ABI tag (cp312)`。
  log 裡寫 `Using Python 3.12.3 environment at: /usr`。Isaac Sim 5.1 只支援 Python 3.11。
* **原因：** Antioch 的 base 映像檔有設定讓 `uv` 預設裝到系統的 Python 3.12，就算設定了 `VIRTUAL_ENV` 也一樣。
  所以 torch 被裝到系統 Python，接著 Isaac Sim 找不到適合 3.12 的版本。
* **解法：** 每一個 `uv pip install` 都明確加上 `--python /opt/isaac-venv/bin/python`。

### 問題 3：constraints 裡的 torch 版本在 PyPI 上找不到

* **狀況：** `cluster-constraints.txt` 寫的是 `torch==2.7.0+cu128`，這個版本只在 PyTorch 自己的套件庫裡有。
* **解法：** 用 constraints 安裝的步驟都加上 `--extra-index-url https://download.pytorch.org/whl/cu128` 和 `--index-strategy unsafe-best-match`。

### 問題 4：`service exec` 找不到腳本（exit 127）

* **狀況：** `antioch service exec -- env ... bash hope_training/antioch/train.sh` 回傳 `No such file or directory`，
  但檔案確實在 `/workspace/project/hope_training/antioch/train.sh`。
* **原因：** 用 `env` 開頭的指令，工作目錄不一定是 `/workspace/project`。
* **解法：** 腳本和 checkpoint 路徑一律寫絕對路徑 `/workspace/project/...`。

### 問題 5：Isaac 啟動後載入不了 USD（缺系統函式庫）

* **狀況：** 訓練一開始就出錯：`ImportError: libomni.usd.so: cannot open shared object file`，Isaac Lab 的擴充套件全部載入失敗。
* **怎麼找到原因：** 往 log 前面看，最早的錯誤是 `libgomp.so.1: cannot open shared object file`。
  `libomni.usd.so` 本身存在，但它依賴的 USD 函式庫需要 `libgomp.so.1`，連帶全部載入失敗。
* **怎麼確定沒有漏掉別的：** 為了不要一次修一個、重建好幾次（每次重建要好幾分鐘），在 session 上用 `ldd` 掃了 Isaac Sim 所有的 `.so` 檔，
  列出「找不到、而且 Isaac 自己也沒附」的函式庫。結果真正缺的系統函式庫只有兩個：`libgomp.so.1` 和 `libatomic.so.1`。
  其他找不到的（`libmpi`、Python 3.9/3.10/3.12 的版本、範例外掛）都是用不到的選用功能。
* **解法：** Dockerfile 的 apt 安裝加上 `libgomp1 libatomic1`。改映像檔要開新的 session，所以先釋放舊的再重建。

### 問題 6：重建時從 PyTorch 下載逾時

* **狀況：** 第二次建置在下載 torch 相關套件時失敗：`Request failed after 3 retries in 127.4s ... operation timed out`（`download.pytorch.org`）。
* **原因：** 暫時的網路問題。
* **解法：** Dockerfile 加上 `UV_HTTP_TIMEOUT=300` 和 `UV_HTTP_RETRIES=8`，重建就成功了。

### 問題 7：題庫檔沒有同步上去

* **狀況：** Isaac 正常啟動了，但訓練出錯：`FileNotFoundError: V17 physical tuple bank was not found ... g1_fh_crosscourt_apex_bank_wide.npz`。
* **原因：** `hope_training/whole_body_tracking/.gitignore` 有一行 `assets/venue_tuple/*`。這四個題庫檔雖然已經 commit 進 git，
  但 **Antioch 同步時只看 `.gitignore` 規則，不管檔案有沒有被 commit**，所以被跳過了：
  * `g1_fh_crosscourt_apex_bank.npz` / `.receipt.json`
  * `g1_fh_crosscourt_apex_bank_wide.npz` / `.receipt.json`
* **解法：** 寫了 `antioch_sim/push_ignored.sh`：找出 `hope_training/` 底下「已 commit 但被忽略」的檔案，用 `antioch service cp` 一個一個複製上去。
  用 `service cp` 複製的檔案不屬於同步範圍，之後同步也不會刪掉它。**每開一個新 session 都要跑一次。**
* **腳本本身的小 bug：** 第一版加了 `set -euo pipefail`，結果找檔案的迴圈最後一次 `git check-ignore` 回傳 1，整個腳本就靜靜地結束，什麼都沒複製。
  改成在那一段後面加 `|| true`。另外 `antioch` 指令必須在 `antioch_sim/` 裡執行，所以腳本會先切到那個資料夾。

### 問題 8：出錯的 Isaac 卡住不結束，白佔 GPU 30 分鐘

* **狀況：** 測試完想釋放 session，Antioch 拒絕：`Session ... has running work`。
* **怎麼找到原因：** 在 session 上看 `ps`，發現**第一次失敗的那個訓練程式（找不到題庫那次）還活著**，已經跑了 34 分鐘，佔 125% CPU 和 1.7 GB GPU 記憶體。
  Python 出錯後，Isaac 在關閉時卡住了。這跟叢集上 job 5719 在最後存檔後卡住是同一種問題。
* **影響：** 第一次的兩次速度測試（4096 和 16384 環境）都是在它同時佔用 GPU 和 CPU 的情況下量的。
  把它砍掉後重量 4096 環境：每圈 6.0 秒（之前量到 6.4 秒），結論不變。
* **解法：**
  1. 用 `pkill -f scripts/train.py` 砍掉殘留程式。
  2. `train.sh` 加上**看門狗**：訓練程式在背景跑，腳本每 20 秒檢查一次。如果 log 出現 `[train.py] ERROR during run`，
     或最後一個 checkpoint（`model_<TOTAL_ITERS-1>.pt`）已經存好，但程式超過 `HANG_GRACE_S`（預設 300 秒）還沒結束，就把它砍掉。
  3. **測過：** 故意給一個不存在的 checkpoint 讓它出錯（`HANG_GRACE_S=30`），看門狗印出 `watchdog: still alive 30 s after an error/final save -> kill`，
     程式被砍掉（exit 143），之後 `pgrep` 確認沒有殘留。
* **習慣：** 量速度或釋放 session 之前，先跑 `pgrep -fa scripts/train.py` 確認沒有殘留的程式。

### 問題 9：session 上有一個空的 `antioch_sim` 資料夾

* **狀況：** 檢查同步結果時，看到 session 上有 `/workspace/project/antioch_sim`。
* **結果：** 裡面是空的（同步排除了 `antioch_sim/**`，只留下資料夾本身），沒有影響。

---

### 問題 10：背景執行的訓練一開始就被砍掉（exec 結束時連子程式一起結束）

* **狀況：** 第一個正式訓練用 `service exec` 啟動 `train.sh`（裡面用 `setsid nohup ... &` 放到背景），印出 `started pid 87`，
  但馬上檢查就沒有 `train.py`，連 log 檔都沒有。
* **怎麼確認：** 用最簡單的測試：exec 裡用 `setsid -f sleep 900` 和 `nohup sleep 901 & disown` 各開一個，exec 結束後再看，兩個都不見了。
  **Antioch 會在 `service exec` 結束時，把它開出來的所有程式都結束**，不管有沒有 `setsid`/`nohup`。
* **解法：** 改成讓訓練當作**服務本身的啟動指令**（`antioch.yaml` 的 `command`）：
  * 沒有設定 `TASK` 環境變數時，服務只是 `sleep infinity`（一般的 session 不受影響）。
  * 有設定時（`session new --env TASK=... --env RUN_NAME=...`），服務等 `/workspace/project/.train_go` 這個檔案出現，再在前景跑 `train.sh`。
  * `antioch_sim/start_job.sh` 一次做完：開 session（帶環境變數）→ `push_ignored.sh` → 上傳 checkpoint → 建立 `.train_go`。
* **注意：** 服務的啟動指令**不算**「活動」，所以 session 的閒置計時從開始就在算。`--idle-timeout` 要比訓練時間長（我們用 `1d`）。

### 問題 11：多個 session 時指令要指定 session；zsh 不會拆字

* `antioch` 指令在只有一個 session 時會自動選它；開了三個就要每個都加 `--session ses-...`。`push_ignored.sh` 改成讀 `SESSION` 環境變數。
* 在 zsh 裡用 `set -- $p` 拆字串不會拆開（zsh 預設不做 word splitting），所以多個 job 的檢查改寫成 bash 腳本 `status.sh`。
* macOS 內建的 bash 是 3.2，沒有 `declare -A`，`collect.sh` 改用標記檔記錄哪個 job 處理完了。

## 7. 一次跑多個訓練（2026-10-19 起的做法）

```bash
cd antioch_sim
# 每個 job 開自己的 session：TASK RUN_NAME TOTAL_ITERS [本機 checkpoint] [NUM_ENVS] [idle-timeout]
./start_job.sh HOPEPingPongUnitreeG1_FH_FIST_STILL g1_fh_still16k 44300 \
  ../artifacts/evals/g1_fh_stable_swing_model_40299/model_40299.pt 16384 1d
# 看進度（可以一次列多個 session:run）
./status.sh ses-xxxx:g1_fh_still16k ses-yyyy:g1_fh_planted16k
# 在背景每 15 分鐘把 checkpoint（每 500 圈 + 最後一個）和 log 抓回 artifacts/antioch/<run>/，
# 訓練結束、檔案都拿到後自動釋放那個 session（筆電要開著）
./collect.sh ses-xxxx:g1_fh_still16k:44300 ses-yyyy:g1_fh_planted16k:44300
```

16384 個環境時每圈約 5 秒（4096 時 6 秒），所以樣本量約是 5 倍。

## 8. 注意事項

* **費用：** session 開著就在計費。`--idle-timeout` 設太長、或忘記釋放，都會一直算錢。`$A session status` 可以看是什麼讓它保持開著。
* **檔案會消失：** 釋放 session 時，上面的 log 和 checkpoint 全部刪除。一定要先用 `service cp` 拿回來。
* **改程式碼不用重開 session：** 每次 `service exec` 都會先同步本機的程式碼。只有改 `Dockerfile` 或 `cluster-constraints.txt` 才要開新 session（會重建映像檔）。
* **同步只看 `.gitignore`：** 新增被忽略、但訓練需要的檔案時，要確認 `push_ignored.sh` 有涵蓋（它只找 `hope_training/` 底下）。
* **登入：** 用的是使用者已經登入的 Antioch 帳號（preview 部署，SDK 0.5.338）。
* **叢集的 Slurm 腳本** 留在 `hope_training/slurm/` 裡參考，訓練已經不用它了。
