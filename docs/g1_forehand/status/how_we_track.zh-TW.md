# 怎麼追蹤一個 run

[English](how_we_track.md) · [文件目錄](../README.zh-TW.md)

## 怎麼追蹤一個 run

有四個工具，由便宜到完整排列，都在叢集的 `~/hope/HOPE` 底下執行。

| 工具 | 回答什麼問題 | 成本 | 輸出 |
|---|---|---|---|
| `status.sh` | 還在跑嗎？最新的訓練指標和最新的 checkpoint | 立即 | 螢幕 |
| `progress.sh <job> [every]` | 整個 run 的指標怎麼變化？ | 立即 | markdown 表格（貼到[訓練曲線](runs_and_results.zh-TW.md)） |
| `eval_forehand.sh --in-job auto <run>` | **用社團成功定義算出的各目標區成功率**（Isaac，虛擬球） | 約 20 分鐘，和訓練共用 GPU | `~/hope/evals/<run>/<model_N>/forehand_zones.{json,txt}` |
| `eval_checkpoint.sh --in-job auto <run>` | 同樣的題目，在另一個模擬器（MuJoCo）裡用**真的球**打：影片加上分區成功率，時機也是真的量 | 約 4 分鐘 | `~/hope/evals/<run>/<model_N>/eval_bank.{json,mp4}` |

```bash
ssh sdsc-scc
cd ~/hope/HOPE
bash hope_training/slurm/pip_env/status.sh                               # 一頁看完
bash hope_training/slurm/pip_env/progress.sh 4382                        # 歷史表格
bash hope_training/slurm/pip_env/eval_forehand.sh --in-job auto g1_fh_diag
bash hope_training/slurm/pip_env/eval_checkpoint.sh --in-job auto g1_fh_diag
```

### 訓練指標（`status.sh`、`progress.sh`）

這些數字來自訓練 log，是最近幾千次擊球的移動平均。因為訓練時開著動作雜訊和 domain randomization，
數字會比評估低。

| 指標 | 意思 |
|---|---|
| racket ≤10 cm | 擊球那一幀，球拍中心離預定擊球點 10 cm 以內 |
| all OK | 擊球那一幀，位置、速度、拍面方向全部在容許範圍內 |
| contact | 接到虛擬球：球拍離球 9.5 cm 以內，而且朝球的方向移動 |
| legal return | 接到球、過網、第一彈落在對方半場（任何位置） |
| landing err | 落點和這一題預定落點之間的距離 |
| falls | 所有 episode 結束中，因為跌倒而結束的比例（其餘是時間到） |

### 社團成功率（`eval_forehand.sh`）

這是我們說好的定義（`utils/forehand_success.py`）。一顆回球要**全部**符合才算成功：

1. **時機**：在我方台面彈起後的最高點擊球（來球的 |vz| ≤ 0.3 m/s）；
2. **接觸**：球拍離球 9.5 cm 以內，而且朝球移動；
3. **過網**，而且第一彈**落在對方半場**；
4. 第一彈落在**這一題指定的目標區**裡（對方右手邊的 4 個區之一）。

評估的計算方式：

* 熱身之後發出的每一題（題庫的一列：發球、擊球點、目標區）都算一次嘗試。機器人跌倒、沒碰到的球算該區的**失敗**，
  不會從分母拿掉；
* 評估時不設時間上限，只有跌倒會結束 episode；
* policy 用確定性輸出（沒有動作雜訊），512 個環境，每個跑 60 秒；
* 表格是漏斗形式：接觸 → 再加過網 → 再加落在對方半場 → 再加落在目標區。成功還要再通過時機檢查。

限制：在 Isaac 裡球是虛擬的，擊球那一幀球一定剛好在擊球點，而題庫把那一幀設在最高點。所以這裡的「時機」
檢查的是題目，不是機器人。要測機器人自己的時機，需要真的模擬一顆球（MuJoCo）或用真機，這是之後的步驟。

---

## 怎麼新增一筆紀錄

1. 執行 `bash hope_training/slurm/pip_env/status.sh`，把主要數字寫成新的一筆，放在[狀態紀錄](log.zh-TW.md)最上面
   （日期、時間、job、iteration、有什麼改變）。
2. 大約每 2000 個 iteration，或改了東西之後，執行 `eval_forehand.sh --in-job auto <run>`，
   再從 `~/hope/evals/<run>/<model_N>/forehand_zones.txt` 在[各區成功率](runs_and_results.zh-TW.md)加一列。
3. 要更新[訓練曲線](runs_and_results.zh-TW.md)，就跑 `progress.sh <job>` 換掉表格。
4. 更新最上面的「最新」那一行，英文版也要同步更新。
