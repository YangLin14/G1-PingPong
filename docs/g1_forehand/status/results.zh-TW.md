# 結果摘要

[English](results.md) · [文件目錄](../README.zh-TW.md)

完整的表格、訓練曲線和狀態紀錄在 [runs_and_results.zh-TW.md](runs_and_results.zh-TW.md) 和 [log.zh-TW.md](log.zh-TW.md)。摘要：

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

