# Triton Droids：Unitree G1 正手斜線對打 — 文件目錄

[English](README.md)

Triton Droids 在 HOPE 和社團 G1 移植版上做的東西：讓 G1 把斜線來球用正手回到對方四個目標區之一。
G1 實機還沒到，目前只在模擬中進行，但所有設計都保留了之後上實機（sim-to-real）的路徑。

## 目前狀態（2026-10-19）

* **運算資源：** 訓練在 **Antioch** 上跑（雲端 RTX PRO 6000；[operations/antioch.zh-TW.md](operations/antioch.zh-TW.md)）。正在跑：D22 三個「腳不動」訓練（16384 環境，約 5.5 小時）。
* **部署修正（10-19）：** `hold_last_target` 讓擊球後繼續顯示上一球的目標，跟訓練一樣：MuJoCo 裡兩球之間不再踏步（每球 3.5 步 → 0 步）。修正後的瞄準補償 (+0.23, −0.01) m，目標區 30%（驗證球、不推）。
* **目前最好的策略：`g1_fh_stable_swing` `model_40299`**（D21、握拳、MuJoCo、120 顆驗證用的球、瞄準補償 (+0.27, −0.08) m）：
  打到 100%、合法 63%、**目標區 38%**；推 1.0 m/s 時：打到 98%、目標區 21%、**沒有跌倒**。
  不加補償會打短（目標區 12%），所以一定要加補償。
  （之前最好的 `g1_fh_fist` `model_32299`：加自己的補償目標區 36%；推 1.0 m/s 跌倒 5%。）
* **平衡：** 推到 1.0 m/s 以內都沒有跌倒；骨盆站在 0.66 m（之前約 0.72）。
* `g1_fh_stable`（只加強平衡）在 40 067 崩潰（value loss 爆掉），而且比較差（目標區 24%）。
* **機器人模型：** 握拳握法，球拍和前臂垂直（D20）。
* 相鄰的 checkpoint 差很多：一定要用 MuJoCo 測試挑。

詳細內容：[status/log.zh-TW.md](status/log.zh-TW.md)（最新的在最上面）。

## 什麼東西在哪裡

| 資料夾 | 檔案 | 內容 |
|---|---|---|
| （最上層） | [overview.zh-TW.md](overview.zh-TW.md) | 背景、社團的成功規則、整套系統怎麼運作 |
| | [decisions.zh-TW.md](decisions.zh-TW.md) | 所有設計決策（D1、D2…）和理由 |
| `implementation/` | [changes.zh-TW.md](implementation/changes.zh-TW.md) | 程式改了什麼，逐檔說明 |
| | [question_bank.zh-TW.md](implementation/question_bank.zh-TW.md) | 餵球和目標區（題庫）怎麼產生 |
| | [pipeline.zh-TW.md](implementation/pipeline.zh-TW.md) | 端到端流程：影片 → clip → 訓練 → 評估 → 部署 |
| `operations/` | [runbook.zh-TW.md](operations/runbook.zh-TW.md) | 日常指令：查訓練、匯出、開 MuJoCo、開始/停止 job |
| | [running.zh-TW.md](operations/running.zh-TW.md) | 安裝和完整執行指令（筆電和叢集） |
| | [deploy_calibration.zh-TW.md](operations/deploy_calibration.zh-TW.md) | 上實機時要量什麼、怎麼量 |
| | [antioch.zh-TW.md](operations/antioch.zh-TW.md) | 在 Antioch（雲端 GPU）上訓練：設定、用法、遇到的問題和解法（只有中文） |
| `status/` | [log.zh-TW.md](status/log.zh-TW.md) | 狀態紀錄：每次檢查、事故、發現，最新的在最上面 |
| | [runs_and_results.zh-TW.md](status/runs_and_results.zh-TW.md) | 所有 run、訓練曲線、各區成功率表 |
| | [how_we_track.zh-TW.md](status/how_we_track.zh-TW.md) | 用哪些指標、怎麼量、怎麼新增紀錄 |
| | [results.zh-TW.md](status/results.zh-TW.md) | 早期結果摘要（2026-10-12） |
| `research/` | [perception.zh-TW.md](research/perception.zh-TW.md) | 攝影機 / 動捕：需要什麼，以及感知實驗 |
| | [humanoid_table_tennis.zh-TW.md](research/humanoid_table_tennis.zh-TW.md) | SMASH、北大、HITTER、PACE：平衡和回球有哪些值得試 |
| `plans/` | [next_steps.zh-TW.md](plans/next_steps.zh-TW.md) | 下一步 |
| | [serve_side_backspin.md](plans/serve_side_backspin.md) | 暫緩的計畫：反手側下旋發球（只有英文） |

這個資料夾以外的相關文件：[../UNITREE_G1.md](../UNITREE_G1.md)（G1 移植：機器人模型、球拍安裝、
模擬器一致性），以及 [`docs/`](..) 裡 HOPE 原本的文件。

## 怎麼新增文件內容

* 檢查、事故、實驗結果 → 在 [status/log.zh-TW.md](status/log.zh-TW.md) 最上面加一筆；
  會取代表格的數字 → [status/runs_and_results.zh-TW.md](status/runs_and_results.zh-TW.md)。
* 設計上的選擇 → 在 [decisions.zh-TW.md](decisions.zh-TW.md) 加一列；對應的程式改動 → [implementation/changes.zh-TW.md](implementation/changes.zh-TW.md)。
* 一律中英文都寫（`*.md` 和 `*.zh-TW.md`），用字淺白，但數字和注意事項要留著。
