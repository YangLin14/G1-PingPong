# Triton Droids：Unitree G1 正手斜線對打 — 文件目錄

[English](README.md)

Triton Droids 在 HOPE 和社團 G1 移植版上做的東西：讓 G1 把斜線來球用正手回到對方四個目標區之一。
G1 實機還沒到，目前只在模擬中進行，但所有設計都保留了之後上實機（sim-to-real）的路徑。

## 目前狀態（2026-10-14）

* **叢集：** zixian 已恢復（10-14 17:05）。`g1_fh_diag`（主要 run）在 17 261 / 20 000，大約再 6 小時；
  GPU idx 5 繼續避開；`g1_fh_rally` 停在 5900，改由新的平衡 run
  （`g1_fh_bal`，D19）取代，還沒送出。
* **目前最好的策略**（`g1_fh_diag` `model_15700`，MuJoCo，部署預設值）：打到球 100%、在最高點擊球 97%、
  合法回球 82%；加上瞄準補償（D18）後 35% 落在目標區。
* **平衡：** 被推 0.5 m/s 時 6% 的球會跌倒，1 m/s 時 26%；原本的訓練完全沒有推擠。正在修（D19）。
* **機器人模型：** 右手改成握拳握住球拍（D20，只改外觀）。

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
| `status/` | [log.zh-TW.md](status/log.zh-TW.md) | 狀態紀錄：每次檢查、事故、發現，最新的在最上面 |
| | [runs_and_results.zh-TW.md](status/runs_and_results.zh-TW.md) | 所有 run、訓練曲線、各區成功率表 |
| | [how_we_track.zh-TW.md](status/how_we_track.zh-TW.md) | 用哪些指標、怎麼量、怎麼新增紀錄 |
| | [results.zh-TW.md](status/results.zh-TW.md) | 早期結果摘要（2026-10-12） |
| `research/` | [perception.zh-TW.md](research/perception.zh-TW.md) | 攝影機 / 動捕：需要什麼，以及感知實驗 |
| `plans/` | [next_steps.zh-TW.md](plans/next_steps.zh-TW.md) | 下一步 |
| | [serve_side_backspin.md](plans/serve_side_backspin.md) | 暫緩的計畫：反手側下旋發球（只有英文） |

這個資料夾以外的相關文件：[../UNITREE_G1.md](../UNITREE_G1.md)（G1 移植：機器人模型、球拍安裝、
模擬器一致性），以及 [`docs/`](..) 裡 HOPE 原本的文件。

## 怎麼新增文件內容

* 檢查、事故、實驗結果 → 在 [status/log.zh-TW.md](status/log.zh-TW.md) 最上面加一筆；
  會取代表格的數字 → [status/runs_and_results.zh-TW.md](status/runs_and_results.zh-TW.md)。
* 設計上的選擇 → 在 [decisions.zh-TW.md](decisions.zh-TW.md) 加一列；對應的程式改動 → [implementation/changes.zh-TW.md](implementation/changes.zh-TW.md)。
* 一律中英文都寫（`*.md` 和 `*.zh-TW.md`），用字淺白，但數字和注意事項要留著。
