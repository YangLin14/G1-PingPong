# 執行方式（筆電和叢集）

[English](running.md) · [文件目錄](../README.zh-TW.md)

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

* 帳號 `tritondroids`，全社團共用 **32 CPU / 96 GB / 8 GPU**。Isaac 需要 RTX 顯卡，所以要用 `--gres=rtx2080ti:1`（idx 0 壞掉期間用 `:2`，見 runbook）（節點 `zixian`，11 GB，driver 610.43）。一個訓練 run 只用一張 GPU。
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

鏡頭固定在側前方約 45 度，畫面要拍到全身和球拍，30–60 fps，一次完整的正手從準備姿勢開始、揮完再回到準備姿勢，錄 5–10 次。之後照 [`slurm/video_to_g1/README.md`](../../../hope_training/slurm/video_to_g1/README.md) 的步驟轉換。SMPL/SMPL-X 的授權是個人的，**絕對不能 commit**（本機已加進 `.git/info/exclude`）。

