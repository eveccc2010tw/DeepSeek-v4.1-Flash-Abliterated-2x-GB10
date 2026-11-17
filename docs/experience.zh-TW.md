# 實機經驗與調校取捨

## 去限制是權重 graft，不是啟動參數

這套配方把 drowzeys 的 `exl3_wo_b_l10_35_mia29.safetensors` 接到 Mia EAQ v1.1：覆蓋 layers 10–35 的 26 個 attention output projection，共 104 個 EXL3 tensors。不是把 GLP vector 加到 runtime，也不是改 system prompt。其餘專家、vision、MTP 與 Engram 仍來自固定底座。

覆層最初的目標是 Mia EAQ v1。官方固定 recipe 的 `ABLIT=1` **拒絕 v1.1**；這裡記錄的是已部署驗收的客製組合，不能把 `EAQ_V1_1=1 ABLIT=1 ./start.sh` 當成一鍵安裝指令。shape／dtype 對得上及成功載入不代表已證明與原始模型品質等價。本次沒有廣泛品質或拒答率評估。

graft 應另建目錄，stock 留存。保留 MTP 原始 dtype，不加 `--jay-zig`；檢查 `n_edited=104`、`n_cast_f32=0`、`jay_zig_casts=false`。改模型後重建兩端原生 rank cache，Python 版本的 cache 不能直接交給原生 Zig 引擎。

## 48-row 與普通生成測速

round_rows／copy_rows 都設 48，兩個 rank 的 pool／streams／graphs 必須一致。helper 的相容預設仍是 32-row，**必須使用本專案範例 JSON 的明確設定**，不能只靠預設值重現現役。保留 copy skip、batch skip、自複製 cap 與 multi 的既有策略，沒有加入未採用的 `0104-prefetch-overlap.patch`。

Copy workload、普通生成、prefill 與並行總吞吐是不同測項。複製既有程式碼時的高 tok/s 不等於一般解題速度；並行 aggregate 也不能當成單一路速度。最新 A/B 兩邊都已是 48-row，不能用它推論 32→48 的效益。

## 記憶體要看 MemAvailable

GB10 的 GPU 和主機共用記憶體。`context=1M` 是請求上限，不是可配置記憶體的保證；載入時的 reserve 計算也不是另外保留的實體 RAM。先量兩端 MemAvailable，長文、graphs 與外部工作負載都會影響餘裕。

| 設定 | 本配方值 | 用意 |
| --- | ---: | --- |
| KV sharding／pool | 開啟／2,101,248 | 每端保留自己的 KV 分片 |
| graph cap | 128 | 限制長時間累積的 CUDA graphs |
| NCCL channels | 8 | 控制通訊資源占用 |
| prompt cache | 0.5 GiB | 快取狀態預算，與 KV pool 不是同一物件 |
| SSD spill | 每端 32 GiB | 保留可恢復的長對話狀態 |
| spill free floor | 35 GiB | 剩餘磁碟低於門檻時限制 spill |
| memguard soft／hold | 2.0 GiB／4 秒 | 持續低水位才停止 rank |
| memguard hard | 1.0 GiB | 連續兩次約 0.25 秒取樣低於門檻即處理 |

guard 先於 rank 啟動；一端 rank 結束，由 supervisor 停整對。停止 rank 失敗時不能先拆掉它的 guard。Supervisor 啟動新 pair 前要求兩端各 110 GiB MemAvailable；接管已健康運行的雙 rank 不要求再空出一份模型記憶體。

本配方的 2.0 GiB soft floor 是這組硬體的實測設定，不能保證其他桌面工作負載也足夠。最近測試仍未刻意觸發 hard guard 或 rank crash。

## 長文快取要分清楚 RAM 與 SSD

`fill_chunks=1`、`fill_first=1`、`fill_cut_in=8192` 讓長 prefill 之間可插入短請求。實測短請求 4.618 秒完成，早於長請求約 146 秒首字。這是該 workload 的結果，不是所有請求的延遲保證。

0101 每 65,536 tokens 保留 RAM 進度點，75 秒取消後重送命中 131,072 tokens；這不等於中途進度已落 SSD。雙 rank 重啟後，另確認 `disk_hits_committed` 由 0 增至正值，才把約 299K 的命中稱為真正 SSD restore。快取、spill 及其他已有能力不能全部歸因成此次更新新增。

## 編譯與回退

四核心編譯曾撞上 3,584 MiB build cgroup 上限；改成 CPU 0 單核心，RAM 和 memory-swap 同為 3584m、cpu-shares=2 後成功。memory-swap 等於 memory 表示沒有額外 swap 額度。容器 cgroup 能限制編譯工作，仍需監看共用主機總記憶體；請勿把它當成主機不會 OOM 的保證。

換引擎先保存舊 image ID、容器設定、權重／template 指紋與獨立 spill。候選先建不啟動，完整停止舊 pair 後才啟新 pair，不能同時載入兩份完整模型。新舊版本的 spill 身分不同，不要共用路徑或硬改 identity 嘗試命中舊 cache。回退時先停新 pair，還原舊 pair 設定及原 spill，再啟 worker、head 與 smoke 驗收。

原部署使用受管 SSH；公開版 supervisor 改為 OpenSSH alias，沿用相同 pair 狀態機。此移植已做單元驗證，尚未在另一組硬體重新部署。正式 supervisor 曾有 SSH 失敗，後續同服務環境連線與受控重啟正常；根因沒有重現，不宣稱已修復根因。
