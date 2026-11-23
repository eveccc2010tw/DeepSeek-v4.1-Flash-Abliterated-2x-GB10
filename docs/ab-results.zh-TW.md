# 更新前後 A/B 與最終實測

測試區間：2026-11-17T23:20:57+08:00 至 2026-11-17T23:32:58+08:00。這是原雙 GB10 部署的歷史實測，公開版本整理期間沒有再次重啟或重新跑模型。

## 固定條件與測量方法

| 項目 | A | B |
| --- | --- | --- |
| Mia recipe | `31457d05f8917301bbf609c3d12c7f364af8a4a6` | `c7f2b85100f0a7b099c92a4e21825e30d93c9c51` |
| 實機 patch digest | `07d9bcbe5743` | `e6ea17d42858` |
| TensorFold base | `fbbc97132cd004237d9887b8d3185896efaedc5c` | 相同 |
| 權重／overlay／Engram | EAQ v1.1／Keys graft／1017 | 相同 |
| round／copy rows | 48／48 | 相同 |
| window／pool／parallel | 1,048,576／2,101,248／3 | 相同 |

兩組各先受控重載並跑一輪 warmup，之後三輪四提示；不計 warmup 及較早的 A 測試。取樣 `temperature=0`、`top_p=1`，兩種協定的 thinking 都關閉。每筆 HTTP 200、SSE `[DONE]`、finish_reason=length、cache=0、reasoning=0，且 prompt／output SHA256 相同。每組生成總量恰為 2,688 tokens，實機計數確認沒有其他完成推論混入。

吞吐定義為 `(completion_tokens - 1) / (last text event - first text event)`。這是串流事件時間估計；不是逐 token GPU 內部計時。TTFT 從送出請求至首個文字事件。

## 性能結果

| 固定提示 | 輸入／輸出 tokens | A 中位 tok/s | B 中位 tok/s | 差異 |
| --- | ---: | ---: | ---: | ---: |
| 短提示 1 | 103／256 | 91.75 | 91.89 | +0.15% |
| 短提示 2 | 103／256 | 83.87 | 84.10 | +0.27% |
| 短提示 3 | 103／256 | 91.09 | 90.85 | -0.26% |
| 較長提示 | 1735／128 | 85.49 | 86.18 | +0.81% |

| 場景彙總 | A tok/s | B tok/s | 差異 | A TTFT 秒 | B TTFT 秒 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 短提示，9 筆／組 | 90.42 | 90.85 | +0.48% | 0.304 | 0.314 |
| 較長提示，3 筆／組 | 85.49 | 86.18 | +0.81% | 1.106 | 1.097 |

數據支持一般生成性能持平。更新理由為工具解析修正與功能驗收；不推論模型品質提升，也不是 stock／abliterated 的去限制效果評測。逐筆 JSON 見 [records/ab](../records/ab)，核算命令：

```bash
python3 tools/summarize-ab.py --records records/ab
```

## 真實路徑驗收

[原始合成請求驗收結果](../records/acceptance.json) 保留 timings、usage、hash 與 counters，移除部署資料。

- API：401、models、Chat SSE、DSML 工具參數內含 closer 字串、多段工具參數、Responses、Anthropic thinking、合成圖片皆通過。
- ALPHA／BETA prompt：299,485／299,491 tokens；交替重送各命中 299,008 tokens，首字約 1 秒，首次約 146 秒。
- 長 prefill 中短請求 4.618 秒完成，早於長請求首字；三路並行輸出 hash 與 A 單流一致。
- 75 秒取消後重送命中 131,072 RAM progress tokens。
- 雙 rank 受控重啟後 ALPHA 命中 299,008 tokens，`disk_hits_committed` 從 0→1；據此確認 SSD restore。
- 最後再跑 API smoke 與 matched benchmark 均通過。兩端 guard 活著，supervisor active／NRestarts=0；pair_failed／broken／stalled 為 0，未見 Docker OOMKilled。

0065 新增 `TENSORFOLD_TOOL_STREAM` 能力，維持預設 live；其餘 capability 與 A 相同。這次不是重新導入全部 cache／spill 功能。

## 2026-11-20 運行時參數優化 A/B 與合併上線驗收

在相同 B 映像（`fbbc97132cd0+e6ea17d42858`）、相同 EAQ v1.1＋Keys 覆層權重、48-row、`KV_SHARD=1`、`PARALLEL=3` 與 `temperature=0`／thinking off 條件下，分兩輪進行 A1 → B → A2 交叉實測，再合併上線。此兩輪為原雙 GB10 部署記錄之摘要數據，未再重匯額外逐筆 JSON 至 `records/ab/`（`records/ab/` 保留 2026-11-17 引擎更新之 24 筆逐筆檔案）。

### 1. `TF_DS_RDMA_CPUS=0`（2026-11-20T21:49–21:55+08:00）

| 測項 | A1（`unset` 基準） | B（`TF_DS_RDMA_CPUS=0`） | A2（還原 `unset` 復測） | 結論 |
| --- | ---: | ---: | ---: | --- |
| 短提示串列（9 筆中位 tok/s） | 91.70（TTFT 0.302s） | 91.21（TTFT 0.303s） | 90.95（TTFT 0.303s） | 持平（±0.6%） |
| 較長提示串列 1735→128（3 筆中位 tok/s） | 86.87（TTFT 1.121s） | 86.30（TTFT 1.098s） | 86.16（TTFT 1.110s） | 持平 |
| 3-stream 併發總吞吐（3 輪中位 tok/s） | 120.85 | 123.63 | 123.41 | 與 A2 差 +0.18% |
| `tf-rdma-proxy` 實際核心（rank 0 / rank 1） | CPU 5 / CPU 5（X925） | **CPU 0 / CPU 0（A725）** | CPU 8 / CPU 5（X925） | `taskset -pc` 為 `0` |
| 第 3 輪 3-stream `acpitz` 峰值（rank 0 / rank 1） | 82.8 °C / 82.4 °C | **80.2 °C / 80.9 °C** | 82.5 °C / 85.4 °C | B 低約 2–4.5 °C |

所有組別 12 筆串列＋3 輪 3-stream 併發之 `output_sha256` 逐筆一致。

### 2. `TF_DS_ENGRAM_LEAD=1`（2026-11-20T23:10–23:21+08:00）

使用獨立暫存容器與隔離 spill 目錄量測冷啟動（`cached_tokens=0`），A2 於長提示 prefix 加入 `-a2` 以避開跨重啟磁碟快取：

| 指標 | A1（關閉） | B（`TF_DS_ENGRAM_LEAD=1`） | A2（還原關閉） | B 相對 A1／A2 均值 |
| --- | ---: | ---: | ---: | --- |
| 短提示串列中位數（3 題） | 91.87 tok/s | 91.81 tok/s | 91.30 tok/s | 持平（+0.2%） |
| 較長提示串列（1735 tok，單次） | 86.32 tok/s（TTFT 1.095s） | 78.28 tok/s（TTFT 1.118s） | 85.83 tok/s（TTFT 1.105s） | 單次冷啟動波動（見下） |
| Cold 64K_A（`cached=0`） | 29.797s / 2191.4 tok/s | **28.716s / 2273.8 tok/s** | 30.488s / 2141.2 tok/s | TTFT -1.43s |
| Cold 64K_B（`cached=0`） | 29.885s / 2184.9 tok/s | **28.664s / 2278.0 tok/s** | 30.145s / 2165.6 tok/s | TTFT -1.35s |
| Cold 64K 中位數 | **29.841s / 2188.1 tok/s** | **28.690s / 2275.9 tok/s** | **30.316s / 2153.4 tok/s** | **TTFT -4.6%（+4.8% tok/s）** |
| Cold 131K（`cached=0`） | **62.108s / 2102.4 tok/s** | **59.186s / 2206.2 tok/s** | **63.343s / 2061.0 tok/s** | **TTFT -3.54s / -5.6%（+6.0% tok/s）** |
| 131K `MemAvailable` 最低（rank 0 / rank 1） | 6.497 / 6.802 GiB | 6.744 / 7.311 GiB | 6.703 / 6.866 GiB | 無額外記憶體壓力 |

`A1`、`B`、`A2` 之串列、`cold_64k`（`27a9f301bc68`／`0e8ea6b10ade`）與 `cold_131k`（`6a2309646691`）輸出 `output_sha256` 全部一致。

### 3. 合併部署後實測驗證（2026-11-20T23:55–23:59+08:00）

雙機正式容器同步啟用 `TF_DS_RDMA_CPUS=0` 與 `TF_DS_ENGRAM_LEAD=1` 後實測：

- 雙機啟動日誌均確認 `info: engram-lead: on (rank 0 / rank 1): each Engram layer's prompt rows read a chunk ahead`；`taskset -pc` 確認雙端 `tf-rdma-proxy` 皆固定於 CPU 0（Cortex-A725）。
- 3 輪 matched serial 短提示（9 筆）：中位數 **`91.61 tok/s`**，`output_sha256` 與基準一致。
- 3 輪 matched serial 較長提示（1735 tok × 3）：`[80.67, 86.19, 85.78]` tok/s，中位數 **`85.78 tok/s`**（TTFT 中位數 `1.091s`），`output_sha256` 一致（確認單項實測中的 `78.28` 屬冷啟動後單次波動）。
- Cold 64K prefill（`65282` tok，`cached=0`）：TTFT **`28.641s`**（**`2279.4 tok/s`**），`output_sha256`（`27a9f301bc68...`）一致，`MemAvailable` 最低 `rank 0 6.780 GiB`／`rank 1 7.351 GiB`。
- 合併部署後未再重跑 131K 與 3-stream 壓測（已於同晚稍早分項 A/B 各自實測）。

## 已知限制

沒有測滿 1M、24 小時 soak、冷 page-cache 載入、全權重重新 hash、硬 guard 或 rank crash 注入，也沒有廣泛程式題品質／拒答率評測。

原測試主機的預設 `/tmp` Zig host suite 為 585/587，兩個既有 O_DIRECT 故障注入測試失敗；相關 fsio／store_test 在 A/B 位元一致。`TF_SPILL_TEST_DIR=/dev/shm` 完整 suite 通過，另有實機 SSD restore；這不表示兩個預設檔案系統測試已修好。

公開補丁移除作者郵件／日期 metadata，diff hunks 保持完全相同，因此重新組合後的 patch digest 與上述實機 B 身分不同。來源映射見 [recipe.lock.json](../recipe.lock.json)，公開組合 manifest 見 [records/public-manifest.json](../records/public-manifest.json)。補丁身分應隨實際位元內容走，不要把測量版 digest 強塞到公開版映像。
