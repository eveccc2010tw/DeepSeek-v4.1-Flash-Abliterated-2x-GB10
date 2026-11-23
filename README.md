# DeepSeek V4.1 Flash 去限制配方：雙 DGX Spark／GX10

這是 Thx／OSSLab 在兩台 GB10 上部署 **DeepSeek V4.1 Flash 1017 去限制（Abliterated／ABLIT）權重覆層**的經驗配方，採用 Mia EAQ v1.1 EXL3 2.9 bpw、drowzeys Keys attention graft，以及 TensorFold 原生 Zig 引擎。中文文件記錄固定版本、48-row 調校、記憶體保護、長對話快取及更新前後 A/B。

**這裡的「去權重」指去限制權重配方**：以 attention output projection 的權重覆層改變模型行為，沒有使用 runtime GLP steering。這個專案不附模型權重、Engram、覆層檔或容器映像；請從原作者來源取得。名稱說明其技術來源，並不代表已量化證明拒答率下降或模型品質提升。

目前配方固定 Mia recipe `c7f2b85`，以官方 53 支加自有 0100–0103 共 57 支引擎補丁，另有兩支 inputs 補丁；範例容器設定已啟用 `TF_DS_RDMA_CPUS=0`（將 `tf-rdma-proxy` 綁定於 GB10 Cortex-A725 小核 CPU 0）與 `TF_DS_ENGRAM_LEAD=1`（Engram chunk-ahead 預讀）。單對話設定上限 1,048,576 tokens，分片共享池 2,101,248，3 路並行，round／copy rows 均為 48。**實機驗收到約 300K；尚未測滿 1M。**

## 閱讀與使用

| 文件 | 內容 |
| --- | --- |
| [安裝與重現](docs/setup.zh-TW.md) | 固定來源、去限制 graft、引擎建置、兩端容器與常駐監督 |
| [經驗與調校](docs/experience.zh-TW.md) | 48-row、KV／SSD、編譯 OOM、MTP dtype、更新與回退 |
| [性能 A/B 與驗收](docs/ab-results.zh-TW.md) | 方法、24 筆逐筆紀錄、300K 長文與限制 |
| [版本鎖定](recipe.lock.json) | 引擎、模型、覆層、Engram、kit、Zig 的 revision 與 SHA256 |
| [補丁來源](patches/README.zh-TW.md) | 0100–0103 的用途、上游歸因與 metadata 整理 |
| [本次公開版驗證](docs/verification.zh-TW.md) | 可重複命令、測試結果及未驗範圍 |

需要兩台各 128 GB 統一記憶體的 DGX Spark／GX10、可用的 ConnectX-7 RoCE／RDMA 直連、各自的本機 NVMe、aarch64 CUDA／sm121 工具鏈及 NVIDIA Container Toolkit。預留模型、Engram、graft、rank cache、建置與 spill 空間；完整準備通常需要數百 GB，不能只按約 196 GiB 的模型大小估算磁碟。

## 實測結果

2026-11-17（台灣時間），固定相同模型與 48-row 設定，只比較更新前後引擎。每組重載後 warmup，再跑三輪四個合成提示，主要測量共 24 筆；兩組提示與輸出 SHA256 一致。

| 場景 | A tok/s | B tok/s | 差異 |
| --- | ---: | ---: | ---: |
| 短提示，9 筆／組 | 90.42 | 90.85 | +0.48% |
| 1,735-token 提示，3 筆／組 | 85.49 | 86.18 | +0.81% |

普通生成吞吐持平；小樣本差異不能當成穩定加速。保留 B 的理由是工具解析修正及功能驗收通過。A/B 兩邊都使用去限制覆層，**這不是 stock 與 abliteration 的對照實驗**。過去不同權重、context、copy workload 的 100–200+ tok/s 數字不併入這張比較表。

已驗：Chat SSE、工具特殊字串、Responses、Anthropic thinking、合成圖片、三路並行、兩個約 300K 對話交替快取、短請求插入、取消後恢復 131,072 tokens，以及雙 rank 重啟後 SSD restore。詳細紀錄和限制見 [A/B 報告](docs/ab-results.zh-TW.md)。

2026-11-20（台灣時間），在相同 B 映像與 48-row 基礎上完成兩項免重編運行時參數的 A1 → B → A2 實測並合併部署：

- **`TF_DS_RDMA_CPUS=0`**：將 `tf-rdma-proxy` 釘在 Cortex-A725 小核 CPU 0，單流／3-stream 總吞吐持平（`<1%`），3-stream 壓測期間雙機 `acpitz` 峰值溫度降低約 `2–4.5 °C`，輸出 SHA256 一致。
- **`TF_DS_ENGRAM_LEAD=1`**：開啟 Engram chunk-ahead 預讀，64K 冷啟動 prefill TTFT 由 `29.841s`／`30.316s` 降至 `28.690s`（`-4.6%`／`+4.8% tok/s`），131K 冷啟動由 `62.108s`／`63.343s` 降至 `59.186s`（`-5.6%`／`+6.0% tok/s`），輸出 SHA256 一致。
- **合併部署回歸（`TF_DS_RDMA_CPUS=0`＋`TF_DS_ENGRAM_LEAD=1`）**：3 輪短提示中位 `91.61 tok/s`、3 輪 1735-token 提示中位 `85.78 tok/s`、冷啟動 64K TTFT `28.641s`（`2279.4 tok/s`）。

## 上游與授權

- [Mia 的雙 Spark recipe](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks)、[EAQ v1.1 模型](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1.1-2.9-bpw)。
- [drowzeys／Keys 去限制覆層](https://huggingface.co/drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated)。
- [bertholomus／TensorFold](https://github.com/bertholomus/TensorFold)、[雙 GB10 kernel kit](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10)。
- [sehoon123 Tuned](https://github.com/sehoon123/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks-Tuned)：0103 的 Responses buffer 成長策略來源。
- DeepSeek、TensorFold、turboderp／ExLlamaV3，以及上述專案的原始貢獻者。

本專案的整合工具與文件採 [Apache-2.0](LICENSE)。補丁上下文及引用程式保留原授權；模型、覆層、kit、CUDA／NVIDIA 映像各依來源授權與存取條件。歸因及第三方聲明見 [NOTICE](NOTICE) 與 [LICENSES](LICENSES)。
