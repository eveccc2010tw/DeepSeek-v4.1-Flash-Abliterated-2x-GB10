# 本專案的四支整合補丁

固定官方 recipe `c7f2b85100f0a7b099c92a4e21825e30d93c9c51` 的 53 支引擎補丁先按檔名順序套用，再套本目錄 0100–0103；inputs 使用該 recipe 的兩支補丁。TensorFold 固定於 `fbbc97132cd004237d9887b8d3185896efaedc5c`。

| 補丁 | 來源與作用 |
| --- | --- |
| 0100 | OSSLab 寬 copy window 整合；保留官方 draft calibration、copy_proposals 與 no_copies，支援 round／copy rows 16、32、48 |
| 0101 | OSSLab RAM-only 64K prefill 進度點；保留 cacheKey、restore／fill、tail／usable 判定及取消重送回歸 |
| 0102 | Responses／Anthropic SSE keepalive；0065 已有 Responses ping，這裡保留 SSE 開啟前的 guard |
| 0103 | 移植 sehoon123 Tuned `171052a7d40745ba43f1d9fcb01cb3c2643d93923` 的 Responses 文字／工具參數 buffer 幾何成長修復，再 rebase 到 Mia 0065 |

官方的 DSML 與 CUDA context 修正沿用上游，不重疊套用舊 0007／0034。沒有納入 `0104-prefetch-overlap.patch`，不能把 Tuned 整包建立在舊引擎的補丁直接疊上來。

公開整理僅移除 0100／0101 的作者郵件、日期行，保留 OSSLab 歸因、Subject 及完整 diff；引擎程式內容不變。`recipe.lock.json` 保存實測與公開 patch SHA256 的對照。Mia／TensorFold／Tuned 的程式及 diff context 仍保留各自授權，見 NOTICE。
