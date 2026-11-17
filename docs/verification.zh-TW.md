# 公開版驗證

本次交付是從既有實機配方挑選、去除私有部署資料後發布的新專案。沒有把私有 repository 的歷史、Docker inspect、模型或憑證一起上傳。

## 可重複檢查

在專案根目錄：

```bash
python3 -m unittest discover -s tests -v
python3 tools/summarize-ab.py --records records/ab
python3 -m compileall -q tools tests
git diff --check
```

- Python 回歸：35 tests 通過。涵蓋固定來源與拒絕 dirty／錯 pin、容器參數與 48-row、編譯資源限制、pair 啟停及 guard 狀態機、SSH 參數 quoting、SSE 測速、A/B 對照與拒絕不匹配資料。
- A/B 核算：24 筆、每組 2,688 生成 tokens，短提示 +0.4763%、較長提示 +0.8091%，prompt／output hash 一致。
- 公開補丁：57 支在固定 TensorFold base 逐項 apply/check；產生的受影響引擎原始碼與原實測 B 樹逐檔位元一致。去除郵件 metadata 後公開 patch digest 為 `8cb202e1699d`；實測 B digest 為 `e6ea17d42858`。
- 引用與發布檔案檢查：文件相對連結存在；候選檔案沒有內網部署位址、私有家目錄、公司私有服務連結、API token 或模型檔。上游公開原作者聲明原樣保留。

## 歷史實機驗收與本次未驗範圍

[A/B 報告](ab-results.zh-TW.md) 是原雙機服務的實測證據，不是本次公開移植在其他硬體上的測試。公開整理只重新跑 Python／補丁來源與資料檢查，沒有再次部署模型、重跑 GB10 CUDA build、重新下載／hash 全部權重或重跑 300K 推論。

公開 supervisor 的 SSH adapter 有本輪 mock 行為驗證；原 pair 狀態機回歸通過，尚未在另一組主機做 OpenSSH 常駐端到端測試。從零建置映像／首次建容器的文件是重現步驟，不能當作本次空機驗收結果。尚未測滿 1M、24h soak、冷 page-cache、hard guard／rank crash 注入與拒答率，原 `/tmp` 的兩個 O_DIRECT 故障注入測試也沒有宣稱修復。
