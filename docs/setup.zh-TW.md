# 安裝與重現

這是一套供熟悉 Docker／RDMA 的操作者重現的配方；不是一鍵安裝器。公開版保留實測的引擎補丁與容器工具，模型需自行準備。原硬體驗收結果見 [A/B 報告](ab-results.zh-TW.md)；本次公開整理沒有重新下載數百 GB 權重或在空機重裝。

## 1. 固定來源，先組合補丁

在專案根目錄，以 Python 3.12 及 Git 執行：

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt
git clone https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks.git upstream
git -C upstream checkout --detach c7f2b85100f0a7b099c92a4e21825e30d93c9c51
python3 tools/assemble-mia-current.py --recipe upstream --output bundle
python3 -m unittest discover -s tests -v
python3 tools/summarize-ab.py --records records/ab
```

`bundle` 必須是新目錄。assembler 拒絕錯 revision、相關 tracked 修改及補丁撞名；輸出官方 53＋本專案 4 支，以及兩支 inputs patches。不要把工作樹的臨時補丁混入建置。

## 2. 建置 aarch64／sm121 引擎

在 GB10 上、沒有其他完整模型佔記憶體時準備初始映像。以下使用上游 Dockerfile，但將兩個 Zig build 都改為單核心；上游 Dockerfile、kit 等授權聲明仍隨映像保留。模型不進 build context。

```bash
mkdir -p build/image-context/zig
cp -r bundle/patches bundle/patches-inputs build/image-context/zig/
python3 - <<'PY'
from pathlib import Path
text = Path('upstream/zig/Dockerfile').read_text()
assert text.count('-j"$(nproc)"') == 2
Path('build/image-context/Dockerfile').write_text(text.replace('-j"$(nproc)"', '-j1'))
PY
INPUTS_REF=$(python3 -c 'import json; print(json.load(open("recipe.lock.json"))["inputs_revision"])')
PATCHES_HASH=$(python3 -c 'import json; print(json.load(open("records/public-manifest.json"))["public_patch_digest"])')
docker build --network host -t local/deepseek-ablit:recipe \
  --build-arg TF_REPO=https://github.com/bertholomus/TensorFold.git \
  --build-arg TF_REF=fbbc97132cd004237d9887b8d3185896efaedc5c \
  --build-arg INPUTS_REF="$INPUTS_REF" \
  --build-arg ZIG_VERSION=0.17.0 \
  --build-arg ZIG_SHA256=9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8 \
  --build-arg PATCHES_HASH="$PATCHES_HASH" build/image-context
```

`nvcr.io/nvidia/pytorch:26.07-py3` 來自上游的 base image 設定；要有適用的驅動、Docker 存取及網路。此從零建置命令本次僅做來源／參數檢查，尚未在空機執行，不能當作本次硬體建置證據。

若已有同一工具鏈的 Mia 原生映像，可用已在原硬體跑過的限資源重編路徑：

```bash
python3 tools/rebuild-mia-zig-engine.py \
  --base-image local/existing-mia-zig:base \
  --patch-dir bundle/patches --inputs-patch-dir bundle/patches-inputs \
  --work-dir build/rebuild --tag local/deepseek-ablit:recipe \
  --memory 3584m --cpus 0 --allow-capabilities-diff
```

`local/existing-mia-zig:base` 是你自己已有的映像，沒有公開 registry 可直接拉。本 helper 只替換引擎／inputs 工具，base 的檔案集合需相容。`--allow-capabilities-diff` 不代替人工核對；原 A→B 唯一新增能力為 `TENSORFOLD_TOOL_STREAM`。保存 image ID、patch digest、capabilities；另一端用 docker save／load 取得同一 immutable image ID，不分別建置兩份不同映像。

## 3. 準備去限制權重與 Engram

版本及覆層 SHA256 以 [recipe.lock.json](../recipe.lock.json) 為準。用自己的 HF 帳號取得 gated overlay 存取；token 只透過本機憑證或環境讀取。

```bash
hf download Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1.1-2.9-bpw \
  --revision aee640e2afba93be683bb5b9b420115fe7fbac20 --local-dir local/stock
hf download drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated \
  --revision d6681ac778ecbd7b422382ff6af88fb3d2debcb7 --local-dir local/overlay \
  --include apply_mia29_wo_b_graft.py exl3_wo_b_l10_35_mia29.safetensors EXL3_MIA29_ABLIT_META.json README.md
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 --local-dir local/engram \
  --include config.json model.safetensors.index.json model-00047-of-00048.safetensors model-00048-of-00048.safetensors
python3 - <<'PY'
from pathlib import Path
import hashlib, json
meta = json.load(open('recipe.lock.json'))['overlay']
p = Path('local/overlay') / meta['file']
assert p.stat().st_size == meta['bytes']
with p.open('rb') as f: assert hashlib.file_digest(f, 'sha256').hexdigest() == meta['sha256']
PY
docker run --rm --network none --entrypoint python3 \
  -v "$PWD/local:/data" local/deepseek-ablit:recipe \
  /data/overlay/apply_mia29_wo_b_graft.py \
  --src /data/stock --dst /data/model-ablit \
  --wo-b /data/overlay/exl3_wo_b_l10_35_mia29.safetensors
```

先閱讀固定版 graft 腳本的條件與輸出。`local/model-ablit` 必須尚未存在，stock 與目的目錄使用同一檔案系統；不要改原始 snapshot。不要加 `--jay-zig`。核對 `ABLIT_META.json` 中 `n_edited=104`、`n_cast_f32=0`、`jay_zig_casts=false`，並逐 tensor 核對 104 個內容與 sidecar 相同，其餘 shape／dtype／內容與 stock 相同。原部署已做過此準備；公開整理沒有重跑完整權重核對。

**不要用官方 `ABLIT=1` 直接準備這個 v1.1 組合**；官方固定版只接受 v1。也不要拿本專案較早的 Python loader sidecar 掃描方式代替原生引擎的 graft／rank cache。

## 4. Runtime artifacts 與雙端設定

依固定 recipe 的 [prepare.sh](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/blob/c7f2b85100f0a7b099c92a4e21825e30d93c9c51/scripts/prepare.sh) 中 kit、token map、rank-cache 步驟準備：

1. kit 固定 `abe07c7d509a0199cc7fec15263be2aa52ac652f`，核對 MANIFEST；針對本 checkpoint 產生 RoPE、vision gate bias 與 token map。vision bias 是 checkpoint 相關資料，不能直接拿其他 pack 的結果。
2. 用本映像內的 `tf-dsv41-rank-cache /model /rank-cache 0 2` 與 `... 1 2`，針對完成的 graft 產生各 rank cache。先清空其他推論工作以保留建置所需記憶體。
3. head 使用完整模型與 rank 0 cache，worker 依上游所需 metadata／vision shards 與 rank 1 cache；兩端 template、tokenizer、Engram、kit 版本相同。將固定 recipe 的 `model/chat_template.jinja` 放入準備後的模型目錄。
4. 兩端 Engram 各讀本機檔案，避免以 NFS 承擔隨機讀取。保存模型／template／rank-cache／image 指紋。
5. 依自己的拓樸實測 RoCE GID index、HCA、interface、`/dev/infiniband` devices；範例的文件用 IP `192.0.2.10`、`rdma0`、`roce0`、GID 3 都是示例，必須替換。

複製 [head](../examples/hostspec-head.json)／[worker](../examples/hostspec-worker.json) 到不納入 Git 的私有設定目錄，填實際絕對路徑及網路參數，保留明確的 48-row、0.5 GiB cache、32 GiB spill、35 GiB free floor。新 spill 目錄需先建立。Head 的 `api_key_file` 是你自行建立的非空 0600 檔案；spec 只記路徑。

候選容器 helper 要求 `--original` 指向保留的舊容器，以保存回退 inspect；適用既有部署升級，**不支援空機首次建立容器**。首次部署請依 `candidate_config()` 產生的 Docker create request 和上游的啟動／裝置契約自行建立，這部分本公開版沒有一鍵工具。

```bash
python3 tools/prepare-mia-zig-container.py \
  --spec private/hostspec-head.json --image local/deepseek-ablit:recipe \
  --original YOUR_PREVIOUS_RANK0 --name mia-zig-ablit-r0 \
  --state-dir private/candidate-head
```

worker 在自己的主機用 rank 1 spec 執行。工具建立未啟動容器並保存私有原始 inspect；不啟停現役。`tf.ref`／`tf.patches` labels 決定實際引擎與 spill identity；範例不硬寫歷史 B 的 digest。

## 5. Guard、常駐與驗收

把固定 recipe 的 `scripts/memguard.sh` 放到每端 `/srv/deepseek/operations`，設 executable；主機時間設定 Asia/Taipei。[pair.json](../examples/pair.json) 的 host 是你自己 `~/.ssh/config` 的 alias，需以 BatchMode 通過 SSH 並有 Docker 操作權限。公開版 supervisor 使用標準 OpenSSH，沒有公司受管 gateway 的依賴。

參照 [systemd user unit](../examples/mia-native-pair.service)，將 `supervise-native-pair.py` 放至 `%h/.local/share/mia-zig-ablit/`，私有 pair config 放至 `%h/.config/mia-zig-ablit/pair.json`。先執行 `systemd-analyze --user verify` 檢查實際 unit，再啟用；需要登出後存活時，依自己的主機政策配置 user lingering。Supervisor 管控兩個 rank，啟動 guard 後先 worker 再 head，不與上游 `start.sh` 同時管理同一對容器。

驗收工具從環境讀 `MODEL_BASE_URL`（例如 `http://127.0.0.1:8012/v1`）與 `MODEL_API_KEY`。連到自己的端點後執行；long／progress 會送約 300K 合成長提示，先確認兩端記憶體及 guard。

```bash
python3 tools/accept-mia-current.py smoke --output private/smoke.json
python3 tools/benchmark-matched.py --label B --output private/benchmark.json
python3 tools/accept-mia-current.py long --output private/long.json
python3 tools/accept-mia-current.py progress --cancel-after 75 --output private/progress.json
# 先以自己的 supervisor 受控重啟兩端，再核對磁碟命中 counter：
python3 tools/accept-mia-current.py resume --output private/resume.json
```

新測試需從空 cache 開始，固定相同 template 與 token 數。要重現 A/B，兩組各受控重載、warmup，再各跑三輪 benchmark；重複提示先確認 cached_tokens=0，不可把快取加速混入普通生成比較。
