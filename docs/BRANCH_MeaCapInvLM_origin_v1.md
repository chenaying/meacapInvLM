# 分支说明：`MeaCapInvLM_origin_v1`

本分支为 **当前工作区实际磁盘状态** 的快照上传（与 `update_2` 可能差异很大，以本分支为准）。

## 包含内容

- 根目录 ViECap / MeaCap InvLM 源码（`validation.py`、`viecap_inference.py`、`ClipCap.py` 等）
- `meacap_utils/`、`models/`、`scripts/`（以当前目录为准）
- `evaluation/` 评测代码（`cocoeval.py`、`pycocoevalcap` 脚本与小依赖）
- `requirements.txt`、`README.md`、`images/` 等

## 未上传（体积过大或 GitHub 单文件 100MB 限制）

| 路径 | 原因 |
|------|------|
| `gpt2/` | ~5.6GB 模型权重 |
| `annotations/`、`checkpoints/`、`data/` | 数据与权重 |
| `*.pickle` / `*.zip` | 特征与压缩包 |
| `evaluation/.../stanford-corenlp-*-models.jar` | 单文件 >100MB |
| `evaluation/.../spice/cache/`、`meteor/data/` | 缓存/语料，可本地再下 |

评测大文件可按 `evaluation/get_stanford_models.sh` 或 ViECap / MeaCap 原仓库说明补齐。

## 相对 `update_2` 的主要差异（工作区现状）

- 删除：`docs/`、`scripts/eval_coco_meacap.sh`、`scripts/extract_nocaps_hf_clip_embeddings.py`、`scripts/prepare_nocaps_viecap_images.py` 等（磁盘上已不存在）
- 修改：`validation.py`、`meacap_utils/invlm_prompt.py`、`scripts/eval_*_meacap.sh`、`requirements.txt` 等
- 新增纳入仓库：`evaluation/` 评测代码树（不含超大 jar/缓存）
