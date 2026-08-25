# 分支说明：`update_2`

本文档说明 GitHub 分支 **`update_2`**（相对 `update1`）的代码改动、实验运行方式，以及与 `update1` 的差异。

- 仓库：https://github.com/chenaying/meacapInvLM  
- 分支：`update_2`  
- 基于：`update1`（commit `d287559`）  
- 本分支新增提交：`0d932bd` — *Add NoCaps image-free InvLM options and prepare scripts*

---

## 1. 本分支解决什么问题

`update1` 已支持 MeaCap InvLM 批量评测，但 **NoCaps InvLM 默认必须读取磁盘原图**（如 `annotations/nocaps/in_domain/0.jpg`）做 HF CLIP 记忆检索。许多环境只有：

- `nocaps_corpus.json`
- `nocaps_corpus_ViT-B32.pickle`（OpenAI CLIP，给 soft prompt 用）

而缺少 ~4500 张 jpg，导致 `FileNotFoundError`。

`update_2` 在保留「有图时走 HF CLIP」的正确路径之外，增加：

1. **无图近似**：用现有 OpenAI ViT-B32 pickle 做记忆检索（调试用，非论文严格复现）  
2. **无图但更正确**：预计算 HF CLIP pickle，之后评测不再读 jpg  
3. **官方 val 图整理**：把 nocaps.org 的 `val/*.jpg` 整理成 ViECap 的 `in_domain/0.jpg` 布局  
4. **启动前检查**：缺图时给出明确提示，避免加载完模型才失败  

---

## 2. 相对 `update1` 的具体修改

### 2.1 修改的文件

| 文件 | 变动要点 |
|------|----------|
| `meacap_utils/invlm_prompt.py` | 记忆检索可走：磁盘图 / HF pickle / OpenAI pickle；NoCaps id→文件名映射；`invlm_skips_disk_images()` |
| `validation.py` | 新增 CLI：`--invlm_memory_from_openai_pickle`、`--nocaps_hf_clip_pickle`、`--nocaps_meta_json`；无图时不调用 `resolve_image_path` |
| `scripts/eval_nocaps_meacap.sh` | 启动前检查 `in_domain/0.jpg`；若 EXTRA_ARGS 含无图选项则跳过 jpg 检查 |

### 2.2 新增的文件

| 文件 | 作用 |
|------|------|
| `scripts/prepare_nocaps_viecap_images.py` | 官方 `val/` + image_info JSON → ViECap `in_domain|near_domain|out_domain/` |
| `scripts/extract_nocaps_hf_clip_embeddings.py` | 有图时一次性导出 `nocaps_corpus_hf_clip.pickle`（HF CLIP） |
| `docs/BRANCH_update_2.md` | 本分支说明（本文档） |

### 2.3 新增命令行参数（`validation.py`）

| 参数 | 含义 |
|------|------|
| `--invlm_memory_from_openai_pickle` | 用 ViECap `*_ViT-B32.pickle` 中的 OpenAI CLIP 特征做记忆检索（**近似**，无 jpg） |
| `--nocaps_hf_clip_pickle PATH` | 用预计算 HF CLIP 向量 pickle 做记忆检索（无 jpg，与记忆库编码器一致） |
| `--nocaps_meta_json PATH` | 官方 NoCaps COCO 格式 json（`images` 字段：id → file_name） |

---

## 3. 与 `update1` 的区别（对照表）

| 能力 | `update1` | `update_2` |
|------|-----------|------------|
| MeaCap InvLM（COCO / Flickr30k / 有图 NoCaps） | ✅ | ✅（同 `update1`） |
| 单图 `viecap_inference.py` | ✅ | ✅ |
| NoCaps **必须**有 `in_domain/0.jpg` 等 | ✅ 默认且无替代 | ✅ 默认；另有无图路径 |
| NoCaps 无图 + OpenAI pickle 近似检索 | ❌ | ✅ `--invlm_memory_from_openai_pickle` |
| NoCaps 无图 + HF CLIP pickle | ❌ | ✅ `--nocaps_hf_clip_pickle` |
| 官方 val → ViECap 目录整理脚本 | ❌ | ✅ `prepare_nocaps_viecap_images.py` |
| 导出 HF CLIP pickle 脚本 | ❌ | ✅ `extract_nocaps_hf_clip_embeddings.py` |
| NoCaps 启动前缺图即退出并提示 fallback | 弱 | ✅ 更明确 |

**结论：**  
- 做 **COCO / Flickr30k InvLM** 或 **已有完整 NoCaps jpg**：`update1` 与 `update_2` 用法相同。  
- 做 **NoCaps 且缺图**：必须用 **`update_2`**（或补齐图片后再用任一分支）。

---

## 4. 如何拉取本分支

```bash
cd ~/project/MeaCap_InvLM
git fetch origin
git checkout update_2
# 或：git pull origin update_2
```

本地模型路径（teacher5 示例，脚本内可被环境变量覆盖）：

```bash
export LANGUAGE_MODEL=/home/teacher5/data1/cyp/project/NewCap/gpt2
export VL_MODEL=/home/teacher5/data1/cyp/project/NewCap/checkpoints/clip-vit-base-patch32
export PARSER_CKPT=/home/teacher5/data1/cyp/project/NewCap/checkpoints/flan-t5-base-VG-factual-sg
export WTE_MODEL=/home/teacher5/data1/cyp/project/NewCap/checkpoints/all-MiniLM-L6-v2
```

---

## 5. 单图推理（与 `update1` 相同）

```bash
conda activate meacap
cd ~/project/MeaCap_InvLM

bash scripts/run_meacap_invlm.sh images/instance1.jpg

# 或手动
python viecap_inference.py \
  --device cuda:0 \
  --memory_id coco \
  --image_path images/instance1.jpg \
  --weight_path checkpoints/train_coco/coco_prefix-0014.pt \
  --using_hard_prompt \
  --soft_prompt_first \
  --local_files_only
```

成功示例：

```text
memory concepts: ['girl', 'bed', ...]
the generated caption: ...
```

ViECap 基线对比：

```bash
python infer_by_instance.py \
  --device cuda:0 \
  --image_path images/instance1.jpg \
  --weight_path checkpoints/train_coco/coco_prefix-0014.pt \
  --using_hard_prompt --soft_prompt_first --prompt_ensemble \
  --name_of_entities_text coco_entities --threshold 0.4
```

---

## 6. 批量实验（评测）

### 6.1 COCO（MeaCap InvLM）

```bash
# 记忆库默认 coco；权重示例：Flickr 训练、在 COCO 上测
bash scripts/eval_coco_meacap.sh train_flickr30k 0 '--using_greedy_search' 29

# 同域更常见写法
bash scripts/eval_coco_meacap.sh train_coco 0 '' 14

# 换记忆库
MEMORY_ID=flickr30k bash scripts/eval_coco_meacap.sh train_flickr30k 0 '' 29
```

需要：`annotations/coco/test_captions_ViT-B32.pickle` + `annotations/coco/val2014/` 原图（InvLM 检索用）。

### 6.2 Flickr30k（MeaCap InvLM）

```bash
bash scripts/eval_flickr30k_meacap.sh train_flickr30k 0 '' 29

# 记忆检索条数改为 3
bash scripts/eval_flickr30k_meacap.sh train_flickr30k 0 '--memory_caption_num 3' 29
# 或：MEMORY_CAPTION_NUM=3 bash scripts/eval_flickr30k_meacap.sh train_flickr30k 0 '' 29
```

需要：`annotations/flickr30k/test_captions_ViT-B32.pickle` + `flickr30k-images/`。

### 6.3 NoCaps（本分支重点）

#### A. 有 ViECap 布局原图（推荐，与论文设定一致）

目录需存在：`annotations/nocaps/in_domain/0.jpg` 等。

```bash
bash scripts/eval_nocaps_meacap.sh train_coco 0 '--top_k 3 --threshold 0.2' 14
```

#### B. 无图：用现有 OpenAI pickle 近似（仅调试）

只需已有 `nocaps_corpus_ViT-B32.pickle`：

```bash
bash scripts/eval_nocaps_meacap.sh train_coco 0 \
  '--top_k 3 --threshold 0.2 --invlm_memory_from_openai_pickle' 14
```

日志会出现警告：OpenAI CLIP 特征与 HF 记忆库不完全同空间，**CIDEr 勿与论文表硬比**。

#### C. 无图但正确：先导出 HF pickle，再评测

**前提：** 至少有一次能访问 NoCaps 原图（本机或同事机器）。

```bash
# 1) 有图时导出（只需跑一次）
python scripts/extract_nocaps_hf_clip_embeddings.py \
  --vl_model /home/teacher5/data1/cyp/project/NewCap/checkpoints/clip-vit-base-patch32 \
  --local_files_only

# 2) 之后无图评测
bash scripts/eval_nocaps_meacap.sh train_coco 0 \
  '--top_k 3 --threshold 0.2 --nocaps_hf_clip_pickle ./annotations/nocaps/nocaps_corpus_hf_clip.pickle' 14
```

#### D. 从官方 `val/` 整理成 ViECap 布局

```bash
# 准备：
#   annotations/nocaps/val/*.jpg
#   annotations/nocaps/nocaps_val_image_info.json  (COCO 格式，含 images 字段)
#   annotations/nocaps/nocaps_corpus.json          (已有)

python scripts/prepare_nocaps_viecap_images.py --dry_run
python scripts/prepare_nocaps_viecap_images.py          # 复制
# 或：python scripts/prepare_nocaps_viecap_images.py --symlink

ls annotations/nocaps/in_domain/0.jpg
bash scripts/eval_nocaps_meacap.sh train_coco 0 '--top_k 3 --threshold 0.2' 14
```

---

## 7. 提示与解码约定（两分支相同）

| 模式 | 参数 |
|------|------|
| 推荐（软+硬，与 ViECap eval 一致） | `--using_hard_prompt --soft_prompt_first` |
| MeaCap 论文倾向（硬+软） | 仅 `--using_hard_prompt` |
| 仅软提示 | 不推荐（易重复乱码） |

- Soft prompt：mapping 网络 ← **OpenAI CLIP** pickle 特征  
- Hard prompt：记忆检索概念 ← **HF CLIP**（原图或 HF pickle；近似模式才用 OpenAI pickle）  
- `top_k` / `threshold`：ViECap 实体分类用；**InvLM 路径基本无效**；检索条数看 `memory_caption_num`

---

## 8. 双 CLIP 说明（读懂 NoCaps 报错的关键）

| 用途 | 编码器 | 数据 |
|------|--------|------|
| Soft prompt（GPT 前缀） | OpenAI CLIP ViT-B/32 | `*_ViT-B32.pickle` |
| Memory 检索 | HF CLIP（与 `memory_clip_embeddings.pt` 一致） | 原图 或 `nocaps_corpus_hf_clip.pickle` |

因此：**有 ViT-B32 pickle ≠ 可以严格做 InvLM 记忆检索**；缺图时要么补图，要么用 `update_2` 的 B/C 路径。

---

## 9. 服务器同步清单

```bash
git fetch origin && git checkout update_2

# 确认新参数存在
grep -n "invlm_memory_from_openai_pickle" validation.py
ls scripts/prepare_nocaps_viecap_images.py scripts/extract_nocaps_hf_clip_embeddings.py
```

更完整的 InvLM 复现背景见：`docs/MeaCapInvLM_复现.md`。
