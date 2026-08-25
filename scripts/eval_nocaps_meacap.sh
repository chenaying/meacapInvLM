#!/usr/bin/env bash
# MeaCap InvLM batch evaluation on NoCaps (same flow as eval_nocaps.sh).
# Usage (ViECap-compatible, 4 args):
#   bash scripts/eval_nocaps_meacap.sh train_coco 0 '--top_k 3 --threshold 0.2' 14
# Usage (3 args):
#   bash scripts/eval_nocaps_meacap.sh train_coco 0 14

set -e
set -o pipefail
SHELL_FOLDER=$(cd "$(dirname "$0")"; pwd)
cd "${SHELL_FOLDER}/.."

EXP_NAME=${1:-train_coco}
DEVICE=${2:-0}
if [[ $# -ge 4 ]]; then
  EXTRA_ARGS=${3:-}
  EPOCH=${4:-14}
else
  EPOCH=${3:-14}
  EXTRA_ARGS=${4:-}
fi

WEIGHT_PATH="checkpoints/${EXP_NAME}/coco_prefix-$(printf '%04d' "${EPOCH}").pt"
NOCAPS_OUT_PATH="checkpoints/${EXP_NAME}"

TIME_START=$(date "+%Y-%m-%d-%H-%M-%S")
LOG_FOLDER="logs/${EXP_NAME}_EVAL_MEACAP"
mkdir -p "${LOG_FOLDER}"
NOCAPS_LOG_FILE="${LOG_FOLDER}/NOCAPS_MEACAP_${TIME_START}.log"

LANGUAGE_MODEL="${LANGUAGE_MODEL:-/home/teacher5/data1/cyp/project/NewCap/gpt2}"
VL_MODEL="${VL_MODEL:-/home/teacher5/data1/cyp/project/NewCap/checkpoints/clip-vit-base-patch32}"
PARSER_CKPT="${PARSER_CKPT:-/home/teacher5/data1/cyp/project/NewCap/checkpoints/flan-t5-base-VG-factual-sg}"
WTE_MODEL="${WTE_MODEL:-/home/teacher5/data1/cyp/project/NewCap/checkpoints/all-MiniLM-L6-v2}"
MEMORY_ID="${MEMORY_ID:-coco}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

FEATURE_PICKLE="./annotations/nocaps/nocaps_corpus_ViT-B32.pickle"
NOCAPS_IMG_DIR="./annotations/nocaps"
if [[ ! -f "${FEATURE_PICKLE}" ]]; then
  echo "ERROR: missing ${FEATURE_PICKLE}"
  echo "  1) unzip ViECap checkpoints.zip into project root, or"
  echo "  2) python images_features_extraction.py"
  exit 1
fi
echo "Using feature pickle: ${FEATURE_PICKLE} ($(du -h "${FEATURE_PICKLE}" | cut -f1))"

# InvLM memory retrieval needs HF CLIP image vectors. Options:
#   (1) jpg on disk (default)  (2) --nocaps_hf_clip_pickle  (3) --invlm_memory_from_openai_pickle (approx)
if [[ "${EXTRA_ARGS}" != *"invlm_memory_from_openai_pickle"* && "${EXTRA_ARGS}" != *"nocaps_hf_clip_pickle"* ]]; then
  NOCAPS_SAMPLE="${NOCAPS_IMG_DIR}/in_domain/0.jpg"
  if [[ ! -f "${NOCAPS_SAMPLE}" ]]; then
    NOCAPS_SAMPLE="${NOCAPS_IMG_DIR}/in-domain/0.jpg"
  fi
  NOCAPS_JPG_COUNT=$(find "${NOCAPS_IMG_DIR}" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) 2>/dev/null | wc -l)
  if [[ ! -f "${NOCAPS_SAMPLE}" ]]; then
    echo "ERROR: missing ${NOCAPS_IMG_DIR}/in_domain/0.jpg"
    echo "  Found ${NOCAPS_JPG_COUNT} images (need ~4500), or use embedding fallback in EXTRA_ARGS:"
    echo "    '--invlm_memory_from_openai_pickle'   # uses existing ViT-B32.pickle (not paper-faithful)"
    echo "    '--nocaps_hf_clip_pickle ./annotations/nocaps/nocaps_corpus_hf_clip.pickle'"
    exit 1
  fi
  echo "NoCaps images OK (sample: ${NOCAPS_SAMPLE}, total jpg/png: ${NOCAPS_JPG_COUNT})"
else
  echo "NoCaps InvLM: skipping jpg check (using precomputed embeddings from EXTRA_ARGS)"
fi

python validation.py \
  --device "cuda:${DEVICE}" \
  --clip_model ViT-B/32 \
  --language_model "${LANGUAGE_MODEL}" \
  --continuous_prompt_length 10 \
  --clip_project_length 10 \
  --using_image_features \
  --name_of_datasets nocaps \
  --path_of_val_datasets ./annotations/nocaps/nocaps_corpus.json \
  --image_folder ./annotations/nocaps/ \
  --weight_path "${WEIGHT_PATH}" \
  --out_path "${NOCAPS_OUT_PATH}" \
  --using_hard_prompt \
  --soft_prompt_first \
  --use_meacap_invlm \
  --memory_id "${MEMORY_ID}" \
  --memory_caption_num 5 \
  --vl_model "${VL_MODEL}" \
  --parser_checkpoint "${PARSER_CKPT}" \
  --wte_model_path "${WTE_MODEL}" \
  --local_files_only \
  ${EXTRA_ARGS} \
  2>&1 | tee -a "${NOCAPS_LOG_FILE}"

echo "========================== NOCAPS IN-DOMAIN (MeaCap InvLM) =============================="
python evaluation/cocoeval.py --result_file_path "${NOCAPS_OUT_PATH}/indomain"*.json 2>&1 | tee -a "${NOCAPS_LOG_FILE}"
echo "========================== NOCAPS NEAR-DOMAIN (MeaCap InvLM) ============================="
python evaluation/cocoeval.py --result_file_path "${NOCAPS_OUT_PATH}/neardomain"*.json 2>&1 | tee -a "${NOCAPS_LOG_FILE}"
echo "========================== NOCAPS OUT-DOMAIN (MeaCap InvLM) =============================="
python evaluation/cocoeval.py --result_file_path "${NOCAPS_OUT_PATH}/outdomain"*.json 2>&1 | tee -a "${NOCAPS_LOG_FILE}"
echo "========================== NOCAPS OVERALL (MeaCap InvLM) ================================"
python evaluation/cocoeval.py --result_file_path "${NOCAPS_OUT_PATH}/overall"*.json 2>&1 | tee -a "${NOCAPS_LOG_FILE}"
