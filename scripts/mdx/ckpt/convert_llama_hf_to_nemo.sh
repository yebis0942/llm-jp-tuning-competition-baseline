#!/bin/bash
set -eux

# open file limit
ulimit -n 65536 1048576

# source .venv/bin/activate

# PROJECT_DIR="/model/kodama/tuning_competition2025" # FIXME: Change this to your project directory.
PROJECT_DIR="/home/mdxuser/workspace/tuning-competition-baseline/project_dir"
export TMPDIR=${PROJECT_DIR}/tmp

# For 1.8B model
# INPUT_NAME_OR_PATH="llm-jp/llm-jp-3-1.8b"
# OUTPUT_PATH=${PROJECT_DIR}/checkpoints/hf-to-nemo/llm-jp--llm-jp-3-1.8b
# HPARAMS_FILE=./megatron_configs/llm-jp/llm-jp-3-1.8b.yaml

# For 3.7B model
# INPUT_NAME_OR_PATH="llm-jp/llm-jp-3-3.7b"
# OUTPUT_PATH=${PROJECT_DIR}/checkpoints/hf-to-nemo/llm-jp--llm-jp-3-3.7b
# HPARAMS_FILE=./megatron_configs/llm-jp/llm-jp-3-3.7b.yaml

# For 13B model
INPUT_NAME_OR_PATH="llm-jp/llm-jp-3-13b"
OUTPUT_PATH=${PROJECT_DIR}/checkpoints/hf-to-nemo/llm-jp--llm-jp-3-13b
HPARAMS_FILE=./megatron_configs/llm-jp/llm-jp-3-13b.yaml

# run
uv run python scripts/ckpt/convert_llama_hf_to_nemo.py \
  --hparams-file ${HPARAMS_FILE} \
  --input-name-or-path ${INPUT_NAME_OR_PATH} \
  --output-path ${OUTPUT_PATH} \
  # --n-jobs 96
  # --cpu-only \

echo "Extracting the Nemo checkpoint to ${OUTPUT_PATH}"
mkdir -p "${OUTPUT_PATH}"
tar -xvf "${OUTPUT_PATH}.nemo" -C "${OUTPUT_PATH}"

if [ -f "${OUTPUT_PATH}/model_config.yaml" ] && [ -d "${OUTPUT_PATH}/model_weights" ]; then
  echo "Successfully converted the checkpoint to Nemo format. Removing the nemo file."
  rm "${OUTPUT_PATH}.nemo"
fi
