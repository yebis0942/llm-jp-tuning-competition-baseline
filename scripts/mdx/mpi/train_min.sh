#!/bin/bash
set -eux

# open file limit
ulimit -n 65536 1048576

export PATH=/usr/local/cuda/bin:$PATH

source .venv/bin/activate

# PROJECT_DIR="/model/kodama/tuning_competition2025" # FIXME: Change this to your project directory.
PROJECT_DIR="/home/mdxuser/workspace/tuning-competition-baseline/project_dir"
export TMPDIR=${PROJECT_DIR}/tmp
mkdir -p ${TMPDIR}

# GPU settings
export NUM_NODES=1
# export NUM_GPU_PER_NODE=8
export NUM_GPU_PER_NODE=1
NUM_GPUS=$((NUM_NODES * NUM_GPU_PER_NODE))
echo "NUM_NODES=${NUM_NODES}, NUM_GPU_PER_NODE=${NUM_GPU_PER_NODE}, NUM_GPUS=${NUM_GPUS}"
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0

NAME="sft-"$(tr -dc 0-9A-Za-z < /dev/urandom | fold -w 10 | head -1)
MODEL=llm-jp-3-13b
MODEL_PATH=${PROJECT_DIR}/checkpoints/hf-to-nemo/llm-jp--llm-jp-3-13b  # FIXME: Change this to your model path.

  # -x MASTER_ADDR=$MASTER_ADDR \
  # -x MASTER_PORT=$MASTER_PORT \
# run
python train.py \
  trainer.num_nodes=${NUM_NODES} \
  use_mpi=True \
  name=${NAME} \
  model=${MODEL} \
  model.restore_from_path=${MODEL_PATH}
