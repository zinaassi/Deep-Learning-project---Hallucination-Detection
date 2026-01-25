#!/bin/bash
#SBATCH --job-name=train_baseline
#SBATCH --output=logs/train_baseline_%j.out
#SBATCH --error=logs/train_baseline_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

DATASET=$1
MODEL_SHORTNAME=$2
SEED=${3:-0}

if [ -z "$DATASET" ] || [ -z "$MODEL_SHORTNAME" ]; then
    echo "Error: Usage: sbatch run_train_baseline.sh <dataset> <model> [seed]"
    exit 1
fi

case $MODEL_SHORTNAME in
    mistral) MODEL_FULL="mistralai/Mistral-7B-Instruct-v0.2" ;;
    llama) MODEL_FULL="meta-llama/Meta-Llama-3-8B-Instruct" ;;
    qwen) MODEL_FULL="Qwen/Qwen2.5-7B-Instruct" ;;
    *) echo "Unknown model"; exit 1 ;;
esac

case $DATASET in
    hotpotqa) TEST_DATASET="hotpotqa_test" ;;
    imdb) TEST_DATASET="imdb_test" ;;
    movies) TEST_DATASET="movies_test" ;;
    *) echo "Unknown dataset"; exit 1 ;;
esac

BASE_PRE_PROCESSED_DATA_DIR="/home/zina.assi/Deep-Learning-project---Hallucination-Detection/data/pre_processed_data"

echo "================================================"
echo "BASELINE LOS-NET TRAINING"
echo "================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Dataset: $DATASET"
echo "Model: $MODEL_FULL"
echo "Starting at: $(date)"

eval "$(conda shell.bash hook)"
conda activate LOS_Net
cd ~/Deep-Learning-project---Hallucination-Detection

export WANDB_MODE=offline

python main.py \
  --train_dataset $DATASET \
  --test_dataset $TEST_DATASET \
  --LLM "$MODEL_FULL" \
  --probe_model LOS-Net \
  --base_pre_processed_data_dir $BASE_PRE_PROCESSED_DATA_DIR \
  --hidden_dim 128 \
  --num_layers 2 \
  --heads 8 \
  --batch_size 64 \
  --num_epochs 100 \
  --patience 20 \
  --lr 1e-4 \
  --weight_decay 0.001 \
  --dropout 0.1 \
  --topk_dim 1000 \
  --seed $SEED \
  --num_workers 4 \
  --pin_memory 1

EXIT_CODE=$?
echo "Finished at: $(date) with exit code $EXIT_CODE"
exit $EXIT_CODE