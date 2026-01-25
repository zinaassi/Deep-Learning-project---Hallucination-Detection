#!/bin/bash
#SBATCH --job-name=train_improved
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Single ImprovedLOSNet training job
# Usage: sbatch run_train_improved.sh <dataset> <model> [seed]
# Example: sbatch run_train_improved.sh hotpotqa mistral 0

DATASET=$1
MODEL_SHORTNAME=$2
SEED=${3:-0}  # Default seed is 0

if [ -z "$DATASET" ] || [ -z "$MODEL_SHORTNAME" ]; then
    echo "Error: Missing arguments"
    echo "Usage: sbatch run_train_improved.sh <dataset> <model> [seed]"
    echo "  dataset: hotpotqa, imdb, movies"
    echo "  model: mistral, llama, qwen"
    echo "  seed: random seed (optional, default=0)"
    exit 1
fi

# Map short names to full model names
case $MODEL_SHORTNAME in
    mistral)
        MODEL_FULL="mistralai/Mistral-7B-Instruct-v0.2"
        ;;
    llama)
        MODEL_FULL="meta-llama/Meta-Llama-3-8B-Instruct"
        ;;
    qwen)
        MODEL_FULL="Qwen/Qwen2.5-7B-Instruct"
        ;;
    *)
        echo "Error: Unknown model shortname: $MODEL_SHORTNAME"
        echo "Use: mistral, llama, or qwen"
        exit 1
        ;;
esac

# Directories (CHANGE THESE to your paths!)
BASE_PRE_PROCESSED_DATA_DIR="/home/zina.assi/Deep-Learning-project---Hallucination-Detection/data/pre_processed_data"

# Determine test dataset name
case $DATASET in
    hotpotqa)
        TEST_DATASET="hotpotqa_test"
        ;;
    imdb)
        TEST_DATASET="imdb_test"
        ;;
    movies)
        TEST_DATASET="movies_test"
        ;;
    *)
        echo "Error: Unknown dataset: $DATASET"
        exit 1
        ;;
esac

echo "================================================"
echo "IMPROVED LOS-NET TRAINING"
echo "================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Dataset: $DATASET (test: $TEST_DATASET)"
echo "Model: $MODEL_FULL"
echo "Seed: $SEED"
echo "Data dir: $BASE_PRE_PROCESSED_DATA_DIR/$MODEL_FULL/$DATASET/"
echo "Starting at: $(date)"
echo ""

# Load environment
eval "$(conda shell.bash hook)"
conda activate LOS_Net

cd /home/zina.assi/Deep-Learning-project---Hallucination-Detection  # CHANGE THIS to your LOS-Net directory

echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

# Verify preprocessed data exists
DATA_DIR="$BASE_PRE_PROCESSED_DATA_DIR/$MODEL_FULL/$DATASET"
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Preprocessed data not found: $DATA_DIR"
    echo "Please run preprocessing first!"
    exit 1
fi

NUM_FILES=$(ls -1 $DATA_DIR/TDS_topk_*.pt 2>/dev/null | wc -l)
echo "Found $NUM_FILES preprocessed samples"
echo ""

echo "================================================"
echo "TRAINING CONFIGURATION"
echo "================================================"
echo "Architecture: ImprovedLOSNet"
echo "Features:"
echo "  - Rank embeddings: ON"
echo "  - Entropy features: ON"
echo "  - Probability gaps: ON"
echo ""
echo "Hyperparameters:"
echo "  - Hidden dim: 128"
echo "  - Layers: 2"
echo "  - Attention heads: 8"
echo "  - Batch size: 64"
echo "  - Epochs: 100 (with early stopping)"
echo "  - Learning rate: 1e-4"
echo ""

echo "================================================"
echo "STARTING TRAINING"
echo "================================================"

export WANDB_MODE=disabled
# Run training
python main.py \
  --train_dataset $DATASET \
  --test_dataset $TEST_DATASET \
  --LLM "$MODEL_FULL" \
  --probe_model ImprovedLOSNet \
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
  --rank_encoding scale_encoding \
  --use_rank_embed \
  --use_entropy \
  --use_gaps \
  --topk_dim 1000 \
  --seed $SEED \
  --num_workers 4 \
  --pin_memory 1

EXIT_CODE=$?

echo ""
echo "================================================"
echo "TRAINING COMPLETE"
echo "================================================"
echo "Exit code: $EXIT_CODE"
echo "Finished at: $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Success: $DATASET with $MODEL_SHORTNAME (seed $SEED)"
    echo ""
    echo "Results saved to: saved_models/"
    echo "Check validation AUC in the output above"
else
    echo "✗ Failed: $DATASET with $MODEL_SHORTNAME (exit code $EXIT_CODE)"
fi

exit $EXIT_CODE
