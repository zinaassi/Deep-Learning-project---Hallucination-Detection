#!/bin/bash
#SBATCH --job-name=preprocess_hd
#SBATCH --output=logs/preprocess_%j.out
#SBATCH --error=logs/preprocess_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2

# Single HD data preprocessing job
# Usage: sbatch run_single_preprocess_job.sh <dataset> <model>
# Example: sbatch run_single_preprocess_job.sh hotpotqa mistral

DATASET=$1
MODEL_SHORTNAME=$2

if [ -z "$DATASET" ] || [ -z "$MODEL_SHORTNAME" ]; then
    echo "Error: Missing arguments"
    echo "Usage: sbatch run_single_preprocess_job.sh <dataset> <model>"
    echo "  dataset: hotpotqa, imdb, movies"
    echo "  model: mistral, llama, qwen"
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
BASE_RAW_DATA_DIR="/home/zina.assi/Deep-Learning-project---Hallucination-Detection/data/raw_data"
BASE_PRE_PROCESSED_DATA_DIR="/home/zina.assi/Deep-Learning-project---Hallucination-Detection/data/pre_processed_data"

echo "================================================"
echo "HD DATA PREPROCESSING"
echo "================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Dataset: $DATASET"
echo "Model: $MODEL_FULL"
echo "Input dir: $BASE_RAW_DATA_DIR/$MODEL_FULL/$DATASET/"
echo "Output dir: $BASE_PRE_PROCESSED_DATA_DIR/$MODEL_FULL/$DATASET/"
echo "Starting at: $(date)"
echo ""

# Load environment
eval "$(conda shell.bash hook)"
conda activate LOS_Net

cd /home/zina.assi/Deep-Learning-project---Hallucination-Detection  # CHANGE THIS to your LOS-Net directory

echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "CPU cores: $SLURM_CPUS_PER_TASK"
echo ""

# Verify raw data exists
RAW_DIR="$BASE_RAW_DATA_DIR/$MODEL_FULL/$DATASET"
if [ ! -d "$RAW_DIR" ]; then
    echo "ERROR: Raw data directory not found: $RAW_DIR"
    echo "Please run data generation first!"
    exit 1
fi

NUM_RAW_FILES=$(ls -1 $RAW_DIR/probs_output_*.pt 2>/dev/null | wc -l)
echo "Found $NUM_RAW_FILES raw data files to preprocess"
echo ""

echo "================================================"
echo "STARTING PREPROCESSING"
echo "================================================"

# Run preprocessing
python preprocess_datasets.py \
  --LLM "$MODEL_FULL" \
  --dataset $DATASET \
  --base_raw_data_dir $BASE_RAW_DATA_DIR \
  --base_pre_processed_data_dir $BASE_PRE_PROCESSED_DATA_DIR \
  --topk_preprocess 1000 \
  --input_output_type output \
  --N_max 100 \
  --input_type LOS

EXIT_CODE=$?

echo ""
echo "================================================"
echo "PREPROCESSING COMPLETE"
echo "================================================"
echo "Exit code: $EXIT_CODE"
echo "Finished at: $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    # Count preprocessed files
    OUTPUT_DIR="$BASE_PRE_PROCESSED_DATA_DIR/$MODEL_FULL/$DATASET"
    NUM_PROCESSED=$(ls -1 $OUTPUT_DIR/sorted_TDS_*.pt 2>/dev/null | wc -l)
    echo "✓ Success: Preprocessed $NUM_PROCESSED samples for $DATASET with $MODEL_SHORTNAME"
    echo "  Location: $OUTPUT_DIR"
    
    # Show file sizes
    echo ""
    echo "Storage usage:"
    du -sh $OUTPUT_DIR
else
    echo "✗ Failed: $DATASET with $MODEL_SHORTNAME (exit code $EXIT_CODE)"
fi

exit $EXIT_CODE
