#!/bin/bash
#SBATCH --job-name=gen_hd_data
#SBATCH --output=logs/gen_hd_%j.out
#SBATCH --error=logs/gen_hd_%j.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

DATASET=$1
MODEL_SHORTNAME=$2

if [ -z "$DATASET" ] || [ -z "$MODEL_SHORTNAME" ]; then
    echo "Error: Missing arguments"
    exit 1
fi

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
        exit 1
        ;;
esac

BASE_RAW_DATA_DIR="/home/zina.assi/Deep-Learning-project---Hallucination-Detection/data/raw_data"
N_SAMPLES=1000

echo "================================================"
echo "HD DATA GENERATION"
echo "================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Dataset: $DATASET"
echo "Model: $MODEL_FULL"
echo "Samples: $N_SAMPLES"
echo "Starting at: $(date)"
echo ""

# Load environment - FIXED
eval "$(conda shell.bash hook)"
conda activate LOS_Net

cd ~/Deep-Learning-project---Hallucination-Detection

echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo ""

echo "================================================"
echo "STARTING DATA GENERATION"
echo "================================================"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python create_HD_datasets.py \
  --dataset $DATASET \
  --LLM $MODEL_FULL \
  --base_raw_data_dir $BASE_RAW_DATA_DIR \
  --n_samples $N_SAMPLES \
  --chunk 1

EXIT_CODE=$?

echo ""
echo "================================================"
echo "DATA GENERATION COMPLETE"
echo "================================================"
echo "Exit code: $EXIT_CODE"
echo "Finished at: $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    OUTPUT_DIR="$BASE_RAW_DATA_DIR/$MODEL_FULL/$DATASET"
    NUM_FILES=$(ls -1 $OUTPUT_DIR/probs_output_*.pt 2>/dev/null | wc -l)
    echo "✓ Success: Generated $NUM_FILES samples for $DATASET with $MODEL_SHORTNAME"
    echo "  Location: $OUTPUT_DIR"
else
    echo "✗ Failed: $DATASET with $MODEL_SHORTNAME (exit code $EXIT_CODE)"
fi

exit $EXIT_CODE