#!/bin/bash

# Submit HD data generation jobs in batches
# 3 datasets × 3 models = 9 total jobs
# Run 3 jobs at a time using dependencies

mkdir -p logs

# All 9 experiments (3 datasets × 3 models)
# Format: "dataset:model_shortname"
experiments=(
    "hotpotqa:mistral"
    "hotpotqa:llama"
    "hotpotqa:qwen"
    "imdb:mistral"
    "imdb:llama"
    "imdb:qwen"
    "movies:mistral"
    "movies:llama"
    "movies:qwen"
)

echo "================================================"
echo "HD DATA GENERATION - BATCHED SUBMISSION"
echo "================================================"
echo "Total experiments: ${#experiments[@]}"
echo "Batch size: 3 jobs at a time"
echo "Datasets: hotpotqa, imdb, movies"
echo "Models: Mistral-7B, Llama-3-8B, Qwen-2.5-7B"
echo ""
echo "NOTE: Currently set to generate 1000 samples per dataset"
echo "      Edit run_single_generation_job.sh to remove N_SAMPLES"
echo "      for full dataset generation"
echo ""

# Batch 1: Submit first 3 jobs (no dependencies)
# echo "Submitting Batch 1 (3 jobs in parallel)..."
# BATCH1_IDS=()
# for i in 0 1 2; do
#     IFS=':' read -r dataset model <<< "${experiments[$i]}"
#     JOB_ID=$(sbatch --parsable run_single_generation_job.sh "$dataset" "$model")
#     BATCH1_IDS+=($JOB_ID)
#     echo "  Submitted $dataset + $model - Job ID: $JOB_ID"
# done

# Build dependency string for batch 1
# BATCH1_DEPS=$(IFS=:; echo "${BATCH1_IDS[*]}")

# Batch 2: Submit next 3 jobs (depend on batch 1)
echo ""
echo "Submitting Batch 2 (3 jobs, depends on Batch 1)..."
BATCH2_IDS=()
for i in 3 4 5; do
    IFS=':' read -r dataset model <<< "${experiments[$i]}"
    JOB_ID=$(sbatch --parsable run_single_generation_job.sh "$dataset" "$model")
    BATCH2_IDS+=($JOB_ID)
    echo "  Submitted $dataset + $model - Job ID: $JOB_ID"
done

# Build dependency string for batch 2
BATCH2_DEPS=$(IFS=:; echo "${BATCH2_IDS[*]}")

# Batch 3: Submit final 3 jobs (depend on batch 2)
echo ""
echo "Submitting Batch 3 (3 jobs, depends on Batch 2)..."
BATCH3_IDS=()
for i in 6 7 8; do
    IFS=':' read -r dataset model <<< "${experiments[$i]}"
    JOB_ID=$(sbatch --parsable --dependency=afterok:$BATCH2_DEPS run_single_generation_job.sh "$dataset" "$model")
    BATCH3_IDS+=($JOB_ID)
    echo "  Submitted $dataset + $model - Job ID: $JOB_ID"
done

echo ""
echo "================================================"
echo "ALL 9 DATA GENERATION JOBS SUBMITTED!"
echo "================================================"
echo "Batch 1: Jobs ${BATCH1_IDS[@]} (running now)"
echo "Batch 2: Jobs ${BATCH2_IDS[@]} (waits for Batch 1)"
echo "Batch 3: Jobs ${BATCH3_IDS[@]} (waits for Batch 2)"
echo ""
echo "Monitor with: watch -n 30 'squeue -u \$USER'"
echo "View logs: tail -f logs/gen_hd_*.out"
echo ""
echo "Expected completion time:"
echo "  - Batch 1: ~6-8 hours (3 jobs × ~2-3 hours each)"
echo "  - Batch 2: starts after Batch 1 completes"
echo "  - Batch 3: starts after Batch 2 completes"
echo "  - Total: ~6-8 hours (running 3 in parallel)"
echo ""
echo "Storage usage per job: ~5-10GB"
echo "Total storage needed: ~50-100GB"
echo ""
echo "After completion, run: ./run_all_hd_preprocess.sh"
