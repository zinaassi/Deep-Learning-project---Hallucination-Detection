#!/bin/bash

mkdir -p logs

echo "Submitting TEST dataset generation jobs..."
echo ""

# All 9 test dataset combinations
test_experiments=(
    "hotpotqa_test:mistral"
    "hotpotqa_test:qwen"
    "imdb_test:mistral"
    "imdb_test:qwen"
    "movies_test:mistral"
    "movies_test:qwen"
)

# Submit all test generation jobs
for exp in "${test_experiments[@]}"; do
    IFS=':' read -r dataset model <<< "$exp"
    JOB_ID=$(sbatch --parsable run_single_generation_job.sh "$dataset" "$model" 1)
    echo "Submitted $dataset + $model - Job ID: $JOB_ID"
done

echo ""
echo "All 9 test dataset generation jobs submitted!"
echo "After they complete, run preprocessing with: ./submit_all_test_preprocess.sh"