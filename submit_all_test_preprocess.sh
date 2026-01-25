#!/bin/bash

echo "Submitting TEST dataset preprocessing jobs..."

test_experiments=(
    "hotpotqa_test:mistral"
    "hotpotqa_test:qwen"
    "imdb_test:mistral"
    "imdb_test:qwen"
    "movies_test:mistral"
    "movies_test:qwen"
)

for exp in "${test_experiments[@]}"; do
    IFS=':' read -r dataset model <<< "$exp"
    JOB_ID=$(sbatch --parsable run_single_preprocess_job.sh "$dataset" "$model")
    echo "Submitted preprocessing: $dataset + $model - Job ID: $JOB_ID"
done

echo ""
echo "All 9 test preprocessing jobs submitted!"