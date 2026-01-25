#!/bin/bash

# Pre-flight check script
# Run this before submitting jobs to verify everything is set up correctly

echo "================================================"
echo "HPC SETUP VERIFICATION"
echo "================================================"
echo ""

# Track issues
ISSUES=0

# 1. Check conda environment
echo "1. Checking conda environment..."
if command -v conda &> /dev/null; then
    echo "   ✓ Conda found: $(which conda)"
else
    echo "   ✗ ERROR: Conda not found"
    ISSUES=$((ISSUES+1))
fi

# 2. Check if LOS_Net environment exists
if conda env list | grep -q "LOS_Net"; then
    echo "   ✓ LOS_Net environment exists"
else
    echo "   ✗ ERROR: LOS_Net environment not found"
    echo "     Create it with: conda env create -f los_net_env.yml"
    ISSUES=$((ISSUES+1))
fi

# 3. Check CUDA module
echo ""
echo "2. Checking CUDA availability..."
if module avail cuda 2>&1 | grep -q "cuda"; then
    echo "   ✓ CUDA module available"
else
    echo "   ⚠ Warning: CUDA module may not be available"
    echo "     Check with: module avail cuda"
fi

# 4. Check Python packages (requires activating environment)
echo ""
echo "3. Checking Python packages..."
source activate LOS_Net 2>/dev/null
if [ $? -eq 0 ]; then
    echo "   ✓ Can activate LOS_Net environment"
    
    # Check key packages
    python -c "import torch" 2>/dev/null && echo "   ✓ PyTorch installed" || { echo "   ✗ ERROR: PyTorch not found"; ISSUES=$((ISSUES+1)); }
    python -c "import transformers" 2>/dev/null && echo "   ✓ Transformers installed" || { echo "   ✗ ERROR: Transformers not found"; ISSUES=$((ISSUES+1)); }
else
    echo "   ✗ ERROR: Cannot activate LOS_Net environment"
    ISSUES=$((ISSUES+1))
fi

# 5. Check LOS-Net code
echo ""
echo "4. Checking LOS-Net codebase..."
if [ -f "create_HD_datasets.py" ]; then
    echo "   ✓ create_HD_datasets.py found"
else
    echo "   ✗ ERROR: create_HD_datasets.py not found"
    echo "     Are you in the LOS-Net directory?"
    ISSUES=$((ISSUES+1))
fi

if [ -f "preprocess_datasets.py" ]; then
    echo "   ✓ preprocess_datasets.py found"
else
    echo "   ✗ ERROR: preprocess_datasets.py not found"
    ISSUES=$((ISSUES+1))
fi

if [ -f "main.py" ]; then
    echo "   ✓ main.py found"
else
    echo "   ✗ ERROR: main.py not found"
    ISSUES=$((ISSUES+1))
fi

# 6. Check ImprovedLOSNet implementation
echo ""
echo "5. Checking ImprovedLOSNet implementation..."
if grep -q "class ImprovedLOSNet" utils/Architectures.py 2>/dev/null; then
    echo "   ✓ ImprovedLOSNet class found in utils/Architectures.py"
else
    echo "   ✗ ERROR: ImprovedLOSNet class not found"
    echo "     Did you implement the enhanced features?"
    ISSUES=$((ISSUES+1))
fi

# 7. Check scripts are executable
echo ""
echo "6. Checking script permissions..."
for script in run_all_hd_generation.sh run_all_hd_preprocess.sh; do
    if [ -x "$script" ]; then
        echo "   ✓ $script is executable"
    else
        echo "   ⚠ Warning: $script is not executable"
        echo "     Run: chmod +x $script"
    fi
done

# 8. Check logs directory
echo ""
echo "7. Checking directories..."
if [ -d "logs" ]; then
    echo "   ✓ logs/ directory exists"
else
    echo "   ⚠ Warning: logs/ directory doesn't exist"
    echo "     It will be created automatically by the scripts"
fi

# 9. Check storage paths
echo ""
echo "8. Checking storage paths..."
echo "   Note: Edit these paths in the scripts before running!"
echo ""
echo "   In run_single_generation_job.sh:"
grep "BASE_RAW_DATA_DIR=" run_single_generation_job.sh 2>/dev/null | head -1 || echo "   ✗ Script not found"
echo ""
echo "   In run_single_preprocess_job.sh:"
grep "BASE_RAW_DATA_DIR=" run_single_preprocess_job.sh 2>/dev/null | head -1 || echo "   ✗ Script not found"
grep "BASE_PRE_PROCESSED_DATA_DIR=" run_single_preprocess_job.sh 2>/dev/null | head -1 || echo "   ✗ Script not found"
echo ""
echo "   In run_train_improved.sh:"
grep "BASE_PRE_PROCESSED_DATA_DIR=" run_train_improved.sh 2>/dev/null | head -1 || echo "   ✗ Script not found"

# 10. Check available storage
echo ""
echo "9. Checking available storage..."
echo "   Current directory: $(pwd)"
df -h . | tail -1 | awk '{print "   Available space: " $4}'

# 11. Test SLURM
echo ""
echo "10. Checking SLURM..."
if command -v sbatch &> /dev/null; then
    echo "   ✓ sbatch command found"
    echo "   ✓ Can submit jobs"
else
    echo "   ✗ ERROR: sbatch not found"
    echo "     Are you on a SLURM cluster?"
    ISSUES=$((ISSUES+1))
fi

if command -v squeue &> /dev/null; then
    echo "   ✓ squeue command found"
    CURRENT_JOBS=$(squeue -u $USER | wc -l)
    echo "   Current jobs in queue: $((CURRENT_JOBS-1))"
else
    echo "   ✗ ERROR: squeue not found"
    ISSUES=$((ISSUES+1))
fi

# Summary
echo ""
echo "================================================"
echo "VERIFICATION SUMMARY"
echo "================================================"

if [ $ISSUES -eq 0 ]; then
    echo "✓ ALL CHECKS PASSED!"
    echo ""
    echo "You're ready to submit jobs!"
    echo ""
    echo "Next steps:"
    echo "  1. Edit paths in scripts (see section 8 above)"
    echo "  2. Run: ./run_all_hd_generation.sh"
    echo "  3. Monitor: watch -n 30 'squeue -u \$USER'"
else
    echo "✗ FOUND $ISSUES ISSUE(S)"
    echo ""
    echo "Please fix the issues above before submitting jobs."
fi

echo ""
echo "For detailed instructions, see: HPC_SCRIPTS_README.md"
echo "================================================"
