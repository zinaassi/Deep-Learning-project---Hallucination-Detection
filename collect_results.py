#!/usr/bin/env python3
import os
import re
from collections import defaultdict

def extract_results(out_file, err_file):
    """Extract dataset, model, architecture, and AUC scores from log files."""
    results = {}
    
    # Read .out file for metadata
    if os.path.exists(out_file):
        with open(out_file, 'r') as f:
            content = f.read()
            
            # Extract dataset
            dataset_match = re.search(r'Dataset: (\w+)', content)
            if dataset_match:
                results['dataset'] = dataset_match.group(1)
            
            # Extract model
            model_match = re.search(r'Model: ([\w\-/\.]+)', content)
            if model_match:
                model_full = model_match.group(1)
                if 'Mistral' in model_full:
                    results['model'] = 'Mistral'
                elif 'Llama' in model_full:
                    results['model'] = 'Llama'
                elif 'Qwen' in model_full:
                    results['model'] = 'Qwen'
            
            # Detect architecture from job name (baseline vs regular)
            if 'baseline' in out_file.lower():
                results['architecture'] = 'LOS-Net'
            else:
                results['architecture'] = 'ImprovedLOSNet'
    
    # Read .err file for results (both wandb and AppLogger formats)
    if os.path.exists(err_file):
        with open(err_file, 'r') as f:
            content = f.read()
            
            # Try wandb format first (baseline logs)
            wandb_val_match = re.search(r'best_val_AUC\s+([\d\.]+)', content)
            wandb_test_match = re.search(r'best_test_AUC\s+([\d\.]+)', content)
            
            if wandb_val_match and wandb_test_match:
                results['val_auc'] = float(wandb_val_match.group(1))
                results['test_auc'] = float(wandb_test_match.group(1))
            else:
                # Try AppLogger format (ImprovedLOSNet logs)
                # Find all validation AUCs and take the maximum
                val_aucs = re.findall(r'Val AUC: ([\d\.]+)', content)
                test_aucs = re.findall(r'Test AUC: ([\d\.]+)', content)
                
                if val_aucs and test_aucs:
                    results['val_auc'] = max(float(x) for x in val_aucs)
                    results['test_auc'] = max(float(x) for x in test_aucs)
    
    # Only return if we have the essential fields
    required_fields = ['dataset', 'model', 'architecture', 'val_auc', 'test_auc']
    if all(field in results for field in required_fields):
        return results
    else:
        return None

def main():
    # Collect all results
    all_results = []
    
    # Find all .out files
    log_files = [f for f in os.listdir('logs') if f.startswith('train') and f.endswith('.out')]
    
    print(f"Scanning {len(log_files)} log files...")
    
    for out_file in log_files:
        # Corresponding .err file
        err_file = out_file.replace('.out', '.err')
        
        out_path = os.path.join('logs', out_file)
        err_path = os.path.join('logs', err_file)
        
        results = extract_results(out_path, err_path)
        if results:
            all_results.append(results)
    
    print(f"Successfully extracted {len(all_results)} complete results\n")
    
    if not all_results:
        print("ERROR: No complete results found!")
        return
    
    # Group by dataset-model combination
    grouped = defaultdict(dict)
    for result in all_results:
        key = (result['dataset'], result['model'])
        arch = result['architecture']
        grouped[key][arch] = result
    
    # Print results table
    print("\n" + "="*100)
    print("HALLUCINATION DETECTION RESULTS - ImprovedLOSNet vs Baseline")
    print("="*100)
    print(f"{'Dataset':<12} {'Model':<10} {'Baseline Val':<14} {'Improved Val':<14} {'Δ Val':<10} {'Baseline Test':<15} {'Improved Test':<15} {'Δ Test':<10}")
    print("-"*100)
    
    improvements = []
    
    for (dataset, model), archs in sorted(grouped.items()):
        baseline = archs.get('LOS-Net', {})
        improved = archs.get('ImprovedLOSNet', {})
        
        if baseline and improved:
            base_val = baseline.get('val_auc', 0)
            imp_val = improved.get('val_auc', 0)
            base_test = baseline.get('test_auc', 0)
            imp_test = improved.get('test_auc', 0)
            
            delta_val = imp_val - base_val
            delta_test = imp_test - base_test
            
            improvements.append((dataset, model, delta_test))
            
            print(f"{dataset:<12} {model:<10} {base_val:.4f}{'':<10} {imp_val:.4f}{'':<10} {delta_val:+.4f}{'':<4} "
                  f"{base_test:.4f}{'':<11} {imp_test:.4f}{'':<11} {delta_test:+.4f}")
    
    print("="*100)
    
    if improvements:
        print(f"\nSummary:")
        print(f"  Total combinations tested: {len(improvements)}")
        
        positive_improvements = sum(1 for _, _, delta in improvements if delta > 0)
        print(f"  Improved performance: {positive_improvements}/{len(improvements)}")
        
        avg_improvement = sum(delta for _, _, delta in improvements) / len(improvements)
        print(f"  Average test AUC improvement: {avg_improvement:+.4f}")
        
        # Show best and worst
        improvements.sort(key=lambda x: x[2], reverse=True)
        print(f"\n  Top 3 improvements:")
        for i, (dataset, model, delta) in enumerate(improvements[:3], 1):
            print(f"    {i}. {dataset} + {model}: {delta:+.4f}")
    
    print("\n" + "="*100)

if __name__ == '__main__':
    main()