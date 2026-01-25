#!/usr/bin/env python3
"""
Test script for ImprovedLOSNet architecture.
Verifies that the model can be instantiated and runs forward pass correctly.
"""

import torch
import argparse
from utils.Architectures import ImprovedLOSNet, LOS_Net

def create_test_args():
    """Create minimal args for testing."""
    args = argparse.Namespace()

    # Model architecture args
    args.hidden_dim = 128
    args.heads = 8
    args.dropout = 0.3
    args.num_layers = 2
    args.pool = 'cls'
    args.topk_dim = 1000  # K = top-1000 tokens
    args.rank_encoding = 'scale_encoding'

    # ImprovedLOSNet specific args
    args.rank_embed_dim = 128
    args.use_rank_embed = True
    args.use_entropy = True
    args.use_gaps = True

    # Other required args
    args.LLM = 'mistralai/Mistral-7B-Instruct-v0.2'

    return args


def test_baseline_model():
    """Test baseline LOS_Net to ensure it still works."""
    print("\n" + "="*80)
    print("Testing Baseline LOS_Net")
    print("="*80)

    args = create_test_args()
    max_sequence_length = 100
    input_dim = 1000  # K dimension

    # Create model
    model = LOS_Net(args=args, max_sequence_length=max_sequence_length, input_dim=input_dim)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create dummy input
    B, N, K = 4, 50, 1000  # batch=4, seq_len=50, top_k=1000
    sorted_TDS_normalized = torch.randn(B, N, K)
    normalized_ATP = torch.randn(B, N, 1)
    ATP_R = torch.randint(0, 32000, (B, N))

    # Forward pass
    print(f"\nInput shapes:")
    print(f"  sorted_TDS_normalized: {sorted_TDS_normalized.shape}")
    print(f"  normalized_ATP: {normalized_ATP.shape}")
    print(f"  ATP_R: {ATP_R.shape}")

    output = model(sorted_TDS_normalized, normalized_ATP, ATP_R)
    print(f"\nOutput shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")

    assert output.shape == (B,), f"Expected output shape (4,), got {output.shape}"
    assert (output >= 0).all() and (output <= 1).all(), "Output should be in [0, 1] due to sigmoid"

    print("\n✓ Baseline LOS_Net test PASSED!")
    return total_params


def test_improved_model(baseline_params):
    """Test ImprovedLOSNet."""
    print("\n" + "="*80)
    print("Testing ImprovedLOSNet")
    print("="*80)

    args = create_test_args()
    max_sequence_length = 100
    input_dim = 1000

    # Create model
    model = ImprovedLOSNet(args=args, max_sequence_length=max_sequence_length, input_dim=input_dim)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Parameter increase vs baseline: +{total_params - baseline_params:,} ({(total_params/baseline_params - 1)*100:.1f}%)")

    # Create dummy input
    B, N, K = 4, 50, 1000
    sorted_TDS_normalized = torch.randn(B, N, K)
    normalized_ATP = torch.randn(B, N, 1)
    ATP_R = torch.randint(0, 32000, (B, N))

    # Forward pass
    print(f"\nInput shapes:")
    print(f"  sorted_TDS_normalized: {sorted_TDS_normalized.shape}")
    print(f"  normalized_ATP: {normalized_ATP.shape}")
    print(f"  ATP_R: {ATP_R.shape}")

    output = model(sorted_TDS_normalized, normalized_ATP, ATP_R)
    print(f"\nOutput shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")

    assert output.shape == (B,), f"Expected output shape (4,), got {output.shape}"
    assert (output >= 0).all() and (output <= 1).all(), "Output should be in [0, 1] due to sigmoid"

    # Test individual feature computation
    print("\n" + "-"*80)
    print("Testing Individual Feature Components")
    print("-"*80)

    # Test entropy computation
    entropy = model.compute_entropy(sorted_TDS_normalized)
    print(f"Entropy shape: {entropy.shape} (expected: [B={B}, N={N}, 1])")
    print(f"Entropy range: [{entropy.min():.4f}, {entropy.max():.4f}]")
    assert entropy.shape == (B, N, 1), f"Entropy shape mismatch"

    # Test probability gaps
    gaps = model.compute_prob_gaps(sorted_TDS_normalized)
    print(f"Gaps shape: {gaps.shape} (expected: [B={B}, N={N}, 2])")
    print(f"Gap1 range: [{gaps[..., 0].min():.4f}, {gaps[..., 0].max():.4f}]")
    print(f"Gap2 range: [{gaps[..., 1].min():.4f}, {gaps[..., 1].max():.4f}]")
    assert gaps.shape == (B, N, 2), f"Gaps shape mismatch"

    print("\n✓ ImprovedLOSNet test PASSED!")
    return total_params


def test_ablation_configurations():
    """Test different feature configurations."""
    print("\n" + "="*80)
    print("Testing Feature Ablation Configurations")
    print("="*80)

    configs = [
        {"use_rank_embed": True, "use_entropy": False, "use_gaps": False, "name": "Rank Embeddings Only"},
        {"use_rank_embed": False, "use_entropy": True, "use_gaps": False, "name": "Entropy Only"},
        {"use_rank_embed": False, "use_entropy": False, "use_gaps": True, "name": "Gaps Only"},
        {"use_rank_embed": True, "use_entropy": True, "use_gaps": True, "name": "All Features"},
    ]

    max_sequence_length = 100
    input_dim = 1000
    B, N, K = 2, 30, 1000

    for config in configs:
        args = create_test_args()
        args.use_rank_embed = config["use_rank_embed"]
        args.use_entropy = config["use_entropy"]
        args.use_gaps = config["use_gaps"]

        model = ImprovedLOSNet(args=args, max_sequence_length=max_sequence_length, input_dim=input_dim)
        total_params = sum(p.numel() for p in model.parameters())

        # Test forward pass
        sorted_TDS_normalized = torch.randn(B, N, K)
        normalized_ATP = torch.randn(B, N, 1)
        ATP_R = torch.randint(0, 32000, (B, N))

        output = model(sorted_TDS_normalized, normalized_ATP, ATP_R)

        print(f"{config['name']:<25} | Params: {total_params:>8,} | Output shape: {output.shape}")

    print("\n✓ All ablation configurations work correctly!")

export WANDB_MODE=disabled

def main():
    print("\n" + "="*80)
    print("ImprovedLOSNet Architecture Test Suite")
    print("="*80)

    # Test baseline
    baseline_params = test_baseline_model()

    # Test improved model
    improved_params = test_improved_model(baseline_params)

    # Test ablation configurations
    test_ablation_configurations()

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Baseline LOS_Net:     {baseline_params:>8,} parameters")
    print(f"ImprovedLOSNet:       {improved_params:>8,} parameters")
    print(f"Increase:             +{improved_params - baseline_params:>7,} ({(improved_params/baseline_params - 1)*100:.1f}%)")
    print("\n✅ All tests PASSED! Model is ready for training.")


if __name__ == '__main__':
    main()
