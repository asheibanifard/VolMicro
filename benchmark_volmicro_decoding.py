#!/usr/bin/env python3
"""
Benchmark VolMicro (3D Gaussian Splatting) decoding/inference time.

This script measures how long it takes to reconstruct a full 3D volume
from the Gaussian representation using the optimized CUDA kernel.

Usage:
    CXX=g++ CC=gcc python benchmark_volmicro_decoding.py
    
    # Or with custom checkpoint:
    CXX=g++ CC=gcc python benchmark_volmicro_decoding.py --checkpoint path/to/checkpoint.pt
"""

import os
# Set compiler environment before importing torch
os.environ['CXX'] = 'g++'
os.environ['CC'] = 'gcc'

import torch
import time
import argparse
import json
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from load_model import load_model


def benchmark_decoding(checkpoint_path, num_runs=10, warmup_runs=3):
    """
    Benchmark VolMicro decoding time using the optimized CUDA kernel.
    
    Args:
        checkpoint_path: Path to model checkpoint
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs (not counted)
    
    Returns:
        dict with benchmark results
    """
    # Load model
    print(f"Loading model from: {checkpoint_path}")
    model = load_model(checkpoint_path)
    model.eval()
    
    D, H, W = model.D, model.H, model.W
    total_voxels = D * H * W
    
    print(f"\nBenchmarking VolMicro decoding...")
    print(f"  Volume: {D}×{H}×{W} = {total_voxels:,} voxels")
    print(f"  Gaussians: {model.N:,}")
    
    # Warmup
    print(f"\n  Warmup ({warmup_runs} runs)...")
    for i in range(warmup_runs):
        with torch.no_grad():
            _ = model.forward()
        torch.cuda.synchronize()
        print(f"    Warmup {i+1}/{warmup_runs} done")
    
    # Benchmark
    print(f"\n  Benchmark ({num_runs} runs)...")
    times = []
    
    for i in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        with torch.no_grad():
            volume = model.forward()
        
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)
        print(f"    Run {i+1}: {elapsed:.1f} ms")
    
    # Statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    std_time = (sum((t - avg_time)**2 for t in times) / len(times)) ** 0.5
    throughput = total_voxels / (avg_time / 1000) / 1e6  # MVox/s
    
    results = {
        'method': 'VolMicro (3D Gaussian Splatting)',
        'volume_shape': [D, H, W],
        'total_voxels': total_voxels,
        'num_gaussians': model.N,
        'num_runs': num_runs,
        'times_ms': times,
        'avg_time_ms': avg_time,
        'min_time_ms': min_time,
        'max_time_ms': max_time,
        'std_time_ms': std_time,
        'throughput_mvox_s': throughput,
    }
    
    print(f"\n{'='*50}")
    print(f"VolMicro Decoding Results:")
    print(f"{'='*50}")
    print(f"  Average time: {avg_time:.1f} ms")
    print(f"  Min time:     {min_time:.1f} ms")
    print(f"  Max time:     {max_time:.1f} ms")
    print(f"  Std dev:      {std_time:.1f} ms")
    print(f"  Throughput:   {throughput:.1f} MVox/s")
    print(f"{'='*50}")
    
    # Compare with COIN
    coin_time_ms = 19424  # From previous benchmark
    speedup = coin_time_ms / avg_time
    print(f"\nComparison with COIN (SIREN):")
    print(f"  COIN decode time:     {coin_time_ms:,} ms")
    print(f"  VolMicro decode time: {avg_time:.1f} ms")
    print(f"  Speedup:              {speedup:.0f}× faster")
    
    results['coin_time_ms'] = coin_time_ms
    results['speedup_vs_coin'] = speedup
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark VolMicro decoding time')
    parser.add_argument('--checkpoint', type=str, 
                        default='pretrained_models/v019/checkpoints/checkpoint_final.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--num_runs', type=int, default=10,
                        help='Number of benchmark runs')
    parser.add_argument('--warmup', type=int, default=3,
                        help='Number of warmup runs')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for results')
    
    args = parser.parse_args()
    
    # Run benchmark
    results = benchmark_decoding(
        args.checkpoint,
        num_runs=args.num_runs,
        warmup_runs=args.warmup,
    )
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    return results


if __name__ == '__main__':
    main()
