#!/usr/bin/env python3
"""
Simple benchmark script for MXFP4 vs standard linear layers in GPT-OSS.

This script provides a quick way to test if MXFP4 quantization provides
performance benefits for linear layers.
"""

import torch
import time
import gc
import argparse

# Import our MXFP4 quantization module
from gpt_oss.triton.mxfp4_quantization import MXFP4Linear


def simple_benchmark(input_size: int, output_size: int, batch_size: int, 
                    num_iterations: int = 100, device=None):
    """
    Simple benchmark comparing standard vs MXFP4 linear layers.
    
    Args:
        input_size: Input dimension
        output_size: Output dimension  
        batch_size: Batch size
        num_iterations: Number of iterations for benchmarking
        device: Device to run on
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Benchmarking on {device}")
    print(f"Input: {batch_size}x{input_size} -> {output_size}")
    print(f"Iterations: {num_iterations}")
    
    # Create test input
    x = torch.randn(batch_size, input_size, dtype=torch.bfloat16, device=device)
    
    # Create layers
    standard_linear = torch.nn.Linear(input_size, output_size, bias=True, 
                                     device=device, dtype=torch.bfloat16)
    mxfp4_linear = MXFP4Linear(input_size, output_size, bias=True, 
                               device=device, dtype=torch.bfloat16)
    
    # Copy weights for fair comparison
    with torch.no_grad():
        mxfp4_linear.weight.copy_(standard_linear.weight)
        if standard_linear.bias is not None:
            mxfp4_linear.bias.copy_(standard_linear.bias)
    
    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for _ in range(10):
            _ = standard_linear(x)
            _ = mxfp4_linear(x)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark standard
    print("Benchmarking standard linear layer...")
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = standard_linear(x)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    standard_time = time.time() - start_time
    
    # Benchmark MXFP4
    print("Benchmarking MXFP4 linear layer...")
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = mxfp4_linear(x)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    mxfp4_time = time.time() - start_time
    
    # Calculate metrics
    theoretical_flops = 2 * batch_size * input_size * output_size
    theoretical_tflops = theoretical_flops / 1e12
    
    standard_tflops = (theoretical_tflops * num_iterations) / standard_time
    mxfp4_tflops = (theoretical_tflops * num_iterations) / mxfp4_time
    
    speedup = standard_time / mxfp4_time
    
    # Print results
    print(f"\nResults:")
    print(f"  Standard: {standard_time:.4f}s, {standard_tflops:.3f} TFLOPs")
    print(f"  MXFP4:    {mxfp4_time:.4f}s, {mxfp4_tflops:.3f} TFLOPs")
    print(f"  Speedup:  {speedup:.2f}x")
    
    if speedup > 1.0:
        print(f"🚀 MXFP4 is {speedup:.2f}x faster!")
    else:
        print(f"⚠️  MXFP4 is {1/speedup:.2f}x slower")
    
    return {
        'standard_time': standard_time,
        'mxfp4_time': mxfp4_time,
        'speedup': speedup,
        'standard_tflops': standard_tflops,
        'mxfp4_tflops': mxfp4_tflops
    }


def benchmark_different_sizes():
    """Benchmark different layer sizes."""
    print("=== Benchmarking Different Layer Sizes ===")
    
    # Test configurations: (input_size, output_size, batch_size)
    configs = [
        (128, 256, 1),
        (128, 256, 16),
        (512, 1024, 1),
        (512, 1024, 16),
        (2048, 4096, 1),
        (2048, 4096, 16),
        (2880, 2880, 1),  # gpt-oss-like
        (2880, 2880, 16),
    ]
    
    results = []
    
    for input_size, output_size, batch_size in configs:
        print(f"\n--- {input_size}x{output_size}, batch={batch_size} ---")
        try:
            result = simple_benchmark(input_size, output_size, batch_size, num_iterations=50)
            result.update({
                'input_size': input_size,
                'output_size': output_size,
                'batch_size': batch_size
            })
            results.append(result)
        except Exception as e:
            print(f"  Failed: {e}")
            continue
    
    # Print summary
    if results:
        print(f"\n=== Summary ===")
        print(f"{'Config':<20} {'Speedup':<10}")
        print("-" * 30)
        
        total_speedup = 0
        for result in results:
            config = f"{result['input_size']}x{result['output_size']}x{result['batch_size']}"
            speedup = result['speedup']
            print(f"{config:<20} {speedup:<10.2f}")
            total_speedup += speedup
        
        avg_speedup = total_speedup / len(results)
        print("-" * 30)
        print(f"{'AVERAGE':<20} {avg_speedup:<10.2f}")
        
        if avg_speedup > 1.0:
            print(f"\n🚀 MXFP4 provides {avg_speedup:.2f}x average speedup!")
        else:
            print(f"\n⚠️  MXFP4 shows {avg_speedup:.2f}x average performance")


def main():
    parser = argparse.ArgumentParser(description="Benchmark MXFP4 vs standard linear layers")
    parser.add_argument("--input-size", type=int, default=2880, help="Input dimension")
    parser.add_argument("--output-size", type=int, default=2880, help="Output dimension")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations")
    parser.add_argument("--all-sizes", action="store_true", help="Benchmark all sizes")
    
    args = parser.parse_args()
    
    print("MXFP4 vs Standard Linear Layer Benchmark")
    print("=" * 50)
    
    # Check dependencies
    try:
        import triton_kernels
        print("✓ triton_kernels available")
    except ImportError:
        print("✗ triton_kernels not available. Please install it first.")
        return
    
    if args.all_sizes:
        benchmark_different_sizes()
    else:
        simple_benchmark(
            input_size=args.input_size,
            output_size=args.output_size,
            batch_size=args.batch_size,
            num_iterations=args.iterations
        )


if __name__ == "__main__":
    main() 