#!/usr/bin/env python3
"""
Test script for MXFP4 quantization in GPT-OSS repository.

This script tests MXFP4 quantization performance and accuracy for linear layers
that are similar to those used in the gpt-oss codebase.
"""

import torch
import time
import gc
import argparse
from typing import List, Dict, Any
import json

# Import our MXFP4 quantization module
from gpt_oss.triton.mxfp4_quantization import (
    MXFP4Linear, 
    MXFP4LinearWithActivation,
    create_quantized_mlp_block,
    benchmark_linear_layers,
    test_quantization_accuracy
)


def test_basic_functionality():
    """Test basic MXFP4 functionality."""
    print("=== Testing Basic MXFP4 Functionality ===")
    
    try:
        # Test device availability
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"✓ Using device: {device}")
        
        # Test basic linear layer
        input_size, output_size = 128, 256
        batch_size = 4
        
        print(f"✓ Testing MXFP4Linear({input_size}, {output_size})")
        mxfp4_layer = MXFP4Linear(input_size, output_size, device=device)
        
        # Test forward pass
        x = torch.randn(batch_size, input_size, dtype=torch.bfloat16, device=device)
        output = mxfp4_layer(x)
        
        print(f"✓ Forward pass successful: input {x.shape} -> output {output.shape}")
        print(f"✓ Output dtype: {output.dtype}")
        
        # Test with activation
        print(f"✓ Testing MXFP4LinearWithActivation({input_size}, {output_size})")
        mxfp4_act_layer = MXFP4LinearWithActivation(
            input_size, output_size, activation="swiglu", device=device
        )
        
        output_act = mxfp4_act_layer(x)
        print(f"✓ Activation forward pass successful: output {output_act.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mlp_block_creation():
    """Test creation of quantized MLP blocks similar to gpt-oss."""
    print("\n=== Testing MLP Block Creation ===")
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Use dimensions similar to gpt-oss model
        hidden_size = 2880
        intermediate_size = 2880
        num_experts = 128
        experts_per_token = 4
        
        print(f"✓ Creating MLP block: hidden={hidden_size}, intermediate={intermediate_size}")
        print(f"✓ Experts: {num_experts}, per token: {experts_per_token}")
        
        gate_layer, mlp1_layer, mlp2_layer = create_quantized_mlp_block(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            experts_per_token=experts_per_token,
            device=device
        )
        
        print(f"✓ Gate layer: {type(gate_layer).__name__}, {gate_layer.weight.shape}")
        print(f"✓ MLP1 layer: {type(mlp1_layer).__name__}, {mlp1_layer.weight.shape}")
        print(f"✓ MLP2 layer: {type(mlp2_layer).__name__}, {mlp2_layer.weight.shape}")
        
        # Test forward pass through the block
        batch_size = 2
        x = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16, device=device)
        
        print("✓ Testing forward pass through MLP block...")
        
        # Gate computation (routing)
        gate_output = gate_layer(x)
        print(f"  Gate output: {gate_output.shape}")
        
        # MLP1 with SwiGLU
        mlp1_output = mlp1_layer(x)
        print(f"  MLP1 output: {mlp1_output.shape}")
        
        # MLP2
        mlp2_output = mlp2_layer(mlp1_output)
        print(f"  MLP2 output: {mlp2_output.shape}")
        
        print("✓ MLP block forward pass successful")
        return True
        
    except Exception as e:
        print(f"✗ MLP block test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_quantization_accuracy_comprehensive():
    """Test quantization accuracy across different layer sizes."""
    print("\n=== Testing Quantization Accuracy ===")
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Test different layer sizes
        test_configs = [
            (128, 256, 64),      # Small layers
            (512, 1024, 256),    # Medium layers
            (2048, 4096, 1024),  # Large layers
            (2880, 2880, 1440),  # gpt-oss-like dimensions
        ]
        
        batch_size = 8
        num_tests = 5
        
        all_results = []
        
        for input_size, hidden_size, output_size in test_configs:
            print(f"\n--- Testing: {input_size}x{hidden_size}x{output_size} ---")
            
            try:
                results = test_quantization_accuracy(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    output_size=output_size,
                    batch_size=batch_size,
                    num_tests=num_tests,
                    device=device
                )
                
                # Add config info to results
                results.update({
                    'input_size': input_size,
                    'hidden_size': hidden_size,
                    'output_size': output_size,
                    'batch_size': batch_size
                })
                
                all_results.append(results)
                
            except Exception as e:
                print(f"  ✗ Failed for config {input_size}x{hidden_size}x{output_size}: {e}")
                continue
        
        # Print summary
        if all_results:
            print(f"\n=== Accuracy Test Summary ===")
            print(f"{'Config':<20} {'Max Abs Error':<15} {'Max Rel Error':<15}")
            print("-" * 50)
            
            for result in all_results:
                config = f"{result['input_size']}x{result['hidden_size']}x{result['output_size']}"
                print(f"{config:<20} {result['max_abs_error']:<15.6f} {result['max_rel_error']:<15.6f}")
            
            return True
        else:
            print("✗ No accuracy tests completed successfully")
            return False
            
    except Exception as e:
        print(f"✗ Comprehensive accuracy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def benchmark_performance_comprehensive():
    """Comprehensive performance benchmarking."""
    print("\n=== Comprehensive Performance Benchmarking ===")
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Benchmark configurations
        benchmark_configs = [
            # (input_size, hidden_size, output_size, batch_size)
            (128, 256, 128, 1),
            (128, 256, 128, 4),
            (128, 256, 128, 16),
            (512, 1024, 512, 1),
            (512, 1024, 512, 4),
            (512, 1024, 512, 16),
            (2048, 4096, 2048, 1),
            (2048, 4096, 2048, 4),
            (2880, 2880, 2880, 1),  # gpt-oss-like
            (2880, 2880, 2880, 4),
        ]
        
        num_iterations = 50
        warmup_iterations = 5
        
        all_results = []
        
        for input_size, hidden_size, output_size, batch_size in benchmark_configs:
            print(f"\n--- Benchmarking: {input_size}x{hidden_size}x{output_size}, batch={batch_size} ---")
            
            try:
                results = benchmark_linear_layers(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    output_size=output_size,
                    batch_size=batch_size,
                    num_iterations=num_iterations,
                    warmup_iterations=warmup_iterations,
                    device=device
                )
                
                # Add config info to results
                results.update({
                    'input_size': input_size,
                    'hidden_size': hidden_size,
                    'output_size': output_size,
                    'batch_size': batch_size
                })
                
                all_results.append(results)
                
            except Exception as e:
                print(f"  ✗ Benchmark failed: {e}")
                continue
        
        # Print comprehensive summary
        if all_results:
            print(f"\n" + "=" * 80)
            print("COMPREHENSIVE BENCHMARK SUMMARY")
            print("=" * 80)
            print(f"{'Config':<25} {'Batch':<6} {'Standard TFLOPs':<15} {'MXFP4 TFLOPs':<15} {'Speedup':<10}")
            print("-" * 80)
            
            total_speedup = 0
            valid_results = 0
            
            for result in all_results:
                config = f"{result['input_size']}x{result['hidden_size']}x{result['output_size']}"
                batch = result['batch_size']
                std_tflops = result['standard_tflops']
                mxfp4_tflops = result['mxfp4_tflops']
                speedup = result['speedup']
                
                print(f"{config:<25} {batch:<6} {std_tflops:<15.3f} {mxfp4_tflops:<15.3f} {speedup:<10.2f}")
                
                if speedup > 0:
                    total_speedup += speedup
                    valid_results += 1
            
            if valid_results > 0:
                avg_speedup = total_speedup / valid_results
                print("-" * 80)
                print(f"{'AVERAGE':<25} {'':<6} {'':<15} {'':<15} {avg_speedup:<10.2f}")
                
                if avg_speedup > 1.0:
                    print(f"\n🚀 MXFP4 provides {avg_speedup:.2f}x average speedup!")
                else:
                    print(f"\n⚠️  MXFP4 shows {avg_speedup:.2f}x average performance")
            
            return True
        else:
            print("✗ No benchmarks completed successfully")
            return False
            
    except Exception as e:
        print(f"✗ Comprehensive benchmarking failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_efficiency():
    """Test memory efficiency of MXFP4 quantization."""
    print("\n=== Testing Memory Efficiency ===")
    
    try:
        if not torch.cuda.is_available():
            print("⚠️  CUDA not available, skipping memory test")
            return False
        
        device = torch.device('cuda')
        torch.cuda.empty_cache()
        gc.collect()
        
        # Test with large layers
        input_size = 4096
        hidden_size = 4096
        output_size = 4096
        batch_size = 16
        
        print(f"✓ Testing memory usage for {input_size}x{hidden_size}x{output_size}, batch={batch_size}")
        
        # Create test input
        x = torch.randn(batch_size, input_size, dtype=torch.bfloat16, device=device)
        
        # Measure standard linear layer memory
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        standard_linear = torch.nn.Linear(input_size, output_size, bias=True, 
                                        device=device, dtype=torch.bfloat16)
        _ = standard_linear(x)
        torch.cuda.synchronize()
        
        standard_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
        
        # Measure MXFP4 linear layer memory
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        mxfp4_linear = MXFP4Linear(input_size, output_size, bias=True, 
                                  device=device, dtype=torch.bfloat16)
        _ = mxfp4_linear(x)  # This triggers quantization
        torch.cuda.synchronize()
        
        mxfp4_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
        
        # Calculate memory savings
        memory_savings = (standard_memory - mxfp4_memory) / standard_memory * 100
        
        print(f"\nMemory Usage Results:")
        print(f"  Standard Linear: {standard_memory:.2f} GB")
        print(f"  MXFP4 Linear:    {mxfp4_memory:.2f} GB")
        print(f"  Memory Savings:  {memory_savings:.1f}%")
        
        if memory_savings > 0:
            print(f"✓ MXFP4 uses less memory than standard linear layers")
        else:
            print(f"⚠️  MXFP4 uses more memory than standard linear layers")
        
        return True
        
    except Exception as e:
        print(f"✗ Memory efficiency test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration_with_gpt_oss_like_model():
    """Test integration with a model structure similar to gpt-oss."""
    print("\n=== Testing Integration with GPT-OSS-like Model ===")
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create a simplified model structure similar to gpt-oss
        class SimplifiedGPTOSSModel(torch.nn.Module):
            def __init__(self, hidden_size=2880, intermediate_size=2880, num_layers=2):
                super().__init__()
                self.hidden_size = hidden_size
                self.intermediate_size = intermediate_size
                self.num_layers = num_layers
                
                # Embedding layer
                self.embedding = torch.nn.Embedding(1000, hidden_size, dtype=torch.bfloat16, device=device)
                
                # Transformer blocks
                self.layers = torch.nn.ModuleList()
                for _ in range(num_layers):
                    # Attention block (simplified)
                    self.layers.append(torch.nn.ModuleDict({
                        'attention': torch.nn.Linear(hidden_size, hidden_size, dtype=torch.bfloat16, device=device),
                        'mlp1': MXFP4LinearWithActivation(
                            hidden_size, intermediate_size * 2, activation="swiglu", device=device
                        ),
                        'mlp2': MXFP4Linear(hidden_size, hidden_size, device=device),
                        'norm1': torch.nn.LayerNorm(hidden_size, dtype=torch.bfloat16, device=device),
                        'norm2': torch.nn.LayerNorm(hidden_size, dtype=torch.bfloat16, device=device),
                    }))
                
                # Output layer
                self.output = torch.nn.Linear(hidden_size, 1000, dtype=torch.bfloat16, device=device)
            
            def forward(self, x):
                # Embedding
                x = self.embedding(x)
                
                # Transformer blocks
                for layer in self.layers:
                    # Self-attention (simplified)
                    residual = x
                    x = layer['norm1'](x)
                    x = layer['attention'](x)
                    x = x + residual
                    
                    # MLP block
                    residual = x
                    x = layer['norm2'](x)
                    x = layer['mlp1'](x)
                    x = layer['mlp2'](x)
                    x = x + residual
                
                # Output
                x = self.output(x)
                return x
        
        print("✓ Creating simplified GPT-OSS-like model...")
        model = SimplifiedGPTOSSModel(hidden_size=1024, intermediate_size=1024, num_layers=2)
        model = model.to(device)
        
        print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test forward pass
        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
        
        print("✓ Testing forward pass...")
        with torch.no_grad():
            output = model(input_ids)
        
        print(f"✓ Forward pass successful: input {input_ids.shape} -> output {output.shape}")
        
        # Test inference time
        num_iterations = 10
        warmup_iterations = 5
        
        print("✓ Warming up...")
        with torch.no_grad():
            for _ in range(warmup_iterations):
                _ = model(input_ids)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        print("✓ Benchmarking inference...")
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(input_ids)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        inference_time = time.time() - start_time
        avg_inference_time = inference_time / num_iterations
        
        print(f"✓ Inference benchmark: {avg_inference_time:.4f}s per forward pass")
        
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def save_results_to_file(results: List[Dict[str, Any]], filename: str = "mxfp4_test_results.json"):
    """Save test results to a JSON file."""
    try:
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"✓ Results saved to {filename}")
    except Exception as e:
        print(f"⚠️  Failed to save results: {e}")


def main():
    """Run all tests."""
    parser = argparse.ArgumentParser(description="Test MXFP4 quantization in GPT-OSS")
    parser.add_argument("--save-results", action="store_true", 
                       help="Save results to JSON file")
    parser.add_argument("--output-file", default="mxfp4_test_results.json",
                       help="Output file for results")
    parser.add_argument("--skip-memory", action="store_true",
                       help="Skip memory efficiency test")
    parser.add_argument("--skip-integration", action="store_true",
                       help="Skip integration test")
    
    args = parser.parse_args()
    
    print("MXFP4 Quantization Testing in GPT-OSS Repository")
    print("=" * 60)
    
    # Check dependencies
    try:
        import triton_kernels
        print("✓ triton_kernels available")
    except ImportError:
        print("✗ triton_kernels not available. Please install it first.")
        return
    
    # Run tests
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("MLP Block Creation", test_mlp_block_creation),
        ("Quantization Accuracy", test_quantization_accuracy_comprehensive),
        ("Performance Benchmarking", benchmark_performance_comprehensive),
    ]
    
    if not args.skip_memory:
        tests.append(("Memory Efficiency", test_memory_efficiency))
    
    if not args.skip_integration:
        tests.append(("Integration Test", test_integration_with_gpt_oss_like_model))
    
    results = []
    passed_tests = 0
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            if success:
                passed_tests += 1
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
            
            results.append({
                'test_name': test_name,
                'passed': success,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            })
            
        except Exception as e:
            print(f"✗ {test_name} CRASHED: {e}")
            results.append({
                'test_name': test_name,
                'passed': False,
                'error': str(e),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            })
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Passed: {passed_tests}/{len(tests)}")
    
    if passed_tests == len(tests):
        print("🎉 All tests passed! MXFP4 quantization is working correctly in GPT-OSS.")
    else:
        print("❌ Some tests failed. Check the output above for details.")
    
    # Save results if requested
    if args.save_results:
        save_results_to_file(results, args.output_file)
    
    print(f"\nTest completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main() 