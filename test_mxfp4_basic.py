#!/usr/bin/env python3
"""
Basic test script for MXFP4 quantization in GPT-OSS.

This script tests basic functionality to ensure MXFP4 quantization is working.
"""

import torch
import sys

def test_imports():
    """Test that all required modules can be imported."""
    print("=== Testing Imports ===")
    
    try:
        import triton_kernels
        print("✓ triton_kernels imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import triton_kernels: {e}")
        return False
    
    try:
        from gpt_oss.triton.mxfp4_quantization import MXFP4Linear, quantize_mx4
        print("✓ MXFP4 modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import MXFP4 modules: {e}")
        return False
    
    return True


def test_basic_quantization():
    """Test basic quantization functionality."""
    print("\n=== Testing Basic Quantization ===")
    
    try:
        # Create a simple weight tensor
        weight = torch.randn(256, 128, dtype=torch.bfloat16)
        print(f"✓ Created weight tensor: {weight.shape}, {weight.dtype}")
        
        # Test quantization
        from gpt_oss.triton.mxfp4_quantization import quantize_mx4
        quantized_weight, weight_scale = quantize_mx4(weight)
        
        print(f"✓ Quantization successful")
        print(f"  Quantized weight type: {type(quantized_weight)}")
        print(f"  Weight scale type: {type(weight_scale)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic quantization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_linear_layer():
    """Test MXFP4 linear layer."""
    print("\n=== Testing MXFP4 Linear Layer ===")
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"✓ Using device: {device}")
        
        # Create MXFP4 linear layer
        from gpt_oss.triton.mxfp4_quantization import MXFP4Linear
        layer = MXFP4Linear(128, 256, device=device)
        print(f"✓ Created MXFP4Linear layer: {layer}")
        
        # Create test input
        x = torch.randn(4, 128, dtype=torch.bfloat16, device=device)
        print(f"✓ Created test input: {x.shape}, {x.dtype}")
        
        # Test forward pass
        output = layer(x)
        print(f"✓ Forward pass successful: {output.shape}, {output.dtype}")
        
        return True
        
    except Exception as e:
        print(f"✗ Linear layer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_accuracy():
    """Test quantization accuracy."""
    print("\n=== Testing Quantization Accuracy ===")
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create layers
        standard_layer = torch.nn.Linear(128, 256, bias=True, device=device, dtype=torch.bfloat16)
        mxfp4_layer = MXFP4Linear(128, 256, bias=True, device=device, dtype=torch.bfloat16)
        
        # Copy weights for fair comparison
        with torch.no_grad():
            mxfp4_layer.weight.copy_(standard_layer.weight)
            if standard_layer.bias is not None:
                mxfp4_layer.bias.copy_(standard_layer.bias)
        
        # Test with random input
        x = torch.randn(8, 128, dtype=torch.bfloat16, device=device)
        
        with torch.no_grad():
            standard_output = standard_layer(x)
            mxfp4_output = mxfp4_layer(x)
        
        # Calculate error
        abs_error = (standard_output - mxfp4_output).abs()
        rel_error = abs_error / (standard_output.abs().clamp_min(1e-30))
        
        max_abs_error = abs_error.max().item()
        max_rel_error = rel_error.max().item()
        
        print(f"✓ Accuracy test completed")
        print(f"  Max absolute error: {max_abs_error:.6f}")
        print(f"  Max relative error: {max_rel_error:.6f}")
        
        if max_rel_error < 0.1:  # 10% relative error threshold
            print("✓ Quantization accuracy is acceptable")
            return True
        else:
            print("⚠️  Quantization accuracy may be too low")
            return False
        
    except Exception as e:
        print(f"✗ Accuracy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("MXFP4 Basic Functionality Test in GPT-OSS")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Basic Quantization", test_basic_quantization),
        ("Linear Layer", test_linear_layer),
        ("Accuracy", test_accuracy),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append(success)
            
            if success:
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
                
        except Exception as e:
            print(f"✗ {test_name} CRASHED: {e}")
            results.append(False)
    
    # Print summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("🎉 All tests passed! MXFP4 quantization is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 