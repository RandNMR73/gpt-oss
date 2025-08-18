# MXFP4 Quantization Testing in GPT-OSS

This directory contains scripts and modules for testing MXFP4 quantization performance directly in the GPT-OSS repository. The goal is to determine if MXFP4 quantization provides speedup for linear layers compared to standard PyTorch linear layers.

## Overview

MXFP4 (Mixed-Precision Floating Point 4-bit) quantization is a technique that reduces the precision of model weights from 16-bit (bfloat16) to 4-bit while maintaining reasonable accuracy. This can potentially provide:

- **Memory savings**: 4x reduction in weight memory usage
- **Performance improvements**: Faster matrix multiplication on supported hardware
- **Maintained accuracy**: Acceptable precision loss for inference

## Files

### Core Module

- **`gpt_oss/triton/mxfp4_quantization.py`**: Main MXFP4 quantization module with quantized linear layers and utilities

### Test Scripts

- **`test_mxfp4_basic.py`**: Basic functionality test to verify MXFP4 is working
- **`benchmark_mxfp4.py`**: Performance benchmark comparing MXFP4 vs standard linear layers
- **`test_mxfp4_gpt_oss.py`**: Comprehensive testing suite for MXFP4 integration

## Prerequisites

1. **triton_kernels**: The MXFP4 implementation requires the `triton_kernels` library
2. **CUDA-capable GPU**: MXFP4 quantization is designed for GPU acceleration
3. **PyTorch**: With CUDA support
4. **bfloat16 support**: The quantization works with bfloat16 precision

## Quick Start

### 1. Basic Functionality Test

Run the basic test to ensure MXFP4 is working:

```bash
python test_mxfp4_basic.py
```

This will test:

- Module imports
- Basic quantization functionality
- Linear layer creation and forward pass
- Quantization accuracy

### 2. Simple Performance Benchmark

Run a quick benchmark with default settings:

```bash
python benchmark_mxfp4.py
```

Or customize the benchmark:

```bash
python benchmark_mxfp4.py --input-size 2048 --output-size 2048 --batch-size 32 --iterations 200
```

### 3. Comprehensive Testing

Run the full test suite:

```bash
python test_mxfp4_gpt_oss.py
```

## Usage Examples

### Basic MXFP4 Linear Layer

```python
from gpt_oss.triton.mxfp4_quantization import MXFP4Linear

# Create a quantized linear layer
layer = MXFP4Linear(128, 256, bias=True, device='cuda')

# Use it like a standard PyTorch linear layer
x = torch.randn(4, 128, dtype=torch.bfloat16, device='cuda')
output = layer(x)
```

### MXFP4 Linear Layer with SwiGLU Activation

```python
from gpt_oss.triton.mxfp4_quantization import MXFP4LinearWithActivation

# Create a quantized linear layer with SwiGLU activation
layer = MXFP4LinearWithActivation(
    128, 256, activation="swiglu", swiglu_limit=7.0, device='cuda'
)

# Forward pass includes activation
output = layer(x)
```

### Creating Quantized MLP Blocks

```python
from gpt_oss.triton.mxfp4_quantization import create_quantized_mlp_block

# Create MLP blocks similar to GPT-OSS model
gate_layer, mlp1_layer, mlp2_layer = create_quantized_mlp_block(
    hidden_size=2880,
    intermediate_size=2880,
    num_experts=128,
    experts_per_token=4,
    device='cuda'
)
```

## Benchmarking

### Single Layer Benchmark

```python
from gpt_oss.triton.mxfp4_quantization import benchmark_linear_layers

results = benchmark_linear_layers(
    input_size=2880,
    hidden_size=2880,
    output_size=2880,
    batch_size=16,
    num_iterations=100
)
```

### Accuracy Testing

```python
from gpt_oss.triton.mxfp4_quantization import test_quantization_accuracy

results = test_quantization_accuracy(
    input_size=2880,
    hidden_size=2880,
    output_size=2880,
    batch_size=16,
    num_tests=20
)
```

## Expected Results

### Performance

- **Speedup**: MXFP4 should provide 1.2x to 2.0x speedup on supported hardware
- **TFLOPs**: Higher throughput compared to standard linear layers
- **Memory**: Lower peak memory usage due to weight quantization

### Accuracy

- **Relative Error**: Should be < 10% for most use cases
- **Absolute Error**: Depends on the scale of your activations
- **Consistency**: Errors should be consistent across different inputs

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `triton_kernels` is properly installed
2. **CUDA Errors**: Check that you have a CUDA-capable GPU and proper drivers
3. **Memory Errors**: Reduce batch size or layer dimensions
4. **Accuracy Issues**: Check that inputs are in bfloat16 format

### Debug Mode

For debugging, you can run tests with more verbose output:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Integration with GPT-OSS

The MXFP4 quantization is designed to be a drop-in replacement for linear layers in the GPT-OSS codebase. You can:

1. **Replace individual layers**: Swap `torch.nn.Linear` with `MXFP4Linear`
2. **Quantize MLP blocks**: Use the provided MLP block creation functions
3. **Maintain compatibility**: The quantized layers have the same interface as standard layers

## Performance Analysis

### What to Look For

1. **Speedup**: Is MXFP4 faster than standard linear layers?
2. **Memory**: Does quantization reduce memory usage?
3. **Accuracy**: Is the precision loss acceptable for your use case?
4. **Scaling**: How does performance scale with different layer sizes?

### Benchmarking Tips

1. **Warmup**: Always run warmup iterations before benchmarking
2. **Multiple runs**: Run benchmarks multiple times to account for variance
3. **Different sizes**: Test various layer dimensions and batch sizes
4. **Memory profiling**: Monitor GPU memory usage during tests

## Contributing

To improve the MXFP4 testing:

1. **Add new test cases**: Extend the test suite with additional scenarios
2. **Optimize benchmarks**: Improve benchmark accuracy and reliability
3. **Document findings**: Share performance results and insights
4. **Report issues**: Document any problems or unexpected behavior

## References

- **GPT-OSS**: The main repository this testing is designed for
- **triton_kernels**: The underlying quantization library
- **MXFP4 Paper**: Research on mixed-precision floating point quantization
- **SwiGLU**: The activation function used in the MLP blocks

## License

This testing code follows the same license as the GPT-OSS repository.
