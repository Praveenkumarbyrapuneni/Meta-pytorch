# PyTorch Architecture Deep Dive

**Purpose**: Foundation knowledge for contributing to PyTorch compiler and quantization  
**Author**: Praveen Kumar Byrapuneni  
**Date**: February 2026

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     USER PYTHON CODE                        │
│  model = MyModel()                                          │
│  compiled = torch.compile(model)                            │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                    TORCH.COMPILE STACK                      │
├─────────────────────────────────────────────────────────────┤
│  1. TorchDynamo (Python bytecode → FX Graph)               │
│     - Bytecode analysis                                     │
│     - Graph capture                                         │
│     - Guard generation                                      │
├─────────────────────────────────────────────────────────────┤
│  2. AOTAutograd (Autograd decomposition)                    │
│     - Forward/backward split                                │
│     - Gradient computation                                  │
│     - Graph optimization                                    │
├─────────────────────────────────────────────────────────────┤
│  3. TorchInductor (FX Graph → Optimized Kernels)           │
│     - Lowering to primitives                                │
│     - Fusion & scheduling                                   │
│     - Code generation (Triton/C++)                          │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                    EXECUTION RUNTIME                        │
├─────────────────────────────────────────────────────────────┤
│  ATen/C10 Core                                              │
│  ├─ Tensor operations                                       │
│  ├─ Device abstraction                                      │
│  └─ Memory management                                       │
├─────────────────────────────────────────────────────────────┤
│  Hardware Backends                                          │
│  ├─ CUDA (NVIDIA GPUs)                                      │
│  ├─ ROCm (AMD GPUs)                                         │
│  ├─ Metal (Apple Silicon)                                   │
│  └─ CPU (MKL/AVX)                                           │
└─────────────────────────────────────────────────────────────┘
```

---

## Layer 1: TorchDynamo - Graph Capture

### What It Does
Captures Python execution as FX graphs through bytecode analysis.

### Key Components

**Bytecode Analysis** (`torch/_dynamo/symbolic_convert.py`)
```python
class InstructionTranslator:
    """Converts Python bytecode to FX graph."""
    
    def CALL_FUNCTION(self, inst):
        # Intercepts function calls
        fn = self.stack.pop()
        args = self.popn(inst.argval)
        
        if can_inline(fn):
            # Inline into graph
            result = self.inline_call(fn, args)
        else:
            # Graph break
            self.graph_break(reason="unsupported function")
```

**Guard Generation** (`torch/fx/experimental/symbolic_shapes.py`)
```python
class ShapeEnv:
    """Tracks symbolic shapes and generates guards."""
    
    def create_symbol(self, val, source):
        # Create symbolic dimension s0, s1, etc.
        symbol = Symbol(f"s{self.next_symbol_id}")
        
        # Track constraint: s0 >= 1, s0 == 7, etc.
        self.add_guard(symbol, val)
        return symbol
```

### Graph Breaks

Dynamo creates **graph breaks** when it encounters:
- Unsupported operations
- Control flow with data-dependent conditions
- External function calls

---

## Layer 2: AOTAutograd - Autograd Decomposition

### What It Does
Splits graphs into forward and backward, decomposes composite ops.

### Key Concepts

**Forward/Backward Split**
```python
def aot_function(fn, args):
    # Trace forward pass
    with torch.enable_grad():
        out = fn(*args)
    
    # Generate backward graph
    grads = torch.autograd.grad(out, args)
    
    # Return joint graph
    return create_joint_graph(fn, grads)
```

**Functionalization**: Converts mutations to functional form
```python
# Before: x.add_(1)  # in-place
# After:  x = x.add(1)  # functional
```

---

## Layer 3: TorchInductor - Code Generation

### What It Does
Generates optimized GPU (Triton) and CPU (C++) kernels.

### Optimization Passes

1. **Lowering**: High-level ops → primitives
   ```python
   torch.matmul(a, b) → aten.mm(a, b) → triton.dot(a, b)
   ```

2. **Fusion**: Combine operations
   ```python
   # Before: 3 kernels
   x = a + b
   y = x * 2
   z = torch.relu(y)
   
   # After: 1 fused kernel
   z = relu((a + b) * 2)
   ```

3. **Scheduling**: Determine execution order
   ```python
   # Topological sort + heuristics
   # Prioritize: producer-consumer locality
   ```

4. **Code Generation**: Emit Triton/C++ code
   ```python
   @triton.jit
   def fused_kernel(a, b, out, n):
       idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
       mask = idx < n
       a_val = tl.load(a + idx, mask=mask)
       b_val = tl.load(b + idx, mask=mask)
       result = tl.maximum((a_val + b_val) * 2, 0)  # fused add+mul+relu
       tl.store(out + idx, result, mask=mask)
   ```

---

## Quantization Stack (PT2E)

```
Eager Model
    ↓ torch.export.export()
Exported Program (FX Graph)
    ↓ prepare_pt2e(quantizer)
Prepared Model (with observers)
    ↓ calibration
Calibrated Model
    ↓ convert_pt2e()
Quantized Model (INT8/FP8)
```

### Key Components

**Quantizer**: Defines quantization strategy
```python
class MyQuantizer(Quantizer):
    def annotate(self, model):
        # Annotate which tensors to quantize
        for node in model.graph.nodes:
            if node.op == 'call_function':
                self._annotate_node(node)
    
    def _annotate_node(self, node):
        node.meta['quantization_annotation'] = QuantizationAnnotation(
            input_qspec_map={node.args[0]: input_qspec},
            output_qspec=output_qspec
        )
```

**Observer**: Collects statistics during calibration
```python
class MinMaxObserver:
    def forward(self, x):
        self.min_val = min(self.min_val, x.min())
        self.max_val = max(self.max_val, x.max())
        return x
    
    def calculate_qparams(self):
        # Compute scale and zero_point
        scale = (self.max_val - self.min_val) / 255
        zero_point = -round(self.min_val / scale)
        return scale, zero_point
```

**Converter**: Inserts quantize/dequantize ops
```python
def convert_pt2e(prepared_model):
    for node in prepared_model.graph.nodes:
        if has_quantization_annotation(node):
            # Insert q/dq ops
            insert_quant_dequant(node)
    return prepared_model
```

---

## File Structure

### Core Directories

```
torch/
├── _dynamo/              # Graph capture
│   ├── symbolic_convert.py
│   ├── bytecode_transformation.py
│   └── guards.py
├── _inductor/            # Code generation
│   ├── codegen/
│   │   ├── triton.py
│   │   └── cpp.py
│   ├── lowering.py
│   └── scheduler.py
├── _functorch/           # Functional transforms
│   └── _aot_autograd/
├── fx/                   # FX graph IR
│   ├── graph.py
│   ├── node.py
│   └── experimental/
│       └── symbolic_shapes.py
└── ao/                   # Architecture optimization
    └── quantization/
        ├── pt2e/
        │   ├── quantizer/
        │   └── utils.py
        └── observer.py

aten/                     # C++ tensor library
└── src/
    └── ATen/
        ├── core/
        └── native/       # CPU/CUDA kernels

c10/                      # Core abstractions
├── core/
│   ├── TensorImpl.h
│   └── Device.h
└── util/
```

---

## Development Workflow

### 1. Setup Dev Environment

```bash
# Clone repo
git clone https://github.com/pytorch/pytorch
cd pytorch

# Setup dependencies
conda create -n pytorch-dev python=3.11
conda activate pytorch-dev
conda install cmake ninja

# Build from source (dev mode)
python setup.py develop

# Run tests
python test/run_test.py test_dynamo
```

### 2. Enable Debug Logging

```bash
# Dynamo logs
export TORCH_LOGS="+dynamo,+aot,+inductor"
export TORCHDYNAMO_VERBOSE=1

# Symbolic shapes
export TORCH_LOGS="+symbolic_shapes"

# Graph visualization
export TORCH_COMPILE_DEBUG=1
```

### 3. Inspect Generated Code

```python
import torch
import torch._dynamo

torch._dynamo.config.verbose = True
torch._dynamo.config.output_code = True

@torch.compile
def fn(x):
    return x + 1

fn(torch.randn(10))
# Prints generated FX graph and Triton code
```

---

## Key Abstractions

### FX Node

```python
class Node:
    op: str              # 'call_function', 'call_method', etc.
    target: Any          # Function/method being called
    args: Tuple          # Positional arguments
    kwargs: Dict         # Keyword arguments
    meta: Dict           # Metadata (shape, dtype, etc.)
    users: Dict[Node]    # Nodes that use this output
```

### SymInt (Symbolic Integer)

```python
class SymInt:
    """Represents a possibly-symbolic integer."""
    
    node: SymNode        # Symbolic expression tree
    
    def __add__(self, other):
        return SymInt(Add(self.node, other.node))
    
    def guard_eq(self, val):
        # Generate runtime guard: self == val
        return Guard(Eq(self.node, val))
```

---

## Performance Considerations

### Compilation Time

- First run: ~1-5s (graph capture + code gen)
- Subsequent runs: <1ms (uses cached compiled code)
- Recompilation triggers: guard failures (shape changes)

### Runtime Overhead

- Guard checking: ~10-50μs per function call
- Kernel launch: ~5-10μs per kernel
- Negligible for operations >1ms

### Memory

- Compiled code cache: ~10-100MB per model
- Guard state: ~1KB per function
- FX graph: ~1MB for large models

---

## Contributing Guidelines

### Code Style

```python
# Use type hints
def process_node(node: torch.fx.Node) -> List[Node]:
    ...

# Document complex logic
def symbolic_size(tensor, dim):
    """Get symbolic size of tensor dimension.
    
    Returns:
        SymInt: Symbolic integer representing dimension size.
        Guards will be generated to ensure correctness.
    """
```

### Testing

```python
# Unit tests
class TestDynamoFeature(TestCase):
    def test_basic_functionality(self):
        ...

# Integration tests
class TestEndToEnd(TestCase):
    def test_resnet50_compile(self):
        model = torchvision.models.resnet50()
        compiled = torch.compile(model)
        ...
```

---

## Resources

- [PyTorch Internals](http://blog.ezyang.com/2019/05/pytorch-internals/)
- [TorchDynamo Deep Dive](https://dev-discuss.pytorch.org/t/torchdynamo-update-1/630)
- [TorchInductor Design](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747)
- [Quantization Technical Overview](https://pytorch.org/blog/quantization-in-practice/)

---

**Document Version**: 1.0  
**Last Updated**: February 10, 2026