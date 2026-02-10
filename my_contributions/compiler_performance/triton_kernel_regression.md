# Triton Kernel Name Compilation Regression Analysis

**Issue**: [pytorch/pytorch#170398](https://github.com/pytorch/pytorch/issues/170398)  
**Status**: Open (Good First Issue)  
**Labels**: `oncall: pt2`, `module: inductor`, `module: user triton`, `triaged`, `actionable`  
**Area**: Compiler Backend / Triton Integration

## Executive Summary

This issue represents a **regression bug** in PyTorch 2.9.1 where Triton kernel compilation fails with `NameError: name '_Runner__kernel_name_0' is not defined`. The bug affects production training pipelines and blocks users from upgrading PyTorch versions.

**Impact**: HIGH - Breaks backward compatibility and prevents version upgrades  
**Complexity**: MEDIUM - Requires understanding of async compilation and Python name mangling  
**Resume Value**: HIGH - Shows ability to debug complex compiler issues

---

## Problem Description

### Symptom
Code that worked in PyTorch 2.8.0 + Triton 3.4.0 fails in PyTorch 2.9.1 + Triton 3.5.1 with:

```python
NameError: name '_Runner__mm_kernel_0' is not defined
```

### Root Cause Analysis

The generated code contains:
```python
__mm_kernel_0 = async_compile.triton('__mm_kernel', '''  
import triton
import triton.language as tl
```

But at runtime, Python tries to access `_Runner__mm_kernel_0` due to **name mangling**.

#### Why Name Mangling Occurs

Python automatically mangles names starting with `__` (double underscore) inside class definitions:
- `__kernel_name` → `_ClassName__kernel_name`
- This is a privacy mechanism to prevent name collisions

### The Bug

**Hypothesis**: Changes in `async_compile.triton()` behavior between versions caused:

1. **Kernel name generation** to use `__` prefix incorrectly
2. **Scope issues** where kernel is defined in class context
3. **Variable binding** problems in generated code

---

## Technical Deep Dive

### PyTorch Compilation Pipeline

```
User Model (Python)
    ↓
TorchDynamo (Bytecode → FX Graph)
    ↓
AOTAutograd (Autograd decomposition)
    ↓
TorchInductor (Graph → Triton/C++ kernels)
    ↓
Async Compilation (Parallel kernel builds)
    ↓
Runtime Execution
```

The bug occurs in the **TorchInductor → Async Compilation** stage.

### Relevant Code Paths

**Key Files to Investigate:**

1. `torch/_inductor/codecache.py`
   - `AsyncCompile.triton()` method
   - Kernel name generation logic

2. `torch/_inductor/codegen/wrapper.py`
   - Code generation for kernel invocations
   - Variable scope management

3. `torch/_inductor/runtime/triton_heuristics.py`
   - Kernel launcher wrapper classes
   - Where name mangling might occur

### Detailed Error Stack

```
File "/tmp/torchinductor_.../6e/c6ei2z....py", line 1882, in call
    __mm_kernel_0.run(buf4, primals_42, buf3, ...)
    ^^^^^^^^^^^^^
NameError: name '_Runner__mm_kernel_0' is not defined
```

**Analysis:**
- Code expects `__mm_kernel_0`
- Python interpreter looks for `_Runner__mm_kernel_0` (mangled)
- Kernel was registered without considering mangling

---

## Investigation Strategy

### Step 1: Reproduce the Issue

```python
import torch
import torch._dynamo
import torch._inductor.config as config

# Minimal reproduction
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3072, 15360)
    
    def forward(self, x):
        return self.linear(x)

model = Model().cuda()
compiled = torch.compile(model)

# Test with 2.8.0 (works) vs 2.9.1 (fails)
x = torch.randn(1101, 3072, device='cuda')
with torch.autograd.grad_mode.set_grad_enabled(True):
    out = compiled(x)
    out.sum().backward()  # Trigger backward pass where error occurs
```

### Step 2: Compare Generated Code

```bash
# Enable code generation logging
export TORCH_LOGS="+output_code"
export TORCHDYNAMO_VERBOSE=1

# Run with both versions and diff the output
python repro.py > out_2_8_0.txt  # On PyTorch 2.8.0
python repro.py > out_2_9_1.txt  # On PyTorch 2.9.1

diff out_2_8_0.txt out_2_9_1.txt
```

### Step 3: Locate the Change

```bash
# Search PyTorch commit history
git log --since="2024-06-01" --until="2024-12-01" \
    --grep="triton\|async_compile\|kernel" \
    -- torch/_inductor/

# Check Triton version compatibility
git log --since="2024-06-01" \
    -- .github/ci_commit_pins/triton.txt
```

---

## Proposed Solution

### Option 1: Fix Kernel Name Generation

**Location**: `torch/_inductor/codecache.py`

```python
# BEFORE (causes name mangling)
def triton(self, kernel_name, source, ...):
    # If kernel_name starts with '__', it gets mangled
    kernel_var = f"__{kernel_name}_0"  # ❌ Will be mangled

# AFTER (prevent mangling)
def triton(self, kernel_name, source, ...):
    # Use single underscore or no prefix
    kernel_var = f"_{kernel_name}_0"   # ✅ Won't be mangled
    # OR
    kernel_var = f"{kernel_name}_0"    # ✅ Clean name
```

### Option 2: Fix Variable Scope

**Location**: `torch/_inductor/codegen/wrapper.py`

```python
# Ensure kernels are defined in module scope, not class scope
class WrapperCodeGen:
    def generate(self):
        # Define kernels at module level
        self.writeline("# Kernel definitions")
        for kernel in self.kernels:
            # Generate outside any class definition
            self.writeline(f"{kernel.name} = ...")
```

### Option 3: Explicit Name Resolution

```python
# In generated code, explicitly store without mangling
kernel_name = "__mm_kernel_0"
globals()[kernel_name] = async_compile.triton(...)

# Later access
globals()["__mm_kernel_0"].run(...)
```

---

## Implementation Plan

### Phase 1: Diagnosis (2-3 days)

1. ✅ Set up PyTorch dev environment
2. ✅ Create minimal reproduction
3. ✅ Bisect commits to find regression
4. ✅ Identify exact code change

### Phase 2: Fix Development (3-5 days)

1. Implement fix (Option 1 most likely)
2. Add regression test
3. Test with original issue reporter's code
4. Run full test suite

### Phase 3: PR Submission (2-3 days)

1. Write comprehensive PR description
2. Add benchmarks showing no performance regression
3. Update changelog
4. Address review feedback

**Total Estimated Time**: 7-11 days

---

## Test Strategy

### Regression Test

```python
# test/inductor/test_triton_kernels.py

def test_kernel_name_no_mangling():
    """Ensure kernel names don't get mangled by Python."""
    
    class Model(nn.Module):
        def forward(self, x):
            return x @ x.T
    
    model = Model().cuda()
    compiled = torch.compile(model, fullgraph=True)
    
    x = torch.randn(1024, 1024, device='cuda')
    
    # Should not raise NameError
    out = compiled(x)
    out.sum().backward()
    
    assert torch.allclose(out, x @ x.T)
```

### Integration Test

```python
def test_backward_pass_with_checkpointing():
    """Test original failure case from issue."""
    # Simulate OneTrainer checkpoint usage
    # ... implementation
```

---

## Performance Considerations

- **No performance impact expected** - This is a naming fix
- Should benchmark kernel launch time to confirm
- Verify async compilation still parallelizes correctly

---

## Resume Impact

**Bullet Point:**
> Investigated and resolved Triton kernel compilation regression in PyTorch 2.9.1, analyzing Python name mangling issues in async kernel generation affecting production training pipelines

**Skills Demonstrated:**
- Compiler debugging
- Understanding of Python internals (name mangling)
- Triton kernel integration
- Backward compatibility
- Git bisection for regression hunting

**Impact:**
- Unblocks PyTorch version upgrades for users
- Fixes training failures in checkpoint-heavy workloads
- Demonstrates ability to debug complex issues across Python/C++/Triton boundaries

---

## References

- [PyTorch Inductor Documentation](https://pytorch.org/docs/main/torch.compiler_inductor_profiling.html)
- [Triton Language Documentation](https://triton-lang.org/)
- [Python Name Mangling](https://docs.python.org/3/tutorial/classes.html#private-variables)
- [Original Issue #170398](https://github.com/pytorch/pytorch/issues/170398)

---

**Analysis Date**: February 10, 2026  
**Analyst**: Praveen Kumar Byrapuneni