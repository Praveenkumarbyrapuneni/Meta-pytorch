# How to Contribute to PyTorch - Practical Guide

**Based on Analysis of Issues #170398, #135859, #147170**  
**Author**: Praveen Kumar Byrapuneni

---

## Phase 1: Preparation (Before Contributing)

### 1. Development Environment Setup

```bash
# Fork the repository on GitHub first
git clone https://github.com/YOUR_USERNAME/pytorch
cd pytorch

# Add upstream remote
git remote add upstream https://github.com/pytorch/pytorch

# Create conda environment
conda create -n pytorch-dev python=3.11
conda activate pytorch-dev

# Install dependencies
conda install cmake ninja numpy pyyaml setuptools cffi typing_extensions
conda install -c pytorch magma-cuda121  # For CUDA support

# Build in development mode
USE_CUDA=1 python setup.py develop

# Verify installation
python -c "import torch; print(torch.__version__)"
python test/run_test.py test_ops  # Run a test
```

### 2. Understanding the Codebase

**Must-read files for our issues:**

```bash
# For Issue #170398 (Triton kernel regression)
torch/_inductor/codecache.py              # Async compilation
torch/_inductor/codegen/wrapper.py        # Code generation
torch/_inductor/runtime/triton_heuristics.py  # Kernel management

# For Issue #135859 (Recompilation)
torch/fx/experimental/symbolic_shapes.py  # Shape constraints
torch/_dynamo/variables/builder.py        # Guard generation
torch/_inductor/compile_fx.py             # Compilation pipeline

# For Issue #147170 (Quantization determinism)
torch/ao/quantization/pt2e/utils.py       # Graph partitioning
torch/fx/passes/graph_manipulation.py     # Graph traversal
```

---

## Phase 2: Issue Investigation

### Step 1: Reproduce the Issue

**Create test file**: `test_issue_XXXXX.py`

```python
#!/usr/bin/env python3
"""
Reproduction script for issue #XXXXX

Usage:
    python test_issue_XXXXX.py
    
Expected: [describe expected behavior]
Actual: [describe actual behavior]
"""

import torch
import sys

def reproduce_issue():
    # Minimal reproduction code from issue
    pass

if __name__ == "__main__":
    try:
        reproduce_issue()
        print("✓ Issue reproduced successfully")
    except Exception as e:
        print(f"✗ Reproduction failed: {e}")
        sys.exit(1)
```

### Step 2: Enable Debugging

```bash
# Create debug script
cat > debug_issue.sh << 'EOF'
#!/bin/bash

# Enable all relevant logs
export TORCH_LOGS="+dynamo,+aot,+inductor,+symbolic_shapes"
export TORCHDYNAMO_VERBOSE=1
export TORCH_COMPILE_DEBUG=1

# For profiling
export TORCHDYNAMO_REPORT_GUARD_FAILURES=1

# Run test
python test_issue_XXXXX.py 2>&1 | tee debug_output.log
EOF

chmod +x debug_issue.sh
./debug_issue.sh
```

### Step 3: Find the Root Cause

**Use git blame and log:**

```bash
# Find recent changes to relevant files
git log --since="6 months ago" --oneline -- torch/_inductor/codecache.py

# Find who introduced a specific line
git blame torch/_inductor/codecache.py

# Search for related issues
gh issue list --search "triton kernel" --state all
```

**Add strategic print statements:**

```python
# In torch/_inductor/codecache.py
def triton(self, kernel_name, source, ...):
    print(f"[DEBUG] Generating kernel: {kernel_name}")
    print(f"[DEBUG] Scope: {type(self).__name__}")
    
    # ... existing code
    
    print(f"[DEBUG] Final kernel variable: {kernel_var}")
```

---

## Phase 3: Implementing the Fix

### Step 1: Create Feature Branch

```bash
# Update from upstream
git fetch upstream main
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b fix/issue-170398-triton-kernel-name
```

### Step 2: Implement Solution

**Follow PyTorch conventions:**

```python
# Good: Type hints, docstrings, clear variable names
def generate_kernel_name(
    base_name: str, 
    index: int,
    avoid_mangling: bool = True
) -> str:
    """Generate unique kernel name.
    
    Args:
        base_name: Base name for kernel (e.g., 'mm_kernel')
        index: Unique index for this kernel
        avoid_mangling: If True, avoids Python name mangling
        
    Returns:
        Kernel name that won't be mangled by Python
        
    Example:
        >>> generate_kernel_name('mm_kernel', 0)
        '_mm_kernel_0'  # Single underscore, not double
    """
    # Use single underscore to avoid __name mangling
    prefix = "_" if avoid_mangling else "__"
    return f"{prefix}{base_name}_{index}"
```

### Step 3: Add Tests

**Create test file**: `test/inductor/test_kernel_naming.py`

```python
import unittest
import torch
from torch.testing._internal.common_utils import TestCase, run_tests

class TestKernelNaming(TestCase):
    
    def test_kernel_name_no_mangling(self):
        """Test that kernel names don't get mangled."""
        
        @torch.compile(fullgraph=True)
        def model(x):
            return x @ x.T
        
        x = torch.randn(128, 128, device='cuda')
        
        # Should not raise NameError
        out = model(x)
        
        # Verify correctness
        expected = x @ x.T
        self.assertEqual(out, expected)
    
    def test_backward_with_checkpointing(self):
        """Test backward pass with gradient checkpointing.
        
        Reproduces original issue scenario.
        """
        import torch.utils.checkpoint as checkpoint
        
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(1024, 1024)
            
            def forward(self, x):
                return checkpoint.checkpoint(
                    self.linear, x, use_reentrant=False
                )
        
        model = Model().cuda()
        compiled = torch.compile(model)
        
        x = torch.randn(128, 1024, device='cuda', requires_grad=True)
        out = compiled(x)
        
        # Backward should not raise NameError
        out.sum().backward()
        
        self.assertIsNotNone(x.grad)

if __name__ == '__main__':
    run_tests()
```

**Run tests:**

```bash
# Run your new test
python test/inductor/test_kernel_naming.py

# Run related test suites
python test/run_test.py test_inductor
python test/run_test.py test_dynamo
```

---

## Phase 4: Creating the Pull Request

### Step 1: Commit Changes

```bash
# Stage changes
git add torch/_inductor/codecache.py
git add test/inductor/test_kernel_naming.py

# Commit with descriptive message
git commit -m "Fix Triton kernel name mangling causing NameError

- Use single underscore prefix to avoid Python name mangling
- Add test for backward pass with checkpointing
- Fixes #170398

The issue occurred because kernel names starting with '__' were
being mangled by Python (e.g., __kernel → _Runner__kernel).
This change ensures kernel names use a single underscore prefix,
which is not subject to name mangling.

Tested:
- test_kernel_name_no_mangling: Verifies no NameError
- test_backward_with_checkpointing: Original issue scenario
"
```

### Step 2: Push and Create PR

```bash
# Push to your fork
git push origin fix/issue-170398-triton-kernel-name

# Create PR on GitHub
gh pr create --title "Fix Triton kernel name mangling in async compilation" \
             --body "$(cat pr_description.md)"
```

**PR Description Template** (`pr_description.md`):

```markdown
## Description

Fixes #170398 

This PR fixes a regression in PyTorch 2.9.1 where Triton kernel names 
starting with `__` (double underscore) were being mangled by Python,
causing `NameError` at runtime.

## Root Cause

Python automatically mangles names starting with `__` inside class 
definitions:
- `__kernel_name_0` → `_ClassName__kernel_name_0`

The kernel registration used `__` prefix, but the generated code 
expected the unmangled name, causing a mismatch.

## Solution

Use single underscore `_` prefix instead of double `__` for kernel 
names. Single underscore is a Python convention for "internal" names
but is not subject to name mangling.

## Testing

- Added `test_kernel_name_no_mangling` - basic functionality
- Added `test_backward_with_checkpointing` - original issue scenario
- Ran full `test_inductor` suite - all passing
- Tested on PyTorch 2.8.0 compatibility - works

## Performance

No performance impact - this is purely a naming convention change.

Benchmark results:
```
Operation        | Before | After | Δ
-----------------|--------|-------|----
Forward pass     | 1.2ms  | 1.2ms | 0%
Backward pass    | 2.1ms  | 2.1ms | 0%
Compilation time | 1.8s   | 1.8s  | 0%
```

## Checklist

- [x] PR title clearly describes the change
- [x] Added tests for the fix
- [x] Ran relevant test suites
- [x] Updated documentation (if needed)
- [x] Benchmarked performance

cc @chauhang @penguinwu (tagged from original issue)
```

---

## Phase 5: Review Process

### Responding to Feedback

```bash
# Address reviewer comments
git add <modified_files>
git commit -m "Address review feedback: <summary>"
git push origin fix/issue-170398-triton-kernel-name

# If major changes needed, create new branch
git checkout -b fix/issue-170398-v2
# Make changes
git push origin fix/issue-170398-v2
gh pr create --base main
```

### Common Review Requests

1. **Add more tests**
   - Edge cases (empty tensors, 1D tensors, etc.)
   - Different dtypes
   - Different devices

2. **Performance benchmarks**
   ```python
   # Add to benchmarks/dynamo/
   @benchmark
   def bench_kernel_launch():
       ...
   ```

3. **Documentation**
   ```python
   # Add to docs/source/compile/
   def example_usage():
       """How to use this feature..."""
   ```

---

## Resume Impact: Documenting Your Work

### On GitHub Profile

**Create a showcase repository**:

```markdown
# PyTorch Contributions

## Pull Requests

### [#XXXXX] Fix Triton kernel name mangling
**Status**: Merged  
**Impact**: Fixed regression blocking PyTorch 2.9.1 adoption

**Technical Details**:
- Debugged Python name mangling issue in compiler backend
- Reduced kernel launch failures by 100%
- Implemented comprehensive test coverage

**Skills**: Compiler debugging, Python internals, Triton
```

### On Resume

```
OPEN SOURCE CONTRIBUTIONS

PyTorch (Meta) - Deep Learning Framework                    2026
• Fixed critical Triton kernel compilation regression (#170398) 
  preventing users from upgrading to PyTorch 2.9.1
• Optimized torch.compile() recompilation logic (#135859), 
  reducing compilation overhead by 90% for variable batch sizes  
• Resolved quantization non-determinism bug (#147170) enabling
  reproducible model deployment
  
Skills: Python, C++, CUDA, Triton, compiler optimization, 
        performance debugging
```

### In Interviews

**STAR Format Example**:

**Situation**: PyTorch users couldn't upgrade to 2.9.1 due to NameError in compiled models  
**Task**: Debug compilation regression affecting Triton kernel generation  
**Action**: 
- Analyzed generated code and identified Python name mangling issue
- Bisected 500+ commits to find introducing change
- Implemented fix using single underscore prefix
- Added comprehensive tests covering edge cases

**Result**: 
- Fix merged, unblocking thousands of users
- Learned compiler internals deeply
- Gained recognition in PyTorch community

---

## Tips for Success

### 1. Start Small
- Don't tackle multiple issues simultaneously
- Get one PR merged before starting another
- Build credibility with maintainers

### 2. Communication
- Comment on issue before starting work
- Ask for guidance if stuck >2 days
- Be responsive to review feedback

### 3. Quality Over Speed
- Comprehensive tests > quick fix
- Clean code > clever code
- Good documentation > obvious code

### 4. Learn from Rejections
- Not all PRs get merged - that's OK!
- Extract learning even from closed PRs
- Maintainers' feedback is valuable

---

## Resources

### Official Guides
- [PyTorch Contributing Guide](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md)
- [Code Style Guide](https://github.com/pytorch/pytorch/wiki/Code-style-guide)
- [Testing Guidelines](https://github.com/pytorch/pytorch/wiki/Testing-guidelines)

### Community
- [PyTorch Dev Discussions](https://dev-discuss.pytorch.org/)
- [PyTorch Discord](https://discord.gg/pytorch)
- [Weekly Office Hours](https://github.com/pytorch/pytorch/wiki/Dev-Infra-Office-Hours)

### Learning
- [PyTorch Internals Blog](http://blog.ezyang.com/)
- [TorchDynamo Technical Deep Dive](https://dev-discuss.pytorch.org/t/torchdynamo-update-1/630)
- [Code Reading Sessions](https://www.youtube.com/c/PyTorch/playlists)

---

**Good luck with your contributions!**  
**Remember**: Every expert was once a beginner. Start small, learn continuously, and be patient.

---

**Guide Version**: 1.0  
**Last Updated**: February 10, 2026  
**Author**: Praveen Kumar Byrapuneni