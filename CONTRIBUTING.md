# Contributing

## Development setup

```bash
pip install -e ".[dev]"
pytest tests/ -q          # CPU-only; GPU-gated tests skip without CUDA
```

With a CUDA GPU + nvcc, enable the JIT-compiled extension (PowerShell/cmd on Windows):

```powershell
$env:SPEC_DECODE_JIT_CUDA = "1"
pytest tests/ -q
```

## The one rule: reference parity

`src/reference.py` is the executable specification. Any change to a CUDA kernel
must keep `tests/test_ops_dispatch.py` parity tests passing — CUDA output must
match the NumPy reference (exactly for integer/bitmask ops, `rtol=1e-4` for
float reductions). If you change intended behavior, change the reference first,
then make the kernel match.

## Guidelines

- New kernels ship in pairs: NumPy reference + CUDA implementation + parity test
- Benchmarks that back README claims are committed to `results/` as JSON
- Measured numbers and design targets stay clearly separated in docs
- Python: black-compatible formatting; C++/CUDA: match the existing style

## Reporting issues

Include your GPU model, CUDA version, PyTorch version, and whether the failure
occurs on the reference path (`use_cuda=False`) or the CUDA path.
