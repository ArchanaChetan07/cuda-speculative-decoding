"""Setup script for Speculative Decoding Optimizer.

CPU-only by default. To build the CUDA extension (requires nvcc + PyTorch):

    PAGEDKV_FORCE_CUDA=1 pip install -e .
"""

import os

from setuptools import setup, find_packages

ext_modules = []
cmdclass = {}

if os.environ.get("PAGEDKV_FORCE_CUDA"):
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    ext_modules = [
        CUDAExtension(
            name="spec_decode_opt._C",
            sources=[
                "csrc/bindings.cpp",
                "csrc/speculative_decode_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        )
    ]
    cmdclass = {"build_ext": BuildExtension}

setup(
    name="spec-decode-optimizer",
    version="0.1.0",
    description="Speculative Decoding Optimizer: CUDA kernels for conflict-free KV cache eviction",
    author="Speculative Decoding Team",
    license="Apache-2.0",
    python_requires=">=3.10",
    # src/ IS the package: find_packages(where="src") would look for
    # subpackages inside it and find nothing, installing zero code
    packages=["spec_decode_opt"],
    package_dir={"spec_decode_opt": "src"},
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "isort>=5.0",
            "mypy>=1.0",
        ],
        "cuda": [
            "torch>=2.0.0",
        ],
    },
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
