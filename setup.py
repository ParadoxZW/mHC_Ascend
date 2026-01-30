"""
Setup script for mHC Ascend Python package.

Note: The C++ extension (mhc_ascend.so) must be built separately using CMake.
This setup.py only installs the Python wrapper package.

Installation:
    1. Build C++ extension: bash scripts/build.sh
    2. Install Python package: pip install -e .

Or use the combined script: bash scripts/install.sh
"""

from setuptools import setup, find_packages
import os
import sys

# Check if C++ extension is available
def check_cpp_extension():
    try:
        import mhc_ascend
        return True
    except ImportError:
        return False

# Version
VERSION = "0.1.0"

# Read README
readme_path = os.path.join(os.path.dirname(__file__), "README.md")
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "mHC (Manifold-Constrained Hyper-Connections) for Huawei Ascend NPUs"

setup(
    name="mhc-ascend",
    version=VERSION,
    author="mHC Development Team",
    description="mHC implementation for Huawei Ascend 910B NPUs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/anthropics/mHC-ascend",

    # Package discovery
    packages=find_packages(where="src/python"),
    package_dir={"": "src/python"},

    # Python version
    python_requires=">=3.8",

    # Dependencies
    install_requires=[
        "torch>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=23.0",
            "flake8>=6.0",
        ],
    },

    # Classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)

# Post-install check
if "install" in sys.argv or "develop" in sys.argv:
    if not check_cpp_extension():
        print("\n" + "=" * 60)
        print("WARNING: mhc_ascend C++ extension not found!")
        print("Please build the C++ extension first:")
        print("  cd mHC_ascend && bash scripts/build.sh")
        print("=" * 60 + "\n")
