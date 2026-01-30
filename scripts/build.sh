#!/bin/bash

# =============================================================================
# Build script for mHC Ascend implementation
# =============================================================================

set -e  # Exit on error

echo "========================================"
echo "Building mHC for Ascend 910B1"
echo "========================================"

# Check if CANN toolkit is installed
if [ -z "$ASCEND_HOME_PATH" ]; then
    echo "ASCEND_HOME_PATH not set. Trying default location..."
    if [ -d "/usr/local/Ascend/ascend-toolkit/latest" ]; then
        export ASCEND_HOME_PATH="/usr/local/Ascend/ascend-toolkit/latest"
        echo "Found CANN toolkit at: $ASCEND_HOME_PATH"
    else
        echo "ERROR: CANN toolkit not found. Please install CANN or set ASCEND_HOME_PATH"
        exit 1
    fi
fi

# Source CANN environment
if [ -f "$ASCEND_HOME_PATH/set_env.sh" ]; then
    echo "Sourcing CANN environment..."
    source "$ASCEND_HOME_PATH/set_env.sh"
else
    echo "WARNING: set_env.sh not found in $ASCEND_HOME_PATH"
fi

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Project root: $PROJECT_ROOT"

# Create build directory (clean rebuild)
BUILD_DIR="$PROJECT_ROOT/build"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo ""
echo "Configuring with CMake..."
cmake .. \
    -DSOC_VERSION=Ascend910B1 \
    -DASCEND_CANN_PACKAGE_PATH="$ASCEND_HOME_PATH" \
    -DCMAKE_BUILD_TYPE=Release

echo ""
echo "Building..."
make -j$(nproc)

echo ""
echo "Installing..."
make install

echo ""
echo "========================================"
echo "Build completed successfully!"
echo "========================================"
echo ""
echo "Built artifacts:"
echo "  - Kernel library: $PROJECT_ROOT/install/lib/libmhc_kernels.a"
echo "  - Headers: $PROJECT_ROOT/install/include/"
if [ -f "$PROJECT_ROOT/install/python/mhc_ascend"*.so ]; then
    echo "  - Python extension: $PROJECT_ROOT/install/python/mhc_ascend*.so"
fi
echo ""
echo "To install Python package:"
echo "  cd $PROJECT_ROOT && pip install -e ."
echo ""
