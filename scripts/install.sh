#!/bin/bash
# =============================================================================
# mHC Ascend 一键安装脚本
#
# 功能：
#   1. 编译 C++ 扩展 (mhc_ascend.so)
#   2. 安装 Python 包 (mhc)
#
# 使用方法：
#   bash scripts/install.sh          # 完整安装
#   bash scripts/install.sh --skip-build  # 仅安装 Python 包（已编译时）
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "============================================================"
echo "mHC Ascend Installation"
echo "============================================================"
echo "Project directory: $PROJECT_DIR"

# Parse arguments
SKIP_BUILD=false
for arg in "$@"; do
    case $arg in
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
    esac
done

# Step 1: Build C++ extension
if [ "$SKIP_BUILD" = false ]; then
    echo ""
    echo "[Step 1/2] Building C++ extension..."
    echo "------------------------------------------------------------"
    bash "$SCRIPT_DIR/build.sh"
else
    echo ""
    echo "[Step 1/2] Skipping C++ build (--skip-build specified)"
fi

# Step 2: Install Python package
echo ""
echo "[Step 2/2] Installing Python package..."
echo "------------------------------------------------------------"

cd "$PROJECT_DIR"

# Uninstall existing version if present
pip uninstall -y mhc-ascend 2>/dev/null || true

# Install in editable mode for development
pip install -e .

# Verify installation
echo ""
echo "============================================================"
echo "Verifying installation..."
echo "============================================================"

python3 -c "
import sys
print(f'Python: {sys.executable}')

# Import torch_npu first (required for NPU support)
try:
    import torch_npu
    print(f'torch_npu: OK')
except ImportError as e:
    print(f'torch_npu: FAILED ({e})')
    print('Note: torch_npu is required for NPU support')
    sys.exit(1)

# Check C++ extension
try:
    import mhc_ascend
    print(f'mhc_ascend: OK ({mhc_ascend.__file__})')
except ImportError as e:
    print(f'mhc_ascend: FAILED ({e})')
    sys.exit(1)

# Check Python package
try:
    from mhc import MHCLayer, MHCResidualWrapper
    print(f'mhc package: OK')
except ImportError as e:
    print(f'mhc package: FAILED ({e})')
    sys.exit(1)

print('')
print('Installation successful!')
"

echo ""
echo "============================================================"
echo "Installation complete!"
echo ""
echo "Usage:"
echo "  import torch"
echo "  import torch_npu"
echo "  from mhc import MHCLayer, MHCResidualWrapper"
echo "============================================================"
