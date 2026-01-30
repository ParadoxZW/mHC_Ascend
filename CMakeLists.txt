cmake_minimum_required(VERSION 3.16.0)
project(mHC_Ascend LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =============================================================================
# User Configuration
# =============================================================================

set(SOC_VERSION "Ascend910B1" CACHE STRING "System on Chip type")
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest" CACHE PATH "ASCEND CANN package installation directory")
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type Release/Debug")
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/install" CACHE STRING "Installation directory")

# =============================================================================
# Find Ascend C CMake Scripts
# =============================================================================

if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake does not exist, please check CANN package installation.")
endif()

include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

# =============================================================================
# Kernel Libraries
# =============================================================================

message(STATUS "Building mHC Ascend kernels for ${SOC_VERSION}")

# Build all kernel files into a single library
ascendc_library(mhc_kernels STATIC
    src/csrc/kernels/sinkhorn_knopp.cpp
    src/csrc/kernels/rmsnorm.cpp
    src/csrc/kernels/compute_rms.cpp
    src/csrc/kernels/stream_ops.cpp
    src/csrc/kernels/fused_ops.cpp
    src/csrc/kernels/mhc_layer.cpp
    src/csrc/kernels/test_vectorize.cpp
)

# Include directories for kernels
target_include_directories(mhc_kernels PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/include
)

# =============================================================================
# PyTorch Extension Library
# =============================================================================

# Find Python and PyTorch paths
execute_process(COMMAND python3 -c "import os; import torch; print(os.path.dirname(torch.__file__))"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE TORCH_PATH
)
message(STATUS "PyTorch path: ${TORCH_PATH}")

set(ENV{ASCEND_HOME_PATH} ${ASCEND_CANN_PACKAGE_PATH})
execute_process(COMMAND python3 -c "import os; import torch_npu; print(os.path.dirname(torch_npu.__file__))"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE TORCH_NPU_PATH
    RESULT_VARIABLE TORCH_NPU_RESULT
)

if(TORCH_NPU_RESULT EQUAL 0)
    message(STATUS "torch_npu path: ${TORCH_NPU_PATH}")

    # Build PyTorch binding library
    add_library(mhc_pytorch SHARED
        src/python/bindings.cpp
    )

    target_link_libraries(mhc_pytorch PRIVATE
        mhc_kernels
        torch_npu
    )

    target_link_directories(mhc_pytorch PRIVATE
        ${TORCH_PATH}/lib
        ${TORCH_NPU_PATH}/lib
    )

    target_include_directories(mhc_pytorch PRIVATE
        ${TORCH_NPU_PATH}/include
        ${TORCH_PATH}/include
        ${TORCH_PATH}/include/torch/csrc/api/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/include
        ${CMAKE_CURRENT_BINARY_DIR}/include
    )

    # Find pybind11 includes
    execute_process(COMMAND python3 -m pybind11 --includes
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE PYBIND11_INC
    )
    string(REPLACE " " ";" PYBIND11_INC ${PYBIND11_INC})

    target_compile_options(mhc_pytorch PRIVATE
        ${PYBIND11_INC}
        -D_GLIBCXX_USE_CXX11_ABI=0
    )
    target_compile_definitions(mhc_pytorch PRIVATE MHC_HOST_BUILD=1)

    # Get Python extension suffix
    execute_process(COMMAND python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE PYBIND11_SUFFIX
    )

    set_target_properties(mhc_pytorch PROPERTIES
        OUTPUT_NAME mhc_ascend${PYBIND11_SUFFIX}
        PREFIX ""
        SUFFIX ""
    )

    # Get Python site-packages directory
    execute_process(
        COMMAND python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))"
        OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    message(STATUS "Python site-packages: ${PYTHON_SITE_PACKAGES}")

    # Install PyTorch extension to site-packages
    install(TARGETS mhc_pytorch
        LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES}
    )
else()
    message(WARNING "torch_npu not found. PyTorch extension will not be built.")
endif()

# =============================================================================
# Tests (Optional)
# =============================================================================

option(BUILD_TESTS "Build test executables" ON)

if(BUILD_TESTS)
    # Test executables would go here
    # For now, tests are Python-based
    message(STATUS "Python tests will be run separately")
endif()

# =============================================================================
# Installation
# =============================================================================

install(TARGETS mhc_kernels
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)

install(DIRECTORY src/csrc/include/
    DESTINATION ${CMAKE_INSTALL_PREFIX}/include
    FILES_MATCHING PATTERN "*.h"
)

# =============================================================================
# Summary
# =============================================================================

message(STATUS "")
message(STATUS "=========================================")
message(STATUS "mHC Ascend Build Configuration")
message(STATUS "=========================================")
message(STATUS "SOC Version:        ${SOC_VERSION}")
message(STATUS "Build Type:         ${CMAKE_BUILD_TYPE}")
message(STATUS "CANN Path:          ${ASCEND_CANN_PACKAGE_PATH}")
message(STATUS "Install Prefix:     ${CMAKE_INSTALL_PREFIX}")
if(TORCH_NPU_RESULT EQUAL 0)
    message(STATUS "PyTorch Extension:  Enabled")
else()
    message(STATUS "PyTorch Extension:  Disabled")
endif()
message(STATUS "=========================================")
message(STATUS "")
