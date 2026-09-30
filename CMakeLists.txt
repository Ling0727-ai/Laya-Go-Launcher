cmake_minimum_required(VERSION 3.18)
project(qualityscaler_tensorrt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Point these at your installs, or pass -DTENSORRT_ROOT=... -DCUDA_ROOT=...
if(NOT TENSORRT_ROOT)
    set(TENSORRT_ROOT "C:/TensorRT-10.16.0.72")
endif()
if(NOT CUDA_ROOT)
    set(CUDA_ROOT "C:/CUDA")
endif()

find_path(TRT_INCLUDE_DIR
          NvInfer.h
          PATHS "${TENSORRT_ROOT}/include"
          NO_DEFAULT_PATH)
if(NOT TRT_INCLUDE_DIR)
    message(FATAL_ERROR "NvInfer.h not found under ${TENSORRT_ROOT}/include")
endif()

find_library(TRT_LIBRARY
             NAMES nvinfer_10 nvinfer
             PATHS "${TENSORRT_ROOT}/lib"
             NO_DEFAULT_PATH)
if(NOT TRT_LIBRARY)
    message(FATAL_ERROR "nvinfer not found under ${TENSORRT_ROOT}/lib")
endif()

set(CUDAToolkit_ROOT "${CUDA_ROOT}")
find_package(CUDAToolkit REQUIRED)

# The kernel keeps the upstream name so existing cgo LDFLAGS (-lqualityscaler_tensorrt)
# continue to resolve.
add_library(qualityscaler_tensorrt SHARED src/ai_tensorrt.cpp)

target_include_directories(qualityscaler_tensorrt PRIVATE
                           "${TRT_INCLUDE_DIR}"
                           "${CMAKE_CURRENT_SOURCE_DIR}/include"
                           ${CUDAToolkit_INCLUDE_DIRS})

target_link_libraries(qualityscaler_tensorrt PRIVATE "${TRT_LIBRARY}" CUDA::cudart)

set_target_properties(qualityscaler_tensorrt PROPERTIES
                      PREFIX ""
                      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
                      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

if(MSVC)
    target_compile_options(qualityscaler_tensorrt PRIVATE /O2 /wd4819 /utf-8)
    target_compile_definitions(qualityscaler_tensorrt PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
else()
    target_compile_options(qualityscaler_tensorrt PRIVATE -O3)
endif()

# ── ONNX Runtime bridge ─────────────────────────────────────────────────────
#
# The second kernel. It resolves onnxruntime.dll at runtime rather than linking
# it, so building this needs only the C API header and never the import library;
# a machine without ONNX Runtime can still start every binary in this repo.
#
# Pass -DORT_HEADER_DIR=... when the header is not in third_party/onnxruntime.
if(NOT ORT_HEADER_DIR)
    set(ORT_HEADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime")
endif()

find_path(ORT_INCLUDE_DIR
          onnxruntime_c_api.h
          PATHS "${ORT_HEADER_DIR}"
          NO_DEFAULT_PATH)
if(NOT ORT_INCLUDE_DIR)
    message(WARNING
            "onnxruntime_c_api.h not found under ${ORT_HEADER_DIR}; "
            "skipping the ONNX Runtime kernel. Place the header there or pass "
            "-DORT_HEADER_DIR=<dir> to build it.")
else()
    add_library(layatrt_onnx SHARED src/layatrt_onnx.cpp)
    target_include_directories(layatrt_onnx PRIVATE
                               "${ORT_INCLUDE_DIR}"
                               "${CMAKE_CURRENT_SOURCE_DIR}/include")
    set_target_properties(layatrt_onnx PROPERTIES
                          PREFIX ""
                          RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
                          LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
    if(MSVC)
        target_compile_options(layatrt_onnx PRIVATE /O2 /wd4819 /utf-8)
        target_compile_definitions(layatrt_onnx PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
    else()
        target_compile_options(layatrt_onnx PRIVATE -O3)
    endif()
endif()
