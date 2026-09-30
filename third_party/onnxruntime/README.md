# third_party/onnxruntime

`onnxruntime_c_api.h` is the ONNX Runtime C API header, taken from the ONNX
Runtime distribution (MIT licensed, © Microsoft Corporation).

It is vendored rather than fetched because the ONNX bridge in `src/layatrt_onnx.cpp`
resolves `onnxruntime.dll` at runtime and never links against the import library:
the header is the only build-time dependency the ONNX kernel has. Keeping it here
means `cmake` can build both kernels with no network access and no ORT SDK
install, and a machine without ONNX Runtime still starts every binary in this
repo.

The header version declares `ORT_API_VERSION 17`. A runtime that reports an
older API version fails the load with "ONNX Runtime C API version is
incompatible" rather than misbehaving; upgrade the header and the DLL together.

To build against a different ONNX Runtime, point CMake at its include directory:

```
cmake -S . -B build -DORT_HEADER_DIR="C:/path/to/onnxruntime/include"
```
