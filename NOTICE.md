# Laya Go Launcher Notices

Copyright (c) 2026 Ling0727-ai / Lingxin.

## Project source

The original Laya Go Launcher source code is licensed under the terms in `LICENSE`.
This is a source-available research and non-commercial license, not an OSI-approved
open-source license.

## Model and runtime separation

The laya model, exported ONNX files, TensorRT engine files, tokenizer/configuration
files, and any training artifacts are separate assets. Their use and redistribution
rights are not granted by the project source license. Verify the model's upstream
license and obtain permission before redistribution or publication.

The ONNX Runtime packages under `Assets/onnx` are third-party Microsoft packages.
Their package-level `LICENSE`, `ThirdPartyNotices.txt`, `Privacy.md`, version files,
and upstream terms must be preserved when those binaries are redistributed.
CUDA, TensorRT, NVIDIA libraries, DirectML, and Wails are also third-party
components governed by their respective licenses and EULAs.

Do not remove, replace, or merge these notices into the project license.
