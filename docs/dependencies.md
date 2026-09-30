# Dependencies and Licenses

This project separates launcher source code from model assets and native runtime packages.
Check the license files shipped with each Release asset before redistribution.

| Component | Version used for validation | Role | License / terms |
|---|---:|---|---|
| Go | 1.27 | launcher and service | Go license; see upstream distribution |
| Wails | v2.13.0 | desktop shell | MIT; see upstream project |
| React / React DOM | 19.1 | GUI | MIT; see upstream packages |
| Vite | 7.x | frontend build | MIT; see upstream package |
| TypeScript | 5.6.x | frontend build | Apache-2.0; see upstream package |
| ONNX Runtime CUDA | 1.28.0 | ONNX CUDA provider | Microsoft package terms; preserve `LICENSE` and `ThirdPartyNotices.txt` |
| ONNX Runtime DirectML | 1.24.4 | ONNX DirectML provider | Microsoft package terms; preserve `LICENSE` and `ThirdPartyNotices.txt` |
| CUDA Toolkit | 12/13 as installed | TensorRT/ONNX native dependencies | NVIDIA license/EULA |
| TensorRT | 10.16 | TensorRT provider | NVIDIA license/EULA |
| DirectML | system/provider dependency | Windows ONNX provider | Microsoft terms |
| laya model and tokenizer | release-specific | model inference asset | Upstream model license; verify before redistribution |

## Redistribution rules

- The project `LICENSE` applies only to original Laya Go Launcher source code.
- Do not claim that ONNX Runtime, CUDA, TensorRT, DirectML, Wails, or model files
  are authored by this project.
- Keep every upstream `LICENSE`, `Privacy.md`, `VERSION_NUMBER`, and
  `ThirdPartyNotices.txt` file beside the corresponding runtime package.
- A Release asset containing model files is not automatically covered by the
  source license. Its asset notes and upstream model terms control.
- NVIDIA and Microsoft runtime packages may have separate redistribution
  conditions. Obtain the required vendor permission for commercial distribution.

## Reproducibility record

For each experiment, record the project Release tag, asset SHA-256, model SHA-256,
GPU, driver, CUDA version, TensorRT version, and ONNX Runtime provider. Do not
publish local absolute paths, tokens, or private checkpoints.
