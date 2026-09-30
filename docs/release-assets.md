# Release Assets

Large files are intentionally distributed as GitHub Release assets instead of Git history.
The source repository remains reviewable and cloneable without downloading GPU runtimes or
model weights.

## Expected assets

- `onnx-runtime-windows-x64.zip`: unpack at the repository root; contains `Assets/onnx/`
- `laya_ctx8192.onnx`: baseline exported model
- `laya_ctx8192.opt.onnx`: optimized exported model
- `laya_ctx8192.opt.fp16.onnx`: FP16 optimized exported model
- optional `laya-engines-windows-x64.zip`: GPU- and TensorRT-specific engine files
- `SHA256SUMS.txt`: checksums for all assets

## Install

```powershell
Get-FileHash .\onnx-runtime-windows-x64.zip -Algorithm SHA256
Expand-Archive .\onnx-runtime-windows-x64.zip -DestinationPath .
```

Copy model assets to `onnx\`, or set `engine_path` to their absolute path in a local
`layatrt.config.json`. Keep that local configuration untracked.

The model assets and vendor runtime binaries have licenses separate from the source
repository. Review `LICENSE`, `NOTICE.md`, and `docs/dependencies.md` before sharing
the assets or using them in a publication.
