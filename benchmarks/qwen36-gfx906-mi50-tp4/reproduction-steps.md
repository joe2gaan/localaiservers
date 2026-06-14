# Reproduction Steps

The canonical technical deployment package is
[qwen36-gfx906/README.md](../../qwen36-gfx906/README.md).

Use that README for:

- Docker image names.
- Docker Hub digest.
- Docker archive hashes.
- Source pins.
- Build commands.
- Deploy commands.
- Run commands.
- Benchmark commands.
- Hardware requirements.
- Limitations.

## High-Level Prerequisites

- 4x AMD Instinct MI50 32GB / GFX906-class target hardware.
- ROCm-compatible host environment suitable for the documented runtime.
- Docker and local storage sufficient for the documented image/runtime path.
- Access to model weights under the model provider's terms.

## High-Level Flow

1. Read [qwen36-gfx906/README.md](../../qwen36-gfx906/README.md).
2. Use the documented prebuilt image or reproducible build path.
3. Start the vLLM service with the documented command.
4. Run the documented benchmark harness.
5. Compare backend decode TPS and client wall TPS separately.

No private host details are required to reproduce the public method.
