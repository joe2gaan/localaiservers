# Reddit Draft: r/LocalAIServers

## Suggested Title

Qwen3.6-35B-A3B at 90+ tok/s sustained on 4x AMD Instinct MI50 32GB, with a reproducible gfx906 Docker runtime

## Post

I have been tuning `Qwen/Qwen3.6-35B-A3B` on a `4x AMD Instinct MI50 32GB` server.

The result is a reproducible `gfx906` runtime that holds **90+ tokens/sec over a 10,000-token single-request decode** on old MI50 hardware, while still running the model as an unquantized 16-bit deployment.

This started as a low-30 TPS single-request path. The promoted TP4 profile is now in a range where this class of server is genuinely useful again for long local generations.

```text
Model: Qwen/Qwen3.6-35B-A3B
Hardware target: 4x AMD Instinct MI50 32GB
GPU arch: gfx906
Parallelism: TP4
Serving dtype: --dtype half
Context setting: --max-model-len 131072
Runtime: vLLM + ROCm + gfx906 patches + tuned MoE config
```

This is not a quantized AWQ/GGUF result. The vLLM launch uses `--dtype half`, so the precise wording is FP16 execution / BF16-tier local service, not native BF16 math.

## Results

The best promoted reference run:

```text
c1_2000 fixed-token decode:    101.47 TPS backend decode
c1_10000 fixed-token decode:    95.66 TPS backend decode
c1_10000 client wall rate:       95.36 output tokens/sec
```

The release image was then rebuilt cleanly on two separate gfx906 hosts from the same `deploy.sh`, pushed to Docker Hub, and speed-tested again:

```text
Clean rebuild A:
c1_2000 backend decode:          94.73 TPS
c1_10000 backend decode:         90.58 TPS
c1_10000 client wall rate:       90.51 output tokens/sec

Clean rebuild B:
c1_2000 backend decode:          95.17 TPS
c1_10000 backend decode:         90.63 TPS
c1_10000 client wall rate:       90.55 output tokens/sec
```

The conservative claim I am making for the release is therefore **90+ TPS sustained over a 10K-token single-request decode** on 4x MI50 32GB.

The benchmark is intentionally synthetic and fixed-token:

- one request at a time
- `max_tokens=min_tokens`
- `ignore_eos=true`
- live stream enabled
- TPS measured from vLLM generation-token metrics and client wall clock

Natural prompts can measure lower because prefill length, reasoning behavior, stop conditions, and answer style change the workload. The point of this test is to isolate sustained decode throughput.

Because this is a thinking model, I separated correctness checks from throughput checks. Correctness smoke tests are uncapped and only validate after the model has completed the thinking trace through the parser split. The fixed-token `c1_2000` and `c1_10000` tests are throughput measurements, not answer-quality tests.

## What Changed From The Baseline

The biggest gains came from stacking several changes that all mattered:

- TP4 serving shape for 4x 32GB MI50.
- Qwen C1 topk8 MoE fastpath.
- Shared-expert / route path patches from the tuning campaign.
- Tuned `E=256,N=128` MoE config for this model and hardware shape.
- vLLM async scheduling.
- Keeping `-O=3`; `-O=0` is diagnostic-only and should not be used for performance numbers.
- `--language-model-only`.
- Qwen3 reasoning parser and Hermes tool-call parser kept in the serving stack.
- C1 single-request decode used as the primary gate instead of optimizing only for aggregate batch throughput.
- RCCL/NCCL settings that reduce communication overhead for this lane:

```text
NCCL_ALGO=Tree
NCCL_PROTO=LL
NCCL_P2P_DISABLE=1
NCCL_MAX_NCHANNELS=1
```

The lesson from the tuning work is that this was not one magic flag. The win came from making the MoE path, tensor-parallel shape, graph/compile path, and communication settings line up with the actual gfx906 hardware.

## Exact vLLM Launch

The image entrypoint turns the runtime environment into this vLLM command:

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --served-model-name Qwen/Qwen3.6-35B-A3B \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --dtype half \
  --host 0.0.0.0 \
  --port 8001 \
  --tensor-parallel-size 4 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.95 \
  --trust-remote-code \
  --generation-config vllm \
  -O=3 \
  --async-scheduling \
  --reasoning-parser qwen3 \
  --language-model-only
```

The full Docker run command, mounts, cache paths, ROCm devices, and environment variables are in the README.

## Reproducible Package

GitHub:

```text
https://github.com/joe2gaan/localaiservers
```

Docker Hub:

```text
joe2gaan/localaiservers
```

The Docker Hub image is runtime-only, not weight-bundled. Model weights are mounted through the local Hugging Face cache. That keeps the image pull practical while still letting users skip the long native ROCm/vLLM build.

Current runtime tag:

```text
joe2gaan/localaiservers:qwen36-gfx906-c1-topk8-runtime-archive-aa34cb675f83
```

Docker Hub manifest digest:

```text
sha256:f5e69ee127b766960e386e0e4eda8e26c399bd02f57c494847cb9a92ce04d8ac
```

Docker Hub config digest / tested local image ID:

```text
sha256:e45309183e6f35cae6fb8f9d8d6f016253f281a5e7187e1f11a57e5e28ef5e86
```

Two independent clean rebuilds produced the same exported Docker archive:

```text
aa34cb675f83ff6cade31cbbb357b1c31d793bee18da491f501d7c39fda3612a  ./.repro-docker-archives/qwen36-gfx906-c1-topk8-fastpath-reproducible.docker.tar
```

The `deploy.sh` used for that reproducibility run:

```text
0392affe7194f35d5e596c7e0f6b29f65f84c4e38f6e281952332f298a9c1991  deploy.sh
```

The loaded image is about 66 GB. The exported Docker archive observed in testing was about 16 GB. The full working directory can be much larger because it contains the model cache, runtime cache, private Docker root, and archive.

## Run From Docker Hub

```bash
mkdir -p ~/qwen36-gfx906-run
cd ~/qwen36-gfx906-run

curl -fsSL https://raw.githubusercontent.com/joe2gaan/localaiservers/main/qwen36-gfx906/deploy.sh -o deploy.sh
curl -fsSL https://raw.githubusercontent.com/joe2gaan/localaiservers/main/qwen36-gfx906/run_qwen36_live_tps.py -o run_qwen36_live_tps.py
chmod +x deploy.sh

DEPLOY_IMAGE=joe2gaan/localaiservers:qwen36-gfx906-c1-topk8-runtime-archive-aa34cb675f83 \
USE_PREBUILT_IMAGE=1 \
PREBUILT_IMAGE_PULL=1 \
AUTO_STAGE_MODEL=1 \
./deploy.sh
```

After vLLM is ready:

```bash
python3 ./run_qwen36_live_tps.py
```

## Build From Source Instead

The package is designed to build from public sources instead of depending on a private base image. The single `deploy.sh` writes its Dockerfile, entrypoint, runtime patches, MoE config, compose file, and helper files into the directory where it is executed.

```bash
mkdir -p ~/qwen36-gfx906-build
cd ~/qwen36-gfx906-build

curl -fsSL https://raw.githubusercontent.com/joe2gaan/localaiservers/main/qwen36-gfx906/deploy.sh -o deploy.sh
chmod +x deploy.sh

./deploy.sh
```

Current build path:

```text
Base image: pinned Ubuntu 24.04/noble image
ROCm package path: pinned ROCm 6.3.4 package set
PyTorch ROCm wheels: torch 2.9.1+rocm6.3
Triton: pinned gfx906 source commit
FlashAttention: pinned gfx906 source commit
vLLM: pinned ai-infos/vllm-gfx906-mobydick source commit
Runtime: bundled patch overlays + tuned MoE config
Build exporter: pinned daemonless BuildKit with timestamp rewrite
```

The script keeps generated files under the directory where it is executed. Docker/containerd state defaults to:

```text
./.d
```

That matters because large Docker image exports can otherwise fill `/var/lib/docker` or `/var/lib/containerd` even when the intended build directory has plenty of free space.

## Minimum Target Host

```text
4x AMD Instinct MI50 32GB
gfx906-compatible ROCm host driver stack
Docker + docker compose
large NVMe working directory
network access during first build/model staging unless cache/model files are already present
```

The script has guardrails:

- Requires 4 visible GPUs by default.
- Requires at least 32 GiB VRAM per GPU.
- Auto-selects compatible gfx906 GPUs instead of assuming the first four devices are always the right lane.
- Failed disk-space checks are fatal.
- GPU VRAM failures warn and default to NO unless the user explicitly continues.
- Every sudo action explains exactly what it is doing, prints the exact `sudo ...` command, and requires `y` or `yes`; blank input defaults to NO and exits.
- Docker/containerd state is isolated under the execution directory by default.
- The ready check waits for `/v1/models` before reporting deployment complete.

## Why This Is Interesting

At 90.5 output tokens/sec, this profile produces roughly:

```text
325,800 output tokens/hour
7.82 million output tokens/day
```

At the promoted 95.36 output tokens/sec run, it is roughly:

```text
343,296 output tokens/hour
8.24 million output tokens/day
```

This is not a claim that 4x MI50 beats modern datacenter GPUs in absolute throughput. H100-class systems still have higher ceilings, especially with FP8 and high-concurrency serving.

The claim is value per local token. A 4x MI50 32GB server is old hardware, but with the right serving path it can run a current MoE model locally at useful long-decode throughput without quantizing the model down to make it fit.

That is the part I think matters for local AI servers: the machine is fully local, the model is not tiny, the 10K decode number stays above 90 TPS, and the serving profile keeps reasoning-parser and tool-call support in the stack.

## Reproduction Request

I am especially interested in results from other `4x AMD Instinct MI50 32GB` systems.

Useful reports would include:

- build success/failure
- ROCm version
- motherboard / PCIe topology
- strict uncapped thinking smoke result
- `c1_2000` and `c1_10000` fixed-token decode TPS
- whether the result holds with the same TP4 config
- power draw if measured
- tool-calling behavior in your client
- Qwen reasoning parser behavior in your client
- SHA256 of the exported Docker archive if you try the reproducibility path

The current target is Qwen3.6-35B-A3B TP4. The next obvious directions are better single-request latency, better TP8 behavior, and seeing how much of this tuning transfers to other MoE and dense models.

## Short Version

4x AMD Instinct MI50 32GB is not dead hardware for local LLM serving.

With a tuned gfx906 TP4 path, `Qwen/Qwen3.6-35B-A3B` moved from roughly ~33 TPS baseline behavior to 90+ TPS sustained over a 10K-token single-request decode. The release package now has both a Docker Hub runtime image and a source-build path through one deploy script.

That is enough performance to make this class of server genuinely interesting again.
