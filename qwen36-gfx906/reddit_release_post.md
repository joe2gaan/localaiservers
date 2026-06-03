# Reddit Draft: r/LocalAIServers

## Suggested Title

Qwen3.6-35B-A3B on 4x AMD Instinct MI50 32GB: ~33 TPS baseline to 90+ TPS sustained 10K decode

## Post

I am preparing to release a single-script reproducible deployment package for `Qwen/Qwen3.6-35B-A3B`, tuned specifically for a `4x AMD Instinct MI50 32GB` server.

The interesting part is not just that it runs. The interesting part is that older `gfx906` hardware can serve a current MoE model at strong interactive speeds when the kernel path, MoE routing, and inter-GPU communication are tuned for the actual hardware.

This campaign started around the low-30 TPS range for single-request decode. The promoted TP4 profile is now producing:

```text
Model: Qwen/Qwen3.6-35B-A3B
Hardware: 4x AMD Instinct MI50 32GB
GPU arch: gfx906
Parallelism: TP4
Serving precision flag: --dtype half
Context setting: --max-model-len 131072

c1_2000 fixed-token decode:   101.47 TPS backend decode
c1_10000 fixed-token decode:   95.66 TPS backend decode
c1_10000 client wall rate:      95.36 output tokens/sec
```

The 10K-token result is the one I care about most. Short decode tests are useful, but they can hide behavior that falls apart once the request shape starts looking more like a real workload. Holding above 90 TPS over a 10K decode on 4x MI50 32GB is the number that makes this worth sharing.

This is an unquantized 16-bit deployment path. The vLLM command uses `--dtype half`, so the precise wording is FP16 execution / BF16-tier local service, not native BF16 math.

Because this is a thinking model, I separated correctness checks from throughput checks. The smoke test is uncapped and only validates the answer after the model has emitted a complete thinking trace through the parser split. The fixed-token `c1_2000` and `c1_10000` tests are throughput measurements.

## What Changed From The Baseline

The largest gains came from stacking several changes that each attacked a different bottleneck:

- Settled on a TP4 serving shape that fits the model cleanly across 4x 32GB GPUs.
- Enabled the Qwen C1 topk8 MoE fastpath.
- Added/overlaid the Qwen MoE fastpath and shared-expert routing patches used by the tuning campaign.
- Used the tuned `E=256,N=128` MoE kernel config for this model/hardware shape.
- Kept the C1 single-request path as the primary tuning target instead of optimizing only for aggregate batch throughput.
- Enabled vLLM async scheduling and kept `-O=3`.
- Kept `--language-model-only` in the serving command.
- Used the Qwen3 reasoning parser and Hermes tool-call parser in the runtime profile.
- Reduced communication overhead with the promoted NCCL/RCCL settings:

```text
NCCL_ALGO=Tree
NCCL_PROTO=LL
NCCL_P2P_DISABLE=1
NCCL_MAX_NCHANNELS=1
```

The key lesson from this run is that gfx906 performance was not limited by one magic flag. The win came from making the MoE path, tensor-parallel shape, and communication settings line up with the hardware.

We also found a few things that did not belong in the final config:

- Do not use `-O=0` for performance testing. It is useful only as a diagnostic.
- Do not validate thinking-model hashes from truncated output. If the closing think path has not completed, the output is incomplete.
- Do not treat profiler overhead runs as performance candidates. They are useful for attribution, not final TPS claims.

## Reproducible Package

The package is designed to build from public sources instead of depending on a private base image. The release artifact is intended to be a single `deploy.sh` that writes its Dockerfile, entrypoint, runtime patches, MoE config, compose file, and helper files as heredocs into the directory where it is executed.

I am using the same namespace for the release package and the prebuilt runtime image:

```text
GitHub:    https://github.com/joe2gaan/localaiservers
Docker Hub: joe2gaan/localaiservers
```

The Docker Hub tag is runtime-only, not weight-bundled. That keeps the pull practical while still giving users the tuned ROCm/vLLM/gfx906 runtime without waiting through the native build. Model weights are mounted through the local Hugging Face cache.

Currently published runtime tag:

```text
joe2gaan/localaiservers:qwen36-gfx906-c1-topk8-runtime-archive-aa34cb675f83
```

Currently published Docker Hub manifest digest:

```text
sha256:f5e69ee127b766960e386e0e4eda8e26c399bd02f57c494847cb9a92ce04d8ac
```

The source runtime archive used for that published tag was verified at `aa34cb675f83ff6cade31cbbb357b1c31d793bee18da491f501d7c39fda3612a`; clean rebuilds on `.20` and `.30` produced the same archive hash. The Docker Hub registry config digest matches the tested local image ID: `sha256:e45309183e6f35cae6fb8f9d8d6f016253f281a5e7187e1f11a57e5e28ef5e86`.

Source-build path:

```bash
mkdir -p ~/qwen36-gfx906-build
cd ~/qwen36-gfx906-build
./deploy.sh
```

Prebuilt-runtime path after the image is published:

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

python3 ./run_qwen36_live_tps.py
```

The archive-hash tag is the one I would cite for reproduction because it identifies the verified source archive lineage. For exact Docker Hub identity, cite the manifest digest. The `latest` tag is only for convenience.

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

The deploy script writes its generated build/runtime files into the directory where it is executed. The current package also defaults to a private Docker + containerd root under:

```text
./.d
```

That matters because large Docker image exports can otherwise fill `/var/lib/docker` or `/var/lib/containerd` even when the actual build directory has plenty of free space.

The reproducibility contract is the SHA256 of the exported Docker archive, not the Docker Engine image ID after load. The archive-hash tag cited here was produced by a clean rebuild started from a directory containing only `deploy.sh`:

```text
0392affe7194f35d5e596c7e0f6b29f65f84c4e38f6e281952332f298a9c1991  deploy.sh
```

Two validated clean rebuilds, on `.20` and `.30`, produced this Docker archive:

```text
aa34cb675f83ff6cade31cbbb357b1c31d793bee18da491f501d7c39fda3612a  ./.repro-docker-archives/qwen36-gfx906-c1-topk8-fastpath-reproducible.docker.tar
```

The final loaded image is about 66 GB. The exported archive observed in testing was about 16 GB, and the full build/deploy working directory is much larger because it contains the model cache, runtime cache, private Docker root, and archive.

## Minimum Target Host

```text
4x AMD Instinct MI50 32GB
gfx906-compatible ROCm host driver stack
Docker + docker compose
large NVMe working directory
network access during first build/model staging unless the cache/model is already present
```

The script has guardrails:

- Requires 4 visible GPUs by default.
- Requires at least 32 GiB VRAM per GPU.
- Auto-selects compatible gfx906 GPUs instead of assuming the first four devices are always the right lane.
- Failed disk-space checks are fatal.
- GPU VRAM failures warn and default to NO unless the user explicitly continues.
- Every sudo action explains exactly what it is doing, prints the exact `sudo ...` command, and requires `y` or `yes`; blank, non-interactive input, or any other answer defaults to NO and exits.
- Docker/containerd state is isolated under the execution directory by default.
- The ready check waits for `/v1/models` before reporting deployment complete.

## Why This Is Interesting

Using the measured 95.36 output tokens/sec result, this profile is approximately:

```text
343,296 output tokens/hour
8.24 million output tokens/day
```

This is not a claim that 4x MI50 beats modern datacenter GPUs in absolute throughput. H100-class systems still have higher ceilings, especially with FP8 and high-concurrency serving.

The claim is value per local token. A 4x MI50 32GB server is old hardware, but with the right serving path it can produce useful long-decode throughput on a current MoE model without quantizing the model down to make it fit.

That is the part I think matters for local AI servers: the machine is fully local, the model is not tiny, the 10K decode number stays strong, and the serving profile keeps reasoning-parser and tool-call support in the stack.

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

With a tuned gfx906 TP4 path, `Qwen/Qwen3.6-35B-A3B` moved from roughly ~33 TPS baseline behavior to 90+ TPS sustained over a 10K-token single-request decode, with the current reproducible package building from public sources through a single deploy script.

That is enough performance to make this class of server genuinely interesting again.
