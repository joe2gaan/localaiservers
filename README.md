# LocalAIServers

Deployable local AI server configurations and reproducible performance notes.

## Qwen3.6-35B-A3B on 4x MI50 32GB

The first release target is a reproducible `gfx906` runtime for:

```text
Model: Qwen/Qwen3.6-35B-A3B
Hardware target: 4x AMD Instinct MI50 32GB
Parallelism: TP4
Runtime image: joe2gaan/localaiservers:qwen36-gfx906-c1-topk8-runtime-archive-aa34cb675f83
Docker Hub digest: sha256:f5e69ee127b766960e386e0e4eda8e26c399bd02f57c494847cb9a92ce04d8ac
```

## Live TPS Video

<video controls preload="metadata" width="100%" src="https://joe2gaan.github.io/localaiservers/qwen36-gfx906/media/qwen36_ref20_machiavelli_100_to_1000_tps_720p.mp4">
  <a href="https://joe2gaan.github.io/localaiservers/qwen36-gfx906/media/">Watch the Qwen3.6-35B-A3B gfx906 live TPS video.</a>
</video>

Watch the playable GitHub Pages version: https://joe2gaan.github.io/localaiservers/qwen36-gfx906/media/

Start here:

```bash
cd qwen36-gfx906
```

To run from the prebuilt Docker Hub runtime image without rebuilding:

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

The image entrypoint launches vLLM with the tested TP4/O3/Tree-LL command. The command inside the container is:

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

The currently published Docker Hub image is the 29-layer runtime produced by clean source rebuilds on `.20` and `.30`. Both rebuilds produced the same source archive SHA-256: `aa34cb675f83ff6cade31cbbb357b1c31d793bee18da491f501d7c39fda3612a`. The public Docker Hub manifest digest is `sha256:f5e69ee127b766960e386e0e4eda8e26c399bd02f57c494847cb9a92ce04d8ac`, and the registry config digest matches the tested local image ID: `sha256:e45309183e6f35cae6fb8f9d8d6f016253f281a5e7187e1f11a57e5e28ef5e86`.

After the vLLM service is ready:

```bash
python3 ./run_qwen36_live_tps.py
```

Reference fixed-token single-request results on the validated 4x MI50 32GB lane:

```text
c1_2000:  101.47 TPS backend decode
c1_10000:  95.66 TPS backend decode
c1_10000:  95.36 client wall TPS
```

See [qwen36-gfx906/README.md](qwen36-gfx906/README.md) for the full deployment and reproduction notes.
