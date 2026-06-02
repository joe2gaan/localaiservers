# LocalAIServers

Deployable local AI server configurations and reproducible performance notes.

## Qwen3.6-35B-A3B on 4x MI50 32GB

The first release target is a reproducible `gfx906` runtime for:

```text
Model: Qwen/Qwen3.6-35B-A3B
Hardware target: 4x AMD Instinct MI50 32GB
Parallelism: TP4
Runtime image: joe2gaan/localaiservers:qwen36-gfx906-c1-topk8-runtime-archive-235f4780cbe1
```

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

DEPLOY_IMAGE=joe2gaan/localaiservers:qwen36-gfx906-c1-topk8-runtime-archive-235f4780cbe1 \
USE_PREBUILT_IMAGE=1 \
PREBUILT_IMAGE_PULL=1 \
AUTO_STAGE_MODEL=1 \
./deploy.sh
```

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
