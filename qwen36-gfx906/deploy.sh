#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(pwd -P)"
DEFAULT_DEPLOY_ENV_PATH="${RUN_DIR}/.deploy.defaults.env"

write_embedded_deploy_env() {
  local env_file="$1"
  mkdir -p "$(dirname "${env_file}")"
  cat <<'DEPLOY_ENV' > "${env_file}"
#!/usr/bin/env bash
# Default tuning values for the Qwen3.6-35B-A3B async TP4 C1 topk8 fastpath profile.
# Target host: 4x AMD Instinct MI50 32GB.
# Uses ":=" assignment so caller-supplied env vars stay authoritative.
: "${MODEL:=Qwen/Qwen3.6-35B-A3B}"
: "${SERVED_MODEL_NAME:=Qwen/Qwen3.6-35B-A3B}"
: "${PORT:=8001}"
: "${TP_SIZE:=4}"
: "${HOST:=0.0.0.0}"
: "${MAX_MODEL_LEN:=131072}"
: "${GPU_MEMORY_UTILIZATION:=0.95}"
: "${TOOL_CALL_PARSER:=hermes}"
: "${REASONING_PARSER:=qwen3}"
: "${VLLM_ENABLE_C1_TOPK8_MOE_FASTPATH:=1}"
: "${VLLM_DTYPE:=half}"
: "${OPT_LEVEL:=3}"
: "${EXTRA_VLLM_ARGS:=--language-model-only}"
: "${DISABLE_ASYNC_SCHEDULING:=0}"
: "${NCCL_ALGO:=Tree}"
: "${NCCL_PROTO:=LL}"
: "${NCCL_P2P_DISABLE:=1}"
: "${NCCL_MAX_NCHANNELS:=1}"
: "${FLASH_ATTENTION_TRITON_AMD_ENABLE:=TRUE}"
: "${FLASH_ATTENTION_TRITON_AMD_REF:=TRUE}"
: "${OMP_NUM_THREADS:=4}"
: "${HIP_VISIBLE_DEVICES:=auto}"
: "${GPU_AUTO_SELECT:=1}"
: "${GPU_TARGET_ARCH:=gfx906}"
: "${GPU_TARGET_ARCH_ALIASES:=gfx906,gfx9006}"
: "${REQUIRED_GPU_MODEL_HINT:=4x gfx906 AMD GPUs with >=32GiB VRAM each}"
: "${REQUIRED_GPU_COUNT:=4}"
: "${REQUIRED_GPU_VRAM_GIB:=32}"
: "${REQUIRED_GPU_FREE_VRAM_GIB:=}"
: "${SKIP_GPU_MEMORY_CHECK:=0}"
: "${SKIP_GPU_FREE_MEMORY_CHECK:=0}"
: "${AUTO_CONTINUE_GPU_CHECK:=0}"
: "${HF_CACHE_DIR:=./hf_cache}"
: "${RUNTIME_ROOT:=./runtime}"
: "${MOE_CONFIG_DIR:=./runtime/vllm_tuned_moe_configs/native_gfx906_qwen36_tp4_n128_c1_parallel_variants_20260525/m8_n32_k32_w2_wave1_k1_llmm1_rpb2_naivec1_nccl_tree_ll_10k_host30}"
: "${VLLM_TUNED_CONFIG_FOLDER:=/opt/vllm_tuned_moe_configs}"
: "${CONTAINER_NAME:=vllm_qwen36_gfx906_c1_topk8_fastpath_m8n32_rpb2}"
: "${DEPLOY_IMAGE:=qwen36-gfx906-c1-topk8-fastpath-reproducible}"
: "${USE_PREBUILT_IMAGE:=0}"
: "${PREBUILT_IMAGE_PULL:=1}"
: "${STAGE_IMAGE:=python:3.12-slim}"
: "${RESTART_POLICY:=no}"
: "${BUNDLED_HOTFIX_DIR:=./.bundled_hotfixes}"
: "${RUNTIME_PATCH_DIR:=./runtime/patches}"
: "${BUNDLE_PATCH_TARGET:=/opt/vllm_patch_bundle}"
: "${DOCKERFILE_PATH:=./Dockerfile}"
: "${QWEN36_PYTHON_SRC_DIR:=}"
: "${ROCM_PATCH_SRC_FILE:=}"
: "${TRITON_ATTENTION_BACKEND_SRC_FILE:=}"
: "${TRITON_ATTENTION_OP_SRC_FILE:=}"
: "${PATCH_SRC_PR43190_DIR:=}"
: "${PATCH_SRC_FASTPATH_FUSED_MOE:=}"
: "${PATCH_SRC_PR41457:=}"
: "${PATCH_SRC_UTILS:=}"
: "${AUTO_STAGE_MODEL:=0}"
: "${HF_TOKEN:=}"
: "${MODEL_REPO_ID:=Qwen/Qwen3.6-35B-A3B}"
: "${AUTO_STAGE_MODEL_FORCE:=0}"
: "${DOCKER_ISOLATED_DAEMON_ENABLED:=1}"
: "${DOCKER_ISOLATED_DAEMON_DIR:=./.d}"
: "${DOCKER_ISOLATED_CLEANUP_ON_FAILURE:=1}"
: "${DOCKER_ISOLATED_DELETE_ROOT_ON_FAILURE:=1}"
: "${DOCKER_TEMP_DATA_ROOT_ENABLED:=0}"
: "${DOCKER_TEMP_DATA_ROOT_DIR:=./.docker-data-root}"
: "${DOCKER_TEMP_DATA_ROOT_CLEANUP:=1}"
: "${SKIP_DISK_SPACE_CHECK:=0}"
: "${MIN_FREE_GIB_HF_CACHE:=120}"
: "${MIN_FREE_GIB_RUNTIME_ROOT:=120}"
: "${MIN_FREE_GIB_BUILD_CONTEXT:=20}"
: "${MIN_FREE_GIB_DOCKER_ROOT:=200}"
: "${MIN_FREE_GIB_TEMP_DOCKER_DATA_ROOT:=200}"
: "${INSTALL_BUNDLED_MOE_CONFIG:=1}"
: "${REFRESH_MOE_CONFIG:=0}"
: "${MOE_CONFIG_BASENAME:=E=256,N=128,device_name=AMD_GFX906.json}"
: "${PREFER_REASONING_PARSER:=qwen3}"
: "${BUILDKIT_PROGRESS:=plain}"
: "${COMPOSE_PROGRESS:=plain}"
: "${COMPOSE_ANSI:=never}"
: "${FORCE_REBUILD:=0}"
: "${BUILD_ONLY:=0}"
: "${DOCKER_REPRODUCIBLE_BUILDX_EXPORT:=1}"
: "${DOCKER_REPRODUCIBLE_EXPORT_MODE:=buildctl-daemonless}"
: "${DOCKER_REPRODUCIBLE_EXPORTER_PROBE:=1}"
: "${REPRO_EXPORT_DOCKER_ARCHIVE:=1}"
: "${REPRO_DOCKER_LOAD_ARCHIVE:=1}"
: "${REPRO_DOCKER_ARCHIVE_DIR:=./.repro-docker-archives}"
: "${REPRO_DOCKER_ARCHIVE_NAME:=${DEPLOY_IMAGE//\//_}.docker.tar}"
: "${REPRO_DOCKER_ARCHIVE_PATH:=}"
: "${BYTE_FOR_BYTE_VALIDATION_MODE:=auto}"
: "${EXPECTED_REPRO_DOCKER_ARCHIVE_SHA256:=}"
: "${MIN_FREE_GIB_REPRO_DOCKER_ARCHIVE:=180}"
: "${PINNED_BUILDX_ENABLED:=1}"
: "${PINNED_BUILDX_VERSION:=0.30.1}"
: "${PINNED_BUILDX_URL_AMD64:=https://github.com/docker/buildx/releases/download/v${PINNED_BUILDX_VERSION}/buildx-v${PINNED_BUILDX_VERSION}.linux-amd64}"
: "${PINNED_BUILDX_SHA256_AMD64:=c37114fcd034025ec68e224657c8a5a850df472ded3ddcbca75ad3a7ebb9710d}"
: "${PINNED_BUILDKIT_ENABLED:=1}"
: "${PINNED_BUILDKIT_VERSION:=0.26.1}"
: "${PINNED_BUILDKIT_IMAGE:=moby/buildkit:v${PINNED_BUILDKIT_VERSION}@sha256:9b7bf602ee3b53e88a0711f3c6124f169fbbe6301db4e1e499bbddb406d20700}"
: "${PINNED_BUILDKIT_BUILDER_NAME:=qwen36-repro-buildkit}"
: "${PINNED_BUILDKIT_BUILDER_NAME_SCOPE:=run-dir}"
: "${PINNED_BUILDKIT_RM_TIMEOUT_SECONDS:=30}"
: "${PINNED_BUILDKIT_INSPECT_TIMEOUT_SECONDS:=300}"
: "${CHECK_GPU_ONLY:=0}"
: "${DOCKER_TEMP_DATA_ROOT_ACTIVE:=0}"
: "${WAIT_FOR_READY_TIMEOUT:=3600}"
: "${SKIP_READY_CHECK:=0}"
# The runtime cache seed is optional. On some Docker/containerd builds the
# one-shot docker-run attach path can block after image export, so default to
# first-start cache population for portability.
: "${SKIP_TRITON_CACHE_SEED:=1}"
: "${TRITON_CACHE_SEED_TIMEOUT_SECONDS:=180}"
DEPLOY_ENV
}

if [[ -f "${SCRIPT_DIR}/deploy.env" ]]; then
  source "${SCRIPT_DIR}/deploy.env"
else
  write_embedded_deploy_env "${DEFAULT_DEPLOY_ENV_PATH}"
  source "${DEFAULT_DEPLOY_ENV_PATH}"
fi

if ! command -v docker >/dev/null 2>&1; then
  echo 'error: docker is not available in PATH' >&2
  exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
  echo 'error: docker compose is not available' >&2
  exit 1
fi

if ! command -v df >/dev/null 2>&1; then
  echo 'error: df is required for disk-space validation' >&2
  exit 1
fi

if ! command -v tar >/dev/null 2>&1; then
  echo 'error: tar is required for Triton cache seeding' >&2
  exit 1
fi

if ! command -v timeout >/dev/null 2>&1; then
  echo 'warning: timeout is not available; optional Triton cache seeding will be skipped' >&2
  SKIP_TRITON_CACHE_SEED=1
fi

abs_path() {
  local path="$1"
  local base="${2:-${SCRIPT_DIR}}"
  if [[ -z "${path}" ]]; then
    echo ""
    return 0
  fi
  if [[ "${path}" == /* ]]; then
    echo "${path}"
  else
    echo "${base}/${path}"
  fi
}

write_embedded_dockerfile() {
  local dockerfile_path="$1"
  mkdir -p "$(dirname "${dockerfile_path}")"

  cat <<'DOCKERFILE' > "${dockerfile_path}"
ARG SOURCE_DATE_EPOCH=1764000000
FROM ubuntu:noble-20250127@sha256:72297848456d5d37d1262630108ab308d3e9ec7ed1c3286a32fe09856619a782 AS vllm-gfx906-repro-build
ARG SOURCE_DATE_EPOCH=1764000000

# Reproduce ai-infos/vllm-gfx906-mobydick:qwen3.6-35B-A3B internals from public sources.
ARG DEBIAN_FRONTEND=noninteractive
ENV DEBIAN_FRONTEND=${DEBIAN_FRONTEND}
ENV ROCM_VERSION=6.3.4
ENV AMDGPU_VERSION=6.3.4
ENV PYTORCH_INDEX_URL=https://download.pytorch.org/whl/rocm6.3
ENV TORCH_VERSION=2.9.1+rocm6.3
ENV TORCHVISION_VERSION=0.24.1+rocm6.3
ENV TORCHAUDIO_VERSION=2.9.1+rocm6.3
ENV PYTORCH_TRITON_ROCM_VERSION=3.5.1
ENV AMDSMI_VERSION=6.3.3
ENV MAX_JOBS=32
ENV VLLM_COMMIT=6a52668c14dc9cb94a270d94bfec735fe627ed0c
ENV VLLM_PACKAGE_VERSION=0.0.0+gfx906
ENV VLLM_VERSION_OVERRIDE=0.0.0+gfx906
ENV TRITON_KERNELS_FETCHCONTENT_REF=7c56a5e40f7fd928dfd5c72902d5def0097db73a
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0+gfx906
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM=0.0.0+gfx906
ENV SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH}
ENV VLLM_TARGET_DEVICE=rocm
ENV PYTORCH_ROCM_ARCH=gfx906
ENV GPU_ARCHS=gfx906
ENV AMDGPU_TARGETS=gfx906
ENV CMAKE_HIP_ARCHITECTURES=gfx906
ENV FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
ENV FLASH_ATTENTION_TRITON_AMD_REF=TRUE
ENV LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:
ENV PATH=/opt/venv/bin:/opt/rocm/bin:/opt/rocm/llvm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV PYTHONPATH=/opt/qwen36-python
ENV PYTHONUNBUFFERED=1
ENV PYTHONHASHSEED=0
ENV PIP_NO_CACHE_DIR=1
ENV PIP_NO_COMPILE=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV TZ=Etc/UTC
ENV REPRO_BUILD_ROOT=/tmp/repro-build
ENV REPRO_WHEEL_ROOT=/tmp/repro-wheels
ENV REPRO_TMPDIR=/tmp/repro-tmp
ENV UBUNTU_BOOTSTRAP_SNAPSHOT=20250501T000000Z
ENV UBUNTU_VLLM_SNAPSHOT=20260315T000000Z

ARG TRITON_REPO=https://github.com/ai-infos/triton-gfx906.git
ARG TRITON_REF=eee3139afb2f12651d82e5068787a342bebbf57e
ARG FLASH_ATTENTION_REPO=https://github.com/ai-infos/flash-attention-gfx906.git
ARG FLASH_ATTN_REPO_REF=0ac8e77b2a6cf773ecf17bc486e1a11fe1e066e0
ARG VLLM_SRC_REPO=https://github.com/ai-infos/vllm-gfx906-mobydick.git

RUN mkdir -p /var/tmp/apt-cache/partial && \
    printf 'Acquire::Retries "20";\nAcquire::http::Timeout "60";\nAcquire::https::Timeout "60";\nAcquire::http::Pipeline-Depth "0";\nAcquire::https::Pipeline-Depth "0";\nAcquire::Check-Valid-Until "false";\n' > /etc/apt/apt.conf.d/99repro-network-retries && \
    printf "deb https://snapshot.ubuntu.com/ubuntu/${UBUNTU_BOOTSTRAP_SNAPSHOT} noble main universe restricted multiverse\n" > /etc/apt/sources.list && \
    printf "deb https://snapshot.ubuntu.com/ubuntu/${UBUNTU_BOOTSTRAP_SNAPSHOT} noble-updates main universe restricted multiverse\n" >> /etc/apt/sources.list && \
    printf "deb https://snapshot.ubuntu.com/ubuntu/${UBUNTU_BOOTSTRAP_SNAPSHOT} noble-backports main universe restricted multiverse\n" >> /etc/apt/sources.list && \
    printf "deb https://snapshot.ubuntu.com/ubuntu/${UBUNTU_BOOTSTRAP_SNAPSHOT} noble-security main universe restricted multiverse\n" >> /etc/apt/sources.list && \
    rm -f /etc/apt/sources.list.d/ubuntu.sources && \
    apt-get update -o Acquire::https::Verify-Peer=false -o Acquire::Check-Valid-Until=false && \
    apt-get install -y --no-install-recommends -o Acquire::https::Verify-Peer=false -o Dir::Cache::archives=/var/tmp/apt-cache ca-certificates && \
    apt-get update -o Acquire::Check-Valid-Until=false && \
    apt-get install -y --no-install-recommends -o Dir::Cache::archives=/var/tmp/apt-cache \
      curl \
      libnuma-dev \
      gnupg && \
    rm -rf /var/lib/apt/lists/* /var/tmp/apt-cache/*

RUN printf "Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600\n" > /etc/apt/preferences.d/rocm-pin-600 && \
    mkdir -p /etc/apt/keyrings && \
    curl --fail --location --show-error --silent --retry 20 --retry-all-errors --connect-timeout 30 --max-time 300 https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor -o /etc/apt/keyrings/rocm.gpg && \
    printf "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/${ROCM_VERSION}/ noble main\n" > /etc/apt/sources.list.d/rocm.list && \
    printf "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/amdgpu/${AMDGPU_VERSION}/ubuntu noble main\n" > /etc/apt/sources.list.d/amdgpu.list && \
    apt-get update -o Acquire::Check-Valid-Until=false || apt-get update -o Acquire::AllowInsecureRepositories=true -o Acquire::AllowDowngradeToInsecureRepositories=true && \
    apt-get install -y --no-install-recommends -o Dir::Cache::archives=/var/tmp/apt-cache \
      sudo \
      libelf1 \
      kmod \
      file \
      python3-dev \
      python3-pip \
      rocm-dev \
      rocm-libs \
      build-essential && \
    rm -rf /var/lib/apt/lists/* /var/tmp/apt-cache/*

RUN set -eu; \
    for spec in \
      "amd-smi-lib=25.1.0.60304-76~24.04" \
      "amdgpu-core=1:6.3.60304-2125197.24.04" \
      "comgr=2.8.0.60304-76~24.04" \
      "composablekernel-dev=1.1.0.60304-76~24.04" \
      "half=1.12.0.60304-76~24.04" \
      "hip-dev=6.3.42134.60304-76~24.04" \
      "hip-doc=6.3.42134.60304-76~24.04" \
      "hip-runtime-amd=6.3.42134.60304-76~24.04" \
      "hip-samples=6.3.42134.60304-76~24.04" \
      "hipblas=2.3.0.60304-76~24.04" \
      "hipblas-common-dev=1.0.0.60304-76~24.04" \
      "hipblas-dev=2.3.0.60304-76~24.04" \
      "hipblaslt=0.10.0.60304-76~24.04" \
      "hipblaslt-dev=0.10.0.60304-76~24.04" \
      "hipcc=1.1.1.60304-76~24.04" \
      "hipcub-dev=3.3.0.60304-76~24.04" \
      "hipfft=1.0.17.60304-76~24.04" \
      "hipfft-dev=1.0.17.60304-76~24.04" \
      "hipify-clang=18.0.0.60304-76~24.04" \
      "hiprand=2.11.1.60304-76~24.04" \
      "hiprand-dev=2.11.1.60304-76~24.04" \
      "hipsolver=2.3.0.60304-76~24.04" \
      "hipsolver-dev=2.3.0.60304-76~24.04" \
      "hipsparse=3.1.2.60304-76~24.04" \
      "hipsparse-dev=3.1.2.60304-76~24.04" \
      "hipsparselt=0.2.2.60304-76~24.04" \
      "hipsparselt-dev=0.2.2.60304-76~24.04" \
      "hiptensor=1.4.0.60304-76~24.04" \
      "hiptensor-dev=1.4.0.60304-76~24.04" \
      "hsa-amd-aqlprofile=1.0.0.60304-76~24.04" \
      "hsa-rocr=1.14.0.60304-76~24.04" \
      "hsa-rocr-dev=1.14.0.60304-76~24.04" \
      "miopen-hip=3.3.0.60304-76~24.04" \
      "miopen-hip-dev=3.3.0.60304-76~24.04" \
      "openmp-extras-dev=18.63.0.60304-76~24.04" \
      "openmp-extras-runtime=18.63.0.60304-76~24.04" \
      "rccl=2.21.5.60304-76~24.04" \
      "rccl-dev=2.21.5.60304-76~24.04" \
      "rocalution=3.2.1.60304-76~24.04" \
      "rocalution-dev=3.2.1.60304-76~24.04" \
      "rocblas=4.3.0.60304-76~24.04" \
      "rocblas-dev=4.3.0.60304-76~24.04" \
      "rocfft=1.0.31.60304-76~24.04" \
      "rocfft-dev=1.0.31.60304-76~24.04" \
      "rocm-cmake=0.14.0.60304-76~24.04" \
      "rocm-core=6.3.4.60304-76~24.04" \
      "rocm-dbgapi=0.77.0.60304-76~24.04" \
      "rocm-debug-agent=2.0.3.60304-76~24.04" \
      "rocm-dev=6.3.4.60304-76~24.04" \
      "rocm-device-libs=1.0.0.60304-76~24.04" \
      "rocm-gdb=15.2.60304-76~24.04" \
      "rocm-libs=6.3.4.60304-76~24.04" \
      "rocm-llvm=18.0.0.25012.60304-76~24.04" \
      "rocm-opencl=2.0.0.60304-76~24.04" \
      "rocm-opencl-dev=2.0.0.60304-76~24.04" \
      "rocm-smi-lib=7.4.0.60304-76~24.04" \
      "rocm-utils=6.3.4.60304-76~24.04" \
      "rocminfo=1.0.0.60304-76~24.04" \
      "rocprim-dev=3.3.0.60304-76~24.04" \
      "rocprofiler=2.0.60304.60304-76~24.04" \
      "rocprofiler-dev=2.0.60304.60304-76~24.04" \
      "rocprofiler-plugins=2.0.60304.60304-76~24.04" \
      "rocprofiler-register=0.4.0.60304-76~24.04" \
      "rocprofiler-sdk=0.5.0-76~24.04" \
      "rocprofiler-sdk-roctx=0.5.0-76~24.04" \
      "rocrand=3.2.0.60304-76~24.04" \
      "rocrand-dev=3.2.0.60304-76~24.04" \
      "rocsolver=3.27.0.60304-76~24.04" \
      "rocsolver-dev=3.27.0.60304-76~24.04" \
      "rocsparse=3.3.0.60304-76~24.04" \
      "rocsparse-dev=3.3.0.60304-76~24.04" \
      "rocthrust-dev=3.3.0.60304-76~24.04" \
      "roctracer=4.1.60304.60304-76~24.04" \
      "roctracer-dev=4.1.60304.60304-76~24.04" \
      "rocwmma-dev=1.6.0.60304-76~24.04"; do \
        pkg="${spec%%=*}"; want="${spec#*=}"; got="$(dpkg-query -W -f='${Version}' "${pkg}" 2>/dev/null || true)"; \
        if [ "${got}" != "${want}" ]; then \
          echo "ROCm apt version drift for ${pkg}: expected ${want}, got ${got:-missing}" >&2; \
          exit 1; \
        fi; \
    done

RUN printf "deb https://snapshot.ubuntu.com/ubuntu/${UBUNTU_VLLM_SNAPSHOT} noble main universe restricted multiverse\n" > /etc/apt/sources.list && \
    printf "deb https://snapshot.ubuntu.com/ubuntu/${UBUNTU_VLLM_SNAPSHOT} noble-updates main universe restricted multiverse\n" >> /etc/apt/sources.list && \
    printf "deb https://snapshot.ubuntu.com/ubuntu/${UBUNTU_VLLM_SNAPSHOT} noble-backports main universe restricted multiverse\n" >> /etc/apt/sources.list && \
    printf "deb https://snapshot.ubuntu.com/ubuntu/${UBUNTU_VLLM_SNAPSHOT} noble-security main universe restricted multiverse\n" >> /etc/apt/sources.list && \
    apt-get update -o Acquire::Check-Valid-Until=false && \
    apt-get install -y --no-install-recommends -o Dir::Cache::archives=/var/tmp/apt-cache \
      build-essential \
      ca-certificates \
      cmake \
      curl \
      git \
      libelf-dev \
      libffi-dev \
      libjpeg-dev \
      libnuma-dev \
      libopenblas-dev \
      libssl-dev \
      ninja-build \
      pkg-config \
      python3 \
      python3-dev \
      python3-pip \
      python3-venv \
      wget \
      zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

RUN set -eu; \
    for spec in \
      "build-essential=12.10ubuntu1" \
      "curl=8.5.0-2ubuntu10.8" \
      "dpkg=1.22.6ubuntu6.1" \
      "dpkg-dev=1.22.6ubuntu6.1" \
      "git=1:2.43.0-1ubuntu7.3" \
      "libc-bin=2.39-0ubuntu8.4" \
      "libc6:amd64=2.39-0ubuntu8.4" \
      "libcurl4t64:amd64=8.5.0-2ubuntu10.8" \
      "libdpkg-perl=1.22.6ubuntu6.1" \
      "libssl-dev:amd64=3.0.13-0ubuntu3.7" \
      "libssl3t64:amd64=3.0.13-0ubuntu3.7" \
      "openssl=3.0.13-0ubuntu3.7" \
      "python3-dev=3.12.3-0ubuntu2.1" \
      "python3.12=3.12.3-1ubuntu0.12" \
      "python3.12-dev=3.12.3-1ubuntu0.12"; do \
        pkg="${spec%%=*}"; want="${spec#*=}"; got="$(dpkg-query -W -f='${Version}' "${pkg}")"; \
        if [ "${got}" != "${want}" ]; then \
          echo "apt version drift for ${pkg}: expected ${want}, got ${got}" >&2; \
          exit 1; \
        fi; \
    done

RUN set -eu; \
    expected_dpkg_lock="21df06cd008564fb7367a6d995bd196e234ffa0e1de9778b08edfda28f68dc15"; \
    actual_dpkg_lock="$(dpkg-query -W | LC_ALL=C sort | sha256sum | awk '{print $1}')"; \
    if [ "${actual_dpkg_lock}" != "${expected_dpkg_lock}" ]; then \
      echo "dpkg package lock drift: expected ${expected_dpkg_lock}, got ${actual_dpkg_lock}" >&2; \
      dpkg-query -W | LC_ALL=C sort >&2; \
      exit 1; \
    fi

RUN python3.12 -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade \
      pip==26.0.1 \
      setuptools==79.0.1 \
      wheel==0.46.3 \
      setuptools-scm==9.2.2 \
      packaging==26.2 \
      cmake==3.31.10 \
      lit==18.1.8 \
      ninja==1.13.0 \
      pybind11==3.0.2 \
      semantic-version==2.10.0 \
      setuptools-rust==1.10.2 && \
    /opt/venv/bin/pip install --no-deps annotated-doc==0.0.4 && \
    /opt/venv/bin/python - <<'PY'
import importlib.util
import sys

if importlib.util.find_spec("annotated_doc") is None:
    print("error: unable to import pinned annotated_doc runtime dependency", file=sys.stderr)
    raise SystemExit(1)
PY
RUN cat > /usr/local/bin/canonicalize-wheel <<'PY' && chmod +x /usr/local/bin/canonicalize-wheel
#!/opt/venv/bin/python
from pathlib import Path
import os
import sys
import time
import zipfile

epoch = int(os.environ.get("SOURCE_DATE_EPOCH", "0") or "0")
date_time = time.gmtime(epoch)[:6]
if date_time[0] < 1980:
    date_time = (1980, 1, 1, 0, 0, 0)

for raw in sys.argv[1:]:
    wheel = Path(raw)
    tmp = wheel.with_suffix(wheel.suffix + ".tmp")
    with zipfile.ZipFile(wheel, "r") as zin, zipfile.ZipFile(
        tmp,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=False,
    ) as zout:
        for name in sorted(zin.namelist()):
            old = zin.getinfo(name)
            data = zin.read(name) if not name.endswith("/") else b""
            mode = (old.external_attr >> 16) & 0o777
            if mode == 0:
                mode = 0o755 if name.endswith("/") else 0o644
            info = zipfile.ZipInfo(name, date_time=date_time)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = mode << 16
            zout.writestr(info, data)
    tmp.replace(wheel)
PY
RUN cat > /tmp/reference-pip-freeze.txt <<'PIPLOCK'
Jinja2==3.1.6
MarkupSafe==3.0.2
PyJWT==2.12.1
PyYAML==6.0.3
Pygments==2.20.0
accelerate==1.13.0
aiohappyeyeballs==2.6.1
aiohttp==3.13.3
aiosignal==1.4.0
amd-quark==0.11.1
amdsmi==6.3.3
annotated-doc==0.0.4
annotated-types==0.7.0
anthropic==0.86.0
anyio==4.13.0
astor==0.8.1
attrs==26.1.0
blake3==1.0.8
boto3==1.42.73
botocore==1.42.73
cachetools==7.0.5
cbor2==5.8.0
certifi==2026.4.22
cffi==2.0.0
charset-normalizer==3.4.6
click==8.3.3
cloudpickle==3.1.2
cmake==3.31.10
colorama==0.4.6
compressed-tensors==0.13.0
conch-triton-kernels==1.2.1
cryptography==46.0.5
cuda-bindings==12.9.4
cuda-pathfinder==1.4.3
datasets==4.8.3
depyf==0.20.0
dill==0.4.1
diskcache==5.6.3
distro==1.9.0
dnspython==2.8.0
docstring_parser==0.17.0
einops==0.8.2
email-validator==2.3.0
evaluate==0.4.6
fastapi-cli==0.0.24
fastapi-cloud-cli==0.15.0
fastapi==0.135.1
fastar==0.9.0
filelock==3.29.0
flash_attn==2.8.3
frozenlist==1.8.0
fsspec==2025.12.0
gguf==0.18.0
google-api-core==2.30.0
google-auth==2.49.1
google-cloud-core==2.5.0
google-cloud-storage==3.10.0
google-crc32c==1.8.0
google-resumable-media==2.8.0
googleapis-common-protos==1.73.0
grpcio-reflection==1.78.0
grpcio==1.78.0
h11==0.16.0
hf-xet==1.4.3
hiredis==3.3.1
httpcore==1.0.9
httptools==0.7.1
httpx-sse==0.4.3
httpx==0.28.1
huggingface_hub==1.12.0
humanize==4.15.0
idna==3.13
ijson==3.5.0
importlib_metadata==8.7.1
iniconfig==2.3.0
interegular==0.3.3
jiter==0.13.0
jmespath==1.1.0
joblib==1.5.3
jsonschema-specifications==2025.9.1
jsonschema==4.26.0
lark==1.2.2
libnacl==2.1.0
lit==18.1.8
llguidance==1.3.0
llvmlite==0.44.0
lm-format-enforcer==0.11.3
loguru==0.7.3
markdown-it-py==4.0.0
mcp==1.26.0
mdurl==0.1.2
mistral_common==1.10.0
ml_dtypes==0.5.4
model-hosting-container-standards==0.1.14
mpmath==1.3.0
msgspec==0.20.0
multidict==6.7.1
multiprocess==0.70.19
narwhals==2.18.0
networkx==3.6.1
ninja==1.13.0
numba==0.61.2
numpy==2.3.5
nvidia-cublas-cu12==12.8.4.1
nvidia-cuda-cupti-cu12==12.8.90
nvidia-cuda-nvrtc-cu12==12.8.93
nvidia-cuda-runtime-cu12==12.8.90
nvidia-cudnn-cu12==9.10.2.21
nvidia-cufft-cu12==11.3.3.83
nvidia-cufile-cu12==1.13.1.3
nvidia-curand-cu12==10.3.9.90
nvidia-cusolver-cu12==11.7.3.90
nvidia-cusparse-cu12==12.5.8.93
nvidia-cusparselt-cu12==0.7.1
nvidia-nccl-cu12==2.27.5
nvidia-nvjitlink-cu12==12.8.93
nvidia-nvshmem-cu12==3.4.5
nvidia-nvtx-cu12==12.8.90
onnx-ir==0.2.0
onnx==1.19.0
onnxscript==0.6.2
onnxslim==0.1.89
openai-harmony==0.0.8
openai==2.24.0
opencv-python-headless==4.13.0.92
opentelemetry-api==1.40.0
opentelemetry-exporter-otlp-proto-common==1.40.0
opentelemetry-exporter-otlp-proto-grpc==1.40.0
opentelemetry-exporter-otlp-proto-http==1.40.0
opentelemetry-exporter-otlp==1.40.0
opentelemetry-proto==1.40.0
opentelemetry-sdk==1.40.0
opentelemetry-semantic-conventions-ai==0.5.0
opentelemetry-semantic-conventions==0.61b0
outlines_core==0.2.11
packaging==26.2
pandas==3.0.1
partial-json-parser==0.2.1.1.post7
peft==0.18.1
pillow==12.0.0
pip==26.0.1
plotly==6.6.0
pluggy==1.6.0
prometheus-fastapi-instrumentator==7.1.0
prometheus_client==0.24.1
propcache==0.4.1
proto-plus==1.27.1
protobuf==6.33.6
psutil==7.2.2
py-cpuinfo==9.0.0
pyarrow==23.0.1
pyasn1==0.6.3
pyasn1_modules==0.4.2
pybase64==1.4.3
pybind11==3.0.2
pycountry==26.2.16
pycparser==3.0
pydantic-extra-types==2.11.1
pydantic-settings==2.13.1
pydantic==2.12.5
pydantic_core==2.41.5
pytest-asyncio==1.3.0
pytest==9.0.2
python-dateutil==2.9.0.post0
python-dotenv==1.2.2
python-json-logger==4.0.0
python-multipart==0.0.22
pytorch-triton-rocm==3.5.1
pyzmq==27.1.0
redis==7.3.0
referencing==0.37.0
regex==2026.4.4
requests==2.32.5
rich-toolkit==0.19.7
rich==15.0.0
rignore==0.7.6
rpds-py==0.30.0
runai-model-streamer-gcs==0.15.3
runai-model-streamer-s3==0.15.3
runai-model-streamer==0.15.3
s3transfer==0.16.0
safetensors==0.7.0
scipy==1.17.1
sentencepiece==0.2.1
sentry-sdk==2.55.0
setproctitle==1.3.7
setuptools-scm==9.2.2
setuptools==79.0.1
shellingham==1.5.4
six==1.17.0
sniffio==1.3.1
sse-starlette==3.3.3
starlette==0.52.1
supervisor==4.3.0
sympy==1.14.0
tensorizer==2.10.1
tiktoken==0.12.0
timm==1.0.25
tokenizers==0.22.2
torch==2.9.1+rocm6.3
torchaudio==2.9.1+rocm6.3
torchvision==0.24.1+rocm6.3
tqdm==4.67.3
transformers==5.7.0
typer==0.25.0
typing-inspection==0.4.2
typing_extensions==4.15.0
urllib3==2.6.3
uvicorn==0.42.0
uvloop==0.22.1
watchfiles==1.1.1
websockets==16.0
wheel==0.46.3
xgrammar==0.1.29
xxhash==3.6.0
yarl==1.23.0
zipp==3.23.0
zstandard==0.25.0
PIPLOCK
RUN grep -Ev '^(triton @|vllm @|flash_attn==)' /tmp/reference-pip-freeze.txt > /tmp/reference-pip-constraints.txt && \
    grep -Ev '^(triton @|vllm @|flash_attn==|torch==|torchvision==|torchaudio==|pytorch-triton-rocm==)' /tmp/reference-pip-freeze.txt > /tmp/reference-pip-install.txt
RUN /opt/venv/bin/pip install --force-reinstall --extra-index-url "${PYTORCH_INDEX_URL}" -c /tmp/reference-pip-constraints.txt torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION}

WORKDIR /opt/src
RUN git clone ${TRITON_REPO} triton-gfx906 && \
    cd triton-gfx906 && \
    git checkout ${TRITON_REF} && \
    find . -exec touch -h -d "@${SOURCE_DATE_EPOCH}" {} + && \
    rm -rf "${REPRO_BUILD_ROOT}/triton" "${REPRO_WHEEL_ROOT}/triton" "${REPRO_TMPDIR}" && \
    mkdir -p "${REPRO_BUILD_ROOT}/triton-tracker" "${REPRO_WHEEL_ROOT}/triton" "${REPRO_TMPDIR}" && \
    TMPDIR="${REPRO_TMPDIR}" PIP_BUILD_TRACKER="${REPRO_BUILD_ROOT}/triton-tracker" \
      /opt/venv/bin/pip wheel --no-deps --no-build-isolation --no-cache-dir --wheel-dir "${REPRO_WHEEL_ROOT}/triton" . && \
    /usr/local/bin/canonicalize-wheel "${REPRO_WHEEL_ROOT}"/triton/*.whl && \
    sha256sum "${REPRO_WHEEL_ROOT}"/triton/*.whl | tee /opt/repro-wheel-triton.sha256 && \
    /opt/venv/bin/pip install --no-deps --force-reinstall --no-index --find-links="${REPRO_WHEEL_ROOT}/triton" "${REPRO_WHEEL_ROOT}"/triton/*.whl && \
    find /opt/venv/lib/python3.12/site-packages/triton /opt/venv/lib/python3.12/site-packages/triton-*.dist-info -exec touch -h -d "@${SOURCE_DATE_EPOCH}" {} +

RUN git clone ${FLASH_ATTENTION_REPO} flash-attention-gfx906 && \
    cd flash-attention-gfx906 && \
    git checkout ${FLASH_ATTN_REPO_REF} && \
    find . -exec touch -h -d "@${SOURCE_DATE_EPOCH}" {} + && \
    rm -rf "${REPRO_BUILD_ROOT}/flash-attn" "${REPRO_WHEEL_ROOT}/flash-attn" "${REPRO_TMPDIR}" && \
    mkdir -p "${REPRO_BUILD_ROOT}/flash-attn-tracker" "${REPRO_WHEEL_ROOT}/flash-attn" "${REPRO_TMPDIR}" && \
    TMPDIR="${REPRO_TMPDIR}" PIP_BUILD_TRACKER="${REPRO_BUILD_ROOT}/flash-attn-tracker" \
      /opt/venv/bin/pip wheel --no-build-isolation --no-deps --no-cache-dir --wheel-dir "${REPRO_WHEEL_ROOT}/flash-attn" . && \
    /usr/local/bin/canonicalize-wheel "${REPRO_WHEEL_ROOT}"/flash-attn/*.whl && \
    sha256sum "${REPRO_WHEEL_ROOT}"/flash-attn/*.whl | tee /opt/repro-wheel-flash-attn.sha256 && \
    /opt/venv/bin/pip install --no-deps --force-reinstall --no-index --find-links="${REPRO_WHEEL_ROOT}/flash-attn" "${REPRO_WHEEL_ROOT}"/flash-attn/*.whl && \
    find /opt/venv/lib/python3.12/site-packages/flash_attn /opt/venv/lib/python3.12/site-packages/flash_attn-*.dist-info -exec touch -h -d "@${SOURCE_DATE_EPOCH}" {} +

RUN git clone ${VLLM_SRC_REPO} /workspace/vllm && \
    cd /workspace/vllm && \
    git checkout ${VLLM_COMMIT} && \
    find . -exec touch -h -d "@${SOURCE_DATE_EPOCH}" {} + && \
    ln -sfn /workspace/vllm /opt/src/vllm

WORKDIR /workspace/vllm
RUN /opt/venv/bin/python - <<'PY'
from pathlib import Path
import os

setup_py = Path("setup.py")
text = setup_py.read_text()
old = '''def _is_hip() -> bool:
    return (
        VLLM_TARGET_DEVICE == "cuda" or VLLM_TARGET_DEVICE == "rocm"
    ) and torch.version.hip is not None
'''
new = '''def _is_hip() -> bool:
    if VLLM_TARGET_DEVICE == "rocm":
        return True
    return VLLM_TARGET_DEVICE == "cuda" and torch.version.hip is not None
'''
if old not in text:
    if 'if VLLM_TARGET_DEVICE == "rocm":' not in text:
        raise SystemExit("could not patch setup.py ROCm target detection")
else:
    setup_py.write_text(text.replace(old, new))

triton_ref = os.environ["TRITON_KERNELS_FETCHCONTENT_REF"]
replaced = []
for path in Path(".").rglob("*"):
    if ".git" in path.parts or not path.is_file():
        continue
    try:
        text = path.read_text()
    except UnicodeDecodeError:
        continue
    if "v3.6.0" not in text:
        continue
    if "triton" not in text.lower():
        continue
    path.write_text(text.replace("v3.6.0", triton_ref))
    replaced.append(str(path))

if not replaced:
    raise SystemExit("could not pin vLLM Triton FetchContent tag v3.6.0")
print("pinned vLLM Triton FetchContent tag v3.6.0 to", triton_ref)
for path in replaced:
    print("  ", path)
PY
RUN find /workspace/vllm -exec touch -h -d "@${SOURCE_DATE_EPOCH}" {} +

RUN /opt/venv/bin/pip install --no-deps --force-reinstall --extra-index-url "${PYTORCH_INDEX_URL}" -r /tmp/reference-pip-install.txt && \
    /opt/venv/bin/pip install --no-deps --force-reinstall --extra-index-url "${PYTORCH_INDEX_URL}" torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} && \
    /opt/venv/bin/pip install --no-deps --force-reinstall packaging==26.2 setuptools==79.0.1 setuptools-scm==9.2.2 wheel==0.46.3 && \
    /opt/venv/bin/pip install --no-deps --force-reinstall amdsmi==${AMDSMI_VERSION} && \
    find /opt/venv -xdev -exec touch -h -d "@${SOURCE_DATE_EPOCH}" {} +

RUN /opt/venv/bin/python - <<'PY'
from pathlib import Path
import shutil

site = Path("/opt/venv/lib/python3.12/site-packages")
target = Path("/opt/qwen36-python")
patterns = [
    "_yaml",
    "annotated_doc",
    "annotated_doc-*.dist-info",
    "anyio",
    "anyio-*.dist-info",
    "certifi",
    "certifi-*.dist-info",
    "click",
    "click-*.dist-info",
    "filelock",
    "filelock-*.dist-info",
    "h11",
    "h11-*.dist-info",
    "hf_xet",
    "hf_xet-*.dist-info",
    "httpcore",
    "httpcore-*.dist-info",
    "httpx",
    "httpx-*.dist-info",
    "huggingface_hub",
    "huggingface_hub-*.dist-info",
    "idna",
    "idna-*.dist-info",
    "markdown_it",
    "markdown_it_py-*.dist-info",
    "mdurl",
    "mdurl-*.dist-info",
    "numpy.libs",
    "packaging",
    "packaging-*.dist-info",
    "pygments",
    "pygments-*.dist-info",
    "pyyaml-*.dist-info",
    "regex",
    "regex-*.dist-info",
    "rich",
    "rich-*.dist-info",
    "safetensors",
    "safetensors-*.dist-info",
    "shellingham",
    "shellingham-*.dist-info",
    "tokenizers",
    "tokenizers-*.dist-info",
    "tqdm",
    "tqdm-*.dist-info",
    "transformers",
    "transformers-*.dist-info",
    "typer",
    "typer-*.dist-info",
    "typing_extensions.py",
    "typing_extensions-*.dist-info",
    "yaml",
]

if target.exists():
    shutil.rmtree(target)
target.mkdir(parents=True)

missing = []
for pattern in patterns:
    matches = sorted(site.glob(pattern))
    if not matches:
        missing.append(pattern)
        continue
    for src in matches:
        dst = target / src.name
        if src.is_dir():
            shutil.copytree(src, dst, symlinks=True)
        else:
            shutil.copy2(src, dst)

if missing:
    raise SystemExit("missing qwen36-python sidecar inputs: " + ", ".join(missing))
PY

RUN /opt/venv/bin/pip uninstall -y triton-rocm || true && \
    /opt/venv/bin/python - <<'PY'
import importlib.metadata as md

expected = "3.5.1+git90c4b262"
got = md.version("triton")
if got != expected:
    raise SystemExit(f"unexpected gfx906 Triton version before vLLM build: {got} != {expected}")
PY

RUN /opt/venv/bin/python - <<'PY'
import torch

expected = "2.9.1+rocm6.3"
if torch.__version__ != expected:
    raise SystemExit(f"unexpected torch version before vLLM build: {torch.__version__} != {expected}")
PY

RUN rm -rf build "${REPRO_BUILD_ROOT}/vllm" "${REPRO_WHEEL_ROOT}/vllm" "${REPRO_TMPDIR}" && \
    mkdir -p "${REPRO_BUILD_ROOT}/vllm" "${REPRO_WHEEL_ROOT}/vllm" "${REPRO_TMPDIR}" && \
    export TMPDIR="${REPRO_TMPDIR}" && \
    /opt/venv/bin/python3 setup.py bdist_wheel --dist-dir "${REPRO_WHEEL_ROOT}/vllm" && \
    /usr/local/bin/canonicalize-wheel "${REPRO_WHEEL_ROOT}"/vllm/vllm-*.whl && \
    sha256sum "${REPRO_WHEEL_ROOT}"/vllm/vllm-*.whl | tee /opt/repro-wheel-vllm.sha256 && \
    /opt/venv/bin/pip install --no-deps --force-reinstall "${REPRO_WHEEL_ROOT}"/vllm/vllm-*.whl && \
    ( /opt/venv/bin/pip uninstall -y semantic-version setuptools-rust || true ) && \
    find /opt/venv/lib/python3.12/site-packages/vllm /opt/venv/lib/python3.12/site-packages/vllm-*.dist-info -exec touch -h -d "@${SOURCE_DATE_EPOCH}" {} +

RUN /opt/venv/bin/pip uninstall -y triton-rocm || true && \
    /opt/venv/bin/python - <<'PY'
import importlib.metadata as md

expected = "3.5.1+git90c4b262"
got = md.version("triton")
if got != expected:
    raise SystemExit(f"unexpected gfx906 Triton version after vLLM build: {got} != {expected}")
PY


RUN cd /tmp && /opt/venv/bin/python - <<'PY'
import importlib.util
import importlib
import importlib.metadata as importlib_metadata
import pathlib
import torch

expected = "2.9.1+rocm6.3"
if torch.__version__ != expected:
    raise SystemExit(f"unexpected torch version after runtime deps: {torch.__version__} != {expected}")
if not str(torch.version.hip).startswith("6.3"):
    raise SystemExit(f"unexpected torch HIP runtime after runtime deps: {torch.version.hip}")

for dist in importlib_metadata.distributions():
    name = (dist.metadata.get("Name") or "").lower()
    if name == "triton-rocm":
        raise SystemExit("unexpected triton-rocm package in gfx906 image")

required_modules = [
    "fastapi",
    "starlette",
    "pydantic_core",
    "anyio",
    "httpx",
    "huggingface_hub",
    "typing_inspection",
    "uvicorn",
]
missing = [name for name in required_modules if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"missing required runtime modules: {', '.join(missing)}")

vllm_spec = importlib.util.find_spec("vllm")
if vllm_spec is None or not vllm_spec.submodule_search_locations:
    raise SystemExit("missing vllm package")
vllm_root = pathlib.Path(next(iter(vllm_spec.submodule_search_locations)))
missing_cli_files = [
    str(path.relative_to(vllm_root))
    for path in (
        vllm_root / "entrypoints" / "cli" / "main.py",
        vllm_root / "entrypoints" / "cli" / "launch.py",
    )
    if not path.is_file()
]
if missing_cli_files:
    raise SystemExit("missing vLLM CLI files: " + ", ".join(missing_cli_files))
for lib_name in ("_C.abi3.so", "_rocm_C.abi3.so"):
    lib_path = vllm_root / lib_name
    if not lib_path.is_file():
        raise SystemExit(f"missing vLLM custom-op shared object: {lib_path}")
PY

RUN cd /tmp && /opt/venv/bin/python - <<'PY'
import importlib.metadata as md
from pathlib import Path
from packaging.utils import canonicalize_name

expected = {}
for raw in Path("/tmp/reference-pip-freeze.txt").read_text().splitlines():
    raw = raw.strip()
    if not raw or raw.startswith("#") or " @ " in raw:
        continue
    name, version = raw.split("==", 1)
    expected[canonicalize_name(name)] = version
expected["vllm"] = "0.0.0+gfx906"
expected["triton"] = "3.5.1+git90c4b262"

actual = {
    canonicalize_name(dist.metadata["Name"]): dist.version
    for dist in md.distributions()
    if dist.metadata.get("Name")
}

missing = []
mismatched = []
for name, version in sorted(expected.items()):
    got = actual.get(name)
    if got is None:
        missing.append(name)
    elif got != version:
        mismatched.append(f"{name}: {got} != {version}")

if missing or mismatched:
    message = []
    if missing:
        message.append("missing reference packages: " + ", ".join(missing))
    if mismatched:
        message.append("reference package version drift: " + "; ".join(mismatched))
    raise SystemExit("\n".join(message))
PY

RUN /opt/venv/bin/python -c "import torch, vllm; print('torch', torch.__version__, 'hip', torch.version.hip); print('vllm', vllm.__version__)"

RUN set -eu; \
    for path in /opt/venv /opt/qwen36-python /workspace/vllm /opt/src; do \
      if [ -e "${path}" ]; then \
        find "${path}" -type d -name '__pycache__' -prune -exec rm -rf {} +; \
        find "${path}" -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete; \
      fi; \
    done; \
    find /opt/src /workspace/vllm -type d -name '.git' -prune -exec rm -rf {} +; \
    rm -rf /opt/src /workspace/vllm "${REPRO_BUILD_ROOT}" "${REPRO_WHEEL_ROOT}" "${REPRO_TMPDIR}" /var/tmp/apt-cache; \
    rm -rf /root/.cache/pip /tmp/pip-* /tmp/wheels; \
    rm -f /var/lib/dpkg/alternatives/*; \
    rm -rf /var/log/apt/* /var/log/dpkg.log /var/log/alternatives.log /var/log/bootstrap.log /var/cache/ldconfig/aux-cache

WORKDIR /tmp

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

FROM vllm-gfx906-repro-build AS vllm-gfx906-repro-smallroot
RUN rm -rf \
    /opt/rocm-6.3.4/lib \
    /opt/venv/lib/python3.12/site-packages/torch/lib \
    /opt/venv/lib/python3.12/site-packages/nvidia \
    /root/.triton

FROM vllm-gfx906-repro-build AS vllm-gfx906-repro-rocm-lib-rest
RUN rm -rf \
    /opt/rocm-6.3.4/lib/hipblaslt \
    /opt/rocm-6.3.4/lib/rocblas \
    /opt/rocm-6.3.4/lib/rocfft \
    /opt/rocm-6.3.4/lib/llvm \
    /opt/rocm-6.3.4/lib/hipsparselt \
    /opt/rocm-6.3.4/lib/librocsolver.so.0.3.60304 \
    /opt/rocm-6.3.4/lib/librocsparse.so.1.0.60304 \
    /opt/rocm-6.3.4/lib/librccl.so.1.0.60304 \
    /opt/rocm-6.3.4/lib/libMIOpen.so.1.0.60304 \
    /opt/rocm-6.3.4/lib/libdevice_gemm_operations.a \
    /opt/rocm-6.3.4/lib/libdevice_conv_operations.a

FROM vllm-gfx906-repro-build AS vllm-gfx906-repro-torch-lib-rest
RUN rm -rf \
    /opt/venv/lib/python3.12/site-packages/torch/lib/hipblaslt \
    /opt/venv/lib/python3.12/site-packages/torch/lib/rocblas \
    /opt/venv/lib/python3.12/site-packages/torch/lib/librocsolver.so \
    /opt/venv/lib/python3.12/site-packages/torch/lib/librocsparse.so \
    /opt/venv/lib/python3.12/site-packages/torch/lib/libmagma.so \
    /opt/venv/lib/python3.12/site-packages/torch/lib/librccl.so \
    /opt/venv/lib/python3.12/site-packages/torch/lib/aotriton.images \
    /opt/venv/lib/python3.12/site-packages/torch/lib/libMIOpen.so \
    /opt/venv/lib/python3.12/site-packages/torch/lib/libtorch_cpu.so \
    /opt/venv/lib/python3.12/site-packages/torch/lib/libtorch_hip.so

FROM vllm-gfx906-repro-build AS vllm-gfx906-repro-triton-rest
RUN rm -rf \
    /root/.triton/llvm \
    /root/.triton/nvidia

FROM scratch AS vllm-gfx906-repro-runtime
ARG SOURCE_DATE_EPOCH=1764000000
ENV ROCM_VERSION=6.3.4
ENV AMDGPU_VERSION=6.3.4
ENV PYTORCH_INDEX_URL=https://download.pytorch.org/whl/rocm6.3
ENV TORCH_VERSION=2.9.1+rocm6.3
ENV TORCHVISION_VERSION=0.24.1+rocm6.3
ENV TORCHAUDIO_VERSION=2.9.1+rocm6.3
ENV PYTORCH_TRITON_ROCM_VERSION=3.5.1
ENV AMDSMI_VERSION=6.3.3
ENV VLLM_COMMIT=6a52668c14dc9cb94a270d94bfec735fe627ed0c
ENV VLLM_PACKAGE_VERSION=0.0.0+gfx906
ENV VLLM_VERSION_OVERRIDE=0.0.0+gfx906
ENV SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH}
ENV VLLM_TARGET_DEVICE=rocm
ENV PYTORCH_ROCM_ARCH=gfx906
ENV GPU_ARCHS=gfx906
ENV AMDGPU_TARGETS=gfx906
ENV CMAKE_HIP_ARCHITECTURES=gfx906
ENV FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
ENV FLASH_ATTENTION_TRITON_AMD_REF=TRUE
ENV LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:
ENV PATH=/opt/venv/bin:/opt/rocm/bin:/opt/rocm/llvm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV PYTHONPATH=/opt/qwen36-python
ENV PYTHONUNBUFFERED=1
ENV PYTHONHASHSEED=0
ENV PIP_NO_CACHE_DIR=1
ENV PIP_NO_COMPILE=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV TZ=Etc/UTC

COPY --from=vllm-gfx906-repro-smallroot / /

COPY --from=vllm-gfx906-repro-build /opt/rocm-6.3.4/lib/hipblaslt /opt/rocm-6.3.4/lib/hipblaslt
COPY --from=vllm-gfx906-repro-build /opt/rocm-6.3.4/lib/rocblas /opt/rocm-6.3.4/lib/rocblas
COPY --from=vllm-gfx906-repro-build /opt/rocm-6.3.4/lib/rocfft /opt/rocm-6.3.4/lib/rocfft
COPY --from=vllm-gfx906-repro-build /opt/rocm-6.3.4/lib/llvm /opt/rocm-6.3.4/lib/llvm
COPY --from=vllm-gfx906-repro-build /opt/rocm-6.3.4/lib/hipsparselt /opt/rocm-6.3.4/lib/hipsparselt
COPY --from=vllm-gfx906-repro-build /opt/rocm-6.3.4/lib/librocsolver.so.0.3.60304 /opt/rocm-6.3.4/lib/librocsolver.so.0.3.60304
COPY --from=vllm-gfx906-repro-build /opt/rocm-6.3.4/lib/librocsparse.so.1.0.60304 /opt/rocm-6.3.4/lib/librocsparse.so.1.0.60304
COPY --from=vllm-gfx906-repro-build /opt/rocm-6.3.4/lib/librccl.so.1.0.60304 /opt/rocm-6.3.4/lib/librccl.so.1.0.60304
COPY --from=vllm-gfx906-repro-build /opt/rocm-6.3.4/lib/libMIOpen.so.1.0.60304 /opt/rocm-6.3.4/lib/libMIOpen.so.1.0.60304
COPY --from=vllm-gfx906-repro-build /opt/rocm-6.3.4/lib/libdevice_gemm_operations.a /opt/rocm-6.3.4/lib/libdevice_gemm_operations.a
COPY --from=vllm-gfx906-repro-build /opt/rocm-6.3.4/lib/libdevice_conv_operations.a /opt/rocm-6.3.4/lib/libdevice_conv_operations.a
COPY --from=vllm-gfx906-repro-rocm-lib-rest /opt/rocm-6.3.4/lib /opt/rocm-6.3.4/lib

COPY --from=vllm-gfx906-repro-build /opt/venv/lib/python3.12/site-packages/torch/lib/hipblaslt /opt/venv/lib/python3.12/site-packages/torch/lib/hipblaslt
COPY --from=vllm-gfx906-repro-build /opt/venv/lib/python3.12/site-packages/torch/lib/rocblas /opt/venv/lib/python3.12/site-packages/torch/lib/rocblas
COPY --from=vllm-gfx906-repro-build /opt/venv/lib/python3.12/site-packages/torch/lib/librocsolver.so /opt/venv/lib/python3.12/site-packages/torch/lib/librocsolver.so
COPY --from=vllm-gfx906-repro-build /opt/venv/lib/python3.12/site-packages/torch/lib/librocsparse.so /opt/venv/lib/python3.12/site-packages/torch/lib/librocsparse.so
COPY --from=vllm-gfx906-repro-build /opt/venv/lib/python3.12/site-packages/torch/lib/libmagma.so /opt/venv/lib/python3.12/site-packages/torch/lib/libmagma.so
COPY --from=vllm-gfx906-repro-build /opt/venv/lib/python3.12/site-packages/torch/lib/librccl.so /opt/venv/lib/python3.12/site-packages/torch/lib/librccl.so
COPY --from=vllm-gfx906-repro-build /opt/venv/lib/python3.12/site-packages/torch/lib/aotriton.images /opt/venv/lib/python3.12/site-packages/torch/lib/aotriton.images
COPY --from=vllm-gfx906-repro-build /opt/venv/lib/python3.12/site-packages/torch/lib/libMIOpen.so /opt/venv/lib/python3.12/site-packages/torch/lib/libMIOpen.so
COPY --from=vllm-gfx906-repro-build /opt/venv/lib/python3.12/site-packages/torch/lib/libtorch_cpu.so /opt/venv/lib/python3.12/site-packages/torch/lib/libtorch_cpu.so
COPY --from=vllm-gfx906-repro-build /opt/venv/lib/python3.12/site-packages/torch/lib/libtorch_hip.so /opt/venv/lib/python3.12/site-packages/torch/lib/libtorch_hip.so
COPY --from=vllm-gfx906-repro-torch-lib-rest /opt/venv/lib/python3.12/site-packages/torch/lib /opt/venv/lib/python3.12/site-packages/torch/lib
COPY --from=vllm-gfx906-repro-build /opt/venv/lib/python3.12/site-packages/nvidia /opt/venv/lib/python3.12/site-packages/nvidia

COPY --from=vllm-gfx906-repro-build /root/.triton/llvm /root/.triton/llvm
COPY --from=vllm-gfx906-repro-build /root/.triton/nvidia /root/.triton/nvidia
COPY --from=vllm-gfx906-repro-triton-rest /root/.triton /root/.triton

WORKDIR /tmp
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
DOCKERFILE
}

write_embedded_dockerignore() {
  local dockerignore_path="$1"
  mkdir -p "$(dirname "${dockerignore_path}")"

  cat <<'DOCKERIGNORE' > "${dockerignore_path}"
*
!Dockerfile
!docker-entrypoint.sh
!.dockerignore
DOCKERIGNORE
}

write_embedded_compose() {
  local compose_path="$1"
  mkdir -p "$(dirname "${compose_path}")"

  cat <<'COMPOSE' > "${compose_path}"
services:
  qwen36-c1-fastpath:
    image: ${DEPLOY_IMAGE:-qwen36-gfx906-c1-topk8-fastpath-host10}
    build:
      context: .
      dockerfile: ${COMPOSE_DOCKERFILE_PATH:-${DOCKERFILE_PATH:-Dockerfile}}
      network: host
      args:
        SOURCE_DATE_EPOCH: ${SOURCE_DATE_EPOCH:-1764000000}
        BUILDKIT_MULTI_PLATFORM: "1"
    container_name: ${CONTAINER_NAME}
    network_mode: host
    ipc: host
    privileged: true
    group_add:
      - video
    cap_add:
      - IPC_LOCK
    devices:
      - /dev/kfd:/dev/kfd
      - /dev/dri:/dev/dri
    security_opt:
      - seccomp=unconfined
      - label=disable
    ulimits:
      memlock:
        soft: -1
        hard: -1
    restart: "${RESTART_POLICY:-no}"
    volumes:
      - ${HF_CACHE_DIR}:/root/.cache/huggingface
      - ${RUNTIME_ROOT}/root/.triton:/root/.triton
      - ${RUNTIME_ROOT}/root/.cache/vllm:/root/.cache/vllm
      - ${RUNTIME_ROOT}/tmp/torchinductor_root:/tmp/torchinductor_root
      - ${MOE_CONFIG_DIR}:/opt/vllm_tuned_moe_configs:ro
      - ${RUNTIME_PATCH_DIR}:${VLLM_PATCH_BUNDLE:-/opt/vllm_patch_bundle}:ro
    environment:
      MODEL: ${MODEL}
      SERVED_MODEL_NAME: ${SERVED_MODEL_NAME}
      HOST: 0.0.0.0
      PORT: ${PORT}
      TP_SIZE: ${TP_SIZE}
      MAX_MODEL_LEN: ${MAX_MODEL_LEN}
      GPU_MEMORY_UTILIZATION: ${GPU_MEMORY_UTILIZATION}
      TOOL_CALL_PARSER: ${TOOL_CALL_PARSER}
      REASONING_PARSER: ${REASONING_PARSER}
      VLLM_DTYPE: ${VLLM_DTYPE}
      OPT_LEVEL: ${OPT_LEVEL}
      EXTRA_VLLM_ARGS: ${EXTRA_VLLM_ARGS}
      DISABLE_ASYNC_SCHEDULING: ${DISABLE_ASYNC_SCHEDULING}
      VLLM_TUNED_CONFIG_FOLDER: ${VLLM_TUNED_CONFIG_FOLDER}
      VLLM_PATCH_BUNDLE: ${VLLM_PATCH_BUNDLE:-/opt/vllm_patch_bundle}
      QWEN36_PYTHON_DIR: ${QWEN36_PYTHON_DIR:-}
      PYTHONPATH: ${QWEN36_PYTHON_DIR:-}
      VLLM_ENABLE_C1_TOPK8_MOE_FASTPATH: ${VLLM_ENABLE_C1_TOPK8_MOE_FASTPATH}
      NCCL_ALGO: ${NCCL_ALGO}
      NCCL_PROTO: ${NCCL_PROTO}
      NCCL_P2P_DISABLE: ${NCCL_P2P_DISABLE}
      NCCL_MAX_NCHANNELS: ${NCCL_MAX_NCHANNELS}
      FLASH_ATTENTION_TRITON_AMD_ENABLE: ${FLASH_ATTENTION_TRITON_AMD_ENABLE}
      FLASH_ATTENTION_TRITON_AMD_REF: ${FLASH_ATTENTION_TRITON_AMD_REF}
      VLLM_LOGGING_LEVEL: INFO
      DO_NOT_TRACK: "1"
      TORCH_BLAS_PREFER_HIPBLASLT: "0"
      OMP_NUM_THREADS: ${OMP_NUM_THREADS}
      VLLM_WORKER_MULTIPROC_METHOD: spawn
      VLLM_ALLOW_LONG_MAX_MODEL_LEN: "1"
      VLLM_TARGET_DEVICE: rocm
      VLLM_VERSION_OVERRIDE: "0.0.0+gfx906"
      PYTORCH_ROCM_ARCH: gfx906
      GPU_ARCHS: gfx906
      TMPDIR: /tmp/torchinductor_root
      TMP: /tmp/torchinductor_root
      TEMP: /tmp/torchinductor_root
      TRITON_CACHE_DIR: /root/.triton
      TORCHINDUCTOR_CACHE_DIR: /tmp/torchinductor_root
      VLLM_CACHE_ROOT: /root/.cache/vllm
      HIP_VISIBLE_DEVICES: ${HIP_VISIBLE_DEVICES}
COMPOSE
}

write_embedded_entrypoint() {
  local entrypoint_path="$1"
  mkdir -p "$(dirname "${entrypoint_path}")"

  cat <<'ENTRYPOINT' > "${entrypoint_path}"
#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.6-35B-A3B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${MODEL}}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"
TP_SIZE="${TP_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-hermes}"
REASONING_PARSER="${REASONING_PARSER:-${REASONING_PARSER_PREFERRED:-none}}"
VLLM_DTYPE="${VLLM_DTYPE:-half}"
OPT_LEVEL="${OPT_LEVEL:-3}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:---language-model-only}"
DISABLE_ASYNC_SCHEDULING="${DISABLE_ASYNC_SCHEDULING:-0}"

VLLM_TUNED_CONFIG_FOLDER="${VLLM_TUNED_CONFIG_FOLDER:-/opt/vllm_tuned_moe_configs}"
export VLLM_TUNED_CONFIG_FOLDER

export PYTHONPATH="${PYTHONPATH:-/opt/qwen36-python}"
export TMPDIR="${TMPDIR:-/tmp/torchinductor_root}"
export TMP="${TMP:-/tmp/torchinductor_root}"
export TEMP="${TEMP:-/tmp/torchinductor_root}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/root/.triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor_root}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/root/.cache/vllm}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"
export VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-rocm}"
export VLLM_VERSION_OVERRIDE="${VLLM_VERSION_OVERRIDE:-0.0.0+gfx906}"
export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx906}"
export GPU_ARCHS="${GPU_ARCHS:-gfx906}"
export FLASH_ATTENTION_TRITON_AMD_ENABLE="${FLASH_ATTENTION_TRITON_AMD_ENABLE:-TRUE}"
export FLASH_ATTENTION_TRITON_AMD_REF="${FLASH_ATTENTION_TRITON_AMD_REF:-TRUE}"
export DO_NOT_TRACK="${DO_NOT_TRACK:-1}"
export TORCH_BLAS_PREFER_HIPBLASLT="${TORCH_BLAS_PREFER_HIPBLASLT:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

if [[ -n "${NCCL_ALGO:-}" ]]; then
  export NCCL_ALGO
fi
if [[ -n "${NCCL_MAX_NCHANNELS:-}" ]]; then
  export NCCL_MAX_NCHANNELS
fi
if [[ -n "${NCCL_PROTO:-}" ]]; then
  export NCCL_PROTO
fi
if [[ -n "${HIP_VISIBLE_DEVICES:-}" ]]; then
  export HIP_VISIBLE_DEVICES
fi
if [[ -n "${VLLM_ENABLE_C1_TOPK8_MOE_FASTPATH:-}" ]]; then
  export VLLM_ENABLE_C1_TOPK8_MOE_FASTPATH
fi

copy_if_present() {
  local src="$1"
  local dst="$2"
  if [[ -f "${src}" ]]; then
    case "${dst}" in
      /opt/src/*|/workspace/*)
        if [[ ! -d "$(dirname "${dst}")" ]]; then
          return 0
        fi
        ;;
    esac
    mkdir -p "$(dirname "${dst}")"
    cp -f "${src}" "${dst}"
  fi
}

apply_patch_bundle() {
  local bundle_dir="${VLLM_PATCH_BUNDLE:-/opt/vllm_patch_bundle}"
  local targets=(
    "shared_expert_gate.py:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/shared_expert_gate.py"
    "shared_expert_gate.py:/opt/src/vllm/vllm/model_executor/layers/fused_moe/shared_expert_gate.py"
    "shared_expert_gate.py:/workspace/vllm/vllm/model_executor/layers/fused_moe/shared_expert_gate.py"
    "utils.py:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/utils.py"
    "utils.py:/opt/src/vllm/vllm/model_executor/layers/utils.py"
    "utils.py:/workspace/vllm/vllm/model_executor/layers/utils.py"
    "rocm.py:/opt/venv/lib/python3.12/site-packages/vllm/platforms/rocm.py"
    "rocm.py:/opt/src/vllm/vllm/platforms/rocm.py"
    "rocm.py:/workspace/vllm/vllm/platforms/rocm.py"
    "triton_attn.py:/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/triton_attn.py"
    "triton_attn.py:/opt/src/vllm/vllm/v1/attention/backends/triton_attn.py"
    "triton_attn.py:/workspace/vllm/vllm/v1/attention/backends/triton_attn.py"
    "triton_unified_attention.py:/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/ops/triton_unified_attention.py"
    "triton_unified_attention.py:/opt/src/vllm/vllm/v1/attention/ops/triton_unified_attention.py"
    "triton_unified_attention.py:/workspace/vllm/vllm/v1/attention/ops/triton_unified_attention.py"
    "qwen2_moe.py:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen2_moe.py"
    "qwen2_moe.py:/opt/src/vllm/vllm/model_executor/models/qwen2_moe.py"
    "qwen2_moe.py:/workspace/vllm/vllm/model_executor/models/qwen2_moe.py"
    "qwen3_moe.py:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_moe.py"
    "qwen3_moe.py:/opt/src/vllm/vllm/model_executor/models/qwen3_moe.py"
    "qwen3_moe.py:/workspace/vllm/vllm/model_executor/models/qwen3_moe.py"
    "qwen3_5.py:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_5.py"
    "qwen3_5.py:/opt/src/vllm/vllm/model_executor/models/qwen3_5.py"
    "qwen3_5.py:/workspace/vllm/vllm/model_executor/models/qwen3_5.py"
  )

  for item in "${targets[@]}"; do
    src="${bundle_dir}/${item%%:*}"
    dst="${item#*:}"
    copy_if_present "${src}" "${dst}"
  done

  local fused_src="${bundle_dir}/fused_moe.py"
  if [[ ! -f "${fused_src}" && -f "${bundle_dir}/fused_moe_pr39016.py" ]]; then
    fused_src="${bundle_dir}/fused_moe_pr39016.py"
  fi
  if [[ -f "${fused_src}" ]]; then
    copy_if_present "${fused_src}" "/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/fused_moe.py"
    copy_if_present "${fused_src}" "/opt/src/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py"
    copy_if_present "${fused_src}" "/workspace/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py"
  fi

  if [[ -d "${bundle_dir}/qwen36-python" ]]; then
    rm -rf /opt/qwen36-python
    mkdir -p /opt/qwen36-python
    cp -a "${bundle_dir}/qwen36-python/." /opt/qwen36-python/
    export PYTHONPATH="/opt/qwen36-python${PYTHONPATH:+:${PYTHONPATH}}"
  fi

  if [[ -z "${PYTHONPATH}" ]]; then
    export PYTHONPATH="/workspace:${PYTHONPATH}"
  fi

  if [[ "${REASONING_PARSER}" == "qwen3" && ! -d "${bundle_dir}/qwen36-python" && ! -d "/opt/qwen36-python" ]]; then
    echo "info: qwen3 reasoning parser requested; no qwen36-python sidecar was staged, so using vLLM's installed qwen3 parser"
  fi
}

apply_patch_bundle

patch_matcher_utils_missing_c_ops() {
  /opt/venv/bin/python - <<'PY'
from pathlib import Path

paths = [
    Path("/opt/venv/lib/python3.12/site-packages/vllm/compilation/passes/fusion/matcher_utils.py"),
    Path("/opt/src/vllm/vllm/compilation/passes/fusion/matcher_utils.py"),
    Path("/workspace/vllm/vllm/compilation/passes/fusion/matcher_utils.py"),
]

old_ops = '''RMS_OP = torch.ops._C.rms_norm.default
RMS_ADD_OP = torch.ops._C.fused_add_rms_norm.default
ROTARY_OP = torch.ops._C.rotary_embedding.default
FLASHINFER_ROTARY_OP = torch.ops.vllm.flashinfer_rotary_embedding.default

QUANT_OPS: dict[QuantKey, OpOverload] = {
    kFp8StaticTensorSym: torch.ops._C.static_scaled_fp8_quant.default,  # noqa: E501
    kFp8DynamicTensorSym: torch.ops._C.dynamic_scaled_fp8_quant.default,  # noqa: E501
    kFp8DynamicTokenSym: torch.ops._C.dynamic_per_token_scaled_fp8_quant.default,  # noqa: E501
}

if current_platform.is_cuda() and hasattr(torch.ops._C, "scaled_fp4_quant"):
    QUANT_OPS[kNvfp4Dynamic] = torch.ops._C.scaled_fp4_quant.default  # noqa: E501

if current_platform.is_cuda():
    QUANT_OPS[kFp8Dynamic128Sym] = torch.ops._C.per_token_group_fp8_quant.default  # noqa: E501
    QUANT_OPS[kFp8Dynamic64Sym] = torch.ops._C.per_token_group_fp8_quant.default  # noqa: E501

SILU_MUL_OP = torch.ops._C.silu_and_mul.default
'''

new_ops = '''def _vllm_gfx906_op_default(namespace: str, name: str):
    try:
        op = getattr(getattr(torch.ops, namespace), name)
        return op.default
    except AttributeError:
        return None


RMS_OP = _vllm_gfx906_op_default("_C", "rms_norm")
RMS_ADD_OP = _vllm_gfx906_op_default("_C", "fused_add_rms_norm")
ROTARY_OP = _vllm_gfx906_op_default("_C", "rotary_embedding")
FLASHINFER_ROTARY_OP = _vllm_gfx906_op_default(
    "vllm", "flashinfer_rotary_embedding")

QUANT_OPS: dict[QuantKey, OpOverload] = {}
for _key, _name in (
    (kFp8StaticTensorSym, "static_scaled_fp8_quant"),
    (kFp8DynamicTensorSym, "dynamic_scaled_fp8_quant"),
    (kFp8DynamicTokenSym, "dynamic_per_token_scaled_fp8_quant"),
):
    _op = _vllm_gfx906_op_default("_C", _name)
    if _op is not None:
        QUANT_OPS[_key] = _op

if current_platform.is_cuda():
    _op = _vllm_gfx906_op_default("_C", "scaled_fp4_quant")
    if _op is not None:
        QUANT_OPS[kNvfp4Dynamic] = _op

    _op = _vllm_gfx906_op_default("_C", "per_token_group_fp8_quant")
    if _op is not None:
        QUANT_OPS[kFp8Dynamic128Sym] = _op
        QUANT_OPS[kFp8Dynamic64Sym] = _op

SILU_MUL_OP = _vllm_gfx906_op_default("_C", "silu_and_mul")
'''

replacements = [
    (old_ops, new_ops),
    ('''        if enabled is None:
            enabled = RotaryEmbedding.enabled()
        if match_rocm_aiter is None:
            match_rocm_aiter = rocm_aiter_ops.is_triton_rotary_embed_enabled()

        super().__init__(enabled)
''',
'''        if enabled is None:
            enabled = RotaryEmbedding.enabled()
        if match_rocm_aiter is None:
            match_rocm_aiter = rocm_aiter_ops.is_triton_rotary_embed_enabled()
        if enabled and not use_flashinfer and not match_rocm_aiter and ROTARY_OP is None:
            enabled = False
        if enabled and use_flashinfer and FLASHINFER_ROTARY_OP is None:
            enabled = False

        super().__init__(enabled)
'''),
    ('''        if enabled is None:
            enabled = RMSNorm.enabled()

        super().__init__(enabled)
        self.epsilon = epsilon
        self._rmsnorm_op = RMS_OP
        self.match_rocm_aiter = match_rocm_aiter
''',
'''        if enabled is None:
            enabled = RMSNorm.enabled()
        if enabled and not match_rocm_aiter and RMS_OP is None:
            enabled = False

        super().__init__(enabled)
        self.epsilon = epsilon
        self._rmsnorm_op = RMS_OP
        self.match_rocm_aiter = match_rocm_aiter
'''),
    ('''        if enabled is None:
            enabled = RMSNorm.enabled()

        super().__init__(enabled)
        self.epsilon = epsilon
        self.match_rocm_aiter = match_rocm_aiter

        self._rmsnorm_op = RMS_ADD_OP
''',
'''        if enabled is None:
            enabled = RMSNorm.enabled()
        if enabled and not match_rocm_aiter and RMS_ADD_OP is None:
            enabled = False

        super().__init__(enabled)
        self.epsilon = epsilon
        self.match_rocm_aiter = match_rocm_aiter

        self._rmsnorm_op = RMS_ADD_OP
'''),
    ('''        if enabled is None:
            enabled = QuantFP8.enabled()

        super().__init__(enabled)
''',
'''        if enabled is None:
            enabled = QuantFP8.enabled()
        if enabled and not match_rocm_aiter and quant_key not in QUANT_OPS:
            enabled = False

        super().__init__(enabled)
'''),
    ('''        else:
            assert quant_key in QUANT_OPS, (
                f"unsupported quantization scheme {quant_key}"
            )
            self.QUANT_OP = QUANT_OPS[quant_key]

            assert quant_key.dtype == current_platform.fp8_dtype(), (
                "Only QuantFP8 supported by"
            )
            assert quant_key.scale2 is None

        self.quant_fp8 = QuantFP8(
''',
'''        else:
            if self.enabled:
                assert quant_key in QUANT_OPS, (
                    f"unsupported quantization scheme {quant_key}"
                )
                self.QUANT_OP = QUANT_OPS[quant_key]

                assert quant_key.dtype == current_platform.fp8_dtype(), (
                    "Only QuantFP8 supported by"
                )
                assert quant_key.scale2 is None
            else:
                self.QUANT_OP = None

        self.quant_fp8 = QuantFP8(
'''),
    ('''        if enabled is None:
            enabled = SiluAndMul.enabled()
        super().__init__(enabled)
''',
'''        if enabled is None:
            enabled = SiluAndMul.enabled()
        if enabled and SILU_MUL_OP is None:
            enabled = False
        super().__init__(enabled)
'''),
]

for path in paths:
    if not path.exists():
        continue
    text = path.read_text()
    if "_vllm_gfx906_op_default" in text:
        continue
    new_text = text
    for old, new in replacements:
        if old not in new_text:
            raise SystemExit(f"matcher_utils patch anchor not found in {path}")
        new_text = new_text.replace(old, new, 1)
    path.write_text(new_text)
PY
}

patch_matcher_utils_missing_c_ops

patch_rocm_gfx906_capability() {
  /opt/venv/bin/python - <<'PY'
from pathlib import Path

paths = [
    Path("/opt/venv/lib/python3.12/site-packages/vllm/platforms/rocm.py"),
    Path("/opt/src/vllm/vllm/platforms/rocm.py"),
    Path("/workspace/vllm/vllm/platforms/rocm.py"),
]

old_on_gfx9 = '_ON_GFX9 = any(arch in _GCN_ARCH for arch in ["gfx90a", "gfx942", "gfx950"])'
new_on_gfx9 = '_ON_GFX9 = any(arch in _GCN_ARCH for arch in ["gfx9", "gfx90a", "gfx942", "gfx950"])'

old_capability = '''    elif n == 4:
        # 2-digit major: gfx10xx, gfx11xx, gfx12xx
        # major(2) + minor(1) + stepping(1)
        major = int(digits[:2])
        minor = int(digits[2])
'''
new_capability = '''    elif n == 4:
        # AMD SMI on gfx906/MI50 can report "gfx9006".
        # Treat gfx90xx as the gfx9 one-digit-major layout rather than
        # the gfx10/11/12 two-digit-major layout.
        if digits.startswith("90"):
            major = int(digits[0])
            minor = int(digits[1])
        else:
            # 2-digit major: gfx10xx, gfx11xx, gfx12xx
            # major(2) + minor(1) + stepping(1)
            major = int(digits[:2])
            minor = int(digits[2])
'''

for path in paths:
    if not path.exists():
        continue
    text = path.read_text()
    text = text.replace(old_on_gfx9, new_on_gfx9)
    if "AMD SMI on gfx906/MI50 can report" not in text:
        text = text.replace(old_capability, new_capability)
    if "def on_gfx906(" not in text:
        text = text.rstrip() + '''

def on_gfx906() -> bool:
    return "gfx906" in _GCN_ARCH or "gfx9006" in _GCN_ARCH
''' + "\n"
    path.write_text(text)
PY
}

patch_rocm_gfx906_capability

patch_batch_invariant_compat() {
  /opt/venv/bin/python - <<'PY'
from pathlib import Path

paths = [
    Path("/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/batch_invariant.py"),
    Path("/opt/src/vllm/vllm/model_executor/layers/batch_invariant.py"),
    Path("/workspace/vllm/vllm/model_executor/layers/batch_invariant.py"),
]

shim = '''

def vllm_is_batch_invariant() -> bool:
    """Compatibility shim for MoE hotfixes from newer vLLM branches."""
    return False
'''

for path in paths:
    if not path.exists():
        continue
    text = path.read_text()
    if "def vllm_is_batch_invariant(" not in text:
        path.write_text(text.rstrip() + shim + "\n")
PY
}

patch_batch_invariant_compat

patch_fused_moe_utils_compat() {
  /opt/venv/bin/python - <<'PY'
from pathlib import Path

paths = [
    Path("/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/utils.py"),
    Path("/opt/src/vllm/vllm/model_executor/layers/fused_moe/utils.py"),
    Path("/workspace/vllm/vllm/model_executor/layers/fused_moe/utils.py"),
]

shim = '''

def disable_inplace() -> bool:
    """Compatibility shim for MoE hotfixes from newer vLLM branches."""
    return False
'''

for path in paths:
    if not path.exists():
        continue
    text = path.read_text()
    if "def disable_inplace(" not in text:
        path.write_text(text.rstrip() + shim + "\n")
PY
}

patch_fused_moe_utils_compat

patch_activation_custom_op_fallbacks() {
  /opt/venv/bin/python - <<'PY'
from pathlib import Path

paths = [
    Path("/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/activation.py"),
    Path("/opt/src/vllm/vllm/model_executor/layers/activation.py"),
    Path("/workspace/vllm/vllm/model_executor/layers/activation.py"),
]

old = '''        if current_platform.is_cuda_alike() or current_platform.is_xpu():
            self.op = torch.ops._C.silu_and_mul
        elif current_platform.is_cpu():
            self._forward_method = self.forward_native
'''
new = '''        if (
            (current_platform.is_cuda_alike() or current_platform.is_xpu())
            and hasattr(torch.ops._C, "silu_and_mul")
        ):
            self.op = torch.ops._C.silu_and_mul
        else:
            self._forward_method = self.forward_native
'''

for path in paths:
    if not path.exists():
        continue
    text = path.read_text()
    if 'and hasattr(torch.ops._C, "silu_and_mul")' not in text:
        text = text.replace(old, new)
        path.write_text(text)
PY
}

patch_activation_custom_op_fallbacks

patch_moe_activation_custom_op_fallbacks() {
  /opt/venv/bin/python - <<'PY'
from pathlib import Path

paths = [
    Path("/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/activation.py"),
    Path("/opt/src/vllm/vllm/model_executor/layers/fused_moe/activation.py"),
    Path("/workspace/vllm/vllm/model_executor/layers/fused_moe/activation.py"),
]

old = '''    if activation == MoEActivation.SILU:
        torch.ops._C.silu_and_mul(output, input)
'''
new = '''    if activation == MoEActivation.SILU:
        if hasattr(torch.ops._C, "silu_and_mul"):
            torch.ops._C.silu_and_mul(output, input)
        else:
            d = input.shape[-1] // 2
            output.copy_(F.silu(input[:, :d]) * input[:, d:])
'''

for path in paths:
    if not path.exists():
        continue
    text = path.read_text()
    if 'hasattr(torch.ops._C, "silu_and_mul")' not in text:
        text = text.replace(old, new)
        path.write_text(text)
PY
}

patch_moe_activation_custom_op_fallbacks

patch_rms_norm_custom_op_fallbacks() {
  /opt/venv/bin/python - <<'PY'
from pathlib import Path

paths = [
    Path("/opt/venv/lib/python3.12/site-packages/vllm/kernels/vllm_c.py"),
    Path("/opt/src/vllm/vllm/kernels/vllm_c.py"),
    Path("/workspace/vllm/vllm/kernels/vllm_c.py"),
]

rms_old = '''    assert variance_size is None
    # ROCm's vLLM C RMSNorm kernel operates on contiguous 2D tensors.
'''
rms_new = '''    assert variance_size is None
    if not hasattr(torch.ops._C, "rms_norm"):
        return ir.ops.rms_norm.impls["native"].impl_fn(
            x, weight, epsilon, variance_size
        )
    # ROCm's vLLM C RMSNorm kernel operates on contiguous 2D tensors.
'''

fused_old = '''    assert variance_size is None
    if IS_ROCM and (not x.is_contiguous() or not x_residual.is_contiguous()):
'''
fused_new = '''    assert variance_size is None
    if not hasattr(torch.ops._C, "fused_add_rms_norm"):
        output, residual = ir.ops.fused_add_rms_norm.impls["native"].impl_fn(
            x, x_residual, weight, epsilon, variance_size
        )
        x.copy_(output)
        x_residual.copy_(residual)
        return x, x_residual
    if IS_ROCM and (not x.is_contiguous() or not x_residual.is_contiguous()):
'''

for path in paths:
    if not path.exists():
        continue
    text = path.read_text()
    if 'not hasattr(torch.ops._C, "rms_norm")' not in text:
        text = text.replace(rms_old, rms_new)
    if 'not hasattr(torch.ops._C, "fused_add_rms_norm")' not in text:
        text = text.replace(fused_old, fused_new)
    path.write_text(text)
PY
}

patch_rms_norm_custom_op_fallbacks

patch_rotary_custom_op_fallbacks() {
  /opt/venv/bin/python - <<'PY'
from pathlib import Path

paths = [
    Path("/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/rotary_embedding/base.py"),
    Path("/opt/src/vllm/vllm/model_executor/layers/rotary_embedding/base.py"),
    Path("/workspace/vllm/vllm/model_executor/layers/rotary_embedding/base.py"),
]

old = '''        if self.use_aiter:
            cos_sin_cache = self._match_cos_sin_cache_dtype(query)
            self.rocm_aiter_triton_rotary_embedding(
                positions,
                query,
                key,
                self.head_size,
                cos_sin_cache,
                self.is_neox_style,
            )
            return query, key
        return self.forward_cuda(positions, query, key)
'''
new = '''        if self.use_aiter:
            cos_sin_cache = self._match_cos_sin_cache_dtype(query)
            self.rocm_aiter_triton_rotary_embedding(
                positions,
                query,
                key,
                self.head_size,
                cos_sin_cache,
                self.is_neox_style,
            )
            return query, key
        if not hasattr(torch.ops._C, "rotary_embedding"):
            return self.forward_native(positions, query, key)
        return self.forward_cuda(positions, query, key)
'''

for path in paths:
    if not path.exists():
        continue
    text = path.read_text()
    if 'not hasattr(torch.ops._C, "rotary_embedding")' not in text:
        text = text.replace(old, new)
        path.write_text(text)
PY
}

patch_rotary_custom_op_fallbacks

cd /tmp

report_reference_custom_op_surface() {
  if [[ "${OPT_LEVEL}" == "0" ]]; then
    return 0
  fi

  local missing_ops
  missing_ops="$(
    /opt/venv/bin/python - <<'PY'
import importlib
import torch

for module in ("vllm._C", "vllm._rocm_C"):
    try:
        importlib.import_module(module)
    except Exception:
        pass

missing = []
for name in ("rotary_embedding", "silu_and_mul", "rms_norm", "fused_add_rms_norm"):
    if not hasattr(torch.ops._C, name):
        missing.append(name)
print(",".join(missing))
PY
  )"

  if [[ -n "${missing_ops}" ]]; then
    echo "info: requested -O=${OPT_LEVEL}; vLLM _C op(s) missing as on the .10 reference image: ${missing_ops}" >&2
    echo "info: keeping -O=${OPT_LEVEL} and using the bundled gfx906 Python/Triton fallbacks." >&2
  fi
}

report_reference_custom_op_surface

args=(
  vllm serve "${MODEL}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --enable-auto-tool-choice
  --tool-call-parser "${TOOL_CALL_PARSER}"
  --dtype "${VLLM_DTYPE}"
  --host "${HOST}"
  --port "${PORT}"
  --tensor-parallel-size "${TP_SIZE}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --trust-remote-code
  --generation-config vllm
  "-O=${OPT_LEVEL}"
)

if [[ "${DISABLE_ASYNC_SCHEDULING}" == "1" ]]; then
  args+=(--no-async-scheduling)
else
  args+=(--async-scheduling)
fi

if [[ -n "${REASONING_PARSER}" && "${REASONING_PARSER}" != "none" ]]; then
  args+=(--reasoning-parser "${REASONING_PARSER}")
fi

if [[ -n "${EXTRA_VLLM_ARGS}" ]]; then
  read -r -a extra_args <<<"${EXTRA_VLLM_ARGS}"
  args+=("${extra_args[@]}")
fi

printf 'qwen36 c1-topk8 async deploy starting with command:\n'
printf '%q ' "${args[@]}"
printf '\n'
exec "${args[@]}"
ENTRYPOINT
}

write_embedded_moe_config() {
  local destination="$1"
  mkdir -p "$(dirname "${destination}")"
  cat <<'MOE_CONFIG' > "${destination}"
{
  "triton_version": "3.5.1",
  "1": {
    "BLOCK_SIZE_M": 8,
    "BLOCK_SIZE_N": 32,
    "BLOCK_SIZE_K": 32,
    "GROUP_SIZE_M": 1,
    "num_warps": 2,
    "num_stages": 2,
    "waves_per_eu": 1,
    "matrix_instr_nonkdim": 16,
    "kpack": 1
  },
  "2": {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 16,
    "BLOCK_SIZE_K": 32,
    "GROUP_SIZE_M": 1,
    "num_warps": 2,
    "num_stages": 2,
    "waves_per_eu": 1,
    "matrix_instr_nonkdim": 16,
    "kpack": 1
  },
  "4": {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 32,
    "BLOCK_SIZE_K": 32,
    "GROUP_SIZE_M": 1,
    "num_warps": 4,
    "num_stages": 2,
    "waves_per_eu": 1,
    "matrix_instr_nonkdim": 16,
    "kpack": 2
  }
}
MOE_CONFIG
}

write_embedded_hotfixes() {
  local bundle_dir="$1"
  mkdir -p "${bundle_dir}"

  cat <<'HOTFIX_SHARED_EXPERT_GATE' > "${bundle_dir}/shared_expert_gate.py"
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Triton kernel for the Qwen2/3-MoE shared-expert sigmoid gate.

Replaces the three-kernel `F.sigmoid(linear(x)) * out` tail of
`Qwen2MoeMLP.forward` / `Qwen3MoeMLP.forward` with a single row-fused
pass that removes the two HBM-resident intermediates.

The wrapper is shape-guarded and silently falls back to the PyTorch
reference (`F.sigmoid(F.linear(x, weight)) * out`) for any input shape
this kernel does not handle, so it is safe to use behind the existing
`expert_gate` call sites without further checks.
"""

import torch
import torch.nn.functional as F

from vllm.triton_utils import tl, triton


@triton.jit
def _fused_shared_expert_gate_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    y_ptr,
    stride_x_n,
    stride_out_n,
    stride_y_n,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < K

    x = tl.load(x_ptr + row * stride_x_n + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gate = tl.sigmoid(tl.sum(x * weight, axis=0))

    out = tl.load(out_ptr + row * stride_out_n + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    tl.store(y_ptr + row * stride_y_n + offsets, out * gate, mask=mask)


def fused_shared_expert_gate(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Compute ``F.sigmoid(F.linear(x, weight)) * out`` in a single pass.

    Specialised for a one-row gate weight (``weight.shape == [1, K]``), as
    produced by ``ReplicatedLinear(hidden_size, 1)`` in the Qwen2/3-MoE
    shared-expert blocks. The kernel handles arbitrary row strides on ``x``,
    ``out``, and the output ``y`` (so views with non-K row stride are fine),
    but assumes unit stride along the inner ``K`` dimension. For any input
    that violates these requirements -- including the unit-inner-stride
    requirement on ``weight`` -- the function falls back to the PyTorch
    reference so callers can use it unconditionally.

    Args:
        x: Shared-expert input, shape ``[N, K]``.
        weight: Gate weight, shape ``[1, K]``.
        out: Shared-expert MLP output, shape ``[N, K]``.

    Returns:
        Contiguous ``[N, K]`` tensor equal to ``sigmoid(x @ weight.T) * out``
        within bf16/fp16 tolerance.
    """
    if (
        x.ndim != 2
        or out.ndim != 2
        or weight.ndim != 2
        or weight.shape[0] != 1
        or x.shape != out.shape
        or weight.shape[1] != x.shape[1]
        or x.stride(1) != 1
        or out.stride(1) != 1
        or weight.stride(1) != 1
    ):
        return F.sigmoid(F.linear(x, weight)) * out

    y = torch.empty_like(out, memory_format=torch.contiguous_format)
    _fused_shared_expert_gate_kernel[(x.shape[0],)](
        x,
        weight,
        out,
        y,
        x.stride(0),
        out.stride(0),
        y.stride(0),
        K=x.shape[1],
        BLOCK_K=triton.next_power_of_2(x.shape[1]),
        num_warps=8,
    )
    return y
HOTFIX_SHARED_EXPERT_GATE

  cat <<'HOTFIX_QWEN3_MOE' > "${bundle_dir}/qwen3_moe.py"
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Qwen3MoE model compatible with HuggingFace weights."""

import typing
from collections.abc import Callable, Iterable
from itertools import islice
from typing import Any

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import SharedFusedMoE
from vllm.model_executor.layers.fused_moe.shared_expert_gate import (
    fused_shared_expert_gate,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.utils import sequence_parallel_chunk
from vllm.sequence import IntermediateTensors

from .interfaces import MixtureOfExperts, SupportsEagle3, SupportsLoRA, SupportsPP
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)


class Qwen3MoeMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        expert_gate: torch.nn.Linear | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()
        self.expert_gate = expert_gate

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        out = self.act_fn(gate_up)
        out, _ = self.down_proj(out)

        if self.expert_gate is not None:
            out = fused_shared_expert_gate(x, self.expert_gate.weight, out)

        return out


class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        parallel_config = vllm_config.parallel_config
        quant_config = vllm_config.quant_config

        self.tp_size = get_tensor_model_parallel_world_size()

        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts = config.num_experts

        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )

        # Load balancing settings.
        vllm_config = get_current_vllm_config()
        eplb_config = vllm_config.parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb

        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate",
        )

        shared_expert_intermediate_size = getattr(
            config, "shared_expert_intermediate_size", 0
        )
        if shared_expert_intermediate_size > 0:
            self.shared_expert_gate = ReplicatedLinear(
                config.hidden_size,
                1,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.shared_expert_gate",
            )
            self.shared_expert = Qwen3MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=shared_expert_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                expert_gate=self.shared_expert_gate,
                prefix=f"{prefix}.shared_expert",
            )
        else:
            self.shared_expert_gate = None
            self.shared_expert = None

        self.experts = SharedFusedMoE(
            shared_experts=self.shared_expert,
            gate=self.gate,
            num_experts=self.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=False,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=self.is_sequence_parallel,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert hidden_states.dim() <= 2, (
            "Qwen3MoeSparseMoeBlock only supports 1D or 2D inputs"
        )
        is_input_1d = hidden_states.dim() == 1
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        if self.is_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)
        shared_out, fused_out = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )
        final_hidden_states = (
            shared_out + fused_out if shared_out is not None else fused_out
        )

        if self.is_sequence_parallel:
            final_hidden_states = tensor_model_parallel_all_gather(
                final_hidden_states, 0
            )
            final_hidden_states = final_hidden_states[:num_tokens]
        elif self.tp_size > 1:
            final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(  # noqa E501
                final_hidden_states
            )

        # return to 1d if input is 1d
        return final_hidden_states.squeeze(0) if is_input_1d else final_hidden_states


class Qwen3MoeAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict[str, Any],
        max_position_embeddings: int = 8192,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        dual_chunk_attention_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or (hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings
        self.dual_chunk_attention_config = dual_chunk_attention_config

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position_embeddings,
            rope_parameters=rope_parameters,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            **{
                "layer_idx": extract_layer_index(prefix),
                "dual_chunk_attention_config": dual_chunk_attention_config,
            }
            if dual_chunk_attention_config
            else {},
        )

        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # Add qk-norm
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)

        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen3MoeDecoderLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        self.self_attn = Qwen3MoeAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_parameters=config.rope_parameters,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            dual_chunk_attention_config=dual_chunk_attention_config,
        )

        # `mlp_only_layers` in the config.
        layer_idx = extract_layer_index(prefix)
        mlp_only_layers = (
            [] if not hasattr(config, "mlp_only_layers") else config.mlp_only_layers
        )
        if (layer_idx not in mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen3MoeSparseMoeBlock(
                vllm_config=vllm_config, prefix=f"{prefix}.mlp"
            )
        else:
            self.mlp = Qwen3MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile
class Qwen3MoeModel(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        decoder_layer_type: type[torch.nn.Module] = Qwen3MoeDecoderLayer,
    ):
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config
        eplb_config = parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts

        self.vocab_size = config.vocab_size
        self.config = config
        self.quant_config = quant_config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: decoder_layer_type(vllm_config=vllm_config, prefix=prefix),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )
        # Track layers for auxiliary hidden state outputs (EAGLE3)
        self.aux_hidden_state_layers: tuple[int, ...] = ()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        aux_hidden_states = []
        for layer_idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            # Collect auxiliary hidden states if specified
            if layer_idx in self.aux_hidden_state_layers:
                aux_hidden_state = (
                    hidden_states + residual if residual is not None else hidden_states
                )
                aux_hidden_states.append(aux_hidden_state)
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)

        # Return auxiliary hidden states if collected
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return SharedFusedMoE.make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
            num_redundant_experts=self.num_redundant_experts,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Skip loading extra parameters for GPTQ/modelopt models.
        ignore_suffixes = (
            ".bias",
            "_bias",
            ".weight_scale",
            "_weight_scale",
            ".input_scale",
            "_input_scale",
        )

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()
        for name, loaded_weight in weights:
            if self.quant_config is not None and (
                scale_name := self.quant_config.get_cache_scale(name)
            ):
                # Loading kv cache quantization scales
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                assert loaded_weight.numel() == 1, (
                    f"KV scale numel {loaded_weight.numel()} != 1"
                )
                loaded_weight = loaded_weight.squeeze()
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            if "scale" in name or "zero_point" in name:
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)

                # Skip loading extra parameters for GPTQ/modelopt models.
                if name.endswith(ignore_suffixes) and name not in params_dict:
                    continue

                # Skip layers on other devices.
                if is_pp_missing_parameter(name, self):
                    continue
                if name.endswith("scale"):
                    # Remapping the name of FP8 kv-scale.
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if weight_loader == default_weight_loader:
                    weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight, shard_id)
                break
            else:
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue

                    # Anyway, this is an expert weight and should not be
                    # attempted to load as other weights later
                    is_expert_weight = True

                    # Do not modify `name` since the loop may continue here
                    # Instead, create a new variable
                    name_mapped = name.replace(weight_name, param_name)

                    if is_pp_missing_parameter(name_mapped, self):
                        continue

                    # Skip loading extra parameters for GPTQ/modelopt models.
                    if (
                        name_mapped.endswith(ignore_suffixes)
                        and name_mapped not in params_dict
                    ):
                        continue

                    param = params_dict[name_mapped]
                    # We should ask the weight loader to return success or not
                    # here since otherwise we may skip experts with other
                    # available replicas.
                    weight_loader = typing.cast(
                        Callable[..., bool], param.weight_loader
                    )
                    success = weight_loader(
                        param,
                        loaded_weight,
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                    if success:
                        name = name_mapped
                        break
                else:
                    if is_expert_weight:
                        # We've checked that this is an expert weight
                        # However it's not mapped locally to this rank
                        # So we simply skip it
                        continue

                    # Skip loading extra parameters for GPTQ/modelopt models.
                    if name.endswith(ignore_suffixes) and name not in params_dict:
                        continue
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class Qwen3MoeForCausalLM(
    nn.Module, SupportsPP, SupportsLoRA, SupportsEagle3, MixtureOfExperts
):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ]
    }

    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        # Only perform the following mapping when Qwen3MoeMLP exists
        if getattr(config, "mlp_only_layers", []):
            self.packed_modules_mapping["gate_up_proj"] = ["gate_proj", "up_proj"]
        self.model = Qwen3MoeModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        # Set MoE hyperparameters
        self.expert_weights = []

        self.moe_layers = []
        example_layer = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue

            assert isinstance(layer, Qwen3MoeDecoderLayer)
            if isinstance(layer.mlp, Qwen3MoeSparseMoeBlock):
                example_layer = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        if example_layer is None:
            raise RuntimeError("No Qwen3MoE layer found in the model.layers.")

        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_logical_experts = example_layer.n_logical_experts
        self.num_physical_experts = example_layer.n_physical_experts
        self.num_local_physical_experts = example_layer.n_local_physical_experts
        self.num_routed_experts = example_layer.n_routed_experts
        self.num_redundant_experts = example_layer.n_redundant_experts

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.model.layers:
            if isinstance(layer.mlp, Qwen3MoeSparseMoeBlock):
                moe = layer.mlp
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model.aux_hidden_state_layers = layers

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        num_layers = len(self.model.layers)
        return (2, num_layers // 2, num_layers - 3)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()
HOTFIX_QWEN3_MOE

  cat <<'HOTFIX_QWEN2_MOE' > "${bundle_dir}/qwen2_moe.py"
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/qwen2_moe/modeling_qwen2_moe.py
# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Qwen2MoE model compatible with HuggingFace weights."""

from collections.abc import Iterable
from itertools import islice
from typing import Any

import torch
from torch import nn
from transformers import Qwen2MoeConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import SharedFusedMoE
from vllm.model_executor.layers.fused_moe.shared_expert_gate import (
    fused_shared_expert_gate,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsLoRA, SupportsPP
from .utils import (
    AutoWeightsLoader,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)


class Qwen2MoeMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        expert_gate: torch.nn.Linear | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()
        self.expert_gate = expert_gate

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        out = self.act_fn(gate_up)
        out, _ = self.down_proj(out)

        if self.expert_gate is not None:
            out = fused_shared_expert_gate(x, self.expert_gate.weight, out)

        return out


class Qwen2MoeSparseMoeBlock(nn.Module):
    def __init__(
        self,
        config: Qwen2MoeConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()

        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )

        self.shared_expert_gate = ReplicatedLinear(
            config.hidden_size,
            1,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.shared_expert_gate",
        )

        if config.shared_expert_intermediate_size > 0:
            self.shared_expert = Qwen2MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.shared_expert_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                expert_gate=self.shared_expert_gate,
                prefix=f"{prefix}.shared_expert",
            )
        else:
            self.shared_expert = None

        self.experts = SharedFusedMoE(
            shared_experts=self.shared_expert,
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=False,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # NOTE: hidden_states can have either 1D or 2D shape.
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)

        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )
        if self.shared_expert is not None:
            final_hidden_states = final_hidden_states[0] + final_hidden_states[1]
        if self.tp_size > 1:
            final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(  # noqa E501
                final_hidden_states
            )

        return final_hidden_states.view(orig_shape)


class Qwen2MoeAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict[str, Any] | None = None,
        max_position_embeddings: int = 8192,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        dual_chunk_attention_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings
        self.dual_chunk_attention_config = dual_chunk_attention_config

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position_embeddings,
            rope_parameters=rope_parameters,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            **{
                "layer_idx": extract_layer_index(prefix),
                "dual_chunk_attention_config": dual_chunk_attention_config,
            }
            if dual_chunk_attention_config
            else {},
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen2MoeDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen2MoeConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        self.self_attn = Qwen2MoeAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_parameters=config.rope_parameters,
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            dual_chunk_attention_config=dual_chunk_attention_config,
        )

        # Note: Qwen/Qwen2-57B-A14B-Instruct does not have
        # `mlp_only_layers` in the config.
        layer_idx = extract_layer_index(prefix)
        mlp_only_layers = (
            [] if not hasattr(config, "mlp_only_layers") else config.mlp_only_layers
        )
        if (layer_idx not in mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen2MoeSparseMoeBlock(
                config=config, quant_config=quant_config, prefix=f"{prefix}.mlp"
            )
        else:
            self.mlp = Qwen2MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> torch.Tensor:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile
class Qwen2MoeModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.vocab_size = config.vocab_size
        self.config = config

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Qwen2MoeDecoderLayer(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(positions, hidden_states, residual)
        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return SharedFusedMoE.make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()
        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if (
                    name.endswith(".bias") or name.endswith("_bias")
                ) and name not in params_dict:
                    continue
                # Skip layers on other devices.
                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)

                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Skip loading extra bias for GPTQ models.
                    if (
                        name.endswith(".bias") or name.endswith("_bias")
                    ) and name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if (
                        name.endswith(".bias") or name.endswith("_bias")
                    ) and name not in params_dict:
                        continue
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Remapping the name of FP8 kv-scale.
                    if name.endswith("kv_scale"):
                        remapped_kv_scale_name = name.replace(
                            ".kv_scale", ".attn.kv_scale"
                        )
                        if remapped_kv_scale_name not in params_dict:
                            logger.warning_once(
                                "Found kv_scale in the checkpoint (e.g. %s), but not found the expected name in the model (e.g. %s). kv_scale is not loaded.",  #  noqa: E501
                                name,
                                remapped_kv_scale_name,
                            )
                            continue
                        else:
                            name = remapped_kv_scale_name
                    # GGUF: make sure that shared_expert_gate is a 2D tensor.
                    if (
                        "mlp.shared_expert_gate" in name
                        and len(loaded_weight.shape) == 1
                    ):
                        loaded_weight = loaded_weight[None, :]
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class Qwen2MoeForCausalLM(nn.Module, SupportsPP, SupportsLoRA):
    fall_back_to_pt_during_load = False
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ]
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        # Only perform the following mapping when Qwen2MoeMLP exists
        if (
            getattr(config, "mlp_only_layers", [])
            or config.shared_expert_intermediate_size > 0
        ):
            self.packed_modules_mapping["gate_up_proj"] = ["gate_proj", "up_proj"]

        self.model = Qwen2MoeModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()
HOTFIX_QWEN2_MOE

  cat <<'HOTFIX_QWEN3_5' > "${bundle_dir}/qwen3_5.py"
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2025 The vLLM team.
# Copyright 2025 The Qwen Team.
# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Qwen3.5 Series compatible with HuggingFace weights."""

import typing
from collections.abc import Callable, Iterable

import torch
from einops import rearrange
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import (
    VllmConfig,
)
from vllm.distributed import (
    get_pp_group,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import (
    GemmaRMSNorm as Qwen3_5RMSNorm,
)
from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.qwen3_5 import (
    Qwen3_5Config,
    Qwen3_5TextConfig,
)
from vllm.transformers_utils.configs.qwen3_5_moe import (
    Qwen3_5MoeConfig,
    Qwen3_5MoeTextConfig,
)

from .interfaces import (
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsPP,
    _require_is_multimodal,
)
from .qwen2_moe import Qwen2MoeMLP as Qwen3NextMLP
from .qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextDecoderLayer,
    Qwen3NextGatedDeltaNet,
    Qwen3NextModel,
    Qwen3NextSparseMoeBlock,
    QwenNextMixtureOfExperts,
)
from .qwen3_vl import (
    Qwen3_VisionTransformer,
    Qwen3VLDummyInputsBuilder,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
)
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    _merge_multimodal_embeddings,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)


class Qwen3_5ProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen3_5Config)


class Qwen3_5MoeProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen3_5MoeConfig)


class Qwen3_5GatedDeltaNet(Qwen3NextGatedDeltaNet):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Parent Qwen3NextGatedDeltaNet still builds in_proj_ba separately.
        # This adapted Qwen3.5 path fuses b/a into in_proj_qkvz instead.
        if hasattr(self, "in_proj_ba"):
            delattr(self, "in_proj_ba")

    def fix_query_key_value_ordering(
        self,
        mixed_qkvz: torch.Tensor,
        mixed_ba: torch.Tensor,
    ):
        raise NotImplementedError(
            "Qwen3.5 Series dont need to fix query key value ordering"
        )

    def create_qkvz_proj(
        self,
        hidden_size: int,
        key_dim: int,
        value_dim: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> MergedColumnParallelLinear:
        return MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[
                key_dim,
                key_dim,
                value_dim,
                value_dim,
                self.num_v_heads,
                self.num_v_heads,
            ],
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
        )

    def create_ba_proj(
        self,
        hidden_size: int,
        num_v_heads: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> MergedColumnParallelLinear:
        # Qwen3.5 has separate in_proj_b and in_proj_a weights in the
        # checkpoint, which are loaded into the fused in_proj_ba parameter
        # via stacked_params_mapping with shard_id 0 and 1 respectively.
        return MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[num_v_heads] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        """
        Forward pass with three parts:
        1. Input projection
        2. Core attention (custom op)
        3. Output projection
        """
        num_tokens = hidden_states.size(0)

        # ============================================================
        # Part 1: Input Projection
        # ============================================================
        mixed_qkvzba, _ = self.in_proj_qkvz(hidden_states)
        qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
        z_size = self.value_dim // self.tp_size
        ba_size = self.num_v_heads // self.tp_size
        mixed_qkv, z, b, a = mixed_qkvzba.split(
            [qkv_size, z_size, ba_size, ba_size], dim=-1
        )
        z = z.reshape(z.size(0), -1, self.head_v_dim)

        b = b.contiguous()
        a = a.contiguous()

        # ============================================================
        # Part 2: Core Attention (Custom Op)
        # ============================================================
        # Note: we should not use torch.empty here like other attention backends,
        # see discussions in https://github.com/vllm-project/vllm/pull/28182
        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        torch.ops.vllm.gdn_attention_core(
            mixed_qkv,
            b,
            a,
            core_attn_out,
            self.prefix,
        )

        # ============================================================
        # Part 3: Output Projection
        # ============================================================
        z_shape_og = z.shape
        # Reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
        output[:num_tokens], _ = self.out_proj(core_attn_out)


class Qwen3_5DecoderLayer(Qwen3NextDecoderLayer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        prefix: str = "",
    ) -> None:
        super(Qwen3NextDecoderLayer, self).__init__()

        config = vllm_config.model_config.hf_text_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        speculative_config = vllm_config.speculative_config

        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)

        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(
                config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                speculative_config=speculative_config,
                prefix=f"{prefix}.linear_attn",
            )
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3NextAttention(
                config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            raise ValueError(f"Invalid layer_type {self.layer_type}")

        # NOTE: Determine the MLP type based on the model type
        # Qwen3.5 use all layers for MLP / Qwen3.5-MoE use sparse MoE blocks
        if config.model_type == "qwen3_5_moe_text":
            self.mlp = Qwen3NextSparseMoeBlock(
                vllm_config=vllm_config,
                prefix=f"{prefix}.mlp",
            )
        elif config.model_type == "qwen3_5_text":
            self.mlp = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            raise ValueError(f"Invalid model_type {config.model_type}")

        self.input_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_scale = getattr(config, "layer_scale", False)
        if self.layer_scale:
            self.attn_layer_scale = torch.nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    config.hidden_size,
                ),
            )
            self.ffn_layer_scale = torch.nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    config.hidden_size,
                ),
            )


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class Qwen3_5Model(Qwen3NextModel):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super(Qwen3NextModel, self).__init__()

        config: Qwen3_5TextConfig | Qwen3_5MoeTextConfig = (
            vllm_config.model_config.hf_text_config
        )
        parallel_config = vllm_config.parallel_config

        eplb_config = parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts

        self.config = config

        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        def get_layer(prefix: str):
            return Qwen3_5DecoderLayer(
                vllm_config,
                layer_type=config.layer_types[extract_layer_index(prefix)],
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        if get_pp_group().is_last_rank:
            self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

    def load_fused_expert_weights(
        self,
        name: str,
        params_dict: dict,
        loaded_weight: torch.Tensor,
        shard_id: str,
        num_experts: int,
    ) -> bool:
        param = params_dict[name]
        weight_loader = typing.cast(Callable[..., bool], param.weight_loader)
        loaded_local_expert = False
        for expert_id in range(num_experts):
            curr_expert_weight = loaded_weight[expert_id]
            success = weight_loader(
                param,
                curr_expert_weight,
                name,
                shard_id,
                expert_id,
                return_success=True,
            )
            if success:
                loaded_local_expert = True

        return loaded_local_expert

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            # self attention
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            # mlp
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            # GDN
            ("in_proj_qkvz", "in_proj_qkv", (0, 1, 2)),
            ("in_proj_qkvz", "in_proj_z", 3),
            ("in_proj_qkvz", "in_proj_b", 4),
            ("in_proj_qkvz", "in_proj_a", 5),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()
        is_fused_expert = False
        fused_expert_params_mapping = [
            ("experts.w13_weight", "experts.gate_up_proj", 0, "w1"),
            ("experts.w2_weight", "experts.down_proj", 0, "w2"),
        ]
        num_experts = (
            self.config.num_experts if hasattr(self.config, "num_experts") else 0
        )
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            if name.startswith("mtp."):
                continue

            # Remapping the name of FP8 kv-scale.
            if name.endswith("scale"):
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if "experts.gate_up_proj" in name or "experts.down_proj" in name:
                    is_fused_expert = True
                    expert_params_mapping = fused_expert_params_mapping

                if weight_name not in name:
                    continue

                if "mlp.experts" in name:
                    continue

                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip layers on other devices.
                if is_pp_missing_parameter(name, self):
                    continue
                # name = apply_attn_prefix(name, params_dict)
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    is_expert_weight = True
                    name_mapped = name.replace(weight_name, param_name)
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name_mapped, self):
                        continue
                    if is_fused_expert:
                        # qwen3.5 no need to transpose
                        # loaded_weight = loaded_weight.transpose(-1, -2)
                        if "experts.gate_up_proj" in name:
                            loaded_weight = loaded_weight.chunk(2, dim=-2)
                            success_w1 = self.load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight[0],
                                "w1",
                                num_experts,
                            )
                            success_w3 = self.load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight[1],
                                "w3",
                                num_experts,
                            )
                            success = success_w1 and success_w3
                        else:
                            # down_proj
                            success = self.load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight,
                                shard_id,
                                num_experts,
                            )
                        if success:
                            name = name_mapped
                            break
                    else:
                        # Skip loading extra bias for GPTQ models.
                        if (
                            name_mapped.endswith(".bias")
                            or name_mapped.endswith("_bias")
                        ) and name_mapped not in params_dict:
                            continue
                        param = params_dict[name_mapped]
                        weight_loader = param.weight_loader
                        success = weight_loader(
                            param,
                            loaded_weight,
                            name_mapped,
                            shard_id=shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                    if success:
                        name = name_mapped
                        break
                else:
                    if is_expert_weight:
                        # We've checked that this is an expert weight
                        # However it's not mapped locally to this rank
                        # So we simply skip it
                        continue
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if is_pp_missing_parameter(name, self):
                        continue
                    if name not in params_dict:
                        logger.warning_once(
                            f"Parameter {name} not found in params_dict, skip loading"
                        )
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class Qwen3_5ForCausalLMBase(
    nn.Module,
    HasInnerState,
    SupportsLoRA,
    SupportsPP,
):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": ["gate_proj", "up_proj"],
        # GDN fused projections.
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config

        scheduler_config = vllm_config.scheduler_config
        if cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "Qwen3.5 currently does not support 'all' prefix caching, "
                "please use '--mamba-cache-mode=align' instead"
            )
        self.quant_config = vllm_config.quant_config

        super().__init__()
        self.config = config
        self.scheduler_config = scheduler_config
        self.model = Qwen3_5Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def get_top_tokens(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.logits_processor.get_top_tokens(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["mtp."],
        )
        return loader.load_weights(weights)


class Qwen3_5ForCausalLM(Qwen3_5ForCausalLMBase):
    pass


class Qwen3_5MoeForCausalLM(Qwen3_5ForCausalLMBase, QwenNextMixtureOfExperts):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        # set MoE hyperparameters
        self.set_moe_parameters()

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()


########################################################
# Qwen3_5-Dense
########################################################


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_5ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Qwen3_5ForConditionalGeneration(Qwen3VLForConditionalGeneration, IsHybrid):
    # Qwen3.5 does not support multimodal pruning (EVS).
    supports_multimodal_pruning = False

    packed_modules_mapping = Qwen3VLForConditionalGeneration.packed_modules_mapping | {
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        # protocols have not __init__ method, so we need to use nn.Module.__init__
        nn.Module.__init__(self)
        config: Qwen3_5Config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        # Qwen3.5 does not support multimodal pruning (EVS).
        self.is_multimodal_pruning_enabled = False

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Qwen3_VisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "visual"),
            )

        with self._mark_language_model(vllm_config):
            self.language_model = Qwen3_5ForCausalLM(
                vllm_config=vllm_config, prefix=maybe_prefix(prefix, "language_model")
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.language_model.embed_input_ids,
            is_multimodal=is_multimodal,
        )

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        is_multimodal = _require_is_multimodal(is_multimodal)

        inputs_embeds = _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

        return inputs_embeds

    def recompute_mrope_positions(self, *args, **kwargs):
        raise NotImplementedError(
            "Qwen3.5 does not support multimodal pruning (EVS). "
            "recompute_mrope_positions should never be called."
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        """Run forward pass for Qwen3.5.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            positions: Flattened (concatenated) position ids corresponding to a
                batch.
                **NOTE**: If mrope is enabled (default setting for Qwen3VL
                opensource models), the shape will be `(3, seq_len)`,
                otherwise it will be `(seq_len,).
            intermediate_tensors: Intermediate tensors from previous pipeline
                stages.
            inputs_embeds: Pre-computed input embeddings.
            **kwargs: Additional keyword arguments including:
                - pixel_values: Pixel values to be fed to a model.
                    `None` if no images are passed.
                - image_grid_thw: Tensor `(n_images, 3)` of image 3D grid in
                    LLM. `None` if no images are passed.
                - pixel_values_videos: Pixel values of videos to be fed to a
                    model. `None` if no videos are passed.
                - video_grid_thw: Tensor `(n_videos, 3)` of video 3D grid in
                    LLM. `None` if no videos are passed.
        """

        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        return hidden_states

    def get_top_tokens(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.language_model.get_top_tokens(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["mtp."],
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_text_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()


########################################################
# Qwen3_5-MoE
########################################################


class Qwen3_5_MoeMixtureOfExperts(MixtureOfExperts):
    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.language_model.model.layers:
            if isinstance(layer.mlp, Qwen3NextSparseMoeBlock):
                moe = layer.mlp
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

    def set_moe_parameters(self):
        self.expert_weights = []

        self.moe_layers = []
        example_moe = None
        for layer in self.language_model.model.layers:
            if isinstance(layer, Qwen3_5DecoderLayer) and isinstance(
                layer.mlp, Qwen3NextSparseMoeBlock
            ):
                example_moe = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        if example_moe is None:
            raise RuntimeError(
                "No Qwen3_5 layer found in the language_model.model.layers."
            )

        # Set MoE hyperparameters
        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.num_routed_experts = example_moe.n_routed_experts
        self.num_redundant_experts = example_moe.n_redundant_experts


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_5MoeProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Qwen3_5MoeForConditionalGeneration(
    Qwen3_5ForConditionalGeneration, Qwen3_5_MoeMixtureOfExperts
):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        # protocols have not __init__ method, so we need to use nn.Module.__init__
        nn.Module.__init__(self)
        config: Qwen3_5MoeConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        # Qwen3.5 does not support multimodal pruning (EVS).
        self.is_multimodal_pruning_enabled = False

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Qwen3_VisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "visual"),
            )

        with self._mark_language_model(vllm_config):
            self.language_model = Qwen3_5MoeForCausalLM(
                vllm_config=vllm_config, prefix=maybe_prefix(prefix, "language_model")
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

        # set MoE hyperparameters
        self.set_moe_parameters()
HOTFIX_QWEN3_5

  cat <<'HOTFIX_UTILS_RPB2' > "${bundle_dir}/utils_llmm1_rpb2.py"
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utility methods for model layers."""

from collections.abc import Callable

import torch

from vllm import _custom_ops as ops
from vllm import envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.logger import init_logger
from vllm.platforms import CpuArchEnum, current_platform
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.triton_utils import triton
from vllm.triton_utils import tl

logger = init_logger(__name__)

MOE_LAYER_ROUTER_GATE_SUFFIXES = {
    "gate",
    "router",
    "router_gate",
    "shared_expert_gate",
    "expert_gate",
}

def get_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1}, num_stages=3, num_warps=2),
    ]

def get_heuristics():
    return {
        'BLOCK_SIZE_M': lambda args: min(16, triton.next_power_of_2(args['M']))
    }

# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K']
)
@triton.heuristics(values=get_heuristics())
@triton.jit
def triton_matmul_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

def triton_matmul(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions" # NOTE(gfx906): b.shape inv
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    N, K = b.shape # NOTE(gfx906): b.shape inv
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    triton_matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(1), b.stride(0),  # NOTE(gfx906): b.stride inv
        c.stride(0), c.stride(1),  #
    )
    return c

def is_layer_moe_router_gate(prefix: str) -> bool:
    if not prefix:
        return False
    return prefix.rsplit(".", 1)[-1] in MOE_LAYER_ROUTER_GATE_SUFFIXES


def get_token_bin_counts_and_mask(
    tokens: torch.Tensor,
    vocab_size: int,
    num_seqs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Compute the bin counts for the tokens.
    # vocab_size + 1 for padding.
    bin_counts = torch.zeros(
        (num_seqs, vocab_size + 1), dtype=torch.long, device=tokens.device
    )
    bin_counts.scatter_add_(1, tokens, torch.ones_like(tokens))
    bin_counts = bin_counts[:, :vocab_size]
    mask = bin_counts > 0

    return bin_counts, mask


def apply_penalties(
    logits: torch.Tensor,
    prompt_tokens_tensor: torch.Tensor,
    output_tokens_tensor: torch.Tensor,
    presence_penalties: torch.Tensor,
    frequency_penalties: torch.Tensor,
    repetition_penalties: torch.Tensor,
) -> torch.Tensor:
    """
    Applies penalties in place to the logits tensor
    logits : The input logits tensor of shape [num_seqs, vocab_size]
    prompt_tokens_tensor: A tensor containing the prompt tokens. The prompts
        are padded to the maximum prompt length within the batch using
        `vocab_size` as the padding value. The value `vocab_size` is used
        for padding because it does not correspond to any valid token ID
        in the vocabulary.
    output_tokens_tensor: The output tokens tensor.
    presence_penalties: The presence penalties of shape (num_seqs, )
    frequency_penalties: The frequency penalties of shape (num_seqs, )
    repetition_penalties: The repetition penalties of shape (num_seqs, )
    """
    num_seqs, vocab_size = logits.shape
    _, prompt_mask = get_token_bin_counts_and_mask(
        prompt_tokens_tensor, vocab_size, num_seqs
    )
    output_bin_counts, output_mask = get_token_bin_counts_and_mask(
        output_tokens_tensor, vocab_size, num_seqs
    )

    # Apply repetition penalties as a custom op
    from vllm._custom_ops import apply_repetition_penalties

    apply_repetition_penalties(logits, prompt_mask, output_mask, repetition_penalties)

    # We follow the definition in OpenAI API.
    # Refer to https://platform.openai.com/docs/api-reference/parameter-details
    logits -= frequency_penalties.unsqueeze(dim=1) * output_bin_counts
    logits -= presence_penalties.unsqueeze(dim=1) * output_mask
    return logits


def default_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
):
    return torch.nn.functional.linear(x, weight, bias)


def use_aiter_triton_gemm(n, m, k, dtype):
    if (
        not rocm_aiter_ops.is_triton_gemm_enabled()
        # MI300's - fp8nuz=True
        or current_platform.is_fp8_fnuz()
        or dtype not in [torch.float16, torch.bfloat16]
    ):
        return False

    # use hipblaslt for the larger GEMMs
    if n > 2048 and m > 512:
        return False
    return (
        (m == 5120 and k == 2880)
        or (m == 2880 and k == 4096)
        or (m == 128 and k == 2880)
        or (m == 640 and k == 2880)
        or (m == 2880 and k == 512)
    )


def rocm_unquantized_gemm_impl(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None) -> torch.Tensor:
    use_skinny = (x.dtype in [torch.float16, torch.bfloat16] \
                  and bias is None)
    if use_skinny is not True:
        return torch.nn.functional.linear(x, weight, bias)

    x_view = x.reshape(-1, x.size(-1))
    n = x_view.shape[0]
    m = weight.shape[0]
    k = weight.shape[1]

    # prefer skinny GEMV kernel
    if m % 4 == 0 and n == 1 and k <= 8192 and k % 8 == 0:
        out = ops.LLMM1(weight, x_view, 2)
        return out.view(*x.shape[:-1], weight.shape[0])

    # low batch size, use triton matmul
    if n <= 16:
        return triton_matmul(x if x.is_contiguous() else x.contiguous(), weight)

    # otherwise, use native torch
    return torch.nn.functional.linear(x, weight, bias)


def rocm_unquantized_gemm_fake(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


def rocm_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.ops.vllm.rocm_unquantized_gemm(x, weight, bias)


direct_register_custom_op(
    op_name="rocm_unquantized_gemm",
    op_func=rocm_unquantized_gemm_impl,
    fake_impl=rocm_unquantized_gemm_fake,
)


def check_cpu_sgl_kernel(n: int, k: int, dtype: torch.dtype) -> bool:
    return (
        torch._C._cpu._is_amx_tile_supported()
        and (dtype in (torch.bfloat16, torch.int8))
        and k % 32 == 0
        and n % 16 == 0
    )


def dispatch_cpu_unquantized_gemm(
    layer: torch.nn.Module,
    remove_weight: bool,
) -> None:
    # skip for missing layers
    if layer.weight.is_meta:
        layer.cpu_linear = torch.nn.functional.linear
        return

    N, K = layer.weight.size()
    dtype = layer.weight.dtype

    if envs.VLLM_CPU_SGL_KERNEL and check_cpu_sgl_kernel(N, K, dtype):
        packed_weight = torch.ops._C.convert_weight_packed(layer.weight)
        if getattr(layer, "bias", None) is not None:
            bias_f32 = layer.bias.to(torch.float32)
        else:
            bias_f32 = None
        layer.cpu_linear = lambda x, weight, bias: torch.ops._C.weight_packed_linear(
            x, packed_weight, bias_f32 if bias is not None else None, True
        )
        if remove_weight:
            layer.weight = torch.nn.Parameter(torch.empty(0), requires_grad=False)
        return
    elif (
        ops._supports_onednn
        and current_platform.get_cpu_architecture() != CpuArchEnum.POWERPC
    ):
        try:
            origin_weight = layer.weight
            handler = ops.create_onednn_mm(origin_weight.t(), 32)
            layer.cpu_linear = lambda x, weight, bias: ops.onednn_mm(handler, x, bias)
            if remove_weight:
                layer.weight = torch.nn.Parameter(torch.empty(0), requires_grad=False)
            return
        except RuntimeError as e:
            logger.warning_once(
                "Failed to create oneDNN linear, fallback to torch linear."
                f" Exception: {e}"
            )

    # fallback case
    layer.cpu_linear = lambda x, weight, bias: torch.nn.functional.linear(
        x, weight, bias
    )


def cpu_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
):
    return layer.cpu_linear(x, weight, bias)


def dispatch_unquantized_gemm() -> Callable[..., torch.Tensor]:
    if current_platform.is_rocm():
        return rocm_unquantized_gemm
    elif current_platform.is_cpu():
        return cpu_unquantized_gemm
    else:
        return default_unquantized_gemm
HOTFIX_UTILS_RPB2
  truncate -s -1 "${bundle_dir}/utils_llmm1_rpb2.py"

  cat <<'HOTFIX_FUSED_MOE_FASTPATH' > "${bundle_dir}/fused_moe_fastpath.py"
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused MoE Triton kernels."""

import functools
import json
import os
from collections.abc import Callable
from typing import Any

import torch

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.batch_invariant import (
    vllm_is_batch_invariant,
)
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    _get_config_dtype_str,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import (
    _resize_cache,
    disable_inplace,
    moe_kernel_quantize_input,
)
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import dequant_mxfp4
from vllm.model_executor.layers.quantization.utils.mxfp6_utils import dequant_mxfp6
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8DynamicTensorSym,
    kFp8DynamicTokenSym,
    kFp8Static128BlockSym,
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx906
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)
logger.info_once("qwen c1 topk8 MoE fastpath overlay loaded")


@triton.jit
def _qwen_c1_topk8_w1_act_kernel(
    x,
    w1,
    topk_ids,
    act,
    W1_STRIDE_E: tl.constexpr,
    W1_STRIDE_I: tl.constexpr,
    W1_STRIDE_H: tl.constexpr,
    H: tl.constexpr,
    I: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_k = tl.program_id(0)
    pid_i = tl.program_id(1)
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    offs_h = tl.arange(0, BLOCK_H)
    expert = tl.load(topk_ids + pid_k).to(tl.int64)
    acc_gate = tl.zeros((BLOCK_I,), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_I,), dtype=tl.float32)
    for h0 in range(0, H, BLOCK_H):
        h = h0 + offs_h
        xv = tl.load(x + h, mask=h < H, other=0.0)
        wg = tl.load(
            w1
            + expert * W1_STRIDE_E
            + offs_i[:, None] * W1_STRIDE_I
            + h[None, :] * W1_STRIDE_H,
            mask=(offs_i[:, None] < I) & (h[None, :] < H),
            other=0.0,
        )
        wu = tl.load(
            w1
            + expert * W1_STRIDE_E
            + (offs_i[:, None] + I) * W1_STRIDE_I
            + h[None, :] * W1_STRIDE_H,
            mask=(offs_i[:, None] < I) & (h[None, :] < H),
            other=0.0,
        )
        acc_gate += tl.sum(wg * xv[None, :], axis=1)
        acc_up += tl.sum(wu * xv[None, :], axis=1)
    silu_gate = acc_gate / (1.0 + tl.exp(-acc_gate))
    out = silu_gate * acc_up
    tl.store(act + pid_k * I + offs_i, out.to(tl.float16), mask=offs_i < I)


@triton.jit
def _qwen_c1_topk8_w2_reduce_kernel(
    w2,
    topk_ids,
    topk_weights,
    act,
    y,
    W2_STRIDE_E: tl.constexpr,
    W2_STRIDE_H: tl.constexpr,
    W2_STRIDE_I: tl.constexpr,
    H: tl.constexpr,
    I: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    pid_h = tl.program_id(0)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_i = tl.arange(0, BLOCK_I)
    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for kidx in range(0, TOP_K):
        expert = tl.load(topk_ids + kidx).to(tl.int64)
        route_w = tl.load(topk_weights + kidx).to(tl.float32)
        expert_acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
        for i0 in range(0, I, BLOCK_I):
            i = i0 + offs_i
            av = tl.load(act + kidx * I + i, mask=i < I, other=0.0)
            ww = tl.load(
                w2
                + expert * W2_STRIDE_E
                + offs_h[:, None] * W2_STRIDE_H
                + i[None, :] * W2_STRIDE_I,
                mask=(offs_h[:, None] < H) & (i[None, :] < I),
                other=0.0,
            )
            expert_acc += tl.sum(ww * av[None, :], axis=1)
        acc += (expert_acc * route_w).to(tl.float16).to(tl.float32)
    tl.store(y + offs_h, acc.to(tl.float16), mask=offs_h < H)


def _qwen_c1_topk8_fastpath_enabled() -> bool:
    return os.environ.get("VLLM_ENABLE_C1_TOPK8_MOE_FASTPATH", "0").lower() in {
        "1",
        "true",
        "yes",
        "force",
    }


def _qwen_c1_topk8_fastpath_debug() -> bool:
    return os.environ.get("VLLM_ENABLE_C1_TOPK8_MOE_FASTPATH", "0").lower() in {
        "force",
        "debug",
    }


def _reject_qwen_c1_topk8_fastpath(reason: str) -> None:
    if _qwen_c1_topk8_fastpath_debug():
        logger.info_once(f"qwen c1 topk8 MoE fastpath rejected: {reason}")


def _try_qwen_c1_topk8_fastpath(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    apply_router_weight_on_input: bool,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    ocp_mx_scheme: str | None,
    per_channel_quant: bool,
    global_num_experts: int,
    expert_map: torch.Tensor | None,
    w1_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
    w1_zp: torch.Tensor | None,
    w2_zp: torch.Tensor | None,
    a1_scale: torch.Tensor | None,
    a2_scale: torch.Tensor | None,
    block_shape: list[int] | None,
    w1_bias: torch.Tensor | None,
    w2_bias: torch.Tensor | None,
) -> torch.Tensor | None:
    if not _qwen_c1_topk8_fastpath_enabled():
        return None
    if activation != "silu" or apply_router_weight_on_input:
        _reject_qwen_c1_topk8_fastpath(
            f"activation={activation} apply_router_weight_on_input={apply_router_weight_on_input}"
        )
        return None
    if (
        use_fp8_w8a8
        or use_int8_w8a8
        or use_int8_w8a16
        or use_int4_w4a16
        or ocp_mx_scheme is not None
        or per_channel_quant
        or expert_map is not None
        or block_shape is not None
    ):
        _reject_qwen_c1_topk8_fastpath(
            "unsupported_quant_or_mapping "
            f"fp8={use_fp8_w8a8} int8={use_int8_w8a8}/{use_int8_w8a16} "
            f"int4={use_int4_w4a16} ocp={ocp_mx_scheme} "
            f"per_channel={per_channel_quant} expert_map={expert_map is not None} "
            f"block_shape={block_shape}"
        )
        return None
    if any(
        x is not None
        for x in (
            w1_scale,
            w2_scale,
            w1_zp,
            w2_zp,
            a1_scale,
            a2_scale,
            w1_bias,
            w2_bias,
        )
    ):
        _reject_qwen_c1_topk8_fastpath("scale_zero_point_or_bias_present")
        return None
    if not (
        hidden_states.is_cuda
        and hidden_states.dtype == torch.float16
        and w1.dtype == torch.float16
        and w2.dtype == torch.float16
        and hidden_states.dim() == 2
        and hidden_states.size(0) == 1
        and hidden_states.is_contiguous()
        and w1.stride(-1) == 1
        and w2.stride(-1) == 1
        and topk_ids.shape == topk_weights.shape
        and topk_ids.dim() == 2
        and topk_ids.size(0) == 1
        and topk_ids.size(1) == 8
        and w1.dim() == 3
        and w2.dim() == 3
        and w1.size(0) == w2.size(0)
        and w2.size(1) == hidden_states.size(1)
        and w2.size(2) > 0
        and w1.size(1) == 2 * w2.size(2)
        and w1.size(2) == hidden_states.size(1)
    ):
        _reject_qwen_c1_topk8_fastpath(
            "shape_or_layout "
            f"hidden_shape={tuple(hidden_states.shape)} hidden_dtype={hidden_states.dtype} "
            f"hidden_cuda={hidden_states.is_cuda} hidden_contig={hidden_states.is_contiguous()} "
            f"w1_shape={tuple(w1.shape)} w1_dtype={w1.dtype} w1_stride={tuple(w1.stride())} "
            f"w2_shape={tuple(w2.shape)} w2_dtype={w2.dtype} w2_stride={tuple(w2.stride())} "
            f"topk_ids_shape={tuple(topk_ids.shape)} topk_weights_shape={tuple(topk_weights.shape)}"
        )
        return None
    if global_num_experts not in (-1, w1.size(0)):
        _reject_qwen_c1_topk8_fastpath(
            f"global_num_experts={global_num_experts} local_experts={w1.size(0)}"
        )
        return None

    logger.info_once("qwen c1 topk8 MoE fastpath active")
    hidden = hidden_states.size(1)
    inter = w2.size(2)
    act = torch.empty((8, inter), device=hidden_states.device, dtype=torch.float16)
    out = torch.empty_like(hidden_states)
    _qwen_c1_topk8_w1_act_kernel[(8, triton.cdiv(inter, 8))](
        hidden_states,
        w1,
        topk_ids.view(-1),
        act,
        W1_STRIDE_E=w1.stride(0),
        W1_STRIDE_I=w1.stride(1),
        W1_STRIDE_H=w1.stride(2),
        H=hidden,
        I=inter,
        BLOCK_I=8,
        BLOCK_H=128,
        num_warps=4,
    )
    _qwen_c1_topk8_w2_reduce_kernel[(triton.cdiv(hidden, 64),)](
        w2,
        topk_ids.view(-1),
        topk_weights.view(-1),
        act,
        out,
        W2_STRIDE_E=w2.stride(0),
        W2_STRIDE_H=w2.stride(1),
        W2_STRIDE_I=w2.stride(2),
        H=hidden,
        I=inter,
        TOP_K=8,
        BLOCK_H=64,
        BLOCK_I=8,
        num_warps=1,
    )
    return out


@triton.jit
def write_zeros_to_output(
    c_ptr,
    stride_cm,
    stride_cn,
    pid_n,
    N,
    offs_token,
    token_mask,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    compute_type,
):
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def fused_moe_kernel_gptq_awq(
    on_gfx906: tl.constexpr,
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    b_scale_ptr,
    b_zp_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N: tl.constexpr,
    K: tl.constexpr,
    EM,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bse,
    stride_bsk,
    stride_bsn,
    stride_bze,
    stride_bzk,
    stride_bzn,
    block_k_diviable: tl.constexpr,
    group_size: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    has_zp: tl.constexpr,
    use_int4_w4a16: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.
    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    # Cast to int64 to prevent overflow in stride*offset products
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        # -----------------------------------------------------------
        # Write back zeros to the output when the expert is not
        # in the current expert parallel rank.
        write_zeros_to_output(
            c_ptr,
            stride_cm,
            stride_cn,
            pid_n,
            N,
            offs_token,
            token_mask,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            compute_type,
        )
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )

    if use_int4_w4a16:
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + (offs_k[:, None] // 2) * stride_bk
            + offs_bn[None, :] * stride_bn
        )
        b_shifter = (offs_k[:, None] % 2) * 4
    elif use_int8_w8a16:
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + offs_k[:, None] * stride_bk
            + offs_bn[None, :] * stride_bn
        )

    if not has_zp and use_int4_w4a16:
        b_zp_num = 8
    if not has_zp and use_int8_w8a16:
        b_zp_num = 128
    elif has_zp and use_int4_w4a16:
        b_zp_shifter = (offs_bn[None, :] % 2) * 4

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to compute_type (--dtype) after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.

        if not block_k_diviable:
            k_mask = offs_k[:, None] < K - k * BLOCK_SIZE_K
            k_other = 0.0
        else:
            k_mask = None
            k_other = None

        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(b_ptrs)
        if use_int4_w4a16:
            b = (b >> b_shifter) & 0xF

        b_scale_ptrs = (
            b_scale_ptr
            + off_experts * stride_bse
            + offs_bn[None, :] * stride_bsn
            + ((offs_k[:, None] + BLOCK_SIZE_K * k) // group_size) * stride_bsk
        )
        b_scale = tl.load(b_scale_ptrs, mask=k_mask, other=k_other)
        b_scale = b_scale.to(tl.float32)

        if has_zp and use_int4_w4a16:
            offs_k_true = (offs_k[:, None] + BLOCK_SIZE_K * k) // group_size
            b_zp_ptrs = (
                b_zp_ptr
                + off_experts * stride_bze
                + (offs_bn[None, :] // 2) * stride_bzn
                + offs_k_true * stride_bzk
            )
            b_zp = tl.load(b_zp_ptrs, mask=k_mask, other=k_other)
            b_zp = (b_zp >> b_zp_shifter) & 0xF
            b_zp = b_zp.to(tl.float32)
        elif has_zp and use_int8_w8a16:
            offs_k_true = (offs_k[:, None] + BLOCK_SIZE_K * k) // group_size
            b_zp_ptrs = (
                b_zp_ptr
                + off_experts * stride_bze
                + offs_bn[None, :] * stride_bzn
                + offs_k_true * stride_bzk
            )
            b_zp = tl.load(b_zp_ptrs, mask=k_mask, other=k_other)
            b_zp = b_zp.to(tl.float32)

        # We accumulate along the K dimension.
        if has_zp:
            if compute_type == tl.float32 and on_gfx906:
                b = ((b.to(tl.float32) - b_zp) * b_scale).to(tl.float16)
            else:
                b = ((b.to(tl.float32) - b_zp) * b_scale).to(compute_type)
        else:
            if compute_type == tl.float32 and on_gfx906:
                b = ((b.to(tl.float32) - b_zp_num) * b_scale).to(tl.float16)
            else:
                b = ((b.to(tl.float32) - b_zp_num) * b_scale).to(compute_type)
        accumulator = tl.dot(a.to(tl.float16) if compute_type == tl.float32 and on_gfx906 else a, b, acc=accumulator)

        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        if use_int4_w4a16:
            b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk
        else:
            b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]

    if not compute_type == tl.float32:
        accumulator = accumulator.to(compute_type)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def fused_moe_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    b_bias_ptr,
    a_scale_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bse,
    stride_bsk,
    stride_bsn,
    stride_bbe,  # bias expert stride
    stride_bbn,  # bias N stride
    # Block size for block-wise quantization
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    naive_block_assignment: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    per_channel_quant: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    on_gfx906: tl.constexpr,
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.
    - naive_block_assignment: A boolean flag indicating whether to use naive
        token wise block assignment. If True, each block corresponds to a
        single token.
    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    offs = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    if not naive_block_assignment:
        offs_token_id = pid_m * BLOCK_SIZE_M + offs
        offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    else:
        offs_token = tl.where(
            offs == 0,
            pid_m,  # first element = pid_m
            num_valid_tokens,  # remaining elements = constant
        )
    # Cast to int64 to prevent overflow in stride*offset products
    # (e.g. stride_cm * offs_token can exceed int32 for large token counts)
    offs_token = offs_token.to(tl.int64)

    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        # -----------------------------------------------------------
        # Write back zeros to the output when the expert is not
        # in the current expert parallel rank.
        write_zeros_to_output(
            c_ptr,
            stride_cm,
            stride_cn,
            pid_n,
            N,
            offs_token,
            token_mask,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            compute_type,
        )
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )

    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )
    if use_int8_w8a16:
        b_scale_ptrs = (
            b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
        )
        b_scale = tl.load(b_scale_ptrs)

    if use_fp8_w8a8 or use_int8_w8a8:
        # block-wise
        if group_k > 0 and group_n > 0:
            if not on_gfx906:
                a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            offs_bsn = offs_bn // group_n
            b_scale_ptrs = (
                b_scale_ptr + off_experts * stride_bse + offs_bsn * stride_bsn
            )
        # channel-wise
        elif per_channel_quant:
            b_scale_ptrs = (
                b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
            )
            b_scale = tl.load(b_scale_ptrs)
            # Load per-token scale for activations
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            a_scale = tl.load(a_scale_ptrs, mask=token_mask, other=0.0)[:, None]
        # tensor-wise
        else:
            a_scale = tl.load(a_scale_ptr)
            b_scale = tl.load(b_scale_ptr + off_experts)
    if HAS_BIAS:
        # bias shape: [num_experts, N]
        bias_ptrs = b_bias_ptr + off_experts * stride_bbe + offs_bn * stride_bbn
        bias = tl.load(bias_ptrs, mask=(offs_bn < N), other=0.0)
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        if on_gfx906:
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0)
        else:
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        if use_int8_w8a16:
            accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
        elif use_fp8_w8a8 or use_int8_w8a8:
            if group_k > 0 and group_n > 0:
                k_start = k * BLOCK_SIZE_K
                offs_ks = k_start // group_k
                if not on_gfx906:
                    a_scale = tl.load(
                        a_scale_ptrs + offs_ks * stride_ask, mask=token_mask, other=0.0
                    )
                else:
                    # Bitwise E4M3 -> FP16 dequant for B (b is already uint8)
                    b_sign = (b & 0x80).to(tl.uint16) << 8
                    b_exp = ((b & 0x78) >> 3).to(tl.uint16)
                    b_exp = tl.where(b_exp == 0, tl.zeros_like(b_exp), b_exp + 8)
                    b_mant = (b & 0x07).to(tl.uint16) << 7
                    b_bits = b_sign | (b_exp << 10) | b_mant
                    b = b_bits.to(tl.float16, bitcast=True)
                
                b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)

                accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :] if not on_gfx906 else tl.dot(a, b) * b_scale[None, :]
            else:
                if use_fp8_w8a8:
                    # acc used to enable fp8_fast_accum
                    accumulator = tl.dot(a, b, acc=accumulator)
                else:
                    accumulator += tl.dot(a, b)
        else:
            accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Dequantization for supported quantization schemes:
    #   - int8_w8a16
    #   - fp8_w8a8
    #   - int8_w8a8
    # Accumulator and scalings are in float32 to preserve numerical accuracy.
    if use_int8_w8a16:
        accumulator = accumulator * b_scale
    elif (use_fp8_w8a8 or use_int8_w8a8) and not (group_k > 0 and group_n > 0):
        accumulator = accumulator * a_scale * b_scale if not on_gfx906 else accumulator * b_scale

    # Bias addition:
    # Bias must be applied after dequantization:
    #   - Since bias is typically not quantized
    #   - Bias should not be scaled by quantization factors
    if HAS_BIAS:
        accumulator += bias[None, :]

    # Router (MoE) weight multiplication:
    # This multiplication MUST be performed in float32 before any precision
    # conversion to ensure numerical stability, which is especially critical
    # on ROCm platforms.
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(
            topk_weights_ptr + offs_token,
            mask=token_mask,
            other=0,
        )
        accumulator *= moe_weight[:, None]

    # Final precision conversion:
    # Cast once at the end to the desired compute/output dtype.
    accumulator = accumulator.to(compute_type)

    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


# NOTE(zyongye): we can remove all the wna16 kernel
# once we drop off sm75 support
def invoke_fused_moe_wna16_cuda_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    B_scale: torch.Tensor | None,
    B_zp: torch.Tensor | None,
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor | None,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict[str, Any],
    block_shape: list[int],
):
    assert B_scale is not None and B_scale.ndim == 3
    assert B_zp is None or B_zp.ndim == 3
    assert block_shape is None or block_shape[0] == 0

    M = A.size(0)
    num_tokens = M * top_k
    bit = 4

    config = config.copy()
    config.update(
        get_moe_wna16_block_config(
            config=config,
            use_moe_wna16_cuda=True,
            num_valid_tokens=num_tokens,
            size_k=A.size(1),
            size_n=B.size(1),
            num_experts=B.size(1),
            group_size=block_shape[1],
            real_top_k=top_k,
            block_size_m=config["BLOCK_SIZE_M"],
        )
    )

    ops.moe_wna16_gemm(
        A,
        C,
        B,
        B_scale,
        B_zp,
        topk_weights if mul_routed_weight else None,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        top_k,
        config["BLOCK_SIZE_M"],
        config["BLOCK_SIZE_N"],
        config["BLOCK_SIZE_K"],
        bit,
    )


# NOTE(zyongye): we can remove all the wna16 kernel
# once we drop off sm75 support
def invoke_fused_moe_wna16_triton_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    B_scale: torch.Tensor | None,
    B_zp: torch.Tensor | None,
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict[str, Any],
    compute_type: tl.dtype,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    block_shape: list[int] | None,
):
    assert B_scale is not None and B_scale.ndim == 3
    assert B_zp is None or B_zp.ndim == 3
    assert block_shape is not None and block_shape[0] == 0

    M = A.size(0)
    num_tokens = M * top_k

    EM = sorted_token_ids.size(0)
    if A.size(0) < config["BLOCK_SIZE_M"]:
        # optimize for small batch_size.
        # We assume that top_ids of each token is unique,
        # so num_valid_experts <= batch_size <= BLOCK_SIZE_M,
        # and we can skip some invalid blocks.
        EM = min(sorted_token_ids.size(0), A.size(0) * top_k * config["BLOCK_SIZE_M"])
    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"])
        * triton.cdiv(B.size(1), META["BLOCK_SIZE_N"]),
    )
    config = config.copy()
    config.update(
        get_moe_wna16_block_config(
            config=config,
            use_moe_wna16_cuda=False,
            num_valid_tokens=num_tokens,
            size_k=A.size(1),
            size_n=B.size(1),
            num_experts=B.size(1),
            group_size=block_shape[1],
            real_top_k=top_k,
            block_size_m=config["BLOCK_SIZE_M"],
        )
    )

    # For gfx906 and compute_type == tl.float32: the kernel internally casts to fp16 before tl.dot,
    # but we still pass compute_type so the OUTPUT is stored correctly.
    # The accumulator is always fp32 inside the kernel regardless.

    fused_moe_kernel_gptq_awq[grid](
        on_gfx906(),
        A,
        B,
        C,
        B_scale,
        B_zp,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.size(1),
        A.size(1),
        EM,
        num_tokens,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
        B_scale.stride(0),
        B_scale.stride(2),
        B_scale.stride(1),
        B_zp.stride(0) if B_zp is not None else 0,
        B_zp.stride(2) if B_zp is not None else 0,
        B_zp.stride(1) if B_zp is not None else 0,
        block_k_diviable=A.size(1) % config["BLOCK_SIZE_K"] == 0,
        group_size=block_shape[1],
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        has_zp=B_zp is not None,
        use_int4_w4a16=use_int4_w4a16,
        use_int8_w8a16=use_int8_w8a16,
        **config,
    )


def invoke_fused_moe_triton_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: torch.Tensor | None,
    B_scale: torch.Tensor | None,
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor | None,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict[str, Any],
    compute_type: tl.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    per_channel_quant: bool,
    block_shape: list[int] | None = None,
    B_bias: torch.Tensor | None = None,
):
    assert topk_weights is not None or not mul_routed_weight
    assert topk_weights is None or topk_weights.stride(1) == 1
    assert sorted_token_ids is None or sorted_token_ids.stride(0) == 1

    if use_fp8_w8a8 or use_int8_w8a8:
        assert B_scale is not None
        assert block_shape is None or triton.cdiv(
            B.size(-2), block_shape[0]
        ) == B_scale.size(-2)
        assert block_shape is None or triton.cdiv(
            B.size(-1), block_shape[1]
        ) == B_scale.size(-1)
    elif use_int8_w8a16 or use_int4_w4a16:
        assert B_scale is not None
        assert block_shape is None or block_shape[0] == 0
    else:
        assert A_scale is None
        assert B_scale is None

    M = A.size(0)
    num_tokens = M * top_k
    if sorted_token_ids is not None:
        EM = sorted_token_ids.size(0)
        if A.size(0) < config["BLOCK_SIZE_M"]:
            # optimize for small batch_size.
            # We assume that top_ids of each token is unique,
            # so num_valid_experts <= batch_size <= BLOCK_SIZE_M,
            # and we can skip some invalid blocks.
            EM = min(
                sorted_token_ids.size(0), A.size(0) * top_k * config["BLOCK_SIZE_M"]
            )
    else:
        EM = num_tokens * config["BLOCK_SIZE_M"]
    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"])
        * triton.cdiv(B.size(1), META["BLOCK_SIZE_N"]),
    )
    HAS_BIAS = B_bias is not None

    config = config.copy()
    config["SPLIT_K"] = 1
    BLOCK_SIZE_K = config.pop("BLOCK_SIZE_K")
    if block_shape is not None:
        BLOCK_SIZE_K = min(BLOCK_SIZE_K, min(block_shape[0], block_shape[1]))
    fused_moe_kernel[grid](
        A,
        B,
        C,
        B_bias,
        A_scale,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.size(1),
        B.size(2),
        EM,
        num_tokens,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
        A_scale.stride(0) if A_scale is not None and A_scale.ndim == 2 else 0,
        A_scale.stride(1) if A_scale is not None and A_scale.ndim == 2 else 0,
        B_scale.stride(0) if B_scale is not None and B_scale.ndim >= 2 else 0,
        B_scale.stride(2) if B_scale is not None and B_scale.ndim == 3 else 0,
        B_scale.stride(1) if B_scale is not None and B_scale.ndim >= 2 else 0,
        B_bias.stride(0) if B_bias is not None else 0,
        B_bias.stride(1) if B_bias is not None else 0,
        0 if block_shape is None else block_shape[0],
        0 if block_shape is None else block_shape[1],
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        per_channel_quant=per_channel_quant,
        naive_block_assignment=(sorted_token_ids is None),
        HAS_BIAS=HAS_BIAS,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        **config,
        on_gfx906=on_gfx906(),
    )


def dispatch_fused_moe_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: torch.Tensor | None,
    B_scale: torch.Tensor | None,
    B_zp: torch.Tensor | None,
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor | None,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict[str, Any],
    compute_type: tl.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    per_channel_quant: bool,
    block_shape: list[int] | None = None,
    B_bias: torch.Tensor | None = None,
) -> None:
    assert topk_weights is not None or not mul_routed_weight
    assert topk_weights is None or topk_weights.stride(1) == 1
    assert sorted_token_ids is None or sorted_token_ids.stride(0) == 1

    M = A.size(0)
    num_tokens = M * top_k

    if (use_int8_w8a16 or use_int4_w4a16) and (
        block_shape is not None and block_shape[1] > 0
    ):
        assert B_bias is None

        use_moe_wna16_cuda = should_moe_wna16_use_cuda(
            num_valid_tokens=num_tokens,
            group_size=block_shape[1],
            num_experts=B.size(0),
            bit=4 if use_int4_w4a16 else 8,
        )

        if use_moe_wna16_cuda:
            invoke_fused_moe_wna16_cuda_kernel(
                A,
                B,
                C,
                B_scale,
                B_zp,
                topk_weights,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                mul_routed_weight,
                top_k,
                config,
                block_shape,
            )
            return
        invoke_fused_moe_wna16_triton_kernel(
            A,
            B,
            C,
            B_scale,
            B_zp,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            mul_routed_weight,
            top_k,
            config,
            compute_type,
            use_int8_w8a16,
            use_int4_w4a16,
            block_shape,
        )

    else:
        invoke_fused_moe_triton_kernel(
            A,
            B,
            C,
            A_scale,
            B_scale,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            mul_routed_weight,
            top_k,
            config,
            compute_type,
            use_fp8_w8a8,
            use_int8_w8a8,
            use_int8_w8a16,
            use_int4_w4a16,
            per_channel_quant,
            block_shape,
            B_bias,
        )


@triton.jit
def compute_identity_kernel(
    top_k: int,
    hidden_states_ptr: tl.tensor,
    expert_scales_ptr: tl.tensor,
    num_tokens: int,
    output_ptr: tl.tensor,
    hidden_dim: int,
    scales_stride: int,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    pid = tl.program_id(0)

    batch_id = pid // (hidden_dim // BLOCK_SIZE)
    dim_offset = pid % (hidden_dim // BLOCK_SIZE) * BLOCK_SIZE

    if batch_id >= num_tokens or dim_offset >= hidden_dim:
        return

    h = tl.load(
        hidden_states_ptr
        + batch_id * hidden_dim
        + dim_offset
        + tl.arange(0, BLOCK_SIZE),
        mask=(dim_offset + tl.arange(0, BLOCK_SIZE)) < hidden_dim,
    )

    result = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(top_k):
        scale = tl.load(expert_scales_ptr + batch_id * scales_stride + i)
        result += h * scale

    tl.store(
        output_ptr + batch_id * hidden_dim + dim_offset + tl.arange(0, BLOCK_SIZE),
        result,
        mask=(dim_offset + tl.arange(0, BLOCK_SIZE)) < hidden_dim,
    )


def zero_experts_compute_triton(
    expert_indices: torch.Tensor,
    expert_scales: torch.Tensor,
    num_experts: int,
    zero_expert_type: str,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    N = expert_indices.numel()
    top_k = expert_indices.size(-1)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)

    if zero_expert_type == "identity":
        zero_expert_mask = expert_indices < num_experts
        zero_expert_scales = expert_scales.clone()
        zero_expert_scales[zero_expert_mask] = 0.0

    normal_expert_mask = expert_indices >= num_experts
    expert_indices[normal_expert_mask] = 0
    expert_scales[normal_expert_mask] = 0.0

    output = torch.zeros_like(hidden_states).to(hidden_states.device)
    hidden_dim = hidden_states.size(-1)
    num_tokens = hidden_states.size(0)

    grid = lambda meta: (num_tokens * (hidden_dim // meta["BLOCK_SIZE"]),)
    compute_identity_kernel[grid](
        top_k,
        hidden_states,
        zero_expert_scales,
        num_tokens,
        output,
        hidden_dim,
        zero_expert_scales.stride(0),
        BLOCK_SIZE=256,
    )

    return output


# Adapted from: https://github.com/sgl-project/sglang/pull/2628
def get_config_file_name(
    E: int, N: int, dtype: str | None, block_shape: list[int] | None = None
) -> str:
    device_name = current_platform.get_device_name().replace(" ", "_")
    # Set device_name to H200 if a device from the H200 family is detected
    if "H200" in device_name.split("_"):
        device_name = "NVIDIA_H200"
    dtype_selector = "" if not dtype else f",dtype={dtype}"
    block_shape_selector = (
        "" if not block_shape or not all(block_shape) else
        f",block_shape={block_shape}"
    ).replace(" ", "")
    gfx906_names = [ "Instinct_MI50", "Instinct_MI60", "AMD_Radeon_Graphics", "Radeon_Pro_VII", "Radeon_VII", "Vega_20" ]
    if any(s in device_name for s in gfx906_names):
        device_name = "AMD_GFX906"
    return f"E={E},N={N},device_name={device_name}{dtype_selector}{block_shape_selector}.json"  # noqa: E501


# Adapted from: https://github.com/sgl-project/sglang/pull/2628
@functools.lru_cache
def get_moe_configs(
    E: int,
    N: int,
    dtype: str | None,
    block_n: int | None = None,
    block_k: int | None = None,
) -> dict[int, Any] | None:
    """
    Return optimized configurations for the fused MoE kernel.

    The return value will be a dictionary that maps an irregular grid of
    batch sizes to configurations of the fused_moe kernel. To evaluate the
    kernel on a given batch size bs, the closest batch size in the grid should
    be picked and the associated configuration chosen to invoke the kernel.
    """

    # Avoid optimizing for the batch invariant case. Use default config
    if vllm_is_batch_invariant():
        return None

    # First look up if an optimized configuration is available in the configs
    # directory
    block_shape = [block_n, block_k] if block_n and block_k else None
    json_file_name = get_config_file_name(E, N, dtype, block_shape)

    config_file_paths = []

    # note that we prioritize user defined config
    user_defined_config_folder = envs.VLLM_TUNED_CONFIG_FOLDER
    if user_defined_config_folder is not None:
        user_defined_config_file_path = os.path.join(
            user_defined_config_folder, json_file_name
        )
        config_file_paths.append(user_defined_config_file_path)

    default_config_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "configs", json_file_name
    )
    config_file_paths.append(default_config_file_path)

    for config_file_path in config_file_paths:
        if os.path.exists(config_file_path):
            with open(config_file_path) as f:
                logger.info_once(
                    "Using configuration from %s for MoE layer.",
                    config_file_path,
                    scope="global",
                )
                # If a configuration has been found, return it
                tuned_config = json.load(f)
                # Delete triton_version from tuned_config
                tuned_config.pop("triton_version", None)
                return {int(key): val for key, val in tuned_config.items()}

    # If no optimized configuration is available, we will use the default
    # configuration
    logger.warning_once(
        "Using default MoE config. Performance might be sub-optimal! "
        "Config file not found at %s",
        ", ".join(config_file_paths),
        scope="local",
    )
    return None


def get_moe_wna16_block_config(
    config: dict[str, int],
    use_moe_wna16_cuda: bool,
    num_valid_tokens: int,
    size_k: int,
    size_n: int,
    num_experts: int,
    group_size: int,
    real_top_k: int,
    block_size_m: int,
):
    if "BLOCK_SIZE_N" in config and "BLOCK_SIZE_K" in config:
        # optimal block config is set
        return {}
    if not use_moe_wna16_cuda:
        # triton moe wna16 kernel
        if num_valid_tokens // real_top_k == 1:
            # if bs=1, use a smaller BLOCK_SIZE_N
            block_size_n = 32
            block_size_k = 64
        else:
            block_size_n = 64
            block_size_k = 32
        while block_size_k > 16 and size_k % block_size_k != 0:
            block_size_k //= 2
        if size_k % block_size_k != 0:
            block_size_k = 1 << (size_k.bit_length() - 1)
            while block_size_k > size_k or size_k % block_size_k != 0:
                block_size_k //= 2
        return {"BLOCK_SIZE_N": block_size_n, "BLOCK_SIZE_K": block_size_k}
            
    else:
        # cuda moe wna16 kernel
        # set default block_size 128, and increase them when num_blocks
        # is too large.
        block_size_n = 128
        block_size_k = 128
        if block_size_k <= group_size:
            block_size_k = group_size

        num_n_blocks = size_k // block_size_k
        num_k_blocks = size_n // block_size_k
        num_m_blocks = (
            num_valid_tokens + block_size_m - 1
        ) / block_size_m + num_experts
        if num_valid_tokens // real_top_k <= block_size_m:
            num_m_blocks = min(num_m_blocks, num_valid_tokens)
        num_blocks = num_m_blocks * num_n_blocks * num_k_blocks

        if size_k % 256 == 0 and num_blocks >= 256 and block_size_k < 256:
            block_size_k = 256
            num_blocks = num_blocks // (256 // block_size_k)

        if (
            num_m_blocks <= 16
            and size_k % (block_size_k * 2) == 0
            and size_k % (block_size_k * 2) == 0
            and block_size_k <= 512
            and num_blocks >= 512
        ):
            block_size_k = block_size_k * 2
            num_blocks = num_blocks // 2

        if num_blocks > 1024:
            block_size_n = 256
            num_n_blocks = num_n_blocks // 2
            num_blocks = num_blocks // 2

        if size_n <= 1024 and num_blocks >= 1024:
            # The kernel performance got much better with BLOCK_SIZE_N=1024
            # when num_blocks is large, event when N is small.
            # Not sure why, maybe it force the CUDA SM process only one block
            # at the same time.
            block_size_n = 1024

        while block_size_k > group_size and size_k % block_size_k != 0:
            block_size_k //= 2
        if size_k % block_size_k != 0:
            block_size_k = 1 << (size_k.bit_length() - 1)
            while block_size_k > group_size and size_k % block_size_k != 0:
                block_size_k //= 2

        return {"BLOCK_SIZE_N": block_size_n, "BLOCK_SIZE_K": block_size_k}


def should_moe_wna16_use_cuda(
    num_valid_tokens: int, group_size: int, num_experts: int, bit: int
):
    return (
        current_platform.is_cuda_alike() 
        and bit == 4
        and group_size in [32, 64, 128]
        and num_valid_tokens / num_experts <= 6
    )


def get_default_config(
    M: int,
    E: int,
    N: int,
    K: int,
    topk: int,
    dtype: str | None,
    block_shape: list[int] | None = None,
) -> dict[str, int]:
    if vllm_is_batch_invariant():
        return {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
            "SPLIT_K": 1,
        }

    # num_stages can cause triton.runtime.errors.OutOfResources on ROCm.
    num_stages_rocm = 2

    if dtype == "fp8_w8a8" and block_shape is not None:
        # Block-wise quant: BLOCK_SIZE_N must be divisible by block_shape[0]
        # BLOCK_SIZE_K must be divisible by block_shape[1]
        # num_stages=3 can cause triton.runtime.errors.OutOfResources
        # on ROCm, set it to 2 instead.
        # gfx906: Optimize for MI50 based on benchmark results.
        if on_gfx906():
            if M <= 1:
                bm, gm, nw, ns = 16, 16, 4, 1
            elif M <= 8:
                bm, gm, nw, ns = 16, 1, 4, 1
            elif M <= 16:
                bm, gm, nw, ns = 16, 8, 4, 1
            elif M <= 64:
                bm, gm, nw, ns = 32, 4, 4, 1
            else:
                bm, gm, nw, ns = 32, 8, 4, 1
            config = {
                "BLOCK_SIZE_M": bm,
                "BLOCK_SIZE_N": block_shape[0],
                "BLOCK_SIZE_K": block_shape[1],
                "GROUP_SIZE_M": gm,
                "SPLIT_K": 1,
                "num_warps": nw,
                "num_stages": ns,
                }
        else:
            # Block-wise quant: tile sizes are constrained by block_shape.
            # Use a small M tile for decode-like batches where tokens are
            # spread thin across experts. Larger batches benefit from
            # GROUP_SIZE_M > 1 because the per-block scales add memory
            # traffic that benefits from L2 tile reuse.
            config = {
                "BLOCK_SIZE_M": 16 if M <= 64 else 64,
                "BLOCK_SIZE_N": block_shape[0],
                "BLOCK_SIZE_K": block_shape[1],
                "GROUP_SIZE_M": 1 if M <= 16 else 32,
                "SPLIT_K": 1,
                "num_warps": 4,
                "num_stages": 3 if not current_platform.is_rocm() else num_stages_rocm,
            }
    elif dtype in ["int4_w4a16", "int8_w8a16"] and block_shape is not None:
        # moe wna16 kernels
        # only set BLOCK_SIZE_M
        # BLOCK_SIZE_N and BLOCK_SIZE_K would be set later
        bit = 4 if dtype == "int4_w4a16" else 8
        use_moe_wna16_cuda = should_moe_wna16_use_cuda(M * topk, block_shape[1], E, bit)
        if use_moe_wna16_cuda:
            config = {"BLOCK_SIZE_M": min(16, M), "SPLIT_K": 1}
        elif M <= 20:
            config = {"BLOCK_SIZE_M": 16, "GROUP_SIZE_M": 1, "SPLIT_K": 1}
        elif M <= 40:
            config = {"BLOCK_SIZE_M": 32, "GROUP_SIZE_M": 1, "SPLIT_K": 1}
        else:
            config = {"BLOCK_SIZE_M": 64, "GROUP_SIZE_M": 1, "SPLIT_K": 1}
    else:
        # General defaults for bf16/fp16 and fp8 per-tensor.
        # Tile sizes scale with batch: small batches are memory-bound
        # (favor tall-K tiles), large batches are compute-bound (favor
        # large M/N tiles with more warps).
        if M <= 32:
            block_m = 16
        elif M <= 96:
            block_m = 32
        elif M <= 512:
            block_m = 64
        else:
            block_m = 128

        block_n = 64 if M <= 64 else 128

        # Small batches benefit from longer reduction (larger K tile),
        # while large batches prefer more output parallelism.
        # FP8 elements are half-width so larger K tiles are always cheap.
        block_k = 128 if dtype == "fp8_w8a8" or M <= 64 else 64

        # Grouping adjacent M-blocks lets them share weight tiles in L2.
        # Only helps when there are enough M-blocks per expert to group;
        # with many experts each one sees few tokens so grouping is useless.
        tokens_per_expert = M // max(E, 1)
        group_m = 16 if tokens_per_expert > 128 else 1

        # Large batches have enough blocks to saturate the GPU, so we
        # use more warps per block to increase arithmetic intensity.
        num_warps = 4 if M <= 128 else 8

        if current_platform.is_rocm():
            num_stages = num_stages_rocm
        elif M <= 32:
            num_stages = 4
        else:
            num_stages = 3

        config = {
            "BLOCK_SIZE_M": block_m,
            "BLOCK_SIZE_N": block_n,
            "BLOCK_SIZE_K": block_k,
            "GROUP_SIZE_M": group_m,
            "SPLIT_K": 1,
            "num_warps": num_warps,
            "num_stages": num_stages,
        }
    return config


def try_get_optimal_moe_config(
    w1_shape: tuple[int, ...],
    w2_shape: tuple[int, ...],
    top_k: int,
    dtype: str | None,
    M: int,
    block_shape: list[int] | None = None,
) -> dict[str, int]:
    from vllm.model_executor.layers.fused_moe import get_config

    override_config = get_config()
    if override_config:
        config = override_config
    else:
        # First try to load optimal config from the file
        E, _, N = w2_shape
        if dtype == "int4_w4a16":
            N = N * 2
        block_n = block_shape[0] if block_shape else 0
        block_k = block_shape[1] if block_shape else 0
        configs = get_moe_configs(E, N, dtype, block_n, block_k)

        if configs:
            # If an optimal configuration map has been found, look up the
            # optimal config
            config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
        else:
            # Else use the default config
            config = get_default_config(M, E, N, w1_shape[2], top_k, dtype, block_shape)
    return config


def inplace_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    ocp_mx_scheme: str | None = None,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    w1_zp: torch.Tensor | None = None,
    w2_zp: torch.Tensor | None = None,
    a1_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    block_shape: list[int] | None = None,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
) -> None:
    fast_out = _try_qwen_c1_topk8_fastpath(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation,
        apply_router_weight_on_input,
        use_fp8_w8a8,
        use_int8_w8a8,
        use_int8_w8a16,
        use_int4_w4a16,
        ocp_mx_scheme,
        per_channel_quant,
        global_num_experts,
        expert_map,
        w1_scale,
        w2_scale,
        w1_zp,
        w2_zp,
        a1_scale,
        a2_scale,
        block_shape,
        w1_bias,
        w2_bias,
    )
    if fast_out is not None:
        hidden_states.copy_(fast_out)
        return
    fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        True,
        activation,
        apply_router_weight_on_input,
        use_fp8_w8a8,
        use_int8_w8a8,
        use_int8_w8a16,
        use_int4_w4a16,
        ocp_mx_scheme,
        per_channel_quant,
        global_num_experts,
        expert_map,
        w1_scale,
        w2_scale,
        w1_zp,
        w2_zp,
        a1_scale,
        a2_scale,
        block_shape,
        w1_bias,
        w2_bias,
    )


def inplace_fused_experts_fake(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    ocp_mx_scheme: str | None = None,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    w1_zp: torch.Tensor | None = None,
    w2_zp: torch.Tensor | None = None,
    a1_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    block_shape: list[int] | None = None,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
) -> None:
    pass


direct_register_custom_op(
    op_name="inplace_fused_experts",
    op_func=inplace_fused_experts,
    mutates_args=["hidden_states"],
    fake_impl=inplace_fused_experts_fake,
)


def outplace_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    ocp_mx_scheme: str | None = None,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    w1_zp: torch.Tensor | None = None,
    w2_zp: torch.Tensor | None = None,
    a1_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    block_shape: list[int] | None = None,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    fast_out = _try_qwen_c1_topk8_fastpath(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation,
        apply_router_weight_on_input,
        use_fp8_w8a8,
        use_int8_w8a8,
        use_int8_w8a16,
        use_int4_w4a16,
        ocp_mx_scheme,
        per_channel_quant,
        global_num_experts,
        expert_map,
        w1_scale,
        w2_scale,
        w1_zp,
        w2_zp,
        a1_scale,
        a2_scale,
        block_shape,
        w1_bias,
        w2_bias,
    )
    if fast_out is not None:
        return fast_out
    return fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        False,
        activation,
        apply_router_weight_on_input,
        use_fp8_w8a8,
        use_int8_w8a8,
        use_int8_w8a16,
        use_int4_w4a16,
        ocp_mx_scheme,
        per_channel_quant,
        global_num_experts,
        expert_map,
        w1_scale,
        w2_scale,
        w1_zp,
        w2_zp,
        a1_scale,
        a2_scale,
        block_shape,
        w1_bias,
        w2_bias,
    )


def outplace_fused_experts_fake(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    ocp_mx_scheme: str | None = None,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    w1_zp: torch.Tensor | None = None,
    w2_zp: torch.Tensor | None = None,
    a1_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    block_shape: list[int] | None = None,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="outplace_fused_experts",
    op_func=outplace_fused_experts,
    fake_impl=outplace_fused_experts_fake,
)


def torch_vllm_inplace_fused_experts(**kwargs) -> torch.Tensor:
    torch.ops.vllm.inplace_fused_experts(**kwargs)
    hidden_states = kwargs["hidden_states"]
    return hidden_states


def torch_vllm_outplace_fused_experts(**kwargs) -> torch.Tensor:
    return torch.ops.vllm.outplace_fused_experts(**kwargs)


def dispatch_fused_experts_func(inplace: bool) -> Callable[..., torch.Tensor]:
    if inplace:
        return torch_vllm_inplace_fused_experts
    return torch_vllm_outplace_fused_experts


# TODO (bnell): replace this with modular op.  Can get rid of inplace/outplace
# torch ops.
def fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: MoEActivation = MoEActivation.SILU,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    quant_config: FusedMoEQuantConfig | None = None,
) -> torch.Tensor:
    """Run fused MoE expert computation using Triton kernels."""
    if quant_config is None:
        quant_config = FUSED_MOE_UNQUANTIZED_CONFIG

    assert not inplace or not disable_inplace()

    return dispatch_fused_experts_func(inplace)(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=activation.value,
        apply_router_weight_on_input=apply_router_weight_on_input,
        use_fp8_w8a8=quant_config.use_fp8_w8a8,
        use_int8_w8a8=quant_config.use_int8_w8a8,
        use_int8_w8a16=quant_config.use_int8_w8a16,
        use_int4_w4a16=quant_config.use_int4_w4a16,
        ocp_mx_scheme=quant_config.ocp_mx_scheme,
        per_channel_quant=quant_config.per_act_token_quant,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
        w1_scale=quant_config.w1_scale,
        w2_scale=quant_config.w2_scale,
        w1_zp=quant_config.w1_zp,
        w2_zp=quant_config.w2_zp,
        a1_scale=quant_config.a1_scale,
        a2_scale=quant_config.a2_scale,
        block_shape=quant_config.block_shape,
        w1_bias=quant_config.w1_bias,
        w2_bias=quant_config.w2_bias,
    )


def _get_config_quant_dtype(
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    ocp_mx_scheme: str | None,
) -> None | torch.dtype | str:
    """
    Get the quantization type based on the quantization strategy flags.
    We don't have a quant_config at this point so we need to work backwards.
    A return type of None means no quantization is required because the
    input is unquantized or has been quantized prior to calling
    fused_experts_impl.
    """
    if use_fp8_w8a8:
        return torch.float8_e4m3fn
    elif use_int8_w8a8:
        return torch.int8
    elif ocp_mx_scheme == "w_mxfp4_a_mxfp4":
        return "mxfp4"
    elif ocp_mx_scheme in {"w_mxfp4_a_mxfp6_e3m2", "w_mxfp6_e3m2_a_mxfp6_e3m2"}:
        return "mxfp6_e3m2"
    elif ocp_mx_scheme in {"w_mxfp4_a_mxfp6_e2m3", "w_mxfp6_e2m3_a_mxfp6_e2m3"}:
        return "mxfp6_e2m3"
    elif ocp_mx_scheme in {"w_mxfp4", "w_mxfp6_e3m2", "w_mxfp6_e2m3"}:
        return torch.bfloat16
    elif ocp_mx_scheme in {"w_mxfp4_a_fp8", "w_mxfp6_e3m2_a_fp8", "w_mxfp6_e2m3_a_fp8"}:
        return torch.float8_e4m3fn

    return None


def fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    ocp_mx_scheme: str | None = None,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    w1_zp: torch.Tensor | None = None,
    w2_zp: torch.Tensor | None = None,
    a1_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    block_shape: list[int] | None = None,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    # Convert string activation to enum for internal use
    activation_enum = MoEActivation.from_str(activation)

    # Check constraints.
    if use_int4_w4a16:
        assert hidden_states.size(1) // 2 == w1.size(2), "Hidden size mismatch"
    elif ocp_mx_scheme is not None:
        if ocp_mx_scheme.startswith("w_mxfp4"):
            # 16bit activation and fp4x2 packed weight
            assert hidden_states.size(1) == w1.size(2) * 2, "hidden size mismatch"
        elif ocp_mx_scheme.startswith("w_mxfp6"):
            assert hidden_states.size(1) == (w1.size(2) * 4) // 3, (
                "hidden size mismatch"
            )
        else:
            raise NotImplementedError(f"Unsupported ocp_mx_scheme={ocp_mx_scheme}")
    else:
        assert hidden_states.size(1) == w1.size(2), (
            f"Hidden size mismatch {hidden_states.size(1)} != {w1.size(2)}"
        )

    assert topk_weights.size() == topk_ids.size(), "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
    assert w2.stride(-1) == 1, "Stride of last dimension must be 1"
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]

    num_tokens = hidden_states.size(0)
    E, N, _ = w1.size()
    K = w2.size(1)
    if global_num_experts == -1:
        global_num_experts = E
    top_k_num = topk_ids.size(1)
    # We execute the fused_moe kernel in chunks to circumvent this issue:
    # https://github.com/vllm-project/vllm/issues/5938
    CHUNK_SIZE = envs.VLLM_FUSED_MOE_CHUNK_SIZE
    M = min(num_tokens, CHUNK_SIZE)

    config_dtype = _get_config_dtype_str(
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        ocp_mx_scheme=ocp_mx_scheme,
        dtype=hidden_states.dtype,
    )

    # Note: for use_int8_w8a16 or use_int4_w4a16, the activations are
    # quantized prior to calling fused_experts.
    quant_dtype = _get_config_quant_dtype(
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        ocp_mx_scheme=ocp_mx_scheme,
    )

    get_config_func = functools.partial(
        try_get_optimal_moe_config,
        w1.size(),
        w2.size(),
        top_k_num,
        config_dtype,
        block_shape=block_shape,
    )

    config = get_config_func(M)

    # We can reuse the memory between these because by the time we need
    # cache3, we're done with cache1
    cache13 = torch.empty(
        M * top_k_num * max(N, K),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache1 = cache13[: M * top_k_num * N].view(M, top_k_num, N)
    intermediate_cache3 = cache13[: M * top_k_num * K].view(M, top_k_num, K)

    # This needs separate memory since it's used concurrently with cache1
    activation_out_dim = mk.FusedMoEExpertsModular.adjust_N_for_activation(
        N, activation_enum
    )
    intermediate_cache2 = torch.empty(
        (M * top_k_num, activation_out_dim),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    if hidden_states.dtype == torch.bfloat16:
        compute_type = tl.bfloat16
    elif hidden_states.dtype == torch.float16:
        compute_type = tl.float16
    elif hidden_states.dtype == torch.float32:
        compute_type = tl.float32
    else:
        raise ValueError(f"Unsupported compute_type: {hidden_states.dtype}")

    out_hidden_states = hidden_states if inplace else torch.empty_like(hidden_states)

    if ocp_mx_scheme is not None:
        # TODO: On platforms for which `current_platform.supports_mx()` is True
        # and for which we have a native OCP mx fused MOE kernel,
        # this dequantization step should not be done.
        if ocp_mx_scheme.startswith("w_mxfp4"):
            # Weight has to be dequantized for mxfp4 emulation.
            w1 = dequant_mxfp4(w1, w1_scale, hidden_states.dtype)
            w1_scale = None
            w2 = dequant_mxfp4(w2, w2_scale, hidden_states.dtype)
            w2_scale = None
        elif ocp_mx_scheme.startswith("w_mxfp6_e3m2"):
            w1 = dequant_mxfp6(
                w1, w1_scale, quant_dtype="fp6_e3m2", float_dtype=hidden_states.dtype
            )
            w1_scale = None
            w2 = dequant_mxfp6(
                w2, w2_scale, quant_dtype="fp6_e3m2", float_dtype=hidden_states.dtype
            )
            w2_scale = None
        elif ocp_mx_scheme.startswith("w_mxfp6_e2m3"):
            w1 = dequant_mxfp6(
                w1, w1_scale, quant_dtype="fp6_e2m3", float_dtype=hidden_states.dtype
            )
            w1_scale = None
            w2 = dequant_mxfp6(
                w2, w2_scale, quant_dtype="fp6_e2m3", float_dtype=hidden_states.dtype
            )
            w2_scale = None
        else:
            raise NotImplementedError(f"Unsupported ocp_mx_scheme={ocp_mx_scheme}")

    for chunk in range((num_tokens // CHUNK_SIZE) + 1):
        begin_chunk_idx, end_chunk_idx = (
            chunk * CHUNK_SIZE,
            min((chunk + 1) * CHUNK_SIZE, num_tokens),
        )
        curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
        tokens_in_chunk, _ = curr_hidden_states.size()

        if tokens_in_chunk == 0:
            break

        if tokens_in_chunk < CHUNK_SIZE and chunk > 0:
            # Adjust the intermediate cache size and config for the last
            # chunk. Note that in most cases we only have one chunk
            # so the cache size and config are already set correctly and
            # do not need to be adjusted.
            intermediate_cache1 = intermediate_cache1[:tokens_in_chunk]
            intermediate_cache2 = intermediate_cache2[
                : tokens_in_chunk * topk_ids.size(1)
            ]
            intermediate_cache3 = intermediate_cache3[:tokens_in_chunk]
            config = get_config_func(tokens_in_chunk)

        curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
        curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]
        qcurr_hidden_states, a1q_scale = moe_kernel_quantize_input(
            A=curr_hidden_states,
            A_scale=a1_scale,
            quant_dtype=quant_dtype,
            per_act_token_quant=per_channel_quant,
            block_shape=block_shape,
            ocp_mx_scheme=ocp_mx_scheme,
        )

        # SPARSITY_FACTOR is a heuristic margin ensuring tokens_in_chunk * top_k
        # activates only a small fraction of total experts
        SPARSITY_FACTOR = 4
        # block quantized code path is not implemented yet.
        naive_block_assignment = (
            expert_map is None
            and tokens_in_chunk * top_k_num * SPARSITY_FACTOR <= global_num_experts
            and not (
                (use_int8_w8a16 or use_int4_w4a16)
                and block_shape is not None
                and block_shape[1] > 0
            )
        )

        if not naive_block_assignment:
            sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
                curr_topk_ids,
                config["BLOCK_SIZE_M"],
                global_num_experts,
                expert_map,
                ignore_invalid_experts=True,
            )
        else:
            max_num_tokens_padded = topk_ids.numel() * config["BLOCK_SIZE_M"]
            expert_ids = curr_topk_ids.view(-1)
            num_tokens_post_padded = torch.empty(
                (1), dtype=torch.int32, device=topk_ids.device
            )
            num_tokens_post_padded.fill_(max_num_tokens_padded)
            sorted_token_ids = None

        dispatch_fused_moe_kernel(
            qcurr_hidden_states,
            w1,
            intermediate_cache1,
            a1q_scale,
            w1_scale,
            w1_zp,
            curr_topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            apply_router_weight_on_input,
            top_k_num,
            config,
            compute_type=compute_type,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            per_channel_quant=per_channel_quant,
            block_shape=block_shape,
            B_bias=w1_bias,
        )

        apply_moe_activation(
            activation_enum, intermediate_cache2, intermediate_cache1.view(-1, N)
        )

        qintermediate_cache2, a2q_scale = moe_kernel_quantize_input(
            A=intermediate_cache2,
            A_scale=a2_scale,
            quant_dtype=quant_dtype,
            per_act_token_quant=per_channel_quant,
            block_shape=block_shape,
            ocp_mx_scheme=ocp_mx_scheme,
        )

        if expert_map is not None:
            intermediate_cache3.zero_()

        dispatch_fused_moe_kernel(
            qintermediate_cache2,
            w2,
            intermediate_cache3,
            a2q_scale,
            w2_scale,
            w2_zp,
            curr_topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            not apply_router_weight_on_input,
            1,
            config,
            compute_type=compute_type,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            per_channel_quant=per_channel_quant,
            block_shape=block_shape,
            B_bias=w2_bias,
        )

        ops.moe_sum(
            intermediate_cache3.view(*intermediate_cache3.size()),
            out_hidden_states[begin_chunk_idx:end_chunk_idx],
        )

    return out_hidden_states


class TritonExperts(mk.FusedMoEExpertsModular):
    """Triton-based fused MoE expert implementation."""

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return current_platform.is_cuda_alike()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        p = current_platform
        if p.is_rocm():
            from vllm.platforms.rocm import on_gfx9

            is_rocm_on_gfx9 = on_gfx9()
            is_rocm_on_gfx906 = on_gfx906()
        else:
            is_rocm_on_gfx9 = False
            is_rocm_on_gfx906 = False

        device_supports_fp8 = is_rocm_on_gfx9 or is_rocm_on_gfx906 or (
            p.is_cuda() and p.has_device_capability((8, 9))
        )

        if not device_supports_fp8:
            return (weight_key, activation_key) == (None, None)

        SUPPORTED_W_A = [
            (None, None),
            (kFp8Static128BlockSym, kFp8Dynamic128Sym),
            (kFp8StaticChannelSym, kFp8DynamicTokenSym),
            (kFp8StaticTensorSym, kFp8DynamicTokenSym),
            (kFp8StaticTensorSym, kFp8StaticTensorSym),
            (kFp8StaticTensorSym, kFp8DynamicTensorSym),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in [
            MoEActivation.SILU,
            MoEActivation.GELU,
            MoEActivation.SWIGLUOAI,
            MoEActivation.SWIGLUSTEP,
            MoEActivation.SILU_NO_MUL,
            MoEActivation.GELU_NO_MUL,
            MoEActivation.RELU2_NO_MUL,
        ]

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return not moe_parallel_config.use_fi_all2allv_kernels

    def supports_chunking(self) -> bool:
        return True

    def supports_expert_map(self) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        workspace1 = (M, topk, max(activation_out_dim, K))
        workspace2 = (M, topk, max(N, K))
        output = (M, K)
        return (workspace1, workspace2, output)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        # ── GFX906 FP8: bitwise Triton dequant path ────────────────────
        _gfx906_fp8 = (self.quant_config.use_fp8_w8a8
                        and self.block_shape is not None
                        and on_gfx906())
        if _gfx906_fp8:
            logger.info_once(
                "GFX906: bitwise FP8 dequant + FP16 GEMM path active",
                scope="local",
            )

            # hidden_states must be FP16 (fp8 quantized avoided in prepare_finalize.py)
            #assert hidden_states.dtype == torch.float16, "hidden_states must be FP16"
            #assert a1q_scale is None, "a1q_scale must be None"


            E_orig, num_tokens, N, K, top_k_num = self.moe_problem_size(
                hidden_states, w1, w2, topk_ids
            )

            if global_num_experts == -1:
                global_num_experts = E_orig

            config = try_get_optimal_moe_config(
                w1.size(),
                w2.size(),
                top_k_num,
                "fp8_w8a8",
                num_tokens,
                block_shape=self.block_shape,
            )

            compute_type = tl.float16

            intermediate_cache1 = _resize_cache(
                workspace2, (num_tokens, top_k_num, N))
            cache2_dim = self.adjust_N_for_activation(N, activation)
            intermediate_cache2 = _resize_cache(
                workspace13, (num_tokens * top_k_num, cache2_dim))
            intermediate_cache3 = _resize_cache(
                workspace2, (num_tokens, top_k_num, K))

            sorted_token_ids, expert_ids, num_tokens_post_padded = (
                moe_align_block_size(
                    topk_ids,
                    config["BLOCK_SIZE_M"],
                    E_orig,
                    expert_map,
                )
            )

            # First GEMM: hidden * w1
            invoke_fused_moe_triton_kernel(
                hidden_states,           # FP16
                w1.view(torch.uint8),    # FP8
                intermediate_cache1,
                None,          # a_scale
                self.w1_scale, # b_scale
                None,          # topk_weights
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                False,         # mul_routed_weights
                top_k_num,
                config,
                compute_type=compute_type,
                use_fp8_w8a8=self.quant_config.use_fp8_w8a8,
                use_int8_w8a8=self.quant_config.use_int8_w8a8,
                use_int8_w8a16=self.quant_config.use_int8_w8a16,
                use_int4_w4a16=self.quant_config.use_int4_w4a16,
                per_channel_quant=self.per_act_token_quant,
                block_shape=self.block_shape,
                B_bias=self.w1_bias,
            )

            self.activation(
                activation, intermediate_cache2,
                intermediate_cache1.view(-1, N))

            assert intermediate_cache2.dtype == torch.float16, "intermediate_cache2 must be FP16"

            # Second GEMM: intermediate * w2
            invoke_fused_moe_triton_kernel(
                intermediate_cache2,   # FP16
                w2.view(torch.uint8),  # FP8
                intermediate_cache3,
                None,                  # a_scale
                self.w2_scale,         # b_scale
                topk_weights,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                not apply_router_weight_on_input,
                1,
                config,
                compute_type=compute_type,
                use_fp8_w8a8=self.quant_config.use_fp8_w8a8,
                use_int8_w8a8=self.quant_config.use_int8_w8a8,
                use_int8_w8a16=self.quant_config.use_int8_w8a16,
                use_int4_w4a16=self.quant_config.use_int4_w4a16,
                per_channel_quant=self.per_act_token_quant,
                block_shape=self.block_shape,
                B_bias=self.w2_bias,
            )

            self.moe_sum(intermediate_cache3, output)
            return

        # ── Standard (non-gfx906) path ────────────────────────────────
        # Check constraints.
        if self.quant_config.use_int4_w4a16:
            assert hidden_states.size(-1) // 2 == w1.size(2), "Hidden size mismatch"
        else:
            assert hidden_states.size(-1) == w1.size(2), (
                f"Hidden size mismatch {hidden_states.size(-1)} != {w1.size(2)}"
            )

        assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
        assert hidden_states.dim() == 2
        assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
        assert w2.stride(-1) == 1, "Stride of last dimension must be 1"
        assert hidden_states.dtype in [
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
        ]

        E, num_tokens, N, K, top_k_num = self.moe_problem_size(
            hidden_states, w1, w2, topk_ids
        )

        if global_num_experts == -1:
            global_num_experts = E

        if _qwen_c1_topk8_fastpath_enabled():
            logger.info_once(
                "qwen c1 topk8 MoE fastpath apply shape seen "
                f"E={E} num_tokens={num_tokens} N={N} K={K} top_k={top_k_num} "
                f"activation={activation.value} dtype={hidden_states.dtype}"
            )
            fast_out = _try_qwen_c1_topk8_fastpath(
                hidden_states,
                w1,
                w2,
                topk_weights,
                topk_ids,
                activation.value,
                apply_router_weight_on_input,
                self.quant_config.use_fp8_w8a8,
                self.quant_config.use_int8_w8a8,
                self.quant_config.use_int8_w8a16,
                self.quant_config.use_int4_w4a16,
                self.quant_config.ocp_mx_scheme,
                self.per_act_token_quant,
                global_num_experts,
                expert_map,
                self.w1_scale,
                self.w2_scale,
                self.w1_zp,
                self.w2_zp,
                a1q_scale,
                a2_scale,
                self.block_shape,
                self.w1_bias,
                self.w2_bias,
            )
            if fast_out is not None:
                output.copy_(fast_out)
                return

        config = try_get_optimal_moe_config(
            w1.size(),
            w2.size(),
            top_k_num,
            self.quant_config.config_name(hidden_states.dtype),
            num_tokens,
            block_shape=self.block_shape,
        )

        if hidden_states.dtype == torch.bfloat16:
            compute_type = tl.bfloat16
        elif hidden_states.dtype == torch.float16:
            compute_type = tl.float16
        elif hidden_states.dtype == torch.float32:
            compute_type = tl.float32
        elif (
            hidden_states.dtype == torch.float8_e4m3fn
            or hidden_states.dtype == torch.float8_e4m3fnuz
        ):
            compute_type = tl.bfloat16
        else:
            raise ValueError(f"Unsupported compute_type: {hidden_states.dtype}")

        # Note that the output tensor might be in workspace1
        intermediate_cache1 = _resize_cache(workspace2, (num_tokens, top_k_num, N))
        cache2_dim = self.adjust_N_for_activation(N, activation)
        intermediate_cache2 = _resize_cache(
            workspace13, (num_tokens * top_k_num, cache2_dim)
        )
        intermediate_cache3 = _resize_cache(workspace2, (num_tokens, top_k_num, K))

        # For C1 decode, routing touches only a few expert slots per layer. The
        # standard unquantized path can skip the align kernel in that sparse case
        # and let the Triton kernel use naive block assignment directly.
        SPARSITY_FACTOR = 4
        naive_block_assignment = (
            expert_map is None
            and num_tokens * top_k_num * SPARSITY_FACTOR <= global_num_experts
            and not (
                (self.quant_config.use_int8_w8a16 or self.quant_config.use_int4_w4a16)
                and self.block_shape is not None
                and self.block_shape[1] > 0
            )
        )

        if not naive_block_assignment:
            sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
                topk_ids, config["BLOCK_SIZE_M"], global_num_experts, expert_map
            )
        else:
            max_num_tokens_padded = topk_ids.numel() * config["BLOCK_SIZE_M"]
            expert_ids = topk_ids.view(-1)
            num_tokens_post_padded = torch.empty(
                (1), dtype=torch.int32, device=topk_ids.device
            )
            num_tokens_post_padded.fill_(max_num_tokens_padded)
            sorted_token_ids = None

        invoke_fused_moe_triton_kernel(
            hidden_states,
            w1,
            intermediate_cache1,
            a1q_scale,
            self.w1_scale,
            None,  # topk_weights
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            False,  # mul_routed_weights
            top_k_num,
            config,
            compute_type=compute_type,
            use_fp8_w8a8=self.quant_config.use_fp8_w8a8,
            use_int8_w8a8=self.quant_config.use_int8_w8a8,
            use_int8_w8a16=self.quant_config.use_int8_w8a16,
            use_int4_w4a16=self.quant_config.use_int4_w4a16,
            per_channel_quant=self.per_act_token_quant,
            block_shape=self.block_shape,
            B_bias=self.w1_bias,
        )

        self.activation(
            activation, intermediate_cache2, intermediate_cache1.view(-1, N)
        )

        a2q_scale: torch.Tensor | None = None

        qintermediate_cache2, a2q_scale = moe_kernel_quantize_input(
            intermediate_cache2,
            a2_scale,
            self.quant_dtype,
            self.per_act_token_quant,
            self.block_shape,
        )

        invoke_fused_moe_triton_kernel(
            qintermediate_cache2,
            w2,
            intermediate_cache3,
            a2q_scale,
            self.w2_scale,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            not apply_router_weight_on_input,
            1,
            config,
            compute_type=compute_type,
            use_fp8_w8a8=self.quant_config.use_fp8_w8a8,
            use_int8_w8a8=self.quant_config.use_int8_w8a8,
            use_int8_w8a16=self.quant_config.use_int8_w8a16,
            use_int4_w4a16=self.quant_config.use_int4_w4a16,
            per_channel_quant=self.per_act_token_quant,
            block_shape=self.block_shape,
            B_bias=self.w2_bias,
        )

        # separate function is required for MoE + LoRA
        self.moe_sum(intermediate_cache3, output)

    def moe_sum(self, input: torch.Tensor, output: torch.Tensor) -> None:
        ops.moe_sum(input, output)


class TritonWNA16Experts(TritonExperts):
    @staticmethod
    def _supports_current_device() -> bool:
        raise NotImplementedError(
            "TritonWNA16Experts is not yet used by an Oracle. "
            "This method should not be called."
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        raise NotImplementedError(
            "TritonWNA16Experts is not yet used by an Oracle. "
            "This method should not be called."
        )

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        raise NotImplementedError(
            "TritonWNA16Experts is not yet used by an Oracle. "
            "This method should not be called."
        )

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        raise NotImplementedError(
            "TritonWNA16Experts is not yet used by an Oracle. "
            "This method should not be called."
        )

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        raise NotImplementedError(
            "TritonWNA16Experts is not yet used by an Oracle. "
            "This method should not be called."
        )

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        # Check constraints.
        if self.quant_config.use_int4_w4a16:
            assert hidden_states.size(-1) // 2 == w1.size(2), "Hidden size mismatch"
        else:
            assert hidden_states.size(-1) == w1.size(2), (
                f"Hidden size mismatch {hidden_states.size(-1)} != {w1.size(2)}"
            )

        assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
        assert hidden_states.dim() == 2
        assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
        assert w2.stride(-1) == 1, "Stride of last dimension must be 1"
        assert hidden_states.dtype in [
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
        ]

        E, num_tokens, N, K, top_k_num = self.moe_problem_size(
            hidden_states, w1, w2, topk_ids
        )

        if global_num_experts == -1:
            global_num_experts = E

        config = try_get_optimal_moe_config(
            w1.size(),
            w2.size(),
            top_k_num,
            self.quant_config.config_name(hidden_states.dtype),
            num_tokens,
            block_shape=self.block_shape,
        )

        if hidden_states.dtype == torch.bfloat16:
            compute_type = tl.bfloat16
        elif hidden_states.dtype == torch.float16:
            compute_type = tl.float16
        elif hidden_states.dtype == torch.float32:
            compute_type = tl.float32
        elif (
            hidden_states.dtype == torch.float8_e4m3fn
            or hidden_states.dtype == torch.float8_e4m3fnuz
        ):
            compute_type = tl.bfloat16
        else:
            raise ValueError(f"Unsupported compute_type: {hidden_states.dtype}")

        # Note that the output tensor might be in workspace1
        intermediate_cache1 = _resize_cache(workspace2, (num_tokens, top_k_num, N))
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        intermediate_cache2 = _resize_cache(
            workspace13, (num_tokens * top_k_num, activation_out_dim)
        )
        intermediate_cache3 = _resize_cache(workspace2, (num_tokens, top_k_num, K))

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, config["BLOCK_SIZE_M"], global_num_experts, expert_map
        )

        invoke_fused_moe_wna16_triton_kernel(
            hidden_states,
            w1,
            intermediate_cache1,
            self.w1_scale,
            self.quant_config.w1_zp,
            None,  # topk_weights
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            False,  # mul_routed_weights
            top_k_num,
            config,
            compute_type=compute_type,
            use_int8_w8a16=self.quant_config.use_int8_w8a16,
            use_int4_w4a16=self.quant_config.use_int4_w4a16,
            block_shape=self.block_shape,
        )

        self.activation(
            activation, intermediate_cache2, intermediate_cache1.view(-1, N)
        )

        a2q_scale: torch.Tensor | None = None

        qintermediate_cache2, a2q_scale = moe_kernel_quantize_input(
            intermediate_cache2,
            a2_scale,
            self.quant_dtype,
            self.per_act_token_quant,
            self.block_shape,
        )

        invoke_fused_moe_wna16_triton_kernel(
            qintermediate_cache2,
            w2,
            intermediate_cache3,
            self.w2_scale,
            self.quant_config.w2_zp,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            not apply_router_weight_on_input,
            1,
            config,
            compute_type=compute_type,
            use_int8_w8a16=self.quant_config.use_int8_w8a16,
            use_int4_w4a16=self.quant_config.use_int4_w4a16,
            block_shape=self.block_shape,
        )

        # separate function is required for MoE + LoRA
        self.moe_sum(intermediate_cache3, output)
HOTFIX_FUSED_MOE_FASTPATH

  cat <<'HOTFIX_FUSED_MOE_PR39016' > "${bundle_dir}/fused_moe_pr39016.py"
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused MoE Triton kernels."""

import functools
import json
import os
from collections.abc import Callable
from typing import Any

import torch

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.batch_invariant import (
    vllm_is_batch_invariant,
)
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    _get_config_dtype_str,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import (
    _resize_cache,
    disable_inplace,
    moe_kernel_quantize_input,
)
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import dequant_mxfp4
from vllm.model_executor.layers.quantization.utils.mxfp6_utils import dequant_mxfp6
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8DynamicTensorSym,
    kFp8DynamicTokenSym,
    kFp8Static128BlockSym,
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx906
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)


@triton.jit
def write_zeros_to_output(
    c_ptr,
    stride_cm,
    stride_cn,
    pid_n,
    N,
    offs_token,
    token_mask,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    compute_type,
):
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def fused_moe_kernel_gptq_awq(
    on_gfx906: tl.constexpr,
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    b_scale_ptr,
    b_zp_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N: tl.constexpr,
    K: tl.constexpr,
    EM,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bse,
    stride_bsk,
    stride_bsn,
    stride_bze,
    stride_bzk,
    stride_bzn,
    block_k_diviable: tl.constexpr,
    group_size: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    has_zp: tl.constexpr,
    use_int4_w4a16: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.
    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    # Cast to int64 to prevent overflow in stride*offset products
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        # -----------------------------------------------------------
        # Write back zeros to the output when the expert is not
        # in the current expert parallel rank.
        write_zeros_to_output(
            c_ptr,
            stride_cm,
            stride_cn,
            pid_n,
            N,
            offs_token,
            token_mask,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            compute_type,
        )
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )

    if use_int4_w4a16:
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + (offs_k[:, None] // 2) * stride_bk
            + offs_bn[None, :] * stride_bn
        )
        b_shifter = (offs_k[:, None] % 2) * 4
    elif use_int8_w8a16:
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + offs_k[:, None] * stride_bk
            + offs_bn[None, :] * stride_bn
        )

    if not has_zp and use_int4_w4a16:
        b_zp_num = 8
    if not has_zp and use_int8_w8a16:
        b_zp_num = 128
    elif has_zp and use_int4_w4a16:
        b_zp_shifter = (offs_bn[None, :] % 2) * 4

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to compute_type (--dtype) after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.

        if not block_k_diviable:
            k_mask = offs_k[:, None] < K - k * BLOCK_SIZE_K
            k_other = 0.0
        else:
            k_mask = None
            k_other = None

        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(b_ptrs)
        if use_int4_w4a16:
            b = (b >> b_shifter) & 0xF

        b_scale_ptrs = (
            b_scale_ptr
            + off_experts * stride_bse
            + offs_bn[None, :] * stride_bsn
            + ((offs_k[:, None] + BLOCK_SIZE_K * k) // group_size) * stride_bsk
        )
        b_scale = tl.load(b_scale_ptrs, mask=k_mask, other=k_other)
        b_scale = b_scale.to(tl.float32)

        if has_zp and use_int4_w4a16:
            offs_k_true = (offs_k[:, None] + BLOCK_SIZE_K * k) // group_size
            b_zp_ptrs = (
                b_zp_ptr
                + off_experts * stride_bze
                + (offs_bn[None, :] // 2) * stride_bzn
                + offs_k_true * stride_bzk
            )
            b_zp = tl.load(b_zp_ptrs, mask=k_mask, other=k_other)
            b_zp = (b_zp >> b_zp_shifter) & 0xF
            b_zp = b_zp.to(tl.float32)
        elif has_zp and use_int8_w8a16:
            offs_k_true = (offs_k[:, None] + BLOCK_SIZE_K * k) // group_size
            b_zp_ptrs = (
                b_zp_ptr
                + off_experts * stride_bze
                + offs_bn[None, :] * stride_bzn
                + offs_k_true * stride_bzk
            )
            b_zp = tl.load(b_zp_ptrs, mask=k_mask, other=k_other)
            b_zp = b_zp.to(tl.float32)

        # We accumulate along the K dimension.
        if has_zp:
            if compute_type == tl.float32 and on_gfx906:
                b = ((b.to(tl.float32) - b_zp) * b_scale).to(tl.float16)
            else:
                b = ((b.to(tl.float32) - b_zp) * b_scale).to(compute_type)
        else:
            if compute_type == tl.float32 and on_gfx906:
                b = ((b.to(tl.float32) - b_zp_num) * b_scale).to(tl.float16)
            else:
                b = ((b.to(tl.float32) - b_zp_num) * b_scale).to(compute_type)
        accumulator = tl.dot(a.to(tl.float16) if compute_type == tl.float32 and on_gfx906 else a, b, acc=accumulator)

        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        if use_int4_w4a16:
            b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk
        else:
            b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]

    if not compute_type == tl.float32:
        accumulator = accumulator.to(compute_type)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def fused_moe_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    b_bias_ptr,
    a_scale_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bse,
    stride_bsk,
    stride_bsn,
    stride_bbe,  # bias expert stride
    stride_bbn,  # bias N stride
    # Block size for block-wise quantization
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    naive_block_assignment: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    per_channel_quant: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    on_gfx906: tl.constexpr,
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.
    - naive_block_assignment: A boolean flag indicating whether to use naive
        token wise block assignment. If True, each block corresponds to a
        single token.
    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    offs = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    if not naive_block_assignment:
        offs_token_id = pid_m * BLOCK_SIZE_M + offs
        offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    else:
        offs_token = tl.where(
            offs == 0,
            pid_m,  # first element = pid_m
            num_valid_tokens,  # remaining elements = constant
        )
    # Cast to int64 to prevent overflow in stride*offset products
    # (e.g. stride_cm * offs_token can exceed int32 for large token counts)
    offs_token = offs_token.to(tl.int64)

    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        # -----------------------------------------------------------
        # Write back zeros to the output when the expert is not
        # in the current expert parallel rank.
        write_zeros_to_output(
            c_ptr,
            stride_cm,
            stride_cn,
            pid_n,
            N,
            offs_token,
            token_mask,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            compute_type,
        )
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )

    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )
    if use_int8_w8a16:
        b_scale_ptrs = (
            b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
        )
        b_scale = tl.load(b_scale_ptrs)

    if use_fp8_w8a8 or use_int8_w8a8:
        # block-wise
        if group_k > 0 and group_n > 0:
            if not on_gfx906:
                a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            offs_bsn = offs_bn // group_n
            b_scale_ptrs = (
                b_scale_ptr + off_experts * stride_bse + offs_bsn * stride_bsn
            )
        # channel-wise
        elif per_channel_quant:
            b_scale_ptrs = (
                b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
            )
            b_scale = tl.load(b_scale_ptrs)
            # Load per-token scale for activations
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            a_scale = tl.load(a_scale_ptrs, mask=token_mask, other=0.0)[:, None]
        # tensor-wise
        else:
            a_scale = tl.load(a_scale_ptr)
            b_scale = tl.load(b_scale_ptr + off_experts)
    if HAS_BIAS:
        # bias shape: [num_experts, N]
        bias_ptrs = b_bias_ptr + off_experts * stride_bbe + offs_bn * stride_bbn
        bias = tl.load(bias_ptrs, mask=(offs_bn < N), other=0.0)
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        if on_gfx906:
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0)
        else:
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        if use_int8_w8a16:
            accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
        elif use_fp8_w8a8 or use_int8_w8a8:
            if group_k > 0 and group_n > 0:
                k_start = k * BLOCK_SIZE_K
                offs_ks = k_start // group_k
                if not on_gfx906:
                    a_scale = tl.load(
                        a_scale_ptrs + offs_ks * stride_ask, mask=token_mask, other=0.0
                    )
                else:
                    # Bitwise E4M3 -> FP16 dequant for B (b is already uint8)
                    b_sign = (b & 0x80).to(tl.uint16) << 8
                    b_exp = ((b & 0x78) >> 3).to(tl.uint16)
                    b_exp = tl.where(b_exp == 0, tl.zeros_like(b_exp), b_exp + 8)
                    b_mant = (b & 0x07).to(tl.uint16) << 7
                    b_bits = b_sign | (b_exp << 10) | b_mant
                    b = b_bits.to(tl.float16, bitcast=True)
                
                b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)

                accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :] if not on_gfx906 else tl.dot(a, b) * b_scale[None, :]
            else:
                if use_fp8_w8a8:
                    # acc used to enable fp8_fast_accum
                    accumulator = tl.dot(a, b, acc=accumulator)
                else:
                    accumulator += tl.dot(a, b)
        else:
            accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Dequantization for supported quantization schemes:
    #   - int8_w8a16
    #   - fp8_w8a8
    #   - int8_w8a8
    # Accumulator and scalings are in float32 to preserve numerical accuracy.
    if use_int8_w8a16:
        accumulator = accumulator * b_scale
    elif (use_fp8_w8a8 or use_int8_w8a8) and not (group_k > 0 and group_n > 0):
        accumulator = accumulator * a_scale * b_scale if not on_gfx906 else accumulator * b_scale

    # Bias addition:
    # Bias must be applied after dequantization:
    #   - Since bias is typically not quantized
    #   - Bias should not be scaled by quantization factors
    if HAS_BIAS:
        accumulator += bias[None, :]

    # Router (MoE) weight multiplication:
    # This multiplication MUST be performed in float32 before any precision
    # conversion to ensure numerical stability, which is especially critical
    # on ROCm platforms.
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(
            topk_weights_ptr + offs_token,
            mask=token_mask,
            other=0,
        )
        accumulator *= moe_weight[:, None]

    # Final precision conversion:
    # Cast once at the end to the desired compute/output dtype.
    accumulator = accumulator.to(compute_type)

    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


# NOTE(zyongye): we can remove all the wna16 kernel
# once we drop off sm75 support
def invoke_fused_moe_wna16_cuda_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    B_scale: torch.Tensor | None,
    B_zp: torch.Tensor | None,
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor | None,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict[str, Any],
    block_shape: list[int],
):
    assert B_scale is not None and B_scale.ndim == 3
    assert B_zp is None or B_zp.ndim == 3
    assert block_shape is None or block_shape[0] == 0

    M = A.size(0)
    num_tokens = M * top_k
    bit = 4

    config = config.copy()
    config.update(
        get_moe_wna16_block_config(
            config=config,
            use_moe_wna16_cuda=True,
            num_valid_tokens=num_tokens,
            size_k=A.size(1),
            size_n=B.size(1),
            num_experts=B.size(1),
            group_size=block_shape[1],
            real_top_k=top_k,
            block_size_m=config["BLOCK_SIZE_M"],
        )
    )

    ops.moe_wna16_gemm(
        A,
        C,
        B,
        B_scale,
        B_zp,
        topk_weights if mul_routed_weight else None,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        top_k,
        config["BLOCK_SIZE_M"],
        config["BLOCK_SIZE_N"],
        config["BLOCK_SIZE_K"],
        bit,
    )


# NOTE(zyongye): we can remove all the wna16 kernel
# once we drop off sm75 support
def invoke_fused_moe_wna16_triton_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    B_scale: torch.Tensor | None,
    B_zp: torch.Tensor | None,
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict[str, Any],
    compute_type: tl.dtype,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    block_shape: list[int] | None,
):
    assert B_scale is not None and B_scale.ndim == 3
    assert B_zp is None or B_zp.ndim == 3
    assert block_shape is not None and block_shape[0] == 0

    M = A.size(0)
    num_tokens = M * top_k

    EM = sorted_token_ids.size(0)
    if A.size(0) < config["BLOCK_SIZE_M"]:
        # optimize for small batch_size.
        # We assume that top_ids of each token is unique,
        # so num_valid_experts <= batch_size <= BLOCK_SIZE_M,
        # and we can skip some invalid blocks.
        EM = min(sorted_token_ids.size(0), A.size(0) * top_k * config["BLOCK_SIZE_M"])
    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"])
        * triton.cdiv(B.size(1), META["BLOCK_SIZE_N"]),
    )
    config = config.copy()
    config.update(
        get_moe_wna16_block_config(
            config=config,
            use_moe_wna16_cuda=False,
            num_valid_tokens=num_tokens,
            size_k=A.size(1),
            size_n=B.size(1),
            num_experts=B.size(1),
            group_size=block_shape[1],
            real_top_k=top_k,
            block_size_m=config["BLOCK_SIZE_M"],
        )
    )

    # For gfx906 and compute_type == tl.float32: the kernel internally casts to fp16 before tl.dot,
    # but we still pass compute_type so the OUTPUT is stored correctly.
    # The accumulator is always fp32 inside the kernel regardless.

    fused_moe_kernel_gptq_awq[grid](
        on_gfx906(),
        A,
        B,
        C,
        B_scale,
        B_zp,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.size(1),
        A.size(1),
        EM,
        num_tokens,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
        B_scale.stride(0),
        B_scale.stride(2),
        B_scale.stride(1),
        B_zp.stride(0) if B_zp is not None else 0,
        B_zp.stride(2) if B_zp is not None else 0,
        B_zp.stride(1) if B_zp is not None else 0,
        block_k_diviable=A.size(1) % config["BLOCK_SIZE_K"] == 0,
        group_size=block_shape[1],
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        has_zp=B_zp is not None,
        use_int4_w4a16=use_int4_w4a16,
        use_int8_w8a16=use_int8_w8a16,
        **config,
    )


def invoke_fused_moe_triton_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: torch.Tensor | None,
    B_scale: torch.Tensor | None,
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor | None,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict[str, Any],
    compute_type: tl.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    per_channel_quant: bool,
    block_shape: list[int] | None = None,
    B_bias: torch.Tensor | None = None,
):
    assert topk_weights is not None or not mul_routed_weight
    assert topk_weights is None or topk_weights.stride(1) == 1
    assert sorted_token_ids is None or sorted_token_ids.stride(0) == 1

    if use_fp8_w8a8 or use_int8_w8a8:
        assert B_scale is not None
        assert block_shape is None or triton.cdiv(
            B.size(-2), block_shape[0]
        ) == B_scale.size(-2)
        assert block_shape is None or triton.cdiv(
            B.size(-1), block_shape[1]
        ) == B_scale.size(-1)
    elif use_int8_w8a16 or use_int4_w4a16:
        assert B_scale is not None
        assert block_shape is None or block_shape[0] == 0
    else:
        assert A_scale is None
        assert B_scale is None

    M = A.size(0)
    num_tokens = M * top_k
    if sorted_token_ids is not None:
        EM = sorted_token_ids.size(0)
        if A.size(0) < config["BLOCK_SIZE_M"]:
            # optimize for small batch_size.
            # We assume that top_ids of each token is unique,
            # so num_valid_experts <= batch_size <= BLOCK_SIZE_M,
            # and we can skip some invalid blocks.
            EM = min(
                sorted_token_ids.size(0), A.size(0) * top_k * config["BLOCK_SIZE_M"]
            )
    else:
        EM = num_tokens * config["BLOCK_SIZE_M"]
    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"])
        * triton.cdiv(B.size(1), META["BLOCK_SIZE_N"]),
    )
    HAS_BIAS = B_bias is not None

    config = config.copy()
    config["SPLIT_K"] = 1
    BLOCK_SIZE_K = config.pop("BLOCK_SIZE_K")
    if block_shape is not None:
        BLOCK_SIZE_K = min(BLOCK_SIZE_K, min(block_shape[0], block_shape[1]))
    fused_moe_kernel[grid](
        A,
        B,
        C,
        B_bias,
        A_scale,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.size(1),
        B.size(2),
        EM,
        num_tokens,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
        A_scale.stride(0) if A_scale is not None and A_scale.ndim == 2 else 0,
        A_scale.stride(1) if A_scale is not None and A_scale.ndim == 2 else 0,
        B_scale.stride(0) if B_scale is not None and B_scale.ndim >= 2 else 0,
        B_scale.stride(2) if B_scale is not None and B_scale.ndim == 3 else 0,
        B_scale.stride(1) if B_scale is not None and B_scale.ndim >= 2 else 0,
        B_bias.stride(0) if B_bias is not None else 0,
        B_bias.stride(1) if B_bias is not None else 0,
        0 if block_shape is None else block_shape[0],
        0 if block_shape is None else block_shape[1],
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        per_channel_quant=per_channel_quant,
        naive_block_assignment=(sorted_token_ids is None),
        HAS_BIAS=HAS_BIAS,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        **config,
        on_gfx906=on_gfx906(),
    )


def dispatch_fused_moe_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: torch.Tensor | None,
    B_scale: torch.Tensor | None,
    B_zp: torch.Tensor | None,
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor | None,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict[str, Any],
    compute_type: tl.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    per_channel_quant: bool,
    block_shape: list[int] | None = None,
    B_bias: torch.Tensor | None = None,
) -> None:
    assert topk_weights is not None or not mul_routed_weight
    assert topk_weights is None or topk_weights.stride(1) == 1
    assert sorted_token_ids is None or sorted_token_ids.stride(0) == 1

    M = A.size(0)
    num_tokens = M * top_k

    if (use_int8_w8a16 or use_int4_w4a16) and (
        block_shape is not None and block_shape[1] > 0
    ):
        assert B_bias is None

        use_moe_wna16_cuda = should_moe_wna16_use_cuda(
            num_valid_tokens=num_tokens,
            group_size=block_shape[1],
            num_experts=B.size(0),
            bit=4 if use_int4_w4a16 else 8,
        )

        if use_moe_wna16_cuda:
            invoke_fused_moe_wna16_cuda_kernel(
                A,
                B,
                C,
                B_scale,
                B_zp,
                topk_weights,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                mul_routed_weight,
                top_k,
                config,
                block_shape,
            )
            return
        invoke_fused_moe_wna16_triton_kernel(
            A,
            B,
            C,
            B_scale,
            B_zp,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            mul_routed_weight,
            top_k,
            config,
            compute_type,
            use_int8_w8a16,
            use_int4_w4a16,
            block_shape,
        )

    else:
        invoke_fused_moe_triton_kernel(
            A,
            B,
            C,
            A_scale,
            B_scale,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            mul_routed_weight,
            top_k,
            config,
            compute_type,
            use_fp8_w8a8,
            use_int8_w8a8,
            use_int8_w8a16,
            use_int4_w4a16,
            per_channel_quant,
            block_shape,
            B_bias,
        )


@triton.jit
def compute_identity_kernel(
    top_k: int,
    hidden_states_ptr: tl.tensor,
    expert_scales_ptr: tl.tensor,
    num_tokens: int,
    output_ptr: tl.tensor,
    hidden_dim: int,
    scales_stride: int,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    pid = tl.program_id(0)

    batch_id = pid // (hidden_dim // BLOCK_SIZE)
    dim_offset = pid % (hidden_dim // BLOCK_SIZE) * BLOCK_SIZE

    if batch_id >= num_tokens or dim_offset >= hidden_dim:
        return

    h = tl.load(
        hidden_states_ptr
        + batch_id * hidden_dim
        + dim_offset
        + tl.arange(0, BLOCK_SIZE),
        mask=(dim_offset + tl.arange(0, BLOCK_SIZE)) < hidden_dim,
    )

    result = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(top_k):
        scale = tl.load(expert_scales_ptr + batch_id * scales_stride + i)
        result += h * scale

    tl.store(
        output_ptr + batch_id * hidden_dim + dim_offset + tl.arange(0, BLOCK_SIZE),
        result,
        mask=(dim_offset + tl.arange(0, BLOCK_SIZE)) < hidden_dim,
    )


def zero_experts_compute_triton(
    expert_indices: torch.Tensor,
    expert_scales: torch.Tensor,
    num_experts: int,
    zero_expert_type: str,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    N = expert_indices.numel()
    top_k = expert_indices.size(-1)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)

    if zero_expert_type == "identity":
        zero_expert_mask = expert_indices < num_experts
        zero_expert_scales = expert_scales.clone()
        zero_expert_scales[zero_expert_mask] = 0.0

    normal_expert_mask = expert_indices >= num_experts
    expert_indices[normal_expert_mask] = 0
    expert_scales[normal_expert_mask] = 0.0

    output = torch.zeros_like(hidden_states).to(hidden_states.device)
    hidden_dim = hidden_states.size(-1)
    num_tokens = hidden_states.size(0)

    grid = lambda meta: (num_tokens * (hidden_dim // meta["BLOCK_SIZE"]),)
    compute_identity_kernel[grid](
        top_k,
        hidden_states,
        zero_expert_scales,
        num_tokens,
        output,
        hidden_dim,
        zero_expert_scales.stride(0),
        BLOCK_SIZE=256,
    )

    return output


# Adapted from: https://github.com/sgl-project/sglang/pull/2628
def get_config_file_name(
    E: int, N: int, dtype: str | None, block_shape: list[int] | None = None
) -> str:
    device_name = current_platform.get_device_name().replace(" ", "_")
    # Set device_name to H200 if a device from the H200 family is detected
    if "H200" in device_name.split("_"):
        device_name = "NVIDIA_H200"
    dtype_selector = "" if not dtype else f",dtype={dtype}"
    block_shape_selector = (
        "" if not block_shape or not all(block_shape) else
        f",block_shape={block_shape}"
    ).replace(" ", "")
    gfx906_names = [ "Instinct_MI50", "Instinct_MI60", "AMD_Radeon_Graphics", "Vega_20" ]
    if any(s in device_name for s in gfx906_names):
        device_name = "AMD_GFX906"
    return f"E={E},N={N},device_name={device_name}{dtype_selector}{block_shape_selector}.json"  # noqa: E501


# Adapted from: https://github.com/sgl-project/sglang/pull/2628
@functools.lru_cache
def get_moe_configs(
    E: int,
    N: int,
    dtype: str | None,
    block_n: int | None = None,
    block_k: int | None = None,
) -> dict[int, Any] | None:
    """
    Return optimized configurations for the fused MoE kernel.

    The return value will be a dictionary that maps an irregular grid of
    batch sizes to configurations of the fused_moe kernel. To evaluate the
    kernel on a given batch size bs, the closest batch size in the grid should
    be picked and the associated configuration chosen to invoke the kernel.
    """

    # Avoid optimizing for the batch invariant case. Use default config
    if vllm_is_batch_invariant():
        return None

    # First look up if an optimized configuration is available in the configs
    # directory
    block_shape = [block_n, block_k] if block_n and block_k else None
    json_file_name = get_config_file_name(E, N, dtype, block_shape)

    config_file_paths = []

    # note that we prioritize user defined config
    user_defined_config_folder = envs.VLLM_TUNED_CONFIG_FOLDER
    if user_defined_config_folder is not None:
        user_defined_config_file_path = os.path.join(
            user_defined_config_folder, json_file_name
        )
        config_file_paths.append(user_defined_config_file_path)

    default_config_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "configs", json_file_name
    )
    config_file_paths.append(default_config_file_path)

    for config_file_path in config_file_paths:
        if os.path.exists(config_file_path):
            with open(config_file_path) as f:
                logger.info_once(
                    "Using configuration from %s for MoE layer.",
                    config_file_path,
                    scope="global",
                )
                # If a configuration has been found, return it
                tuned_config = json.load(f)
                # Delete triton_version from tuned_config
                tuned_config.pop("triton_version", None)
                return {int(key): val for key, val in tuned_config.items()}

    # If no optimized configuration is available, we will use the default
    # configuration
    logger.warning_once(
        "Using default MoE config. Performance might be sub-optimal! "
        "Config file not found at %s",
        ", ".join(config_file_paths),
        scope="local",
    )
    return None


def get_moe_wna16_block_config(
    config: dict[str, int],
    use_moe_wna16_cuda: bool,
    num_valid_tokens: int,
    size_k: int,
    size_n: int,
    num_experts: int,
    group_size: int,
    real_top_k: int,
    block_size_m: int,
):
    if "BLOCK_SIZE_N" in config and "BLOCK_SIZE_K" in config:
        # optimal block config is set
        return {}
    if not use_moe_wna16_cuda:
        # triton moe wna16 kernel
        if num_valid_tokens // real_top_k == 1:
            # if bs=1, use a smaller BLOCK_SIZE_N
            block_size_n = 32
            block_size_k = 64
        else:
            block_size_n = 64
            block_size_k = 32
        while block_size_k > 16 and size_k % block_size_k != 0:
            block_size_k //= 2
        if size_k % block_size_k != 0:
            block_size_k = 1 << (size_k.bit_length() - 1)
            while block_size_k > size_k or size_k % block_size_k != 0:
                block_size_k //= 2
        return {"BLOCK_SIZE_N": block_size_n, "BLOCK_SIZE_K": block_size_k}
            
    else:
        # cuda moe wna16 kernel
        # set default block_size 128, and increase them when num_blocks
        # is too large.
        block_size_n = 128
        block_size_k = 128
        if block_size_k <= group_size:
            block_size_k = group_size

        num_n_blocks = size_k // block_size_k
        num_k_blocks = size_n // block_size_k
        num_m_blocks = (
            num_valid_tokens + block_size_m - 1
        ) / block_size_m + num_experts
        if num_valid_tokens // real_top_k <= block_size_m:
            num_m_blocks = min(num_m_blocks, num_valid_tokens)
        num_blocks = num_m_blocks * num_n_blocks * num_k_blocks

        if size_k % 256 == 0 and num_blocks >= 256 and block_size_k < 256:
            block_size_k = 256
            num_blocks = num_blocks // (256 // block_size_k)

        if (
            num_m_blocks <= 16
            and size_k % (block_size_k * 2) == 0
            and size_k % (block_size_k * 2) == 0
            and block_size_k <= 512
            and num_blocks >= 512
        ):
            block_size_k = block_size_k * 2
            num_blocks = num_blocks // 2

        if num_blocks > 1024:
            block_size_n = 256
            num_n_blocks = num_n_blocks // 2
            num_blocks = num_blocks // 2

        if size_n <= 1024 and num_blocks >= 1024:
            # The kernel performance got much better with BLOCK_SIZE_N=1024
            # when num_blocks is large, event when N is small.
            # Not sure why, maybe it force the CUDA SM process only one block
            # at the same time.
            block_size_n = 1024

        while block_size_k > group_size and size_k % block_size_k != 0:
            block_size_k //= 2
        if size_k % block_size_k != 0:
            block_size_k = 1 << (size_k.bit_length() - 1)
            while block_size_k > group_size and size_k % block_size_k != 0:
                block_size_k //= 2

        return {"BLOCK_SIZE_N": block_size_n, "BLOCK_SIZE_K": block_size_k}


def should_moe_wna16_use_cuda(
    num_valid_tokens: int, group_size: int, num_experts: int, bit: int
):
    return (
        current_platform.is_cuda_alike() 
        and bit == 4
        and group_size in [32, 64, 128]
        and num_valid_tokens / num_experts <= 6
    )


def get_default_config(
    M: int,
    E: int,
    N: int,
    K: int,
    topk: int,
    dtype: str | None,
    block_shape: list[int] | None = None,
) -> dict[str, int]:
    if vllm_is_batch_invariant():
        return {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
            "SPLIT_K": 1,
        }

    # num_stages can cause triton.runtime.errors.OutOfResources on ROCm.
    num_stages_rocm = 2

    if dtype == "fp8_w8a8" and block_shape is not None:
        # Block-wise quant: BLOCK_SIZE_N must be divisible by block_shape[0]
        # BLOCK_SIZE_K must be divisible by block_shape[1]
        # num_stages=3 can cause triton.runtime.errors.OutOfResources
        # on ROCm, set it to 2 instead.
        # gfx906: Optimize for MI50 based on benchmark results.
        if on_gfx906():
            if M <= 1:
                bm, gm, nw, ns = 16, 16, 4, 1
            elif M <= 8:
                bm, gm, nw, ns = 16, 1, 4, 1
            elif M <= 16:
                bm, gm, nw, ns = 16, 8, 4, 1
            elif M <= 64:
                bm, gm, nw, ns = 32, 4, 4, 1
            else:
                bm, gm, nw, ns = 32, 8, 4, 1
            config = {
                "BLOCK_SIZE_M": bm,
                "BLOCK_SIZE_N": block_shape[0],
                "BLOCK_SIZE_K": block_shape[1],
                "GROUP_SIZE_M": gm,
                "SPLIT_K": 1,
                "num_warps": nw,
                "num_stages": ns,
                }
        else:
            # Block-wise quant: tile sizes are constrained by block_shape.
            # Use a small M tile for decode-like batches where tokens are
            # spread thin across experts. Larger batches benefit from
            # GROUP_SIZE_M > 1 because the per-block scales add memory
            # traffic that benefits from L2 tile reuse.
            config = {
                "BLOCK_SIZE_M": 16 if M <= 64 else 64,
                "BLOCK_SIZE_N": block_shape[0],
                "BLOCK_SIZE_K": block_shape[1],
                "GROUP_SIZE_M": 1 if M <= 16 else 32,
                "SPLIT_K": 1,
                "num_warps": 4,
                "num_stages": 3 if not current_platform.is_rocm() else num_stages_rocm,
            }
    elif dtype in ["int4_w4a16", "int8_w8a16"] and block_shape is not None:
        # moe wna16 kernels
        # only set BLOCK_SIZE_M
        # BLOCK_SIZE_N and BLOCK_SIZE_K would be set later
        bit = 4 if dtype == "int4_w4a16" else 8
        use_moe_wna16_cuda = should_moe_wna16_use_cuda(M * topk, block_shape[1], E, bit)
        if use_moe_wna16_cuda:
            config = {"BLOCK_SIZE_M": min(16, M), "SPLIT_K": 1}
        elif M <= 20:
            config = {"BLOCK_SIZE_M": 16, "GROUP_SIZE_M": 1, "SPLIT_K": 1}
        elif M <= 40:
            config = {"BLOCK_SIZE_M": 32, "GROUP_SIZE_M": 1, "SPLIT_K": 1}
        else:
            config = {"BLOCK_SIZE_M": 64, "GROUP_SIZE_M": 1, "SPLIT_K": 1}
    else:
        # General defaults for bf16/fp16 and fp8 per-tensor.
        # Tile sizes scale with batch: small batches are memory-bound
        # (favor tall-K tiles), large batches are compute-bound (favor
        # large M/N tiles with more warps).
        if M <= 32:
            block_m = 16
        elif M <= 96:
            block_m = 32
        elif M <= 512:
            block_m = 64
        else:
            block_m = 128

        block_n = 64 if M <= 64 else 128

        # Small batches benefit from longer reduction (larger K tile),
        # while large batches prefer more output parallelism.
        # FP8 elements are half-width so larger K tiles are always cheap.
        block_k = 128 if dtype == "fp8_w8a8" or M <= 64 else 64

        # Grouping adjacent M-blocks lets them share weight tiles in L2.
        # Only helps when there are enough M-blocks per expert to group;
        # with many experts each one sees few tokens so grouping is useless.
        tokens_per_expert = M // max(E, 1)
        group_m = 16 if tokens_per_expert > 128 else 1

        # Large batches have enough blocks to saturate the GPU, so we
        # use more warps per block to increase arithmetic intensity.
        num_warps = 4 if M <= 128 else 8

        if current_platform.is_rocm():
            num_stages = num_stages_rocm
        elif M <= 32:
            num_stages = 4
        else:
            num_stages = 3

        config = {
            "BLOCK_SIZE_M": block_m,
            "BLOCK_SIZE_N": block_n,
            "BLOCK_SIZE_K": block_k,
            "GROUP_SIZE_M": group_m,
            "SPLIT_K": 1,
            "num_warps": num_warps,
            "num_stages": num_stages,
        }
    return config


def try_get_optimal_moe_config(
    w1_shape: tuple[int, ...],
    w2_shape: tuple[int, ...],
    top_k: int,
    dtype: str | None,
    M: int,
    block_shape: list[int] | None = None,
) -> dict[str, int]:
    from vllm.model_executor.layers.fused_moe import get_config

    override_config = get_config()
    if override_config:
        config = override_config
    else:
        # First try to load optimal config from the file
        E, _, N = w2_shape
        if dtype == "int4_w4a16":
            N = N * 2
        block_n = block_shape[0] if block_shape else 0
        block_k = block_shape[1] if block_shape else 0
        configs = get_moe_configs(E, N, dtype, block_n, block_k)

        if configs:
            # If an optimal configuration map has been found, look up the
            # optimal config
            config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
        else:
            # Else use the default config
            config = get_default_config(M, E, N, w1_shape[2], top_k, dtype, block_shape)
    return config


def inplace_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    ocp_mx_scheme: str | None = None,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    w1_zp: torch.Tensor | None = None,
    w2_zp: torch.Tensor | None = None,
    a1_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    block_shape: list[int] | None = None,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
) -> None:
    fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        True,
        activation,
        apply_router_weight_on_input,
        use_fp8_w8a8,
        use_int8_w8a8,
        use_int8_w8a16,
        use_int4_w4a16,
        ocp_mx_scheme,
        per_channel_quant,
        global_num_experts,
        expert_map,
        w1_scale,
        w2_scale,
        w1_zp,
        w2_zp,
        a1_scale,
        a2_scale,
        block_shape,
        w1_bias,
        w2_bias,
    )


def inplace_fused_experts_fake(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    ocp_mx_scheme: str | None = None,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    w1_zp: torch.Tensor | None = None,
    w2_zp: torch.Tensor | None = None,
    a1_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    block_shape: list[int] | None = None,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
) -> None:
    pass


direct_register_custom_op(
    op_name="inplace_fused_experts",
    op_func=inplace_fused_experts,
    mutates_args=["hidden_states"],
    fake_impl=inplace_fused_experts_fake,
)


def outplace_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    ocp_mx_scheme: str | None = None,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    w1_zp: torch.Tensor | None = None,
    w2_zp: torch.Tensor | None = None,
    a1_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    block_shape: list[int] | None = None,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        False,
        activation,
        apply_router_weight_on_input,
        use_fp8_w8a8,
        use_int8_w8a8,
        use_int8_w8a16,
        use_int4_w4a16,
        ocp_mx_scheme,
        per_channel_quant,
        global_num_experts,
        expert_map,
        w1_scale,
        w2_scale,
        w1_zp,
        w2_zp,
        a1_scale,
        a2_scale,
        block_shape,
        w1_bias,
        w2_bias,
    )


def outplace_fused_experts_fake(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    ocp_mx_scheme: str | None = None,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    w1_zp: torch.Tensor | None = None,
    w2_zp: torch.Tensor | None = None,
    a1_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    block_shape: list[int] | None = None,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="outplace_fused_experts",
    op_func=outplace_fused_experts,
    fake_impl=outplace_fused_experts_fake,
)


def torch_vllm_inplace_fused_experts(**kwargs) -> torch.Tensor:
    torch.ops.vllm.inplace_fused_experts(**kwargs)
    hidden_states = kwargs["hidden_states"]
    return hidden_states


def torch_vllm_outplace_fused_experts(**kwargs) -> torch.Tensor:
    return torch.ops.vllm.outplace_fused_experts(**kwargs)


def dispatch_fused_experts_func(inplace: bool) -> Callable[..., torch.Tensor]:
    if inplace:
        return torch_vllm_inplace_fused_experts
    return torch_vllm_outplace_fused_experts


def _prepare_expert_assignment(
    topk_ids: torch.Tensor,
    config: dict[str, Any],
    num_tokens: int,
    top_k_num: int,
    global_num_experts: int,
    expert_map: torch.Tensor | None,
    *,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    block_shape: list[int] | None = None,
    ignore_invalid_experts: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """Prepare expert assignments for the aligned and low-latency Triton paths."""
    # SPARSITY_FACTOR is a heuristic margin ensuring tokens_in_chunk * top_k
    # activates only a small fraction of total experts.
    # Skips moe_align_block_size and activates the `sorted_token_ids is None`
    # path of the fused_moe_kernel kernel.
    naive_block_assignment = (
        expert_map is None
        and num_tokens * top_k_num * 4 <= global_num_experts
        and not (
            (use_int8_w8a16 or use_int4_w4a16)
            and block_shape is not None
            and block_shape[1] > 0
        )
    )

    if naive_block_assignment:
        return (
            None,
            topk_ids.view(-1),
            torch.full(
                (1,),
                topk_ids.numel() * config["BLOCK_SIZE_M"],
                dtype=torch.int32,
                device=topk_ids.device,
            ),
        )

    return moe_align_block_size(
        topk_ids,
        config["BLOCK_SIZE_M"],
        global_num_experts,
        expert_map,
        ignore_invalid_experts=ignore_invalid_experts,
    )


# TODO (bnell): replace this with modular op.  Can get rid of inplace/outplace
# torch ops.
def fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: MoEActivation = MoEActivation.SILU,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    quant_config: FusedMoEQuantConfig | None = None,
) -> torch.Tensor:
    """Run fused MoE expert computation using Triton kernels."""
    if quant_config is None:
        quant_config = FUSED_MOE_UNQUANTIZED_CONFIG

    assert not inplace or not disable_inplace()

    return dispatch_fused_experts_func(inplace)(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=activation.value,
        apply_router_weight_on_input=apply_router_weight_on_input,
        use_fp8_w8a8=quant_config.use_fp8_w8a8,
        use_int8_w8a8=quant_config.use_int8_w8a8,
        use_int8_w8a16=quant_config.use_int8_w8a16,
        use_int4_w4a16=quant_config.use_int4_w4a16,
        ocp_mx_scheme=quant_config.ocp_mx_scheme,
        per_channel_quant=quant_config.per_act_token_quant,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
        w1_scale=quant_config.w1_scale,
        w2_scale=quant_config.w2_scale,
        w1_zp=quant_config.w1_zp,
        w2_zp=quant_config.w2_zp,
        a1_scale=quant_config.a1_scale,
        a2_scale=quant_config.a2_scale,
        block_shape=quant_config.block_shape,
        w1_bias=quant_config.w1_bias,
        w2_bias=quant_config.w2_bias,
    )


def _get_config_quant_dtype(
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    ocp_mx_scheme: str | None,
) -> None | torch.dtype | str:
    """
    Get the quantization type based on the quantization strategy flags.
    We don't have a quant_config at this point so we need to work backwards.
    A return type of None means no quantization is required because the
    input is unquantized or has been quantized prior to calling
    fused_experts_impl.
    """
    if use_fp8_w8a8:
        return torch.float8_e4m3fn
    elif use_int8_w8a8:
        return torch.int8
    elif ocp_mx_scheme == "w_mxfp4_a_mxfp4":
        return "mxfp4"
    elif ocp_mx_scheme in {"w_mxfp4_a_mxfp6_e3m2", "w_mxfp6_e3m2_a_mxfp6_e3m2"}:
        return "mxfp6_e3m2"
    elif ocp_mx_scheme in {"w_mxfp4_a_mxfp6_e2m3", "w_mxfp6_e2m3_a_mxfp6_e2m3"}:
        return "mxfp6_e2m3"
    elif ocp_mx_scheme in {"w_mxfp4", "w_mxfp6_e3m2", "w_mxfp6_e2m3"}:
        return torch.bfloat16
    elif ocp_mx_scheme in {"w_mxfp4_a_fp8", "w_mxfp6_e3m2_a_fp8", "w_mxfp6_e2m3_a_fp8"}:
        return torch.float8_e4m3fn

    return None


def fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    ocp_mx_scheme: str | None = None,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    w1_zp: torch.Tensor | None = None,
    w2_zp: torch.Tensor | None = None,
    a1_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    block_shape: list[int] | None = None,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    # Convert string activation to enum for internal use
    activation_enum = MoEActivation.from_str(activation)

    # Check constraints.
    if use_int4_w4a16:
        assert hidden_states.size(1) // 2 == w1.size(2), "Hidden size mismatch"
    elif ocp_mx_scheme is not None:
        if ocp_mx_scheme.startswith("w_mxfp4"):
            # 16bit activation and fp4x2 packed weight
            assert hidden_states.size(1) == w1.size(2) * 2, "hidden size mismatch"
        elif ocp_mx_scheme.startswith("w_mxfp6"):
            assert hidden_states.size(1) == (w1.size(2) * 4) // 3, (
                "hidden size mismatch"
            )
        else:
            raise NotImplementedError(f"Unsupported ocp_mx_scheme={ocp_mx_scheme}")
    else:
        assert hidden_states.size(1) == w1.size(2), (
            f"Hidden size mismatch {hidden_states.size(1)} != {w1.size(2)}"
        )

    assert topk_weights.size() == topk_ids.size(), "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
    assert w2.stride(-1) == 1, "Stride of last dimension must be 1"
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]

    num_tokens = hidden_states.size(0)
    E, N, _ = w1.size()
    K = w2.size(1)
    if global_num_experts == -1:
        global_num_experts = E
    top_k_num = topk_ids.size(1)
    # We execute the fused_moe kernel in chunks to circumvent this issue:
    # https://github.com/vllm-project/vllm/issues/5938
    CHUNK_SIZE = envs.VLLM_FUSED_MOE_CHUNK_SIZE
    M = min(num_tokens, CHUNK_SIZE)

    config_dtype = _get_config_dtype_str(
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        ocp_mx_scheme=ocp_mx_scheme,
        dtype=hidden_states.dtype,
    )

    # Note: for use_int8_w8a16 or use_int4_w4a16, the activations are
    # quantized prior to calling fused_experts.
    quant_dtype = _get_config_quant_dtype(
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        ocp_mx_scheme=ocp_mx_scheme,
    )

    get_config_func = functools.partial(
        try_get_optimal_moe_config,
        w1.size(),
        w2.size(),
        top_k_num,
        config_dtype,
        block_shape=block_shape,
    )

    config = get_config_func(M)

    # We can reuse the memory between these because by the time we need
    # cache3, we're done with cache1
    cache13 = torch.empty(
        M * top_k_num * max(N, K),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache1 = cache13[: M * top_k_num * N].view(M, top_k_num, N)
    intermediate_cache3 = cache13[: M * top_k_num * K].view(M, top_k_num, K)

    # This needs separate memory since it's used concurrently with cache1
    activation_out_dim = mk.FusedMoEExpertsModular.adjust_N_for_activation(
        N, activation_enum
    )
    intermediate_cache2 = torch.empty(
        (M * top_k_num, activation_out_dim),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    if hidden_states.dtype == torch.bfloat16:
        compute_type = tl.bfloat16
    elif hidden_states.dtype == torch.float16:
        compute_type = tl.float16
    elif hidden_states.dtype == torch.float32:
        compute_type = tl.float32
    else:
        raise ValueError(f"Unsupported compute_type: {hidden_states.dtype}")

    out_hidden_states = hidden_states if inplace else torch.empty_like(hidden_states)

    if ocp_mx_scheme is not None:
        # TODO: On platforms for which `current_platform.supports_mx()` is True
        # and for which we have a native OCP mx fused MOE kernel,
        # this dequantization step should not be done.
        if ocp_mx_scheme.startswith("w_mxfp4"):
            # Weight has to be dequantized for mxfp4 emulation.
            w1 = dequant_mxfp4(w1, w1_scale, hidden_states.dtype)
            w1_scale = None
            w2 = dequant_mxfp4(w2, w2_scale, hidden_states.dtype)
            w2_scale = None
        elif ocp_mx_scheme.startswith("w_mxfp6_e3m2"):
            w1 = dequant_mxfp6(
                w1, w1_scale, quant_dtype="fp6_e3m2", float_dtype=hidden_states.dtype
            )
            w1_scale = None
            w2 = dequant_mxfp6(
                w2, w2_scale, quant_dtype="fp6_e3m2", float_dtype=hidden_states.dtype
            )
            w2_scale = None
        elif ocp_mx_scheme.startswith("w_mxfp6_e2m3"):
            w1 = dequant_mxfp6(
                w1, w1_scale, quant_dtype="fp6_e2m3", float_dtype=hidden_states.dtype
            )
            w1_scale = None
            w2 = dequant_mxfp6(
                w2, w2_scale, quant_dtype="fp6_e2m3", float_dtype=hidden_states.dtype
            )
            w2_scale = None
        else:
            raise NotImplementedError(f"Unsupported ocp_mx_scheme={ocp_mx_scheme}")

    for chunk in range((num_tokens // CHUNK_SIZE) + 1):
        begin_chunk_idx, end_chunk_idx = (
            chunk * CHUNK_SIZE,
            min((chunk + 1) * CHUNK_SIZE, num_tokens),
        )
        curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
        tokens_in_chunk, _ = curr_hidden_states.size()

        if tokens_in_chunk == 0:
            break

        if tokens_in_chunk < CHUNK_SIZE and chunk > 0:
            # Adjust the intermediate cache size and config for the last
            # chunk. Note that in most cases we only have one chunk
            # so the cache size and config are already set correctly and
            # do not need to be adjusted.
            intermediate_cache1 = intermediate_cache1[:tokens_in_chunk]
            intermediate_cache2 = intermediate_cache2[
                : tokens_in_chunk * topk_ids.size(1)
            ]
            intermediate_cache3 = intermediate_cache3[:tokens_in_chunk]
            config = get_config_func(tokens_in_chunk)

        curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
        curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]
        qcurr_hidden_states, a1q_scale = moe_kernel_quantize_input(
            A=curr_hidden_states,
            A_scale=a1_scale,
            quant_dtype=quant_dtype,
            per_act_token_quant=per_channel_quant,
            block_shape=block_shape,
            ocp_mx_scheme=ocp_mx_scheme,
        )

        sorted_token_ids, expert_ids, num_tokens_post_padded = (
            _prepare_expert_assignment(
                curr_topk_ids,
                config,
                tokens_in_chunk,
                top_k_num,
                global_num_experts,
                expert_map,
                use_int8_w8a16=use_int8_w8a16,
                use_int4_w4a16=use_int4_w4a16,
                block_shape=block_shape,
                ignore_invalid_experts=True,
            )
        )

        dispatch_fused_moe_kernel(
            qcurr_hidden_states,
            w1,
            intermediate_cache1,
            a1q_scale,
            w1_scale,
            w1_zp,
            curr_topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            apply_router_weight_on_input,
            top_k_num,
            config,
            compute_type=compute_type,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            per_channel_quant=per_channel_quant,
            block_shape=block_shape,
            B_bias=w1_bias,
        )

        apply_moe_activation(
            activation_enum, intermediate_cache2, intermediate_cache1.view(-1, N)
        )

        qintermediate_cache2, a2q_scale = moe_kernel_quantize_input(
            A=intermediate_cache2,
            A_scale=a2_scale,
            quant_dtype=quant_dtype,
            per_act_token_quant=per_channel_quant,
            block_shape=block_shape,
            ocp_mx_scheme=ocp_mx_scheme,
        )

        if expert_map is not None:
            intermediate_cache3.zero_()

        dispatch_fused_moe_kernel(
            qintermediate_cache2,
            w2,
            intermediate_cache3,
            a2q_scale,
            w2_scale,
            w2_zp,
            curr_topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            not apply_router_weight_on_input,
            1,
            config,
            compute_type=compute_type,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            per_channel_quant=per_channel_quant,
            block_shape=block_shape,
            B_bias=w2_bias,
        )

        ops.moe_sum(
            intermediate_cache3.view(*intermediate_cache3.size()),
            out_hidden_states[begin_chunk_idx:end_chunk_idx],
        )

    return out_hidden_states


class TritonExperts(mk.FusedMoEExpertsModular):
    """Triton-based fused MoE expert implementation."""

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return current_platform.is_cuda_alike()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        p = current_platform
        if p.is_rocm():
            from vllm.platforms.rocm import on_gfx9

            is_rocm_on_gfx9 = on_gfx9()
            is_rocm_on_gfx906 = on_gfx906()
        else:
            is_rocm_on_gfx9 = False
            is_rocm_on_gfx906 = False

        device_supports_fp8 = is_rocm_on_gfx9 or is_rocm_on_gfx906 or (
            p.is_cuda() and p.has_device_capability((8, 9))
        )

        if not device_supports_fp8:
            return (weight_key, activation_key) == (None, None)

        SUPPORTED_W_A = [
            (None, None),
            (kFp8Static128BlockSym, kFp8Dynamic128Sym),
            (kFp8StaticChannelSym, kFp8DynamicTokenSym),
            (kFp8StaticTensorSym, kFp8DynamicTokenSym),
            (kFp8StaticTensorSym, kFp8StaticTensorSym),
            (kFp8StaticTensorSym, kFp8DynamicTensorSym),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in [
            MoEActivation.SILU,
            MoEActivation.GELU,
            MoEActivation.SWIGLUOAI,
            MoEActivation.SWIGLUSTEP,
            MoEActivation.SILU_NO_MUL,
            MoEActivation.GELU_NO_MUL,
            MoEActivation.RELU2_NO_MUL,
        ]

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return not moe_parallel_config.use_fi_all2allv_kernels

    def supports_chunking(self) -> bool:
        return True

    def supports_expert_map(self) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        workspace1 = (M, topk, max(activation_out_dim, K))
        workspace2 = (M, topk, max(N, K))
        output = (M, K)
        return (workspace1, workspace2, output)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        # ── GFX906 FP8: bitwise Triton dequant path ────────────────────
        _gfx906_fp8 = (self.quant_config.use_fp8_w8a8
                        and self.block_shape is not None
                        and on_gfx906())
        if _gfx906_fp8:
            logger.info_once(
                "GFX906: bitwise FP8 dequant + FP16 GEMM path active",
                scope="local",
            )

            # hidden_states must be FP16 (fp8 quantized avoided in prepare_finalize.py)
            #assert hidden_states.dtype == torch.float16, "hidden_states must be FP16"
            #assert a1q_scale is None, "a1q_scale must be None"


            E_orig, num_tokens, N, K, top_k_num = self.moe_problem_size(
                hidden_states, w1, w2, topk_ids
            )

            if global_num_experts == -1:
                global_num_experts = E_orig

            config = try_get_optimal_moe_config(
                w1.size(),
                w2.size(),
                top_k_num,
                "fp8_w8a8",
                num_tokens,
                block_shape=self.block_shape,
            )

            compute_type = tl.float16

            intermediate_cache1 = _resize_cache(
                workspace2, (num_tokens, top_k_num, N))
            cache2_dim = self.adjust_N_for_activation(N, activation)
            intermediate_cache2 = _resize_cache(
                workspace13, (num_tokens * top_k_num, cache2_dim))
            intermediate_cache3 = _resize_cache(
                workspace2, (num_tokens, top_k_num, K))

            sorted_token_ids, expert_ids, num_tokens_post_padded = (
                moe_align_block_size(
                    topk_ids,
                    config["BLOCK_SIZE_M"],
                    E_orig,
                    expert_map,
                )
            )

            # First GEMM: hidden * w1
            invoke_fused_moe_triton_kernel(
                hidden_states,           # FP16
                w1.view(torch.uint8),    # FP8
                intermediate_cache1,
                None,          # a_scale
                self.w1_scale, # b_scale
                None,          # topk_weights
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                False,         # mul_routed_weights
                top_k_num,
                config,
                compute_type=compute_type,
                use_fp8_w8a8=self.quant_config.use_fp8_w8a8,
                use_int8_w8a8=self.quant_config.use_int8_w8a8,
                use_int8_w8a16=self.quant_config.use_int8_w8a16,
                use_int4_w4a16=self.quant_config.use_int4_w4a16,
                per_channel_quant=self.per_act_token_quant,
                block_shape=self.block_shape,
                B_bias=self.w1_bias,
            )

            self.activation(
                activation, intermediate_cache2,
                intermediate_cache1.view(-1, N))

            assert intermediate_cache2.dtype == torch.float16, "intermediate_cache2 must be FP16"

            # Second GEMM: intermediate * w2
            invoke_fused_moe_triton_kernel(
                intermediate_cache2,   # FP16
                w2.view(torch.uint8),  # FP8
                intermediate_cache3,
                None,                  # a_scale
                self.w2_scale,         # b_scale
                topk_weights,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                not apply_router_weight_on_input,
                1,
                config,
                compute_type=compute_type,
                use_fp8_w8a8=self.quant_config.use_fp8_w8a8,
                use_int8_w8a8=self.quant_config.use_int8_w8a8,
                use_int8_w8a16=self.quant_config.use_int8_w8a16,
                use_int4_w4a16=self.quant_config.use_int4_w4a16,
                per_channel_quant=self.per_act_token_quant,
                block_shape=self.block_shape,
                B_bias=self.w2_bias,
            )

            self.moe_sum(intermediate_cache3, output)
            return

        # ── Standard (non-gfx906) path ────────────────────────────────
        # Check constraints.
        if self.quant_config.use_int4_w4a16:
            assert hidden_states.size(-1) // 2 == w1.size(2), "Hidden size mismatch"
        else:
            assert hidden_states.size(-1) == w1.size(2), (
                f"Hidden size mismatch {hidden_states.size(-1)} != {w1.size(2)}"
            )

        assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
        assert hidden_states.dim() == 2
        assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
        assert w2.stride(-1) == 1, "Stride of last dimension must be 1"
        assert hidden_states.dtype in [
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
        ]

        E, num_tokens, N, K, top_k_num = self.moe_problem_size(
            hidden_states, w1, w2, topk_ids
        )

        if global_num_experts == -1:
            global_num_experts = E

        config = try_get_optimal_moe_config(
            w1.size(),
            w2.size(),
            top_k_num,
            self.quant_config.config_name(hidden_states.dtype),
            num_tokens,
            block_shape=self.block_shape,
        )

        if hidden_states.dtype == torch.bfloat16:
            compute_type = tl.bfloat16
        elif hidden_states.dtype == torch.float16:
            compute_type = tl.float16
        elif hidden_states.dtype == torch.float32:
            compute_type = tl.float32
        elif (
            hidden_states.dtype == torch.float8_e4m3fn
            or hidden_states.dtype == torch.float8_e4m3fnuz
        ):
            compute_type = tl.bfloat16
        else:
            raise ValueError(f"Unsupported compute_type: {hidden_states.dtype}")

        # Note that the output tensor might be in workspace1
        intermediate_cache1 = _resize_cache(workspace2, (num_tokens, top_k_num, N))
        cache2_dim = self.adjust_N_for_activation(N, activation)
        intermediate_cache2 = _resize_cache(
            workspace13, (num_tokens * top_k_num, cache2_dim)
        )
        intermediate_cache3 = _resize_cache(workspace2, (num_tokens, top_k_num, K))

        sorted_token_ids, expert_ids, num_tokens_post_padded = (
            _prepare_expert_assignment(
                topk_ids,
                config,
                num_tokens,
                top_k_num,
                global_num_experts,
                expert_map,
                use_int8_w8a16=self.quant_config.use_int8_w8a16,
                use_int4_w4a16=self.quant_config.use_int4_w4a16,
                block_shape=self.block_shape,
            )
        )

        invoke_fused_moe_triton_kernel(
            hidden_states,
            w1,
            intermediate_cache1,
            a1q_scale,
            self.w1_scale,
            None,  # topk_weights
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            False,  # mul_routed_weights
            top_k_num,
            config,
            compute_type=compute_type,
            use_fp8_w8a8=self.quant_config.use_fp8_w8a8,
            use_int8_w8a8=self.quant_config.use_int8_w8a8,
            use_int8_w8a16=self.quant_config.use_int8_w8a16,
            use_int4_w4a16=self.quant_config.use_int4_w4a16,
            per_channel_quant=self.per_act_token_quant,
            block_shape=self.block_shape,
            B_bias=self.w1_bias,
        )

        self.activation(
            activation, intermediate_cache2, intermediate_cache1.view(-1, N)
        )

        a2q_scale: torch.Tensor | None = None

        qintermediate_cache2, a2q_scale = moe_kernel_quantize_input(
            intermediate_cache2,
            a2_scale,
            self.quant_dtype,
            self.per_act_token_quant,
            self.block_shape,
        )

        invoke_fused_moe_triton_kernel(
            qintermediate_cache2,
            w2,
            intermediate_cache3,
            a2q_scale,
            self.w2_scale,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            not apply_router_weight_on_input,
            1,
            config,
            compute_type=compute_type,
            use_fp8_w8a8=self.quant_config.use_fp8_w8a8,
            use_int8_w8a8=self.quant_config.use_int8_w8a8,
            use_int8_w8a16=self.quant_config.use_int8_w8a16,
            use_int4_w4a16=self.quant_config.use_int4_w4a16,
            per_channel_quant=self.per_act_token_quant,
            block_shape=self.block_shape,
            B_bias=self.w2_bias,
        )

        # separate function is required for MoE + LoRA
        self.moe_sum(intermediate_cache3, output)

    def moe_sum(self, input: torch.Tensor, output: torch.Tensor) -> None:
        ops.moe_sum(input, output)


class TritonWNA16Experts(TritonExperts):
    @staticmethod
    def _supports_current_device() -> bool:
        raise NotImplementedError(
            "TritonWNA16Experts is not yet used by an Oracle. "
            "This method should not be called."
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        raise NotImplementedError(
            "TritonWNA16Experts is not yet used by an Oracle. "
            "This method should not be called."
        )

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        raise NotImplementedError(
            "TritonWNA16Experts is not yet used by an Oracle. "
            "This method should not be called."
        )

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        raise NotImplementedError(
            "TritonWNA16Experts is not yet used by an Oracle. "
            "This method should not be called."
        )

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        raise NotImplementedError(
            "TritonWNA16Experts is not yet used by an Oracle. "
            "This method should not be called."
        )

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        # Check constraints.
        if self.quant_config.use_int4_w4a16:
            assert hidden_states.size(-1) // 2 == w1.size(2), "Hidden size mismatch"
        else:
            assert hidden_states.size(-1) == w1.size(2), (
                f"Hidden size mismatch {hidden_states.size(-1)} != {w1.size(2)}"
            )

        assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
        assert hidden_states.dim() == 2
        assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
        assert w2.stride(-1) == 1, "Stride of last dimension must be 1"
        assert hidden_states.dtype in [
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
        ]

        E, num_tokens, N, K, top_k_num = self.moe_problem_size(
            hidden_states, w1, w2, topk_ids
        )

        if global_num_experts == -1:
            global_num_experts = E

        config = try_get_optimal_moe_config(
            w1.size(),
            w2.size(),
            top_k_num,
            self.quant_config.config_name(hidden_states.dtype),
            num_tokens,
            block_shape=self.block_shape,
        )

        if hidden_states.dtype == torch.bfloat16:
            compute_type = tl.bfloat16
        elif hidden_states.dtype == torch.float16:
            compute_type = tl.float16
        elif hidden_states.dtype == torch.float32:
            compute_type = tl.float32
        elif (
            hidden_states.dtype == torch.float8_e4m3fn
            or hidden_states.dtype == torch.float8_e4m3fnuz
        ):
            compute_type = tl.bfloat16
        else:
            raise ValueError(f"Unsupported compute_type: {hidden_states.dtype}")

        # Note that the output tensor might be in workspace1
        intermediate_cache1 = _resize_cache(workspace2, (num_tokens, top_k_num, N))
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        intermediate_cache2 = _resize_cache(
            workspace13, (num_tokens * top_k_num, activation_out_dim)
        )
        intermediate_cache3 = _resize_cache(workspace2, (num_tokens, top_k_num, K))

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, config["BLOCK_SIZE_M"], global_num_experts, expert_map
        )

        invoke_fused_moe_wna16_triton_kernel(
            hidden_states,
            w1,
            intermediate_cache1,
            self.w1_scale,
            self.quant_config.w1_zp,
            None,  # topk_weights
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            False,  # mul_routed_weights
            top_k_num,
            config,
            compute_type=compute_type,
            use_int8_w8a16=self.quant_config.use_int8_w8a16,
            use_int4_w4a16=self.quant_config.use_int4_w4a16,
            block_shape=self.block_shape,
        )

        self.activation(
            activation, intermediate_cache2, intermediate_cache1.view(-1, N)
        )

        a2q_scale: torch.Tensor | None = None

        qintermediate_cache2, a2q_scale = moe_kernel_quantize_input(
            intermediate_cache2,
            a2_scale,
            self.quant_dtype,
            self.per_act_token_quant,
            self.block_shape,
        )

        invoke_fused_moe_wna16_triton_kernel(
            qintermediate_cache2,
            w2,
            intermediate_cache3,
            self.w2_scale,
            self.quant_config.w2_zp,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            not apply_router_weight_on_input,
            1,
            config,
            compute_type=compute_type,
            use_int8_w8a16=self.quant_config.use_int8_w8a16,
            use_int4_w4a16=self.quant_config.use_int4_w4a16,
            block_shape=self.block_shape,
        )

        # separate function is required for MoE + LoRA
        self.moe_sum(intermediate_cache3, output)
HOTFIX_FUSED_MOE_PR39016

  cat <<'HOTFIX_ROCM' > "${bundle_dir}/rocm.py"
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from datetime import timedelta
from functools import cache, lru_cache, wraps
from typing import TYPE_CHECKING

import regex as re
import torch
from torch.distributed import PrefixStore, ProcessGroup
from torch.distributed.distributed_c10d import is_nccl_available

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.utils.torch_utils import cuda_device_count_stateless
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from .interface import DeviceCapability, Platform, PlatformEnum

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.attention.selector import AttentionSelectorConfig

logger = init_logger(__name__)

try:
    from amdsmi import (
        AmdSmiException,
        amdsmi_get_gpu_asic_info,
        amdsmi_get_processor_handles,
        amdsmi_init,
        amdsmi_shut_down,
        amdsmi_topo_get_link_type,
    )
except ImportError as e:
    logger.warning("Failed to import from amdsmi with %r", e)

try:
    import vllm._C  # noqa: F401
except ImportError as e:
    logger.warning("Failed to import from vllm._C with %r", e)

# import custom ops, trigger op registration
try:
    import vllm._rocm_C  # noqa: F401
except ImportError as e:
    logger.warning("Failed to import from vllm._rocm_C with %r", e)

# Models not supported by ROCm.
_ROCM_UNSUPPORTED_MODELS: list[str] = []

# Models partially supported by ROCm.
# Architecture -> Reason.
_ROCM_PARTIALLY_SUPPORTED_MODELS: dict[str, str] = {}
_ROCM_DEVICE_ID_NAME_MAP: dict[str, str] = {
    "0x74a0": "AMD_Instinct_MI300A",
    "0x74a1": "AMD_Instinct_MI300X",
    "0x74b5": "AMD_Instinct_MI300X",  # MI300X VF
    "0x74a2": "AMD_Instinct_MI308X",
    "0x74a5": "AMD_Instinct_MI325X",
    "0x74b9": "AMD_Instinct_MI325X",  # MI325X VF
    "0x74a9": "AMD_Instinct_MI300X_HF",
    "0x74bd": "AMD_Instinct_MI300X_HF",
    "0x744c": "AMD_Radeon_RX7900XTX",
}


def _sync_hip_cuda_env_vars():
    """Ensure HIP_VISIBLE_DEVICES and CUDA_VISIBLE_DEVICES are consistent.
    Treats empty string as unset. Raises on genuine conflicts."""
    hip_val = os.environ.get("HIP_VISIBLE_DEVICES") or None
    cuda_val = os.environ.get("CUDA_VISIBLE_DEVICES") or None

    if hip_val is not None and cuda_val is not None:
        if hip_val != cuda_val:
            raise ValueError(
                f"Inconsistent GPU visibility env vars: "
                f"HIP_VISIBLE_DEVICES='{hip_val}' vs "
                f"CUDA_VISIBLE_DEVICES='{cuda_val}'. "
                f"Please set only one, or ensure they match."
            )
    elif hip_val is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = hip_val
    elif cuda_val is not None:
        os.environ["HIP_VISIBLE_DEVICES"] = cuda_val


# Sync at import time - catches misconfigurations from process start.
_sync_hip_cuda_env_vars()


def _set_rocm_nccl_workarounds():
    """Disable NCCL watchdog/monitoring threads on ROCm to prevent
    hipErrorCapturedEvent during CUDA graph capture with tensor parallelism.

    The PyTorch NCCL watchdog thread queries HIP events via hipEventQuery,
    which throws hipErrorCapturedEvent when events belong to a capturing
    stream. TORCH_NCCL_BLOCKING_WAIT=1 is the only way to prevent the
    watchdog from starting (TORCH_NCCL_ASYNC_ERROR_HANDLING=0 only changes
    the error handling mode but does NOT stop the watchdog).
    (Workadounds implemented during Qwen 3.5 27b AWQ MTP+TP2 testing)
    """
    os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
    os.environ.setdefault("TORCH_NCCL_ENABLE_MONITORING", "0")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "0")
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "0")


_set_rocm_nccl_workarounds()

# AMDSMI utils
# Note that NVML is not affected by `{CUDA/HIP}_VISIBLE_DEVICES`,
# all the related functions work on real physical device ids.
# the major benefit of using AMDSMI is that it will not initialize CUDA


def with_amdsmi_context(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        amdsmi_init()
        try:
            return fn(*args, **kwargs)
        finally:
            amdsmi_shut_down()

    return wrapper


# Known amdsmi target_graphics_version quirks.
# Some ROCm versions (e.g. 6.3.4) return non-standard names like
# "gfx9006" instead of "gfx906".  Map them to canonical names here.
_AMDSMI_GFX_NORMALIZATION: dict[str, str] = {
    "gfx9006": "gfx906",
    "gfx8003": "gfx906",
}


@with_amdsmi_context
def _query_gcn_arch_from_amdsmi() -> str:
    """Query GCN arch from amdsmi. Raises if not available."""
    handles = amdsmi_get_processor_handles()
    if handles:
        asic_info = amdsmi_get_gpu_asic_info(handles[0])
        # Use target_graphics_version which contains the gfx name
        # e.g., 'gfx942' for MI300X/MI325X
        target_gfx = asic_info.get("target_graphics_version", "")
        if target_gfx:
            normalized = _AMDSMI_GFX_NORMALIZATION.get(target_gfx, target_gfx)
            if normalized != target_gfx:
                logger.warning(
                    "amdsmi returned non-standard GCN arch '%s', "
                    "normalizing to '%s'.",
                    target_gfx,
                    normalized,
                )
            return normalized
    raise RuntimeError("amdsmi did not return valid GCN arch")


def _get_gcn_arch() -> str:
    """
    Get GCN arch via amdsmi (no CUDA init), fallback to torch.cuda.
    Called once at module level; result stored in _GCN_ARCH.
    """
    try:
        return _query_gcn_arch_from_amdsmi()
    except Exception as e:
        logger.debug("Failed to get GCN arch via amdsmi: %s", e)
        logger.warning_once(
            "Failed to get GCN arch via amdsmi, falling back to torch.cuda. "
            "This will initialize CUDA and may cause "
            "issues if CUDA_VISIBLE_DEVICES is not set yet."
        )
    # Ultimate fallback: use torch.cuda (will initialize CUDA)
    return torch.cuda.get_device_properties("cuda").gcnArchName


# Resolve once at module load. Uses amdsmi (no CUDA init) so Ray workers
# can still set CUDA_VISIBLE_DEVICES after import.
# These are plain Python bools — fully torch.compile/Dynamo safe.
_GCN_ARCH = _get_gcn_arch()

_ON_GFX1X = any(arch in _GCN_ARCH for arch in ["gfx11", "gfx12"])
_ON_MI3XX = any(arch in _GCN_ARCH for arch in ["gfx942", "gfx950"])
_ON_GFX9 = any(arch in _GCN_ARCH for arch in ["gfx90a", "gfx942", "gfx950"])
_ON_GFX942 = "gfx942" in _GCN_ARCH
_ON_GFX950 = "gfx950" in _GCN_ARCH
_ON_GFX906 = "gfx906" in _GCN_ARCH


def _capability_from_gcn_arch(gcn_arch: str) -> tuple[int, int] | None:
    """
    Parse (major, minor) from a GCN arch string, mirroring how
    HIP derives hipDeviceProp_t.major / .minor.

    Format: gfx<MAJOR><MINOR><STEPPING>
      - 1-digit major  (gfx9xx):  "gfx" + M + m + stepping
      - 2-digit major  (gfx1xxx): "gfx" + MM + m + stepping

    Examples:
      gfx90a  -> (9, 0)    gfx942  -> (9, 4)    gfx950 -> (9, 5)
      gfx1100 -> (11, 0)   gfx1101 -> (11, 0)   gfx1200 -> (12, 0)

    Returns None only when the string is not gfx-prefixed at all
    (i.e. not a ROCm arch string). Raises on any string that looks
    like a GCN arch but does not match a known layout.
    """
    m = re.match(r"gfx(\d+)", gcn_arch)
    if not m:
        # Not a gfx string at all — caller should fall back to torch.cuda
        return None

    digits = m.group(1)
    n = len(digits)

    if n < 2:
        raise ValueError(
            f"GCN arch '{gcn_arch}' has too few digits ({n}) after 'gfx' "
            f"to derive a (major, minor) capability. "
            f"Please file a vLLM issue with your GPU model."
        )

    if n in (2, 3):
        # 1-digit major: gfx9 family
        # len 2: major + minor          (e.g. gfx90 from gfx90a)
        # len 3: major + minor + step   (e.g. gfx942)
        major = int(digits[0])
        minor = int(digits[1])
    elif n == 4:
        # 2-digit major: gfx10xx, gfx11xx, gfx12xx
        # major(2) + minor(1) + stepping(1)
        major = int(digits[:2])
        minor = int(digits[2])
    elif n >= 5:
        raise ValueError(
            f"GCN arch '{gcn_arch}' has {n} digits after 'gfx', which "
            f"exceeds the known 4-digit layout (MMms). Cannot determine "
            f"major/minor split unambiguously. "
            f"Please file a vLLM issue with your GPU model."
        )

    if major < 9:
        raise ValueError(
            f"Parsed unknown ROCm architecture from GCN arch '{gcn_arch}': "
            f"major={major}, minor={minor}. "
            f"Major version < 9 is not expected for any supported AMD GPU. "
            f"Please file a vLLM issue with your GPU model."
        )

    if major > 12:
        raise ValueError(
            f"Parsed unknown ROCm architecture from GCN arch '{gcn_arch}': "
            f"major={major}, minor={minor}. "
            f"Major version > 12 is beyond currently known AMD generations. "
            f"Please file a vLLM issue with your GPU model so support "
            f"can be added."
        )

    return (major, minor)


def on_gfx1x() -> bool:
    return _ON_GFX1X


def on_mi3xx() -> bool:
    return _ON_MI3XX


def on_gfx906() -> bool:
    return _ON_GFX906


def on_gfx9() -> bool:
    return _ON_GFX9


def on_gfx942() -> bool:
    return _ON_GFX942


def on_gfx950() -> bool:
    return _ON_GFX950


@cache
def use_rocm_custom_paged_attention(
    qtype: torch.dtype,
    head_size: int,
    block_size: int,
    gqa_ratio: int,
    max_seq_len: int,
    sliding_window: int,
    kv_cache_dtype: str,
    alibi_slopes: torch.Tensor | None = None,
    sinks: torch.Tensor | None = None,
) -> bool:
    # custom paged attn always supported on V0. On V1, requires sliding window
    # disabled due to observed numerical discrepancy.
    if _ON_GFX9:
        return (
            (sliding_window == 0 or sliding_window == (-1, -1))
            and (qtype == torch.half or qtype == torch.bfloat16)
            and (head_size == 64 or head_size == 128)
            and (block_size == 16 or block_size == 32)
            and (gqa_ratio >= 1 and gqa_ratio <= 16)
            and max_seq_len <= 128 * 1024
            and (envs.VLLM_ROCM_CUSTOM_PAGED_ATTN)
            and sinks is None
        )

    else:
        return (
            _ON_GFX1X
            and (sliding_window == 0 or sliding_window == (-1, -1))
            and (qtype == torch.half or qtype == torch.bfloat16)
            and head_size == 128
            and block_size == 16
            and (gqa_ratio >= 3 and gqa_ratio <= 16)
            and max_seq_len <= 128 * 1024
            and alibi_slopes is None
            and kv_cache_dtype == "auto"
            and envs.VLLM_ROCM_CUSTOM_PAGED_ATTN
            and sinks is None
        )


@cache
def flash_attn_triton_available() -> bool:
    try:
        from importlib.util import find_spec

        if find_spec("flash_attn") is None:
            return False
        if find_spec("flash_attn.flash_attn_triton_amd") is None:
            return False
        if os.environ.get("FLASH_ATTENTION_TRITON_AMD_ENABLE") != "TRUE":
            logger.info_once(
                "Set FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE to enable "
                "Flash Attention Triton backend on RDNA or GFX906."
            )
            return False
        return True
    except ImportError:
        return False


def _get_backend_priorities(
    use_mla: bool,
    use_sparse: bool,
) -> list[AttentionBackendEnum]:
    from vllm._aiter_ops import rocm_aiter_ops

    if use_sparse:
        return [AttentionBackendEnum.ROCM_AITER_MLA_SPARSE]

    if use_mla:
        if rocm_aiter_ops.is_mla_enabled():
            return [
                AttentionBackendEnum.ROCM_AITER_MLA,
                AttentionBackendEnum.TRITON_MLA,
                AttentionBackendEnum.ROCM_AITER_TRITON_MLA,
            ]
        else:
            return [
                AttentionBackendEnum.TRITON_MLA,
            ]

    backends = []

    # Priority 1: Check for AITER Unified Attention (must check before MHA)
    if envs.VLLM_ROCM_USE_AITER and envs.VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION:
        backends.append(AttentionBackendEnum.ROCM_AITER_UNIFIED_ATTN)

    # Priority 2: Check for AITER MHA (Flash Attention)
    if envs.VLLM_ROCM_USE_AITER and envs.VLLM_ROCM_USE_AITER_MHA:
        backends.append(AttentionBackendEnum.ROCM_AITER_FA)

    # Priority 3: Check for ROCM_ATTN (prefill-decode split)
    from vllm.config import get_current_vllm_config_or_none

    vllm_config = get_current_vllm_config_or_none()
    if (
        vllm_config is not None
        and vllm_config.attention_config.use_prefill_decode_attention
    ):
        backends.append(AttentionBackendEnum.ROCM_ATTN)

    # Default: Triton Unified Attention
    backends.append(AttentionBackendEnum.TRITON_ATTN)
    return backends


class RocmPlatform(Platform):
    _enum = PlatformEnum.ROCM
    device_name: str = "rocm"
    device_type: str = "cuda"
    dispatch_key: str = "CUDA"
    ray_device_key: str = "GPU"
    dist_backend: str = "nccl"
    # rocm shares the same device control env var as CUDA
    device_control_env_var: str = "CUDA_VISIBLE_DEVICES"
    ray_noset_device_env_vars: list[str] = [
        "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES",
    ]

    supported_quantization: list[str] = [
        "awq",
        "awq_marlin",  # will be overwritten with awq
        "gptq",
        "gptq_marlin",  # will be overwritten with gptq
        "fp8",
        "compressed-tensors",
        "fbgemm_fp8",
        "gguf",
        "quark",
        "ptpc_fp8",
        "mxfp4",
        "petit_nvfp4",
        "torchao",
        "inc",
        "bitsandbytes",
    ]

    @classmethod
    def import_kernels(cls) -> None:
        """Import ROCm-specific kernels."""
        super().import_kernels()

        import contextlib

        # Import ROCm-specific extension
        with contextlib.suppress(ImportError):
            import vllm._rocm_C  # noqa: F401

    @classmethod
    def get_valid_backends(
        cls,
        device_capability: DeviceCapability,
        attn_selector_config: "AttentionSelectorConfig",
        num_heads: int | None = None,
    ) -> tuple[
        list[tuple["AttentionBackendEnum", int]],
        dict["AttentionBackendEnum", list[str]],
    ]:
        valid_backends_priorities = []
        invalid_reasons = {}

        backend_priorities = _get_backend_priorities(
            attn_selector_config.use_mla,
            attn_selector_config.use_sparse,
        )
        for priority, backend in enumerate(backend_priorities):
            try:
                backend_class = backend.get_class()
                invalid_reasons_i = backend_class.validate_configuration(
                    device_capability=device_capability,
                    **attn_selector_config._asdict(),
                )
            except ImportError:
                invalid_reasons_i = ["ImportError"]
            if invalid_reasons_i:
                invalid_reasons[backend] = invalid_reasons_i
            else:
                valid_backends_priorities.append((backend, priority))

        return valid_backends_priorities, invalid_reasons

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: "AttentionBackendEnum",
        attn_selector_config: "AttentionSelectorConfig",
        num_heads: int | None = None,
    ) -> str:
        device_capability = cls.get_device_capability()
        assert device_capability is not None

        # First try checking just the selected backend, if there is one.
        if selected_backend is not None:
            try:
                backend_class = selected_backend.get_class()
                invalid_reasons = backend_class.validate_configuration(
                    device_capability=device_capability,
                    **attn_selector_config._asdict(),
                )
            except ImportError:
                invalid_reasons = ["ImportError"]
            if invalid_reasons:
                raise ValueError(
                    f"Selected backend {selected_backend} is not valid for "
                    f"this configuration. Reason: {invalid_reasons}"
                )
            else:
                logger.info("Using %s backend.", selected_backend)
                return selected_backend.get_path()

        # No selected backend or the selected backend is invalid,
        # so we try finding a valid backend.
        valid_backends_priorities, invalid_reasons = cls.get_valid_backends(
            device_capability=device_capability,
            attn_selector_config=attn_selector_config,
            num_heads=num_heads,
        )
        reasons_str = (
            "{"
            + ", ".join(
                f"{backend.name}: [{', '.join(reasons)}]"
                for backend, reasons in invalid_reasons.items()
            )
            + "}"
        )
        config_str = attn_selector_config.__repr__()
        logger.debug_once(
            f"Some attention backends are not valid for {cls.device_name} with "
            f"{config_str}. Reasons: {reasons_str}."
        )
        if len(valid_backends_priorities) == 0:
            raise ValueError(
                f"No valid attention backend found for {cls.device_name} "
                f"with {config_str}. Reasons: {reasons_str}."
            )

        # We have found some valid backends. Select the one with the
        # highest priority.
        sorted_indices = sorted(
            range(len(valid_backends_priorities)),
            key=lambda i: valid_backends_priorities[i][1],
        )
        selected_index = sorted_indices[0]
        selected_backend = valid_backends_priorities[selected_index][0]
        logger.info_once(
            "Using %s attention backend out of potential backends: %s.",
            selected_backend.name,
            "[" + ", ".join(f"'{b[0].name}'" for b in valid_backends_priorities) + "]",
            scope="local",
        )

        return selected_backend.get_path()

    @classmethod
    def get_supported_vit_attn_backends(cls) -> list["AttentionBackendEnum"]:
        return [
            AttentionBackendEnum.FLASH_ATTN,
            AttentionBackendEnum.ROCM_AITER_FA,
            AttentionBackendEnum.TRITON_ATTN,
            AttentionBackendEnum.TORCH_SDPA,
        ]

    @classmethod
    def get_vit_attn_backend(
        cls,
        head_size: int,
        dtype: torch.dtype,
        backend: "AttentionBackendEnum | None" = None,
    ) -> "AttentionBackendEnum":
        if backend is not None:
            assert backend in cls.get_supported_vit_attn_backends(), (
                f"Backend {backend} is not supported for vit attention. "
                f"Supported backends are: {cls.get_supported_vit_attn_backends()}"
            )
            logger.info_once(f"Using backend {backend} for vit attention")
            return backend

        from importlib.util import find_spec

        from vllm._aiter_ops import rocm_aiter_ops

        if rocm_aiter_ops.is_enabled() and on_gfx9():
            logger.info_once("Using AITER Flash Attention backend for ViT model.")
            return AttentionBackendEnum.ROCM_AITER_FA

        if (
            on_gfx9()
            and find_spec("flash_attn") is not None
            and (dtype == torch.float16 or dtype == torch.bfloat16)
        ):
            logger.info_once("Using Flash Attention backend for ViT model.")
            return AttentionBackendEnum.FLASH_ATTN

        # RDNA3/RDNA4 (gfx11xx/gfx12xx): Use Flash Attention Triton backend
        if (
            (on_gfx1x() or on_gfx906())
            and flash_attn_triton_available()
            and (dtype == torch.float16 or dtype == torch.bfloat16)
        ):
            logger.info_once(
                "Using Flash Attention (Triton backend) for ViT model on RDNA or GFX906."
            )
            return AttentionBackendEnum.FLASH_ATTN

        logger.info_once("Using Torch SDPA backend for ViT model.")
        return AttentionBackendEnum.TORCH_SDPA

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        """
        Set the device for the current platform.
        """
        torch.cuda.set_device(device)

    @classmethod
    @lru_cache(maxsize=8)
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability | None:
        cap = _capability_from_gcn_arch(_GCN_ARCH)
        if cap is not None:
            return DeviceCapability(major=cap[0], minor=cap[1])

        logger.warning_once(
            "Could not derive device capability from GCN arch '%s', "
            "falling back to torch.cuda (this will initialize CUDA).",
            _GCN_ARCH,
        )
        major, minor = torch.cuda.get_device_capability(device_id)
        return DeviceCapability(major=major, minor=minor)

    @classmethod
    @with_amdsmi_context
    def is_fully_connected(cls, physical_device_ids: list[int]) -> bool:
        """
        Query if the set of gpus are fully connected by xgmi (1 hop)
        """
        handles = [amdsmi_get_processor_handles()[i] for i in physical_device_ids]
        for i, handle in enumerate(handles):
            for j, peer_handle in enumerate(handles):
                if i < j:
                    try:
                        link_type = amdsmi_topo_get_link_type(handle, peer_handle)
                        # type is 2 for XGMI
                        if link_type["hops"] != 1 or link_type["type"] != 2:
                            return False
                    except AmdSmiException as error:
                        logger.error("AMD 1 hop XGMI detection failed.", exc_info=error)
                        return False
        return True

    @classmethod
    @with_amdsmi_context
    @lru_cache(maxsize=8)
    def get_device_name(cls, device_id: int = 0) -> str:
        physical_device_id = cls.device_id_to_physical_device_id(device_id)
        handle = amdsmi_get_processor_handles()[physical_device_id]
        asic_info = amdsmi_get_gpu_asic_info(handle)
        device_name: str = asic_info["device_id"]
        if device_name in _ROCM_DEVICE_ID_NAME_MAP:
            return _ROCM_DEVICE_ID_NAME_MAP[device_name]
        return asic_info["market_name"]

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        device_props = torch.cuda.get_device_properties(device_id)
        return device_props.total_memory

    @classmethod
    def apply_config_platform_defaults(cls, vllm_config: "VllmConfig") -> None:
        from vllm._aiter_ops import rocm_aiter_ops
        from vllm.config.compilation import CUDAGraphMode

        compilation_config = vllm_config.compilation_config
        is_eager_execution = compilation_config.cudagraph_mode == CUDAGraphMode.NONE
        use_aiter_fused_moe = rocm_aiter_ops.is_fused_moe_enabled()
        use_aiter_rms_norm = rocm_aiter_ops.is_rmsnorm_enabled()
        use_aiter_fp8_linear = rocm_aiter_ops.is_linear_fp8_enabled()
        use_aiter_fused_se = rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()
        #  Aiter rms norm perform best when CUDA Graph capture is enabled.
        if (
            use_aiter_rms_norm
            and not is_eager_execution
            and "-rms_norm" not in compilation_config.custom_ops
        ):
            compilation_config.custom_ops.append("+rms_norm")

        if use_aiter_fp8_linear and "-quant_fp8" not in compilation_config.custom_ops:
            compilation_config.custom_ops.append("+quant_fp8")

        if use_aiter_fused_se and "-grouped_topk" in compilation_config.custom_ops:
            logger.warning_once(
                "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS is enabled, which "
                "requires the 'grouped_topk' custom op. Overriding the "
                "user-provided '-grouped_topk'."
            )
            compilation_config.custom_ops.remove("-grouped_topk")
        # Ensure grouped_topk is always enabled when using AITER if
        # its not disabled by user
        if (
            use_aiter_fused_moe
            and "+grouped_topk" not in compilation_config.custom_ops
            and "-grouped_topk" not in compilation_config.custom_ops
        ):
            compilation_config.custom_ops.append("+grouped_topk")
        # Enable rotary embedding customop when using AITER if not disabled by user
        if (
            rocm_aiter_ops.is_enabled()
            and "+rotary_embedding" not in compilation_config.custom_ops
            and "-rotary_embedding" not in compilation_config.custom_ops
        ):
            compilation_config.custom_ops.append("+rotary_embedding")

        # Default dispatch to rocm's sparse_attn_indexer implementation
        compilation_config.custom_ops.append("+sparse_attn_indexer")

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        from vllm.config.compilation import CUDAGraphMode

        cache_config = vllm_config.cache_config
        compilation_config = vllm_config.compilation_config
        parallel_config = vllm_config.parallel_config

        if compilation_config.cudagraph_mode.has_full_cudagraphs():
            # decode context parallel does not support full cudagraphs
            if parallel_config.decode_context_parallel_size > 1:
                logger.warning_once(
                    "Decode context parallel (DCP) is enabled, which is "
                    "incompatible with full CUDA graphs. "
                    "Overriding cudagraph_mode to PIECEWISE."
                )
                compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
            # prefill context parallel do not support full cudagraphs
            elif parallel_config.prefill_context_parallel_size > 1:
                logger.warning_once(
                    "Prefill context parallel (PCP) is enabled, which is "
                    "incompatible with full CUDA graphs. "
                    "Overriding cudagraph_mode to PIECEWISE."
                )
                compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE

        if cache_config and not cache_config.user_specified_block_size:
            if (
                envs.VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION and envs.VLLM_ROCM_USE_AITER
                # NOTE: This block has been deprecated
                # or get_env_variable_attn_backend()
                # == AttentionBackendEnum.ROCM_AITER_UNIFIED_ATTN
                # TODO: monitor https://github.com/vllm-project/vllm/pull/30396
                # to see how we can transition to the new way of selecting
                # attention backends
            ):
                cache_config.block_size = 64
                logger.warning(
                    "[ROCM_AITER_UNIFIED_ATTN]: Setting kv cache block size to 64."
                )
            else:
                cache_config.block_size = 16

        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "vllm.v1.worker.gpu_worker.Worker"

    @classmethod
    def update_block_size_for_backend(cls, vllm_config: "VllmConfig") -> None:
        # TODO: ROCm still sets block_size in check_and_update_config.
        # Move that logic here so block_size is chosen by the backend.
        pass

    @classmethod
    def verify_model_arch(cls, model_arch: str) -> None:
        if model_arch in _ROCM_UNSUPPORTED_MODELS:
            raise ValueError(
                f"Model architecture '{model_arch}' is not supported by ROCm for now."
            )

        if model_arch in _ROCM_PARTIALLY_SUPPORTED_MODELS:
            msg = _ROCM_PARTIALLY_SUPPORTED_MODELS[model_arch]
            logger.warning(
                "Model architecture '%s' is partially supported by ROCm: %s",
                model_arch,
                msg,
            )

    @classmethod
    def verify_quantization(cls, quant: str) -> None:
        super().verify_quantization(quant)
        if quant == "awq" and not envs.VLLM_USE_TRITON_AWQ:
            logger.warning(
                "Using AWQ quantization with ROCm, but VLLM_USE_TRITON_AWQ"
                " is not set, enabling VLLM_USE_TRITON_AWQ."
            )
        os.environ["VLLM_USE_TRITON_AWQ"] = "1"

    @classmethod
    def get_punica_wrapper(cls) -> str:
        return "vllm.lora.punica_wrapper.punica_gpu.PunicaWrapperGPU"

    @classmethod
    def get_current_memory_usage(
        cls, device: torch.types.Device | None = None
    ) -> float:
        torch.cuda.reset_peak_memory_stats(device)
        free_mem, total_mem = torch.cuda.mem_get_info(device)
        return total_mem - free_mem

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return (
            "vllm.distributed.device_communicators.cuda_communicator.CudaCommunicator"  # noqa
        )

    @classmethod
    def supports_mx(cls) -> bool:
        return any(gfx in _GCN_ARCH for gfx in ["gfx95"])

    @classmethod
    def supports_fp8(cls) -> bool:
        return any(gfx in _GCN_ARCH for gfx in ["gfx94", "gfx95", "gfx12"])

    @classmethod
    def is_fp8_fnuz(cls) -> bool:
        # only device 0 is checked, this assumes MI300 platforms are homogeneous
        return "gfx94" in _GCN_ARCH

    @classmethod
    def fp8_dtype(cls) -> torch.dtype:
        if cls.is_fp8_fnuz():
            return torch.float8_e4m3fnuz
        else:
            return torch.float8_e4m3fn

    @classmethod
    def use_custom_allreduce(cls) -> bool:
        # We only enable custom allreduce for MI300 series
        return any(gfx in _GCN_ARCH for gfx in ["gfx94", "gfx95"])

    @classmethod
    def opaque_attention_op(cls) -> bool:
        return True

    @classmethod
    def is_navi(cls) -> bool:
        return "gfx1" in _GCN_ARCH

    @classmethod
    def get_static_graph_wrapper_cls(cls) -> str:
        return "vllm.compilation.cuda_graph.CUDAGraphWrapper"

    @classmethod
    def stateless_init_device_torch_dist_pg(
        cls,
        backend: str,
        prefix_store: PrefixStore,
        group_rank: int,
        group_size: int,
        timeout: timedelta,
    ) -> ProcessGroup:
        assert is_nccl_available()
        pg: ProcessGroup = ProcessGroup(
            prefix_store,
            group_rank,
            group_size,
        )
        from torch.distributed.distributed_c10d import ProcessGroupNCCL

        backend_options = ProcessGroupNCCL.Options()
        backend_options._timeout = timeout

        backend_class = ProcessGroupNCCL(
            prefix_store, group_rank, group_size, backend_options
        )
        backend_type = ProcessGroup.BackendType.NCCL
        device = torch.device("cuda")
        pg._set_default_backend(backend_type)
        backend_class._set_sequence_number_for_group()

        pg._register_backend(device, backend_type, backend_class)
        return pg

    @classmethod
    def device_count(cls) -> int:
        return cuda_device_count_stateless()

    @classmethod
    def check_if_supports_dtype(cls, dtype: torch.dtype):
        if dtype == torch.bfloat16:  # noqa: SIM102
            if not cls.has_device_capability(80):
                capability = cls.get_device_capability()
                gpu_name = cls.get_device_name()

                if capability is None:
                    compute_str = "does not have a compute capability"
                else:
                    version_str = capability.as_version_str()
                    compute_str = f"has compute capability {version_str}"

                raise ValueError(
                    "Bfloat16 is only supported on GPUs "
                    "with compute capability of at least 8.0. "
                    f"Your {gpu_name} GPU {compute_str}. "
                    "You can use float16 instead by explicitly setting the "
                    "`dtype` flag in CLI, for example: --dtype=half."
                )

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        return True

    @classmethod
    def support_static_graph_mode(cls) -> bool:
        return True

    @classmethod
    def num_compute_units(cls, device_id: int = 0) -> int:
        return torch.cuda.get_device_properties(device_id).multi_processor_count

    @classmethod
    def use_custom_op_collectives(cls) -> bool:
        return True
HOTFIX_ROCM
  truncate -s -1 "${bundle_dir}/rocm.py"

  cat <<'HOTFIX_TRITON_ATTN' > "${bundle_dir}/triton_attn.py"
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""High-Performance Triton-only Attention layer."""

from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.utils.math_utils import next_power_of_2
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.ops.triton_prefill_attention import context_attention_fwd
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash,
)
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)


# constants
MIN_LAUNCH_GRID_SIZE_2D = 128  # Minimum launch grid size of 2D kernel
NUM_PAR_SOFTMAX_SEGMENTS = 64  # Number of parallel tiled softmax segments


@dataclass
class TritonAttentionMetadata:
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    seq_threshold_3D: int
    num_par_softmax_segments: int
    softmax_segm_output: torch.Tensor
    softmax_segm_max: torch.Tensor
    softmax_segm_expsum: torch.Tensor

    # For cascade attention.
    use_cascade: bool
    common_prefix_len: int
    cu_prefix_query_lens: torch.Tensor | None
    prefix_kv_lens: torch.Tensor | None
    suffix_kv_lens: torch.Tensor | None

    # Optional aot scheduling
    scheduler_metadata: torch.Tensor | None = None
    prefix_scheduler_metadata: torch.Tensor | None = None
    mm_prefix_range: dict[int, list[tuple[int, int]]] | None = None

    @property
    def mm_prefix_range_tensor(self) -> torch.Tensor | None:
        """Convert mm_prefix_range dict to padded tensor for Triton kernel.

        Returns shape: (num_seqs, max_ranges, 2) with 0-padding for empty ranges.
        Empty ranges have start==end==0, which kernel skips via is_valid check.
        """
        # TODO(Isotr0py): Move to model runner's attention metadata
        # preparation to avoid duplicate computation.
        if self.mm_prefix_range is None:
            return None

        num_seqs = self.seq_lens.shape[0]
        device = self.seq_lens.device

        # Collect ranges, using [(0,0)] for empty sequences to ensure uniform dims
        range_lists = [
            self.mm_prefix_range.get(i, [(0, 0)]) or [(0, 0)] for i in range(num_seqs)
        ]

        # Return None if all ranges are trivial (only (0,0) placeholders)
        if all(r == [(0, 0)] for r in range_lists):
            return None

        # Create 2D tensors with shape (num_ranges, 2) for each sequence
        range_tensors = [
            torch.tensor(r, dtype=torch.int32, device=device).view(-1, 2)
            for r in range_lists
        ]

        return torch.nested.nested_tensor(
            range_tensors, layout=torch.jagged
        ).to_padded_tensor(0)


class TritonAttentionMetadataBuilder(AttentionMetadataBuilder[TritonAttentionMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        self.block_size = kv_cache_spec.block_size

        model_config = vllm_config.model_config
        self.num_heads_q = model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )
        self.num_heads_kv = model_config.get_num_kv_heads(vllm_config.parallel_config)
        self.headdim = model_config.get_head_size()

        # Check if CUDA Graphs are enabled for decode
        self.decode_cudagraph_enabled = (
            self.vllm_config.compilation_config.cudagraph_mode
            in (
                CUDAGraphMode.FULL_AND_PIECEWISE,
                CUDAGraphMode.FULL_DECODE_ONLY,
                CUDAGraphMode.FULL,
            )
        )

        # The launch grid for the 2D kernel is defined as (num_q_blocks, num_heads_kv).
        # A lower bound for num_q_blocks is the number of sequences.
        # To ensure the minimum launch grid size is achieved, the number of sequences
        # must be at least equal to the threshold below.
        # If this threshold is not reached (i.e., the batch size is not large enough),
        # the 3D kernel will be selected instead.
        self.seq_threshold_3D = MIN_LAUNCH_GRID_SIZE_2D // self.num_heads_kv

        # Modify the threshold if needed.
        if self.decode_cudagraph_enabled:
            capture_sizes = self.vllm_config.compilation_config.cudagraph_capture_sizes
            assert capture_sizes, "CUDA Graphs enabled but no capture sizes specified."

            # Select the CUDA Graph capture size closest to self.seq_threshold_3D
            # as threshold. This ensures that each captured graph covers the
            # correct execution path.
            self.seq_threshold_3D = min(
                capture_sizes,
                key=lambda x: abs(x - self.seq_threshold_3D),
            )

        self.num_par_softmax_segments = NUM_PAR_SOFTMAX_SEGMENTS
        headdim_padded = next_power_of_2(self.headdim)
        self.softmax_segm_output = torch.empty(
            (
                self.seq_threshold_3D,
                self.num_heads_q,
                self.num_par_softmax_segments,
                headdim_padded,
            ),
            dtype=torch.float32,
            device=device,
        )
        self.softmax_segm_max = torch.empty(
            (self.seq_threshold_3D, self.num_heads_q, self.num_par_softmax_segments),
            dtype=torch.float32,
            device=device,
        )
        self.softmax_segm_expsum = torch.empty(
            (self.seq_threshold_3D, self.num_heads_q, self.num_par_softmax_segments),
            dtype=torch.float32,
            device=device,
        )

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> TritonAttentionMetadata:
        attn_metadata = self.build(0, common_attn_metadata)
        # When doing full graph capture, setting seq_lens to
        # max_model_len will cause graph capture to be extremely
        # slow, so here we set it to 1.
        attn_metadata.seq_lens.fill_(1)
        return attn_metadata

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> TritonAttentionMetadata:
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len

        max_seq_len = common_attn_metadata.max_seq_len
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping

        use_cascade = common_prefix_len > 0

        if use_cascade:
            cu_prefix_query_lens = torch.tensor(
                [0, num_actual_tokens], dtype=torch.int32, device=self.device
            )
            prefix_kv_lens = torch.tensor(
                [common_prefix_len], dtype=torch.int32, device=self.device
            )
            suffix_kv_lens = common_attn_metadata.seq_lens.cpu() - common_prefix_len
            suffix_kv_lens = suffix_kv_lens.to(self.device)
        else:
            cu_prefix_query_lens = None
            prefix_kv_lens = None
            suffix_kv_lens = None
            prefix_scheduler_metadata = None

        attn_metadata = TritonAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            cu_prefix_query_lens=cu_prefix_query_lens,
            prefix_kv_lens=prefix_kv_lens,
            suffix_kv_lens=suffix_kv_lens,
            prefix_scheduler_metadata=prefix_scheduler_metadata,
            seq_threshold_3D=self.seq_threshold_3D,
            num_par_softmax_segments=self.num_par_softmax_segments,
            softmax_segm_output=self.softmax_segm_output,
            softmax_segm_max=self.softmax_segm_max,
            softmax_segm_expsum=self.softmax_segm_expsum,
        )
        return attn_metadata


class TritonAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        if block_size is None:
            return True
        return block_size % 16 == 0

    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_name() -> str:
        return "TRITON_ATTN"

    @staticmethod
    def get_impl_cls() -> type["TritonAttentionImpl"]:
        return TritonAttentionImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        # `stride_order` indicates the permutation that gets
        # us from `get_kv_cache_shape` to the actual memory layout we want.
        if include_num_layers_dimension:
            # (num_blocks, num_layers, 2, block_size, num_kv_heads, head_size)
            return (1, 0, 2, 3, 4, 5)

        # (num_blocks, 2, block_size, num_kv_heads, head_size)
        return (0, 1, 2, 3, 4)

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False

    @staticmethod
    def get_builder_cls() -> type["TritonAttentionMetadataBuilder"]:
        return TritonAttentionMetadataBuilder

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size >= 32

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        return True

    @classmethod
    def supports_sink(cls) -> bool:
        return True

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        """TritonAttention supports all attention types."""
        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER_DECODER,
        )

    @classmethod
    def supports_alibi_sqrt(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return True


class TritonAttentionImpl(AttentionImpl):
    def fused_output_quant_supported(self, quant_key: QuantKey):
        return quant_key == kFp8StaticTensorSym

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: int | None = None,
        sinks: torch.Tensor | None = None,
        use_alibi_sqrt: bool = False,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        elif attn_type in (AttentionType.ENCODER, AttentionType.ENCODER_ONLY):
            self.sliding_window = (sliding_window - 1, sliding_window - 1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.attn_type = attn_type
        self.fp8_dtype = current_platform.fp8_dtype()

        self.sinks = sinks
        if sinks is not None:
            assert sinks.shape[0] == num_heads, (
                "Sinks must have the same number of heads as the number of "
                f"heads in the layer. Sinks shape: {sinks.shape}, "
                f"num_heads: {num_heads}."
            )
        self.use_alibi_sqrt = use_alibi_sqrt
        self.supports_quant_query_input = current_platform.is_cuda()

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with Paged Attention impl. in Triton.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [num_blocks, 2, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."

        if output_block_scale is not None:
            raise NotImplementedError(
                "fused block_scale output quantization is not yet supported"
                " for TritonAttentionImpl"
            )

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        assert attn_metadata.use_cascade is False

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        num_actual_tokens = attn_metadata.num_actual_tokens

        # Handle encoder attention differently - no KV cache needed
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return self._forward_encoder_attention(
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                output[:num_actual_tokens],
                attn_metadata,
                layer,
            )

        # For decoder and cross-attention, use KV cache as before
        key_cache, value_cache = kv_cache.unbind(1)
        if self.kv_cache_dtype.startswith("fp8"):
            if key_cache.dtype != self.fp8_dtype:
                key_cache = key_cache.view(self.fp8_dtype)
                value_cache = value_cache.view(self.fp8_dtype)
            assert layer._q_scale_float == 1.0, (
                "A non 1.0 q_scale is not currently supported."
            )

        cu_seqlens_q = attn_metadata.query_start_loc
        seqused_k = attn_metadata.seq_lens
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_seq_len
        block_table = attn_metadata.block_table

        seq_threshold_3D = attn_metadata.seq_threshold_3D
        num_par_softmax_segments = attn_metadata.num_par_softmax_segments
        softmax_segm_output = attn_metadata.softmax_segm_output
        softmax_segm_max = attn_metadata.softmax_segm_max
        softmax_segm_expsum = attn_metadata.softmax_segm_expsum

        descale_shape = (cu_seqlens_q.shape[0] - 1, key_cache.shape[2])
        mm_prefix_range_tensor = attn_metadata.mm_prefix_range_tensor

        unified_attention(
            q=query[:num_actual_tokens],
            k=key_cache,
            v=value_cache,
            out=output[:num_actual_tokens],
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=True,
            alibi_slopes=self.alibi_slopes,
            use_alibi_sqrt=self.use_alibi_sqrt,
            window_size=self.sliding_window,
            block_table=block_table,
            softcap=self.logits_soft_cap,
            q_descale=None,  # Not supported
            k_descale=layer._k_scale.expand(descale_shape),
            v_descale=layer._v_scale.expand(descale_shape),
            seq_threshold_3D=seq_threshold_3D,
            num_par_softmax_segments=num_par_softmax_segments,
            softmax_segm_output=softmax_segm_output,
            softmax_segm_max=softmax_segm_max,
            softmax_segm_expsum=softmax_segm_expsum,
            sinks=self.sinks,
            output_scale=output_scale,
            mm_prefix_range=mm_prefix_range_tensor,
        )

        return output

    def _forward_encoder_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        layer: torch.nn.Module,
    ) -> torch.Tensor:
        """Forward pass for encoder attention without KV cache.

        Args:
            query: shape = [num_encoder_tokens, num_heads, head_size]
            key: shape = [num_encoder_tokens, num_kv_heads, head_size]
            value: shape = [num_encoder_tokens, num_kv_heads, head_size]
            output: shape = [num_encoder_tokens, num_heads, head_size]
            attn_metadata: Encoder attention metadata
            layer: The attention layer
        """
        # For encoder attention, process FP8 quantization if needed
        if self.kv_cache_dtype.startswith("fp8"):
            raise NotImplementedError(
                "quantization is not supported for encoder attention"
            )

        # Use encoder-specific metadata for sequence information
        query_start_loc = attn_metadata.query_start_loc
        seq_lens = attn_metadata.seq_lens
        max_query_len = attn_metadata.max_query_len

        # Call flash attention directly on Q, K, V tensors
        context_attention_fwd(
            q=query,
            k=key,
            v=value,
            o=output,
            b_start_loc=query_start_loc,
            b_seq_len=seq_lens,
            max_input_len=max_query_len,
            is_causal=False,  # Encoder attention is bidirectional
            softmax_scale=self.scale,
            sliding_window_q=self.sliding_window[0],
            sliding_window_k=self.sliding_window[1],
        )
        return output

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return
        # For decoder and cross-attention, use KV cache as before
        key_cache, value_cache = kv_cache.unbind(1)

        # Reshape the input keys and values and store them in the cache.
        if self.kv_cache_dtype.startswith("fp8"):
            key_cache = key_cache.view(self.fp8_dtype)
            value_cache = value_cache.view(self.fp8_dtype)
            # triton kernel does not support uint8 kv_cache
            #  (because some explicit casts (e.g. float8_e4m3fnuz)
            #   are not supported)
        triton_reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    def fused_rope_kvcache_supported(self):
        return rocm_aiter_ops.is_enabled()

    def do_rope_and_kv_cache_update(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        is_neox: bool,
        kv_cache: torch.Tensor,
        layer_slot_mapping: torch.Tensor,
    ):
        key_cache, value_cache = kv_cache.unbind(1)
        flash_layout = True

        is_fp8_kv_cache = self.kv_cache_dtype.startswith("fp8")
        if is_fp8_kv_cache:
            key_cache = key_cache.view(self.fp8_dtype)
            value_cache = value_cache.view(self.fp8_dtype)

        rocm_aiter_ops.triton_rope_and_cache(
            query,
            key,
            value,
            positions,
            cos_sin_cache,
            is_neox,
            key_cache,
            value_cache,
            layer_slot_mapping,
            layer._k_scale,
            layer._v_scale,
            flash_layout,
            is_fp8_kv_cache,
        )
HOTFIX_TRITON_ATTN

  cat <<'HOTFIX_TRITON_UNIFIED_ATTENTION' > "${bundle_dir}/triton_unified_attention.py"
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Authors:
#  - Burkhard Ringlein <ngl@zurich.ibm.com>
#  - Jan van Lunteren <jvl@zurich.ibm.com>
#  - Chih-Chieh Yang <chih.chieh.yang@ibm.com>
#  - Thomas Parnell <tpa@zurich.ibm.com>

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.batch_invariant import vllm_is_batch_invariant
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)
is_batch_invariant = vllm_is_batch_invariant()
float8_info = torch.finfo(current_platform.fp8_dtype())


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def apply_softcap(S, x):
    Sdiv = S / x
    p1 = tl.exp(Sdiv)
    p2 = tl.exp(-Sdiv)
    return x * (p1 - p2) / (p1 + p2)


@triton.jit
def find_seq_idx(
    query_start_len_ptr,
    target_idx,
    num_seqs,
    BLOCK_Q: tl.constexpr,
    use_q_block_mode: tl.constexpr,
):
    left: tl.int32 = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = tl.load(query_start_len_ptr + mid)
        mid_val = val // BLOCK_Q + mid if use_q_block_mode else val

        if mid_val <= target_idx:
            left = mid + 1
        else:
            right = mid

    return left - 1


@triton.autotune(
    configs=[triton.Config({}, num_stages=1, num_warps=2)],
    key=[]
)
@triton.jit
def kernel_unified_attention_2d(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    query_ptr,  # [num_tokens, num_query_heads, head_size]
    key_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
    value_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
    sink_ptr,  # [num_query_heads]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    alibi_slopes_ptr,  # [num_query_heads]
    qq_bias_ptr,  # [num_query_tokens, num_query_tokens]
    scale,  # float32
    k_scale,  # float32
    v_scale,  # float32
    out_scale,  # float32
    softcap,  # float32
    num_query_heads: tl.constexpr,  # int
    num_queries_per_kv: tl.constexpr,  # int
    block_table_stride: tl.int64,  # int
    query_stride_0: tl.int64,  # int
    query_stride_1: tl.int64,  # int, should be equal to head_size
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    qq_bias_stride_0: tl.int64,  # int
    BLOCK_SIZE: tl.constexpr,  # int
    TILE_SIZE: tl.constexpr,  # int must be power of 2
    HEAD_SIZE: tl.constexpr,  # int
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    USE_ALIBI_SLOPES: tl.constexpr,  # bool
    USE_ALIBI_SQRT: tl.constexpr,  # bool
    USE_QQ_BIAS: tl.constexpr,  # bool
    USE_SOFTCAP: tl.constexpr,  # bool
    USE_SINKS: tl.constexpr,  # bool
    SLIDING_WINDOW: tl.constexpr,  # int
    USE_MM_PREFIX: tl.constexpr,  # bool
    MAX_MM_RANGES: tl.constexpr,  # int
    mm_prefix_range_ptr,  # [num_seqs] - prefix length for each sequence
    stride_k_cache_0: tl.int64,  # int
    stride_k_cache_1: tl.int64,  # int
    stride_k_cache_2: tl.int64,  # int
    stride_k_cache_3: tl.constexpr,  # int
    stride_v_cache_0: tl.int64,  # int
    stride_v_cache_1: tl.int64,  # int
    stride_v_cache_2: tl.int64,  # int
    stride_v_cache_3: tl.constexpr,  # int
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,  # int
    USE_FP8: tl.constexpr,  # bool
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    seq_idx = find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
    )

    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx

    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)

    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    # Q : (BLOCK_M, HEAD_SIZE_PADDED)
    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    block_table_offset = seq_idx * block_table_stride

    if not USE_SINKS:
        M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    else:
        M = tl.load(
            sink_ptr + query_offset_1,
            mask=query_mask_1,
            other=float("-inf"),
        ).to(dtype=tl.float32)

    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # context length for this particular sequences
    context_len = seq_len - cur_batch_query_len

    # alibi slope for this head
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0
        )

    # query-query attention bias
    if USE_QQ_BIAS:
        qq_bias_row_ptrs = (
            qq_bias_ptr + query_pos[:, None] * qq_bias_stride_0
        )  # shape: [BLOCK_M]

    # compute the length of the longest sequence prefix spanned by any
    # query token in the current q_block (q_block_local_idx)
    max_seq_prefix_len = (
        context_len
        + q_block_local_idx * BLOCK_Q
        + (BLOCK_M - 1) // num_queries_per_kv
        + 1
    )

    if USE_MM_PREFIX:
        # image bidirectional attention ranges require a full range
        # including q_block padding to make sure doc mask is correct
        max_seq_prefix_len = tl.maximum(max_seq_prefix_len, seq_len)
    else:
        # adjust for potential padding in the last q_block by considering the
        # actual sequence length
        max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)

    # calculate the number of tiles that need to be processed to
    # cover the longest sequence prefix (due to causal masking, tiles beyond
    # this prefix can be skipped)
    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)

    # ---- Sliding-window tile pruning --------------------
    # Default: keep previous global behavior
    tile_start = 0
    tile_end = num_tiles
    # TODO(Isotr0py): sliding window pruning with image bidirectional mask
    if SLIDING_WINDOW > 0 and not USE_MM_PREFIX:
        # Query rows covered by this Q-block
        qpos_lo = q_block_local_idx * BLOCK_Q
        qpos_hi = tl.minimum(
            qpos_lo + (BLOCK_M - 1) // num_queries_per_kv,
            cur_batch_query_len - 1,
        )
        # For sliding window, each query position q can only attend to
        # keys in the range [q_abs - SLIDING_WINDOW + 1, q_abs]
        # where q_abs = context_len + q
        # The union of allowed key positions for this Q-block is:
        # [context_len + qpos_lo - SLIDING_WINDOW + 1, context_len + qpos_hi]
        first_allowed_key = context_len + qpos_lo - SLIDING_WINDOW + 1
        last_allowed_key = context_len + qpos_hi
        # Convert to tile indices and clamp
        tile_start = tl.maximum(0, first_allowed_key // TILE_SIZE)
        tile_end = tl.minimum((last_allowed_key // TILE_SIZE) + 1, num_tiles)

    # iterate through tiles (now limited to the sliding window range)
    for j in range(tile_start, tile_end):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len

        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
        ).to(tl.int64)

        v_offset = (
            physical_block_idx[:, None] * stride_v_cache_0
            + kv_head_idx * stride_v_cache_2
            + offs_d[None, :] * stride_v_cache_3
            + (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1
        )

        k_offset = (
            physical_block_idx[None, :] * stride_k_cache_0
            + kv_head_idx * stride_k_cache_2
            + offs_d[:, None] * stride_k_cache_3
            + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1
        )

        # K : (HEAD_SIZE, TILE_SIZE)
        K_load = tl.load(
            key_cache_ptr + k_offset,
            mask=dim_mask[:, None] & tile_mask[None, :],
            other=0.0,
        )

        if K_load.dtype.is_fp8():
            if Q.dtype.is_fp8():
                K = K_load
            else:
                K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q.dtype)
        else:
            K = K_load

        # V : (TILE_SIZE, HEAD_SIZE)
        V_load = tl.load(
            value_cache_ptr + v_offset,
            mask=dim_mask[None, :] & tile_mask[:, None],
            other=0.0,
        )

        if V_load.dtype.is_fp8():
            if Q.dtype.is_fp8():
                V = V_load
            else:
                V = (V_load.to(tl.float32) * tl.load(v_scale)).to(Q.dtype)
        else:
            V = V_load

        # Compute attention mask: causal by default (key <= query)
        query_abs_pos = context_len + query_pos[:, None]
        seq_mask = seq_offset[None, :] <= query_abs_pos

        # Apply sliding window to base mask BEFORE mm_prefix OR.
        # Order must match FlexAttention: (causal AND sliding_window) OR mm_prefix
        if SLIDING_WINDOW > 0:
            seq_mask = seq_mask & ((query_abs_pos - seq_offset) < SLIDING_WINDOW)

        # PrefixLM: extend mask with bidirectional ranges for multimodal tokens.
        # Applied AFTER sliding window so mm_prefix ranges override SW restriction.
        if USE_MM_PREFIX:
            for i in range(MAX_MM_RANGES):
                range_start = tl.load(
                    mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + i * 2
                )
                range_end = tl.load(
                    mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + i * 2 + 1
                )

                is_valid = range_start < range_end
                q_in_range = (
                    (query_abs_pos >= range_start)
                    & (query_abs_pos <= range_end)
                    & is_valid
                )
                k_in_range = (
                    (seq_offset[None, :] >= range_start)
                    & (seq_offset[None, :] <= range_end)
                    & is_valid
                )
                seq_mask |= q_in_range & k_in_range

        # S : (BLOCK_M, TILE_SIZE)
        S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)

        S += scale * tl.dot(Q, K)

        if USE_SOFTCAP:
            S = apply_softcap(S, softcap)

        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
        )

        if USE_ALIBI_SLOPES:
            if USE_ALIBI_SQRT:
                relative_pos = seq_offset - (context_len + query_pos[:, None])
                alibi_offset = tl.where(
                    relative_pos <= 0,
                    -tl.sqrt((-relative_pos).to(tl.float32)),
                    0.0,
                )
            else:
                alibi_offset = seq_offset - context_len
            S += alibi_slope[:, None] * alibi_offset

        if USE_QQ_BIAS:
            # compute key positions relative to query section
            key_rel_pos = seq_offset - context_len  # shape: [BLOCK_SIZE]
            # load bias only for keys that correspond to queries
            is_query_key = key_rel_pos >= 0 and key_rel_pos < qq_bias_stride_0
            qq_bias = tl.load(
                qq_bias_row_ptrs + key_rel_pos[None, :],
                mask=is_query_key[None, :],  # avoid OOB for context keys
                other=0.0,
            )
            S += qq_bias

        # compute running maximum
        # m_j : (BLOCK_M,)
        m_j = tl.maximum(M, tl.max(S, axis=1))

        # For sliding window there's a chance the max is -inf due to masking of
        # the entire row. In this case we need to set m_j 0 to avoid NaN
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)

        # P : (BLOCK_M, TILE_SIZE)
        P = tl.exp(S - m_j[:, None])

        # l_j : (BLOCK_M,)
        l_j = tl.sum(P, axis=1)

        # alpha : (BLOCK_M, )
        alpha = tl.exp(M - m_j)

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        acc = acc * alpha[:, None]

        # update constants
        L = L * alpha + l_j
        M = m_j

        if SLIDING_WINDOW:
            qpos_lo = q_block_local_idx * BLOCK_Q
            V = tl.where(
                (context_len + qpos_lo - seq_offset[:, None]) < SLIDING_WINDOW, V, 0.0
            )

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        acc += tl.dot(P.to(V.dtype), V)

    # epilogue
    acc = acc / L[:, None]
    if USE_FP8:
        acc = acc * tl.load(out_scale)
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)

    output_offset = (
        query_offset_0[:, None] * output_stride_0
        + query_offset_1[:, None] * output_stride_1
        + offs_d[None, :]
    )

    tl.store(
        output_ptr + output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )


@triton.autotune(
    configs=[triton.Config({}, num_stages=1, num_warps=2)],
    key=[]
)
@triton.jit
def kernel_unified_attention_3d(
    segm_output_ptr,
    # [num_tokens, num_query_heads, num_segments, head_size_padded]
    segm_max_ptr,  # [num_tokens, num_query_heads, num_segments]
    segm_expsum_ptr,  # [num_tokens, num_query_heads, num_segments]
    query_ptr,  # [num_tokens, num_query_heads, head_size]
    key_cache_ptr,  # [num_blks, num_kv_heads, head_size // x, blk_size, x]
    value_cache_ptr,  # [num_blks, num_kv_heads, head_size, blk_size]
    sink_ptr,  # [num_query_heads]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    alibi_slopes_ptr,  # [num_query_heads]
    qq_bias_ptr,  # [num_query_tokens, num_query_tokens]
    scale,  # float32
    k_scale,  # float32
    v_scale,  # float32
    softcap,  # float32
    num_query_heads: tl.constexpr,  # int
    num_queries_per_kv: tl.constexpr,  # int
    block_table_stride: tl.int64,  # int
    query_stride_0: tl.int64,  # int
    query_stride_1: tl.int64,  # int, should be equal to head_size
    qq_bias_stride_0: tl.int64,  # int
    BLOCK_SIZE: tl.constexpr,  # int
    TILE_SIZE: tl.constexpr,  # int, must be power of 2
    HEAD_SIZE: tl.constexpr,  # int
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    USE_ALIBI_SLOPES: tl.constexpr,  # bool
    USE_ALIBI_SQRT: tl.constexpr,  # bool
    USE_QQ_BIAS: tl.constexpr,  # bool
    USE_SOFTCAP: tl.constexpr,  # bool
    USE_SINKS: tl.constexpr,  # bool
    SLIDING_WINDOW: tl.constexpr,  # int
    stride_k_cache_0: tl.int64,  # int
    stride_k_cache_1: tl.int64,  # int
    stride_k_cache_2: tl.int64,  # int
    stride_k_cache_3: tl.constexpr,  # int
    stride_v_cache_0: tl.int64,  # int
    stride_v_cache_1: tl.int64,  # int
    stride_v_cache_2: tl.int64,  # int
    stride_v_cache_3: tl.constexpr,  # int
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,  # int
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
    USE_MM_PREFIX: tl.constexpr,  # bool
    MAX_MM_RANGES: tl.constexpr,  # int
    mm_prefix_range_ptr,  # [num_seqs] - prefix length for each sequence
):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2)

    seq_idx = find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
    )

    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx

    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)

    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # number of segments for this particular sequence
    num_segments = NUM_SEGMENTS_PER_SEQ
    tiles_per_segment = cdiv_fn(seq_len, num_segments * TILE_SIZE)

    if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    # Q : (BLOCK_M, HEAD_SIZE_PADDED)
    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    block_table_offset = seq_idx * block_table_stride

    if USE_SINKS:
        if segm_idx == 0:
            M = tl.load(
                sink_ptr + query_offset_1,
                mask=query_mask_1,
                other=float("-inf"),
            ).to(dtype=tl.float32)
        else:
            M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    else:
        M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)

    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    # context length for this particular sequences
    context_len = seq_len - cur_batch_query_len

    # alibi slope for this head
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0
        )

    # query-query attention bias
    if USE_QQ_BIAS:
        qq_bias_row_ptrs = (
            qq_bias_ptr + query_pos[:, None] * qq_bias_stride_0
        )  # shape: [BLOCK_M]

    # compute the length of the longest sequence prefix spanned by any
    # query token in the current q_block (q_block_local_idx)
    max_seq_prefix_len = (
        context_len
        + q_block_local_idx * BLOCK_Q
        + (BLOCK_M - 1) // num_queries_per_kv
        + 1
    )

    # adjust for potential padding in the last q_block by considering the
    # actual sequence length
    max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)

    # calculate the number of tiles that need to be processed to
    # cover the longest sequence prefix (due to causal masking, tiles beyond
    # this prefix can be skipped)
    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)

    # ---- Sliding-window tile pruning --------------------
    # Default: keep previous global behavior
    tile_start = 0
    tile_end = num_tiles
    # TODO(Isotr0py): sliding window pruning with image bidirectional mask
    if SLIDING_WINDOW > 0 and not USE_MM_PREFIX:
        # Query rows covered by this Q-block
        qpos_lo = q_block_local_idx * BLOCK_Q
        qpos_hi = tl.minimum(
            qpos_lo + (BLOCK_M - 1) // num_queries_per_kv,
            cur_batch_query_len - 1,
        )
        # For sliding window, each query position q can only attend to
        # keys in the range [q_abs - SLIDING_WINDOW + 1, q_abs]
        # where q_abs = context_len + q
        # The union of allowed key positions for this Q-block is:
        # [context_len + qpos_lo - SLIDING_WINDOW + 1, context_len + qpos_hi]
        first_allowed_key = context_len + qpos_lo - SLIDING_WINDOW + 1
        last_allowed_key = context_len + qpos_hi
        # Convert to tile indices and clamp
        tile_start = tl.maximum(0, first_allowed_key // TILE_SIZE)
        tile_end = tl.minimum((last_allowed_key // TILE_SIZE) + 1, num_tiles)

    # iterate through tiles (now limited to the sliding window range)
    for j in range(
        max(segm_idx * tiles_per_segment, tile_start),
        min((segm_idx + 1) * tiles_per_segment, tile_end),
    ):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len

        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
        ).to(tl.int64)

        v_offset = (
            physical_block_idx[:, None] * stride_v_cache_0
            + kv_head_idx * stride_v_cache_2
            + offs_d[None, :] * stride_v_cache_3
            + (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1
        )

        k_offset = (
            physical_block_idx[None, :] * stride_k_cache_0
            + kv_head_idx * stride_k_cache_2
            + offs_d[:, None] * stride_k_cache_3
            + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1
        )

        # K : (HEAD_SIZE, TILE_SIZE)
        K_load = tl.load(
            key_cache_ptr + k_offset,
            mask=dim_mask[:, None] & tile_mask[None, :],
            other=0.0,
        )

        if K_load.dtype.is_fp8():
            if Q.dtype.is_fp8():
                K = K_load
            else:
                K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q.dtype)
        else:
            K = K_load

        # V : (TILE_SIZE, HEAD_SIZE)
        V_load = tl.load(
            value_cache_ptr + v_offset,
            mask=dim_mask[None, :] & tile_mask[:, None],
            other=0.0,
        )

        if V_load.dtype.is_fp8():
            if Q.dtype.is_fp8():
                V = V_load
            else:
                V = (V_load.to(tl.float32) * tl.load(v_scale)).to(Q.dtype)
        else:
            V = V_load

        # Compute attention mask: causal by default (key <= query)
        query_abs_pos = context_len + query_pos[:, None]
        seq_mask = seq_offset[None, :] <= query_abs_pos

        # Apply sliding window to base mask BEFORE mm_prefix OR.
        # Order must match FlexAttention: (causal AND sliding_window) OR mm_prefix
        if SLIDING_WINDOW > 0:
            seq_mask = seq_mask & ((query_abs_pos - seq_offset) < SLIDING_WINDOW)

        # PrefixLM: extend mask with bidirectional ranges for multimodal tokens.
        # Applied AFTER sliding window so mm_prefix ranges override SW restriction.
        if USE_MM_PREFIX:
            for i in range(MAX_MM_RANGES):
                range_start = tl.load(
                    mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + i * 2
                )
                range_end = tl.load(
                    mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + i * 2 + 1
                )

                is_valid = range_start < range_end
                q_in_range = (
                    (query_abs_pos >= range_start)
                    & (query_abs_pos <= range_end)
                    & is_valid
                )
                k_in_range = (
                    (seq_offset[None, :] >= range_start)
                    & (seq_offset[None, :] <= range_end)
                    & is_valid
                )
                seq_mask |= q_in_range & k_in_range

        # S : (BLOCK_M, TILE_SIZE)
        S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)
        S += scale * tl.dot(Q, K)

        if USE_SOFTCAP:
            S = apply_softcap(S, softcap)

        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
        )

        if USE_ALIBI_SLOPES:
            if USE_ALIBI_SQRT:
                relative_pos = seq_offset - (context_len + query_pos[:, None])
                alibi_offset = tl.where(
                    relative_pos <= 0,
                    -tl.sqrt((-relative_pos).to(tl.float32)),
                    0.0,
                )
            else:
                alibi_offset = seq_offset - context_len
            S += alibi_slope[:, None] * alibi_offset

        if USE_QQ_BIAS:
            # compute key positions relative to query section
            key_rel_pos = seq_offset - context_len  # shape: [BLOCK_SIZE]
            # load bias only for keys that correspond to queries
            is_query_key = key_rel_pos >= 0 and key_rel_pos < qq_bias_stride_0
            qq_bias = tl.load(
                qq_bias_row_ptrs + key_rel_pos[None, :],
                mask=is_query_key[None, :],  # avoid OOB for context keys
                other=0.0,
            )
            S += qq_bias

        # compute running maximum
        # m_j : (BLOCK_M,)
        m_j = tl.maximum(M, tl.max(S, axis=1))

        # For sliding window there's a chance the max is -inf due to masking of
        # the entire row. In this case we need to set m_j 0 to avoid NaN
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)

        # P : (BLOCK_M, TILE_SIZE,)
        P = tl.exp(S - m_j[:, None])

        # l_j : (BLOCK_M,)
        l_j = tl.sum(P, axis=1)

        # alpha : (BLOCK_M, )
        alpha = tl.exp(M - m_j)

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        acc = acc * alpha[:, None]

        # update constants
        L = L * alpha + l_j
        M = m_j

        if SLIDING_WINDOW:
            qpos_lo = q_block_local_idx * BLOCK_Q
            V = tl.where(
                (context_len + qpos_lo - seq_offset[:, None]) < SLIDING_WINDOW, V, 0.0
            )

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        acc += tl.dot(P.to(V.dtype), V)

    segm_output_offset = (
        query_offset_0[:, None].to(tl.int64)
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + query_offset_1[:, None] * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + segm_idx * HEAD_SIZE_PADDED
        + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
    )
    tl.store(
        segm_output_ptr + segm_output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )
    segm_offset = (
        query_offset_0.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_offset_1 * NUM_SEGMENTS_PER_SEQ
        + segm_idx
    )
    tl.store(segm_max_ptr + segm_offset, M, mask=query_mask_0 & query_mask_1)
    tl.store(segm_expsum_ptr + segm_offset, L, mask=query_mask_0 & query_mask_1)


@triton.jit
def reduce_segments(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    segm_output_ptr,
    # [num_tokens, num_query_heads, max_num_segments, head_size]
    segm_max_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    segm_expsum_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    seq_lens_ptr,  # [num_seqs]
    num_seqs,  # int
    num_query_heads: tl.constexpr,  # int
    out_scale_inv,  # float32
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    block_table_stride: tl.int64,  # int
    TILE_SIZE: tl.constexpr,  # int
    HEAD_SIZE: tl.constexpr,  # int, must be power of 2
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
    USE_FP8: tl.constexpr,  # bool
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    query_token_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)

    seq_idx = find_seq_idx(
        query_start_len_ptr, query_token_idx, num_seqs, BLOCK_Q, False
    )

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # number of segments for this particular sequence
    num_segments = NUM_SEGMENTS_PER_SEQ
    tiles_per_segment = cdiv_fn(seq_len, num_segments * TILE_SIZE)

    # create masks for subsequent loads
    act_num_segments = cdiv_fn(seq_len, tiles_per_segment * TILE_SIZE)
    segm_mask = tl.arange(0, NUM_SEGMENTS_PER_SEQ) < tl.full(
        [NUM_SEGMENTS_PER_SEQ], act_num_segments, dtype=tl.int32
    )
    dim_mask = tl.where(tl.arange(0, HEAD_SIZE_PADDED) < HEAD_SIZE, 1, 0).to(tl.int1)

    # load segment maxima
    segm_offset = (
        query_token_idx.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_head_idx * NUM_SEGMENTS_PER_SEQ
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)
    )
    segm_max = tl.load(segm_max_ptr + segm_offset, mask=segm_mask, other=float("-inf"))
    overall_max = tl.max(segm_max)

    # load and rescale segment exp sums
    segm_expsum = tl.load(segm_expsum_ptr + segm_offset, mask=segm_mask, other=0.0)
    segm_expsum = segm_expsum * tl.exp(segm_max - overall_max)
    overall_expsum = tl.sum(segm_expsum)

    # load, rescale, and add segment attention outputs
    segm_output_offset = (
        query_token_idx.to(tl.int64)
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + query_head_idx * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)[:, None] * HEAD_SIZE_PADDED
        + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
    )
    segm_output = tl.load(
        segm_output_ptr + segm_output_offset,
        mask=segm_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    segm_output *= tl.exp(segm_max - overall_max)[:, None]
    acc_sum = tl.sum(segm_output, axis=0)
    # safely divide by overall_expsum, returning 0.0 if overall_expsum is 0
    acc = tl.where(overall_expsum == 0.0, 0.0, acc_sum / overall_expsum)

    if USE_FP8:
        acc = acc * tl.load(out_scale_inv)
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)

    # write result
    output_offset = (
        query_token_idx * output_stride_0
        + query_head_idx * output_stride_1
        + tl.arange(0, HEAD_SIZE_PADDED)
    )
    tl.store(output_ptr + output_offset, acc, mask=dim_mask)


def _is_gemma3_attention(head_size: int, sliding_window: int) -> bool:
    """Detect Gemma3 models via unique (head_size, sliding_window) signature.

    Gemma3 models are the only ones using sliding_window=1024 with
    head_size 128 (27B) or 256 (1B, 4B, 12B). Other SWA models use
    different window sizes (Mistral=4096, Phi-3=2047).
    """
    return sliding_window == 1024 and head_size in (128, 256)


def _get_tile_size(
    head_size: int,
    sliding_window: int,
    element_size: int,
    is_prefill: bool,
) -> int:
    """Select tile size with Gemma3-specific optimization.

    For Gemma3, use 32 for both prefill and decode to better utilize
    the larger head dimension (128/256). For other models, use
    the default vLLM behavior.
    """
    if _is_gemma3_attention(head_size, sliding_window):
        # Gemma3: use 32 for decode (default is 16)
        return 32

    # Default behavior
    if is_prefill:
        return 32
    return 16 if element_size >= 2 else 32


def unified_attention(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    softcap,
    q_descale,
    k_descale,
    v_descale,
    seq_threshold_3D=None,
    num_par_softmax_segments=None,
    softmax_segm_output=None,
    softmax_segm_max=None,
    softmax_segm_expsum=None,
    alibi_slopes=None,
    output_scale=None,
    qq_bias=None,
    # Optional tensor for sinks
    sinks=None,
    # Optional tensor for prefix lengths (PrefixLM support)
    mm_prefix_range=None,
    use_alibi_sqrt=False,
):
    assert causal, "Only causal attention is supported"
    assert q_descale is None, "Q scales not supported"

    if sinks is not None:
        assert sinks.shape[0] == q.shape[1], "Sinks must be num_query_heads size"

    use_mm_prefix = False
    max_mm_ranges = 0
    if mm_prefix_range is not None:
        if mm_prefix_range.ndim == 3:
            use_mm_prefix = True
            max_mm_ranges = mm_prefix_range.shape[1]
        else:
            raise ValueError(
                f"Unsupported mm_prefix_range shape: {mm_prefix_range.shape}"
            )

    use_alibi_slopes = alibi_slopes is not None
    use_qq_bias = qq_bias is not None

    block_size = v.shape[1]
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = k.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = q.shape[2]
    head_size_padded = triton.next_power_of_2(head_size)
    
    BLOCK_M = (
        16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    )
    BLOCK_Q = BLOCK_M // num_queries_per_kv

    # Ideally we would launch with kernel with:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)] blocks.
    # However, it is slow to realize the query_lens on cpu.
    # Instead we use upper-bound:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)]
    #   <= \sum_i[floor(query_len[i] / BLOCK_Q) + 1]
    #    = \sum_i[floor(query_len[i] / BLOCK_Q)] + num_seqs
    #   <= floor(\sum_i(query_len[i]) / BLOCK_Q) + num_seqs
    #    = floor(q.shape[0] / BLOCK_Q) + num_seqs
    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs

    # Tile sizes for prefill and decode. Gemma3 models use optimized values.
    # Note: tile size must be at least 32 for fp8 (element_size == 1).
    is_fp8 = q.element_size() == 1
    min_tile_size = 32 if is_fp8 else 16
    abs_min_tile_size = 32 if is_fp8 else 8

    def _next_power_of_two(x: int) -> int:
        if x <= 1:
            return 1
        return 1 << (x - 1).bit_length()

    def _prev_power_of_two(x: int) -> int:
        if x <= 1:
            return 1
        return 1 << ((x.bit_length() - 1))

    preferred_prefill = min(_next_power_of_two(block_size), 256)
    preferred_prefill = max(preferred_prefill, min_tile_size)
    preferred_decode = max(min_tile_size,
                           16 if q.element_size() >= 2 else 32)
    preferred_decode = min(preferred_decode, preferred_prefill)

    max_tile_shared = None
    shared_mem_limit = None
    try:
        device_props = torch.cuda.get_device_properties(q.device)
        shared_mem_limit = max(
            getattr(device_props, "sharedMemPerBlockOptin", 0),
            getattr(device_props, "sharedMemPerBlock", 0),
        )
        if shared_mem_limit == 0:
            # Fallback to the architectural minimum when runtime does not report.
            shared_mem_limit = 64 * 1024
        if shared_mem_limit and shared_mem_limit > 0:
            head_bytes = max(2 * head_size_padded *
                             max(k.element_size(), v.element_size(), 1), 1)
            softmax_bytes = max(BLOCK_M * 4, 1)
            max_tile_per_head = shared_mem_limit // head_bytes
            max_tile_softmax = shared_mem_limit // softmax_bytes
            if max_tile_per_head > 0:
                candidates = [max_tile_per_head]
                if max_tile_softmax > 0:
                    candidates.append(max_tile_softmax)
                max_tile_shared = max(abs_min_tile_size, min(candidates))
    except Exception:  # pragma: no cover
        max_tile_shared = None

    max_tile_block = min(_next_power_of_two(block_size), 256)
    kv_elem_bytes = max(k.element_size(), v.element_size(), 1)

    def _estimate_shared(tile: int) -> int:
        if shared_mem_limit is None or shared_mem_limit <= 0:
            return 0
        shared = 0
        shared += 2 * head_size_padded * tile * kv_elem_bytes
        shared += 2 * BLOCK_M * tile * 4
        return shared

    def _final_tile(preferred: int) -> int:
        tile = max(preferred, min_tile_size)
        tile = _next_power_of_two(tile)
        tile = min(tile, max_tile_block)
        if max_tile_shared is not None and max_tile_shared > 0:
            while tile > max_tile_shared and tile > abs_min_tile_size:
                tile //= 2
        if shared_mem_limit and shared_mem_limit > 0:
            while tile > abs_min_tile_size and _estimate_shared(tile) > shared_mem_limit:
                tile //= 2
        return max(tile, abs_min_tile_size)

    TILE_SIZE_PREFILL = _final_tile(preferred_prefill)
    TILE_SIZE_DECODE = 8

    # Launch the 2D kernel if
    # 1. No intermediate tiled softmax buffers for the 3D kernel have been allocated, or
    # 2. The batch includes at least one prefill request, or
    # 3. The number of sequences exceeds the configured threshold, or
    # 4. Batch invariance is enabled
    if (
        seq_threshold_3D is None
        or num_par_softmax_segments is None
        or softmax_segm_output is None
        or softmax_segm_max is None
        or softmax_segm_expsum is None
        or max_seqlen_q > 1
        or num_seqs > seq_threshold_3D
        or is_batch_invariant
    ):
        kernel_unified_attention_2d[
            (
                total_num_q_blocks,
                num_kv_heads,
            )
        ](
            output_ptr=out,
            query_ptr=q,
            key_cache_ptr=k,
            value_cache_ptr=v,
            sink_ptr=sinks,
            block_tables_ptr=block_table,
            seq_lens_ptr=seqused_k,
            alibi_slopes_ptr=alibi_slopes,
            qq_bias_ptr=qq_bias,
            scale=softmax_scale,
            k_scale=k_descale,
            v_scale=v_descale,
            out_scale=1 / output_scale if output_scale is not None else 1.0,
            softcap=softcap,
            num_query_heads=num_query_heads,
            num_queries_per_kv=num_queries_per_kv,
            block_table_stride=block_table.stride(0),
            query_stride_0=q.stride(0),
            query_stride_1=q.stride(1),
            output_stride_0=out.stride(0),
            output_stride_1=out.stride(1),
            qq_bias_stride_0=qq_bias.stride(0) if use_qq_bias else 0,
            BLOCK_SIZE=block_size,
            TILE_SIZE=TILE_SIZE_PREFILL,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
            USE_ALIBI_SLOPES=use_alibi_slopes,
            USE_ALIBI_SQRT=use_alibi_sqrt,
            USE_QQ_BIAS=use_qq_bias,
            USE_SOFTCAP=(softcap > 0),
            USE_SINKS=(sinks is not None),
            USE_MM_PREFIX=use_mm_prefix,
            MAX_MM_RANGES=max_mm_ranges,
            mm_prefix_range_ptr=mm_prefix_range,
            SLIDING_WINDOW=(1 + window_size[0]),
            stride_k_cache_0=k.stride(0),
            stride_k_cache_1=k.stride(1),
            stride_k_cache_2=k.stride(2),
            stride_k_cache_3=k.stride(3),
            stride_v_cache_0=v.stride(0),
            stride_v_cache_1=v.stride(1),
            stride_v_cache_2=v.stride(2),
            stride_v_cache_3=v.stride(3),
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=BLOCK_Q,
            num_seqs=num_seqs,
            BLOCK_M=BLOCK_M,
            USE_FP8=output_scale is not None,
        )
    else:
        kernel_unified_attention_3d[
            (total_num_q_blocks, num_kv_heads, num_par_softmax_segments)
        ](
            segm_output_ptr=softmax_segm_output,
            segm_max_ptr=softmax_segm_max,
            segm_expsum_ptr=softmax_segm_expsum,
            query_ptr=q,
            key_cache_ptr=k,
            value_cache_ptr=v,
            sink_ptr=sinks,
            block_tables_ptr=block_table,
            seq_lens_ptr=seqused_k,
            alibi_slopes_ptr=alibi_slopes,
            qq_bias_ptr=qq_bias,
            scale=softmax_scale,
            k_scale=k_descale,
            v_scale=v_descale,
            softcap=softcap,
            num_query_heads=num_query_heads,
            num_queries_per_kv=num_queries_per_kv,
            block_table_stride=block_table.stride(0),
            query_stride_0=q.stride(0),
            query_stride_1=q.stride(1),
            qq_bias_stride_0=qq_bias.stride(0) if use_qq_bias else 0,
            BLOCK_SIZE=block_size,
            TILE_SIZE=TILE_SIZE_DECODE,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
            USE_ALIBI_SLOPES=use_alibi_slopes,
            USE_ALIBI_SQRT=use_alibi_sqrt,
            USE_QQ_BIAS=use_qq_bias,
            USE_SOFTCAP=(softcap > 0),
            USE_SINKS=(sinks is not None),
            USE_MM_PREFIX=use_mm_prefix,
            MAX_MM_RANGES=max_mm_ranges,
            mm_prefix_range_ptr=mm_prefix_range,
            SLIDING_WINDOW=(1 + window_size[0]),
            stride_k_cache_0=k.stride(0),
            stride_k_cache_1=k.stride(1),
            stride_k_cache_2=k.stride(2),
            stride_k_cache_3=k.stride(3),
            stride_v_cache_0=v.stride(0),
            stride_v_cache_1=v.stride(1),
            stride_v_cache_2=v.stride(2),
            stride_v_cache_3=v.stride(3),
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=BLOCK_Q,
            num_seqs=num_seqs,
            BLOCK_M=BLOCK_M,
            NUM_SEGMENTS_PER_SEQ=num_par_softmax_segments,
        )
        reduce_segments[(q.shape[0], num_query_heads)](
            output_ptr=out,
            segm_output_ptr=softmax_segm_output,
            segm_max_ptr=softmax_segm_max,
            segm_expsum_ptr=softmax_segm_expsum,
            seq_lens_ptr=seqused_k,
            num_seqs=num_seqs,
            num_query_heads=num_query_heads,
            out_scale_inv=1 / output_scale if output_scale is not None else 1.0,
            output_stride_0=out.stride(0),
            output_stride_1=out.stride(1),
            block_table_stride=block_table.stride(0),
            TILE_SIZE=TILE_SIZE_DECODE,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=BLOCK_Q,
            NUM_SEGMENTS_PER_SEQ=num_par_softmax_segments,
            USE_FP8=output_scale is not None,
        )
HOTFIX_TRITON_UNIFIED_ATTENTION

}

resolve_bool() {
  local value="${1,,}"
  case "${value}" in
    1|true|yes|on) echo 1 ;;
    *) echo 0 ;;
  esac
}

validate_repro_docker_archive_sha() {
  local archive_path="$1"
  local actual_sha="$2"
  local mode="${BYTE_FOR_BYTE_VALIDATION_MODE:-auto}"
  mode="${mode,,}"

  if [[ "$(resolve_bool "${mode}")" != "1" && "${mode}" != "auto" ]]; then
    echo "warning: BYTE_FOR_BYTE_VALIDATION_MODE=0; skipping exported Docker archive SHA validation" >&2
    echo "warning: this build is a local deploy artifact, not byte-for-byte evidence for the published release" >&2
    return 0
  fi

  if [[ -z "${actual_sha}" ]]; then
    echo "error: could not read Docker archive SHA for ${archive_path}" >&2
    return 1
  fi

  if [[ -z "${EXPECTED_REPRO_DOCKER_ARCHIVE_SHA256:-}" ]]; then
    if [[ "${mode}" == "auto" ]]; then
      echo "warning: BYTE_FOR_BYTE_VALIDATION_MODE=auto and EXPECTED_REPRO_DOCKER_ARCHIVE_SHA256 is empty" >&2
      echo "warning: recorded archive SHA ${actual_sha}; no release byte-for-byte target was enforced" >&2
      echo "warning: set EXPECTED_REPRO_DOCKER_ARCHIVE_SHA256 for strict release reproduction, or BYTE_FOR_BYTE_VALIDATION_MODE=0 for local deploys" >&2
      return 0
    fi
    echo "error: BYTE_FOR_BYTE_VALIDATION_MODE=1 requires EXPECTED_REPRO_DOCKER_ARCHIVE_SHA256" >&2
    return 1
  fi

  if [[ "${actual_sha}" != "${EXPECTED_REPRO_DOCKER_ARCHIVE_SHA256}" ]]; then
    echo "error: byte-for-byte Docker archive SHA mismatch" >&2
    echo "expected: ${EXPECTED_REPRO_DOCKER_ARCHIVE_SHA256}" >&2
    echo "actual:   ${actual_sha}" >&2
    echo "archive:  ${archive_path}" >&2
    echo "Set BYTE_FOR_BYTE_VALIDATION_MODE=0 only for non-canonical local deploys where release reproduction is not being claimed." >&2
    return 1
  fi

  echo "byte-for-byte Docker archive SHA verified: ${actual_sha}"
}

prompt_continue_on_gpu_gate_failure() {
  local reason="$1"

  if [[ "$(resolve_bool "${AUTO_CONTINUE_GPU_CHECK}")" == "1" ]]; then
    echo "warning: ${reason}"
    echo "warning: AUTO_CONTINUE_GPU_CHECK=1 is set. Continuing despite failed GPU gate."
    return 0
  fi

  echo "warning: ${reason}" >&2

  if [[ ! -t 0 ]]; then
    echo "error: interactive input unavailable, defaulting to NO. Set AUTO_CONTINUE_GPU_CHECK=1 to override." >&2
    return 1
  fi

  while true; do
    read -r -p "GPU gate failed for ${REQUIRED_GPU_MODEL_HINT:-4x AMD Instinct MI50 32GB}. Continue anyway? [y/N]: " choice
    case "${choice,,}" in
      y|yes)
        echo "warning: continuing after explicit user approval."
        return 0
        ;;
      ''|n|no)
        echo "error: GPU VRAM gate failed and execution was aborted." >&2
        return 1
        ;;
      *)
        echo "please answer y or n (default N)"
        ;;
    esac
  done
}

normalize_gpu_device_list() {
  local raw="$1"
  local -n output="$2"
  local -a temp_list=()
  local item

  output=()
  IFS=',' read -r -a temp_list <<< "${raw}"
  for item in "${temp_list[@]:-}"; do
    item="$(echo "${item}" | tr -d '[:space:]')"
    if [[ -z "${item}" ]]; then
      continue
    fi
    if [[ "${item}" =~ ^[0-9]+$ ]]; then
      output+=("${item}")
    fi
  done
}

ROCM_SMI_GPU_PROBE_LOADED=0
ROCM_SMI_GPU_PROBE_CACHE=""

get_rocm_smi_gpu_probe() {
  if [[ "${ROCM_SMI_GPU_PROBE_LOADED}" != "1" ]]; then
    ROCM_SMI_GPU_PROBE_LOADED=1
    if command -v rocm-smi >/dev/null 2>&1; then
      ROCM_SMI_GPU_PROBE_CACHE="$(rocm-smi --showproductname --showmeminfo vram 2>/dev/null || true)"
    else
      ROCM_SMI_GPU_PROBE_CACHE=""
    fi
  fi

  printf '%s\n' "${ROCM_SMI_GPU_PROBE_CACHE}"
}

list_rocm_gpu_ids() {
  local ids
  local card_path

  ids="$(get_rocm_smi_gpu_probe | grep -oE 'GPU\[[0-9]+\]' | sed -E 's/GPU\[([0-9]+)\]/\1/' | sort -n -u || true)"
  if [[ -n "${ids}" ]]; then
    printf '%s\n' "${ids}"
    return 0
  fi

  for card_path in /sys/class/drm/card[0-9]*; do
    [[ -e "${card_path}" ]] || continue
    basename "${card_path}" | sed -E 's/^card//'
  done | sort -n -u
}

gpu_rocm_smi_field() {
  local device_id="$1"
  local field="$2"

  get_rocm_smi_gpu_probe | awk -v gpu="GPU[${device_id}]" -v field="${field}" '
    index($0, gpu) && index($0, field) {
      sub(/^.*:[[:space:]]*/, "", $0)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", $0)
      print $0
      exit
    }
  '
}

canonicalize_gpu_arch() {
  local arch="${1,,}"
  arch="$(printf '%s' "${arch}" | tr -cd '[:alnum:]_')"
  case "${arch}" in
    gfx9006) echo "gfx906" ;;
    *) echo "${arch}" ;;
  esac
}

gpu_arch_rocm_smi() {
  local device_id="$1"
  local arch

  arch="$(gpu_rocm_smi_field "${device_id}" "GFX Version" | awk '{print $1}' || true)"
  if [[ -n "${arch}" ]]; then
    canonicalize_gpu_arch "${arch}"
    return 0
  fi

  return 1
}

gpu_arch_matches_target() {
  local arch
  local aliases
  local -a alias_list=()
  local alias

  arch="$(canonicalize_gpu_arch "$1")"
  aliases="${GPU_TARGET_ARCH_ALIASES:-${GPU_TARGET_ARCH:-gfx906}}"

  if [[ -z "${arch}" ]]; then
    return 1
  fi
  if [[ "${GPU_TARGET_ARCH:-}" == "any" || "${aliases,,}" == "any" ]]; then
    return 0
  fi

  IFS=',' read -r -a alias_list <<< "${aliases}"
  for alias in "${alias_list[@]:-}"; do
    alias="$(canonicalize_gpu_arch "$(echo "${alias}" | tr -d '[:space:]')")"
    if [[ -n "${alias}" && "${arch}" == "${alias}" ]]; then
      return 0
    fi
  done

  return 1
}

gpu_vram_total_bytes_rocm_smi() {
  local device_id="$1"
  local bytes

  bytes="$(gpu_rocm_smi_field "${device_id}" "VRAM Total Memory (B)" | tr -cd '0-9' || true)"
  if [[ "${bytes}" =~ ^[0-9]+$ ]]; then
    echo "${bytes}"
    return 0
  fi

  return 1
}

gpu_vram_used_bytes_rocm_smi() {
  local device_id="$1"
  local bytes

  bytes="$(gpu_rocm_smi_field "${device_id}" "VRAM Total Used Memory (B)" | tr -cd '0-9' || true)"
  if [[ "${bytes}" =~ ^[0-9]+$ ]]; then
    echo "${bytes}"
    return 0
  fi

  return 1
}

gpu_vram_gib_sysfs() {
  local device_id="$1"
  local path
  local bytes

  bytes="$(gpu_vram_total_bytes_rocm_smi "${device_id}" || true)"
  if [[ "${bytes}" =~ ^[0-9]+$ ]]; then
    echo $(( (bytes + 1073741823) / 1073741824 ))
    return 0
  fi

  path="/sys/class/drm/card${device_id}/device/mem_info_vram_total"
  if [[ -r "${path}" ]]; then
    bytes="$(tr -d '[:space:]' < "${path}")"
    if [[ "${bytes}" =~ ^[0-9]+$ ]]; then
      echo $(( (bytes + 1073741823) / 1073741824 ))
      return 0
    fi
  fi
  return 1
}

gpu_vram_bytes_sysfs() {
  local device_id="$1"
  local kind="$2"
  local path
  local bytes

  case "${kind}" in
    total)
      bytes="$(gpu_vram_total_bytes_rocm_smi "${device_id}" || true)"
      ;;
    used)
      bytes="$(gpu_vram_used_bytes_rocm_smi "${device_id}" || true)"
      ;;
    *)
      bytes=""
      ;;
  esac
  if [[ "${bytes}" =~ ^[0-9]+$ ]]; then
    echo "${bytes}"
    return 0
  fi

  path="/sys/class/drm/card${device_id}/device/mem_info_vram_${kind}"
  if [[ -r "${path}" ]]; then
    bytes="$(tr -d '[:space:]' < "${path}")"
    if [[ "${bytes}" =~ ^[0-9]+$ ]]; then
      echo "${bytes}"
      return 0
    fi
  fi
  return 1
}

auto_select_hip_visible_devices() {
  if [[ "$(resolve_bool "${GPU_AUTO_SELECT:-1}")" != "1" ]]; then
    return 0
  fi

  local raw="${HIP_VISIBLE_DEVICES:-}"
  if [[ -n "${raw}" && "${raw,,}" != "auto" ]]; then
    echo "using caller-supplied HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}"
    return 0
  fi

  if ! command -v rocm-smi >/dev/null 2>&1; then
    echo "error: HIP_VISIBLE_DEVICES=auto requires rocm-smi so ROCm device ordinals can be matched to ${GPU_TARGET_ARCH:-gfx906}." >&2
    echo "       install rocm-smi or set HIP_VISIBLE_DEVICES explicitly, for example HIP_VISIBLE_DEVICES=0,1,2,3." >&2
    return 1
  fi

  local min_gpus="${REQUIRED_GPU_COUNT:-4}"
  local min_vram_gib="${REQUIRED_GPU_VRAM_GIB:-32}"
  local gpu_ids
  local gpu_id
  local arch
  local vram_gib
  local old_ifs
  local -a selected_gpus=()

  gpu_ids="$(list_rocm_gpu_ids || true)"
  if [[ -z "${gpu_ids}" ]]; then
    echo "error: rocm-smi did not report any GPUs; cannot auto-select a ${GPU_TARGET_ARCH:-gfx906} lane." >&2
    return 1
  fi

  while IFS= read -r gpu_id; do
    [[ -n "${gpu_id}" ]] || continue

    arch="$(gpu_arch_rocm_smi "${gpu_id}" || true)"
    if ! gpu_arch_matches_target "${arch}"; then
      continue
    fi

    vram_gib="$(gpu_vram_gib_sysfs "${gpu_id}" || true)"
    if [[ -z "${vram_gib}" ]]; then
      continue
    fi
    if (( vram_gib < min_vram_gib )); then
      continue
    fi

    selected_gpus+=("${gpu_id}")
    if (( ${#selected_gpus[@]} >= min_gpus )); then
      break
    fi
  done <<< "${gpu_ids}"

  if (( ${#selected_gpus[@]} < min_gpus )); then
    echo "error: unable to auto-select ${min_gpus} GPU(s) matching ${GPU_TARGET_ARCH:-gfx906} with >= ${min_vram_gib}GiB VRAM." >&2
    echo "       Discovered ROCm GPUs:" >&2
    while IFS= read -r gpu_id; do
      [[ -n "${gpu_id}" ]] || continue
      arch="$(gpu_arch_rocm_smi "${gpu_id}" || echo unknown)"
      vram_gib="$(gpu_vram_gib_sysfs "${gpu_id}" || echo unknown)"
      echo "       GPU[${gpu_id}]: arch=${arch} vram=${vram_gib}GiB" >&2
    done <<< "${gpu_ids}"
    echo "       Set HIP_VISIBLE_DEVICES explicitly to override auto-selection." >&2
    return 1
  fi

  old_ifs="${IFS}"
  IFS=','
  HIP_VISIBLE_DEVICES="${selected_gpus[*]}"
  IFS="${old_ifs}"
  export HIP_VISIBLE_DEVICES

  echo "auto-selected HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES} for ${REQUIRED_GPU_MODEL_HINT:-target GPUs} (target arch ${GPU_TARGET_ARCH:-gfx906}, aliases ${GPU_TARGET_ARCH_ALIASES:-${GPU_TARGET_ARCH:-gfx906}})"
  return 0
}

format_gib() {
  local bytes="$1"
  awk -v bytes="${bytes}" 'BEGIN { printf "%.2f", bytes / 1073741824 }'
}

check_gpu_vram_requirements() {
  if [[ "$(resolve_bool "${SKIP_GPU_MEMORY_CHECK}")" == "1" ]]; then
    echo "warning: SKIP_GPU_MEMORY_CHECK=1, skipping GPU VRAM preflight"
    return 0
  fi

  local min_gpus="${REQUIRED_GPU_COUNT:-4}"
  local min_vram_gib="${REQUIRED_GPU_VRAM_GIB:-32}"
  local -a gpu_list=()
  local gpu_id
  local found_count=0
  local missing_count=0
  local device_label
  local vram_gib

  if ! command -v rocm-smi >/dev/null 2>&1; then
    echo "warning: rocm-smi not found; using sysfs for GPU VRAM probe only"
  fi

  normalize_gpu_device_list "${HIP_VISIBLE_DEVICES}" gpu_list
  if [[ "${#gpu_list[@]}" -eq 0 ]]; then
    echo "error: HIP_VISIBLE_DEVICES is empty; unable to identify target GPUs" >&2
    return 1
  fi

  if (( ${#gpu_list[@]} < min_gpus )); then
    echo "error: HIP_VISIBLE_DEVICES only provides ${#gpu_list[@]} devices, but ${min_gpus} are required for ${REQUIRED_GPU_MODEL_HINT:-this configuration}" >&2
    return 1
  fi

  for gpu_id in "${gpu_list[@]}"; do
    if (( found_count >= min_gpus )); then
      break
    fi

    vram_gib="$(gpu_vram_gib_sysfs "${gpu_id}" || true)"
    if [[ -z "${vram_gib}" ]]; then
      echo "warning: could not read VRAM for ROCm device ${gpu_id}; skipping this device"
      ((missing_count++))
      continue
    fi

    if (( vram_gib < min_vram_gib )); then
      device_label="$(printf '%s (%s GiB)' "${gpu_id}" "${vram_gib}")"
      echo "error: ${device_label} is below minimum ${min_vram_gib}GiB required for ${REQUIRED_GPU_MODEL_HINT:-this config}" >&2
      return 1
    fi

    ((found_count++))
  done

  if (( found_count < min_gpus )); then
    echo "error: insufficient readable GPU VRAM entries for ${REQUIRED_GPU_MODEL_HINT:-this config}; expected ${min_gpus} and found ${found_count} readable GPUs" >&2
    if (( missing_count > 0 )); then
      echo "       this is commonly caused by unsupported ROCm sysfs layout; rerun with SKIP_GPU_MEMORY_CHECK=1 only if you have validated VRAM manually." >&2
    fi
    return 1
  fi

  echo "GPU VRAM preflight passed for ${REQUIRED_GPU_MODEL_HINT:-target GPUs}: checking first ${found_count} GPU(s) from HIP_VISIBLE_DEVICES for >= ${min_vram_gib}GiB each"
  return 0
}

check_gpu_free_memory_requirements() {
  if [[ "$(resolve_bool "${SKIP_GPU_MEMORY_CHECK}")" == "1" || "$(resolve_bool "${SKIP_GPU_FREE_MEMORY_CHECK}")" == "1" ]]; then
    echo "warning: skipping GPU free-memory preflight"
    return 0
  fi

  local min_gpus="${REQUIRED_GPU_COUNT:-4}"
  local -a gpu_list=()
  local gpu_id
  local checked_count=0
  local total_bytes
  local used_bytes
  local free_bytes
  local required_bytes
  local required_display
  local free_display

  normalize_gpu_device_list "${HIP_VISIBLE_DEVICES}" gpu_list
  if [[ "${#gpu_list[@]}" -eq 0 ]]; then
    echo "error: HIP_VISIBLE_DEVICES is empty; unable to identify target GPUs for free-memory check" >&2
    return 1
  fi

  if (( ${#gpu_list[@]} < min_gpus )); then
    echo "error: HIP_VISIBLE_DEVICES only provides ${#gpu_list[@]} devices, but ${min_gpus} are required for ${REQUIRED_GPU_MODEL_HINT:-this configuration}" >&2
    return 1
  fi

  for gpu_id in "${gpu_list[@]}"; do
    if (( checked_count >= min_gpus )); then
      break
    fi

    total_bytes="$(gpu_vram_bytes_sysfs "${gpu_id}" total || true)"
    used_bytes="$(gpu_vram_bytes_sysfs "${gpu_id}" used || true)"
    if [[ -z "${total_bytes}" || -z "${used_bytes}" ]]; then
      echo "error: could not read current VRAM usage for ROCm device ${gpu_id}; set SKIP_GPU_FREE_MEMORY_CHECK=1 only after checking the lane manually." >&2
      return 1
    fi

    if (( used_bytes > total_bytes )); then
      echo "error: invalid VRAM accounting for ROCm device ${gpu_id}: used bytes exceed total bytes" >&2
      return 1
    fi

    free_bytes=$(( total_bytes - used_bytes ))
    if [[ -n "${REQUIRED_GPU_FREE_VRAM_GIB:-}" ]]; then
      required_bytes="$(awk -v gib="${REQUIRED_GPU_FREE_VRAM_GIB}" 'BEGIN { printf "%.0f", gib * 1073741824 }')"
      required_display="${REQUIRED_GPU_FREE_VRAM_GIB}"
    else
      required_bytes="$(awk -v total="${total_bytes}" -v util="${GPU_MEMORY_UTILIZATION:-0.95}" 'BEGIN { printf "%.0f", total * util }')"
      required_display="$(format_gib "${required_bytes}")"
    fi
    free_display="$(format_gib "${free_bytes}")"

    if (( free_bytes < required_bytes )); then
      echo "error: ROCm device ${gpu_id} has ${free_display}GiB free, but this profile needs at least ${required_display}GiB free before start (GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.95})." >&2
      echo "       stop other GPU processes on the selected HIP_VISIBLE_DEVICES lane, lower GPU_MEMORY_UTILIZATION, or set SKIP_GPU_FREE_MEMORY_CHECK=1 only for build-only runs." >&2
      return 1
    fi

    ((checked_count++))
  done

  echo "GPU free-memory preflight passed for ${REQUIRED_GPU_MODEL_HINT:-target GPUs}: first ${checked_count} GPU(s) have enough free VRAM for GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.95}"
  return 0
}

run_as_root() {
  local purpose="$1"
  shift

  if [[ "${EUID}" == "0" ]]; then
    "$@"
  else
    if ! command -v sudo >/dev/null 2>&1; then
      echo "error: sudo is required for: ${purpose}" >&2
      exit 1
    fi

    echo "sudo is required for: ${purpose}" >&2
    echo "command: sudo $*" >&2
    if [[ ! -t 0 ]]; then
      echo "error: interactive input unavailable, defaulting to NO." >&2
      exit 1
    fi

    local choice
    read -r -p "Type y or yes to continue with sudo [y/N]: " choice
    case "${choice,,}" in
      y|yes)
        ;;
      *)
        echo "error: sudo action declined; exiting." >&2
        exit 1
        ;;
    esac

    sudo "$@"
  fi
}

lazy_unmount_tree() {
  local root="$1"
  if ! command -v findmnt >/dev/null 2>&1; then
    return 0
  fi

  local mounts=()
  while IFS= read -r mount_point; do
    [[ -n "${mount_point}" ]] && mounts+=("${mount_point}")
  done < <(findmnt --raw -n -o TARGET | grep -F "${root}" | sort -r || true)

  if (( ${#mounts[@]} > 0 )); then
    run_as_root "lazy-unmount private Docker/containerd mounts under ${root}" umount -l "${mounts[@]}" || true
  fi
}

wait_for_docker_ready() {
  local timeout_seconds="${1:-60}"
  local start_ts
  start_ts="$(date +%s)"
  while (( $(date +%s) - start_ts < timeout_seconds )); do
    if docker version >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

restart_docker_service() {
  if command -v systemctl >/dev/null 2>&1; then
    run_as_root "restart the system Docker service after changing or restoring Docker daemon settings" systemctl restart docker
  elif command -v service >/dev/null 2>&1; then
    run_as_root "restart the system Docker service after changing or restoring Docker daemon settings" service docker restart
  else
    echo 'error: no supported method found to restart docker service (systemctl/service)' >&2
    return 1
  fi

  if ! wait_for_docker_ready 120; then
    echo 'error: docker did not recover after service restart' >&2
    return 1
  fi
}

shell_quote() {
  local value="$1"
  printf "'%s'" "${value//\'/\'\\\'\'}"
}

start_root_background() {
  local purpose="$1"
  local log_file="$2"
  local pid_file="$3"
  shift 3
  local cmd=""
  local arg

  for arg in "$@"; do
    cmd="${cmd} $(shell_quote "${arg}")"
  done

  run_as_root "${purpose}" sh -c "nohup${cmd} > $(shell_quote "${log_file}") 2>&1 & echo \$! > $(shell_quote "${pid_file}")"
}

wait_for_socket_path() {
  local socket_path="$1"
  local timeout_seconds="${2:-60}"
  local start_ts
  start_ts="$(date +%s)"
  while (( $(date +%s) - start_ts < timeout_seconds )); do
    if [[ -S "${socket_path}" ]]; then
      return 0
    fi
    sleep 1
  done
  return 1
}

write_isolated_containerd_config() {
  local config_path="$1"
  local root_dir="$2"
  local state_dir="$3"
  local socket_path="$4"

  cat > "${config_path}" <<EOF
version = 2
root = "${root_dir}"
state = "${state_dir}"

[grpc]
  address = "${socket_path}"

[plugins."io.containerd.grpc.v1.cri"]
  disable_apparmor = true
EOF
}

start_isolated_docker_daemon() {
  DOCKER_ISOLATED_DAEMON_DIR="$(abs_path "${DOCKER_ISOLATED_DAEMON_DIR:-./.d}" "${RUN_DIR}")"
  mkdir -p "${DOCKER_ISOLATED_DAEMON_DIR}"
  DOCKER_ISOLATED_DAEMON_DIR="$(cd "${DOCKER_ISOLATED_DAEMON_DIR}" && pwd -P)"
  DOCKER_ISOLATED_CONTAINERD_ROOT="${DOCKER_ISOLATED_DAEMON_DIR}/c"
  DOCKER_ISOLATED_CONTAINERD_STATE="${DOCKER_ISOLATED_DAEMON_DIR}/s"
  DOCKER_ISOLATED_DOCKER_ROOT="${DOCKER_ISOLATED_DAEMON_DIR}/d"
  DOCKER_ISOLATED_DOCKER_EXEC="${DOCKER_ISOLATED_DAEMON_DIR}/e"
  DOCKER_ISOLATED_RUN="${DOCKER_ISOLATED_DAEMON_DIR}/r"
  DOCKER_ISOLATED_LOG_DIR="${DOCKER_ISOLATED_DAEMON_DIR}/l"
  DOCKER_ISOLATED_CONTAINERD_SOCKET="${DOCKER_ISOLATED_RUN}/containerd.sock"
  DOCKER_ISOLATED_DOCKER_SOCKET="${DOCKER_ISOLATED_RUN}/docker.sock"
  DOCKER_ISOLATED_CONTAINERD_CONFIG="${DOCKER_ISOLATED_DAEMON_DIR}/containerd.toml"

  mkdir -p \
    "${DOCKER_ISOLATED_CONTAINERD_ROOT}" \
    "${DOCKER_ISOLATED_CONTAINERD_STATE}" \
    "${DOCKER_ISOLATED_DOCKER_ROOT}" \
    "${DOCKER_ISOLATED_DOCKER_EXEC}" \
    "${DOCKER_ISOLATED_RUN}" \
    "${DOCKER_ISOLATED_LOG_DIR}"

  check_free_space "isolated docker/containerd root" "${DOCKER_ISOLATED_DAEMON_DIR}" "${MIN_FREE_GIB_TEMP_DOCKER_DATA_ROOT}"

  if ! command -v containerd >/dev/null 2>&1; then
    echo "error: containerd is required for isolated Docker mode" >&2
    exit 1
  fi
  if ! command -v dockerd >/dev/null 2>&1; then
    echo "error: dockerd is required for isolated Docker mode" >&2
    exit 1
  fi

  export DOCKER_HOST="unix://${DOCKER_ISOLATED_DOCKER_SOCKET}"
  if [[ -S "${DOCKER_ISOLATED_DOCKER_SOCKET}" ]] && docker version >/dev/null 2>&1; then
    echo "reusing isolated docker daemon: ${DOCKER_ISOLATED_DOCKER_SOCKET}"
  else
    run_as_root "remove stale private Docker/containerd sockets and pid files under ${DOCKER_ISOLATED_RUN}" rm -f \
      "${DOCKER_ISOLATED_CONTAINERD_SOCKET}" \
      "${DOCKER_ISOLATED_CONTAINERD_SOCKET}.ttrpc" \
      "${DOCKER_ISOLATED_DOCKER_SOCKET}" \
      "${DOCKER_ISOLATED_RUN}/containerd.pid" \
      "${DOCKER_ISOLATED_RUN}/dockerd-daemon.pid" \
      "${DOCKER_ISOLATED_RUN}/dockerd-wrapper.pid"
    write_isolated_containerd_config \
      "${DOCKER_ISOLATED_CONTAINERD_CONFIG}" \
      "${DOCKER_ISOLATED_CONTAINERD_ROOT}" \
      "${DOCKER_ISOLATED_CONTAINERD_STATE}" \
      "${DOCKER_ISOLATED_CONTAINERD_SOCKET}"

    start_root_background \
      "start the private containerd daemon with root under ${DOCKER_ISOLATED_CONTAINERD_ROOT}" \
      "${DOCKER_ISOLATED_LOG_DIR}/containerd.log" \
      "${DOCKER_ISOLATED_RUN}/containerd.pid" \
      containerd --config "${DOCKER_ISOLATED_CONTAINERD_CONFIG}"

    if ! wait_for_socket_path "${DOCKER_ISOLATED_CONTAINERD_SOCKET}" 60; then
      echo "error: isolated containerd socket did not appear: ${DOCKER_ISOLATED_CONTAINERD_SOCKET}" >&2
      echo "containerd log: ${DOCKER_ISOLATED_LOG_DIR}/containerd.log" >&2
      exit 1
    fi

    start_root_background \
      "start the private Docker daemon with data-root ${DOCKER_ISOLATED_DOCKER_ROOT} and containerd socket ${DOCKER_ISOLATED_CONTAINERD_SOCKET}" \
      "${DOCKER_ISOLATED_LOG_DIR}/dockerd.log" \
      "${DOCKER_ISOLATED_RUN}/dockerd-wrapper.pid" \
      dockerd \
      --config-file /dev/null \
      --host "unix://${DOCKER_ISOLATED_DOCKER_SOCKET}" \
      --data-root "${DOCKER_ISOLATED_DOCKER_ROOT}" \
      --exec-root "${DOCKER_ISOLATED_DOCKER_EXEC}" \
      --pidfile "${DOCKER_ISOLATED_RUN}/dockerd-daemon.pid" \
      --containerd "${DOCKER_ISOLATED_CONTAINERD_SOCKET}" \
      --iptables=false \
      --ip-forward=false \
      --ip-masq=false \
      --bridge=none

    if ! wait_for_docker_ready 120; then
      echo "error: isolated dockerd did not become ready: ${DOCKER_ISOLATED_DOCKER_SOCKET}" >&2
      echo "dockerd log: ${DOCKER_ISOLATED_LOG_DIR}/dockerd.log" >&2
      exit 1
    fi
  fi

  local reported_root
  reported_root="$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || true)"
  if [[ "${reported_root}" != "${DOCKER_ISOLATED_DOCKER_ROOT}" ]]; then
    echo "error: isolated Docker root mismatch" >&2
    echo "       expected: ${DOCKER_ISOLATED_DOCKER_ROOT}" >&2
    echo "       actual:   ${reported_root}" >&2
    exit 1
  fi

  cat > "${RUN_DIR}/.deploy.docker-host.env" <<EOF
export DOCKER_HOST=unix://${DOCKER_ISOLATED_DOCKER_SOCKET}
export DOCKER_ISOLATED_DAEMON_DIR=${DOCKER_ISOLATED_DAEMON_DIR}
EOF

  echo "using isolated docker daemon: ${DOCKER_ISOLATED_DOCKER_SOCKET}"
  echo "isolated docker/containerd root: ${DOCKER_ISOLATED_DAEMON_DIR}"
}

write_docker_daemon_json() {
  local source_path="$1"
  local data_root="$2"
  local output_path="$3"

  if [[ -f "${source_path}" ]]; then
    python3 - "${source_path}" "${data_root}" <<'PY' > "${output_path}"
import json
import sys

source = sys.argv[1]
data_root = sys.argv[2]

with open(source, "r", encoding="utf-8") as fh:
    config = json.load(fh)
if not isinstance(config, dict):
    config = {}
config["data-root"] = data_root
print(json.dumps(config, indent=2, sort_keys=True))
PY
  else
    python3 - "${data_root}" <<'PY' > "${output_path}"
import json
import sys

print(json.dumps({"data-root": sys.argv[1]}, indent=2, sort_keys=True))
PY
  fi
}

check_free_space() {
  local label="$1"
  local path="$2"
  local min_gib="$3"
  local available_gib

  if [[ ! -d "${path}" ]]; then
    echo "error: path for ${label} check does not exist: ${path}" >&2
    exit 1
  fi

  available_gib="$(df -BG "${path}" | awk 'NR==2 { gsub(/G/, "", $4); print $4 }')"
  if ! [[ "${available_gib}" =~ ^[0-9]+$ ]]; then
    echo "error: could not parse available space for ${label} at ${path}; disk-space check failed." >&2
    exit 1
  fi

  if (( available_gib < min_gib )); then
    echo "error: insufficient free space for ${label} (${path})" >&2
    echo "       available: ${available_gib}GiB, required: ${min_gib}GiB" >&2
    exit 1
  fi

  echo "space check passed for ${label}: ${available_gib}GiB available (min ${min_gib}GiB)"
}

required_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "error: required file does not exist: ${path}" >&2
    exit 1
  fi
}

copy_patch() {
  local src="$1"
  local dst="$2"
  if [[ ! -f "${src}" ]]; then
    echo "error: missing patch source: ${src}" >&2
    exit 1
  fi
  mkdir -p "$(dirname "${dst}")"
  cp -a "${src}" "${dst}"
}

copy_if_present() {
  local src="$1"
  local dst="$2"
  if [[ -z "${src}" ]]; then
    return 0
  fi
  if [[ -f "${src}" ]]; then
    mkdir -p "$(dirname "${dst}")"
    cp -a "${src}" "${dst}"
  fi
}

wait_for_container_running() {
  local container="$1"
  local timeout_seconds="${2:-120}"
  local elapsed=0
  local status running exit_code

  while (( elapsed < timeout_seconds )); do
    if ! docker inspect "${container}" >/dev/null 2>&1; then
      sleep 2
      ((elapsed += 2))
      continue
    fi

    status="$(docker inspect -f '{{.State.Status}}' "${container}" 2>/dev/null || true)"
    running="$(docker inspect -f '{{.State.Running}}' "${container}" 2>/dev/null || true)"
    exit_code="$(docker inspect -f '{{.State.ExitCode}}' "${container}" 2>/dev/null || true)"

    if [[ "${running}" == "true" ]]; then
      echo "container ${container} is running (${status})"
      return 0
    fi

    if [[ "${status}" == "exited" || "${status}" == "dead" ]]; then
      echo "error: container ${container} exited during startup (status=${status}, exit_code=${exit_code})" >&2
      docker logs --tail 200 "${container}" || true
      return 1
    fi

    sleep 2
    ((elapsed += 2))
  done

  echo "error: timeout waiting for ${container} to stay up for startup check (${timeout_seconds}s)" >&2
  docker logs --tail 200 "${container}" || true
  return 1
}

probe_vllm_ready() {
  local port="$1"

  if command -v curl >/dev/null 2>&1; then
    curl -fsS --max-time 5 "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1
    return $?
  fi

  if ! { exec 3<>"/dev/tcp/127.0.0.1/${port}"; } 2>/dev/null; then
    return 1
  fi

  printf 'GET /v1/models HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n' >&3
  local status_line
  IFS= read -r status_line <&3 || true
  exec 3<&-
  exec 3>&-
  [[ "${status_line}" == *" 200 "* ]]
}

wait_for_vllm_ready() {
  local container="$1"
  local timeout_seconds="${2:-3600}"
  local start_ts now elapsed running status exit_code
  local last_progress_ts=0

  if [[ "$(resolve_bool "${SKIP_READY_CHECK:-0}")" == "1" ]]; then
    echo "warning: skipping vLLM API readiness check because SKIP_READY_CHECK=1"
    return 0
  fi

  echo "waiting up to ${timeout_seconds}s for vLLM API readiness at http://127.0.0.1:${PORT}/v1/models"
  echo "first run can take a long time while weights download and Triton kernels compile into ${RUNTIME_ROOT}"

  start_ts="$(date +%s)"
  while true; do
    now="$(date +%s)"
    elapsed=$((now - start_ts))
    if (( elapsed >= timeout_seconds )); then
      echo "error: timeout waiting for vLLM API readiness after ${timeout_seconds}s" >&2
      docker logs --tail 240 "${container}" || true
      return 1
    fi

    status="$(docker inspect -f '{{.State.Status}}' "${container}" 2>/dev/null || true)"
    running="$(docker inspect -f '{{.State.Running}}' "${container}" 2>/dev/null || true)"
    exit_code="$(docker inspect -f '{{.State.ExitCode}}' "${container}" 2>/dev/null || true)"

    if [[ "${running}" != "true" ]]; then
      echo "error: container ${container} stopped before vLLM API became ready (status=${status}, exit_code=${exit_code})" >&2
      docker logs --tail 240 "${container}" || true
      return 1
    fi

    if probe_vllm_ready "${PORT}"; then
      echo "vLLM API is ready: http://127.0.0.1:${PORT}/v1"
      return 0
    fi

    if (( now - last_progress_ts >= 60 )); then
      echo "still waiting for vLLM API readiness (${elapsed}s elapsed); recent container logs:"
      docker logs --since 70s --tail 40 "${container}" || true
      last_progress_ts="${now}"
    fi

    sleep 5
  done
}

seed_runtime_triton_cache_from_image() {
  if [[ "$(resolve_bool "${SKIP_TRITON_CACHE_SEED:-0}")" == "1" ]]; then
    echo "skipping Triton cache seed because SKIP_TRITON_CACHE_SEED=1"
    return 0
  fi

  if [[ -d "${TRITON_CACHE_HOST}/llvm" && -d "${TRITON_CACHE_HOST}/nvidia" ]]; then
    echo "reusing existing Triton toolchain cache at ${TRITON_CACHE_HOST}"
    return 0
  fi

  mkdir -p "${RUNTIME_ROOT}/root" "${TRITON_CACHE_HOST}"
  if ! [[ "${TRITON_CACHE_SEED_TIMEOUT_SECONDS}" =~ ^[0-9]+$ ]] || (( TRITON_CACHE_SEED_TIMEOUT_SECONDS <= 0 )); then
    echo "warning: invalid TRITON_CACHE_SEED_TIMEOUT_SECONDS=${TRITON_CACHE_SEED_TIMEOUT_SECONDS}; using 180s" >&2
    TRITON_CACHE_SEED_TIMEOUT_SECONDS=180
  fi

  echo "seeding Triton toolchain cache from image into ${TRITON_CACHE_HOST} (timeout ${TRITON_CACHE_SEED_TIMEOUT_SECONDS}s)"
  if ! timeout "${TRITON_CACHE_SEED_TIMEOUT_SECONDS}s" docker run --rm --pull=never --entrypoint tar "${DEPLOY_IMAGE}" -C /root -cf - .triton | tar --no-same-owner -xf - -C "${RUNTIME_ROOT}/root"; then
    echo "warning: could not seed Triton cache from image; first vLLM start may download/prepare Triton toolchains" >&2
    return 0
  fi
}

download_file() {
  local url="$1"
  local dest="$2"

  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 5 --retry-delay 2 -o "${dest}" "${url}"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "${dest}" "${url}"
  else
    echo "error: curl or wget is required to download pinned Buildx" >&2
    return 1
  fi
}

ensure_pinned_buildx() {
  if [[ "$(resolve_bool "${PINNED_BUILDX_ENABLED:-1}")" != "1" ]]; then
    return 0
  fi

  local arch
  arch="$(uname -m)"
  if [[ "${arch}" != "x86_64" && "${arch}" != "amd64" ]]; then
    echo "error: pinned Buildx bootstrap currently supports linux/amd64 only, got ${arch}" >&2
    exit 1
  fi

  export DOCKER_CONFIG="${DOCKER_CONFIG:-${RUN_DIR}/.docker-cli}"
  local plugin_dir="${DOCKER_CONFIG}/cli-plugins"
  local plugin="${plugin_dir}/docker-buildx"
  local expected_sha="${PINNED_BUILDX_SHA256_AMD64}"
  local url="${PINNED_BUILDX_URL_AMD64}"

  mkdir -p "${plugin_dir}" "${RUN_DIR}/tools"
  if [[ ! -x "${plugin}" ]] || ! printf '%s  %s\n' "${expected_sha}" "${plugin}" | sha256sum -c - >/dev/null 2>&1; then
    local tmp="${RUN_DIR}/tools/docker-buildx-v${PINNED_BUILDX_VERSION}.linux-amd64.tmp"
    echo "installing pinned Docker Buildx v${PINNED_BUILDX_VERSION} into ${plugin}"
    rm -f "${tmp}"
    download_file "${url}" "${tmp}"
    printf '%s  %s\n' "${expected_sha}" "${tmp}" | sha256sum -c -
    chmod +x "${tmp}"
    mv -f "${tmp}" "${plugin}"
  fi

  local version_out
  version_out="$(docker buildx version)"
  echo "using ${version_out}"
  if [[ "${version_out}" != *"v${PINNED_BUILDX_VERSION} "* ]]; then
    echo "error: expected Docker Buildx v${PINNED_BUILDX_VERSION}, got: ${version_out}" >&2
    exit 1
  fi
}

ensure_pinned_buildkit_builder() {
  REPRO_BUILDX_BUILDER=''
  if [[ "$(resolve_bool "${PINNED_BUILDKIT_ENABLED:-1}")" != "1" ]]; then
    return 0
  fi

  local builder="${PINNED_BUILDKIT_BUILDER_NAME}"
  if [[ "${PINNED_BUILDKIT_BUILDER_NAME_SCOPE}" == "run-dir" ]]; then
    local builder_scope
    builder_scope="$(printf '%s' "${RUN_DIR}:${DOCKER_ISOLATED_DAEMON_DIR}:${DEPLOY_IMAGE}" | sha256sum | awk '{print substr($1, 1, 12)}')"
    builder="${builder}-${builder_scope}"
  fi
  local inspect_log="${RUN_DIR}/.repro-buildkit-builder.inspect"
  local rm_failed=0

  if command -v timeout >/dev/null 2>&1; then
    if ! timeout "${PINNED_BUILDKIT_RM_TIMEOUT_SECONDS}s" docker buildx rm -f "${builder}" >/dev/null 2>&1; then
      echo "warning: stale Buildx builder removal for ${builder} failed or timed out after ${PINNED_BUILDKIT_RM_TIMEOUT_SECONDS}s; continuing" >&2
      rm_failed=1
    fi
  else
    if ! docker buildx rm -f "${builder}" >/dev/null 2>&1; then
      rm_failed=1
    fi
  fi
  if [[ "${rm_failed}" == "1" ]]; then
    local fallback_scope
    fallback_scope="$(printf '%s' "${builder}:fallback:$$:$(date +%s)" | sha256sum | awk '{print substr($1, 1, 12)}')"
    builder="${builder}-${fallback_scope}"
    echo "warning: using fallback Buildx builder name ${builder} because stale builder cleanup did not complete" >&2
  fi
  docker buildx create \
    --name "${builder}" \
    --driver docker-container \
    --driver-opt "image=${PINNED_BUILDKIT_IMAGE}" \
    --driver-opt "network=host" \
    --use >/dev/null

  local attempt inspect_rc
  for attempt in 1 2 3 4 5; do
    inspect_rc=0
    if command -v timeout >/dev/null 2>&1; then
      timeout "${PINNED_BUILDKIT_INSPECT_TIMEOUT_SECONDS}s" docker buildx inspect --bootstrap "${builder}" >"${inspect_log}" 2>&1 || inspect_rc=$?
    else
      docker buildx inspect --bootstrap "${builder}" >"${inspect_log}" 2>&1 || inspect_rc=$?
    fi
    if [[ "${inspect_rc}" == "0" ]] && grep -q "BuildKit version:[[:space:]]*v${PINNED_BUILDKIT_VERSION}" "${inspect_log}"; then
      cat "${inspect_log}"
      break
    fi
    if [[ "${inspect_rc}" == "124" ]]; then
      echo "warning: BuildKit bootstrap inspection for ${builder} timed out after ${PINNED_BUILDKIT_INSPECT_TIMEOUT_SECONDS}s" >&2
    fi
    cat "${inspect_log}" >&2
    if (( attempt < 5 )); then
      echo "warning: BuildKit bootstrap inspection failed on attempt ${attempt}; retrying in 10s" >&2
      sleep 10
    fi
  done
  if ! grep -q "BuildKit version:[[:space:]]*v${PINNED_BUILDKIT_VERSION}" "${inspect_log}"; then
    echo "error: expected BuildKit v${PINNED_BUILDKIT_VERSION}; inspect output:" >&2
    cat "${inspect_log}" >&2
    exit 1
  fi

  REPRO_BUILDX_BUILDER="${builder}"
}

probe_reproducible_buildx_exporter() {
  if [[ "$(resolve_bool "${DOCKER_REPRODUCIBLE_EXPORTER_PROBE:-1}")" != "1" ]]; then
    return 0
  fi

  local probe_tag="${DEPLOY_IMAGE}:buildx-rewrite-probe"
  local probe_dir="${RUN_DIR}/.repro-buildx-exporter-probe"
  local marker="${RUN_DIR}/.repro-buildx-exporter-probe.ok"
  local marker_version="buildx=v${PINNED_BUILDX_VERSION},buildkit=v${PINNED_BUILDKIT_VERSION},type=docker,dest,rewrite-timestamp=true,copy-layer-probe=1"
  if [[ -f "${marker}" ]] && [[ "$(cat "${marker}" 2>/dev/null || true)" == "${marker_version}" ]]; then
    return 0
  fi

  rm -rf "${probe_dir}" "${marker}"
  mkdir -p "${probe_dir}"
  local probe_tar="${probe_dir}/probe.docker.tar"
  printf 'source-date-epoch=%s\n' "${SOURCE_DATE_EPOCH}" > "${probe_dir}/marker.txt"
  cat > "${probe_dir}/Dockerfile" <<'EOF'
ARG SOURCE_DATE_EPOCH=1764000000
FROM scratch
ARG SOURCE_DATE_EPOCH=1764000000
LABEL org.opencontainers.image.created="${SOURCE_DATE_EPOCH}"
COPY marker.txt /marker.txt
EOF

  echo "checking BuildKit rewrite-timestamp docker exporter support"
  local probe_args=(
    docker buildx build
  )
  if [[ -n "${REPRO_BUILDX_BUILDER:-}" ]]; then
    probe_args+=(--builder "${REPRO_BUILDX_BUILDER}")
  fi
  probe_args+=(
    --progress=plain \
    --network=none \
    --provenance=false \
    --sbom=false \
    --platform linux/amd64 \
    --build-arg "SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH}" \
    -t "${probe_tag}" \
    --output "type=docker,dest=${probe_tar},rewrite-timestamp=true" \
    "${probe_dir}"
  )
  "${probe_args[@]}"
  sha256sum "${probe_tar}"
  docker load -i "${probe_tar}" >/dev/null
  docker image inspect "${probe_tag}" --format '{{.Id}} {{.Created}} {{json .RootFS.Layers}}'
  docker image rm -f "${probe_tag}" >/dev/null 2>&1 || true
  rm -rf "${probe_dir}"
  printf '%s\n' "${marker_version}" > "${marker}"
}

probe_reproducible_buildctl_daemonless_exporter() {
  if [[ "$(resolve_bool "${DOCKER_REPRODUCIBLE_EXPORTER_PROBE:-1}")" != "1" ]]; then
    return 0
  fi

  local probe_tag="${DEPLOY_IMAGE}:buildctl-daemonless-rewrite-probe"
  local probe_dir="${RUN_DIR}/.repro-buildctl-daemonless-probe"
  local probe_out="${probe_dir}/out"
  local marker="${RUN_DIR}/.repro-buildctl-daemonless-probe.ok"
  local marker_version="buildkit-image=${PINNED_BUILDKIT_IMAGE},type=docker,dest,rewrite-timestamp=true,copy-layer-probe=1"
  if [[ -f "${marker}" ]] && [[ "$(cat "${marker}" 2>/dev/null || true)" == "${marker_version}" ]]; then
    return 0
  fi

  rm -rf "${probe_dir}" "${marker}"
  mkdir -p "${probe_dir}/context" "${probe_out}"
  local probe_tar="${probe_out}/probe.docker.tar"
  printf 'source-date-epoch=%s\n' "${SOURCE_DATE_EPOCH}" > "${probe_dir}/context/marker.txt"
  cat > "${probe_dir}/context/Dockerfile" <<'EOF'
ARG SOURCE_DATE_EPOCH=1764000000
FROM scratch
ARG SOURCE_DATE_EPOCH=1764000000
LABEL org.opencontainers.image.created="${SOURCE_DATE_EPOCH}"
COPY marker.txt /marker.txt
EOF

  echo "checking daemonless BuildKit rewrite-timestamp docker exporter support"
  docker run --rm --privileged --network host \
    --entrypoint buildctl-daemonless.sh \
    -e "BUILDKITD_FLAGS=--allow-insecure-entitlement network.host" \
    -v "${probe_dir}/context:/workspace:ro" \
    -v "${probe_out}:/out" \
    "${PINNED_BUILDKIT_IMAGE}" \
    build \
      --frontend dockerfile.v0 \
      --local context=/workspace \
      --local dockerfile=/workspace \
      --opt filename=Dockerfile \
      --opt platform=linux/amd64 \
      --opt "build-arg:SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH}" \
      --output "type=docker,name=${probe_tag},dest=/out/$(basename "${probe_tar}"),rewrite-timestamp=true" \
      --progress=plain

  sha256sum "${probe_tar}"
  docker load -i "${probe_tar}" >/dev/null
  docker image inspect "${probe_tag}" --format '{{.Id}} {{.Created}} {{json .RootFS.Layers}}'
  docker image rm -f "${probe_tag}" >/dev/null 2>&1 || true
  rm -rf "${probe_dir}"
  printf '%s\n' "${marker_version}" > "${marker}"
}

build_image_reproducibly_with_buildctl_daemonless() {
  probe_reproducible_buildctl_daemonless_exporter

  local archive_path="${REPRO_DOCKER_ARCHIVE_PATH}"
  local keep_archive=1
  if [[ "$(resolve_bool "${REPRO_EXPORT_DOCKER_ARCHIVE:-1}")" != "1" ]]; then
    archive_path="${REPRO_DOCKER_ARCHIVE_DIR}/${DEPLOY_IMAGE//\//_}.daemonless-load.docker.tar"
    keep_archive=0
  fi

  local archive_dir archive_name dockerfile_rel
  archive_dir="$(dirname "${archive_path}")"
  archive_name="$(basename "${archive_path}")"
  mkdir -p "${archive_dir}"
  rm -f "${archive_path}" "${archive_path}.sha256"

  if [[ "${DEPLOY_DOCKERFILE_PATH}" == "${RUN_DIR}"/* ]]; then
    dockerfile_rel="${DEPLOY_DOCKERFILE_PATH#"${RUN_DIR}/"}"
  else
    echo "error: daemonless BuildKit export requires DOCKERFILE_PATH to be inside the run directory: ${RUN_DIR}" >&2
    exit 1
  fi

  echo "building reproducible docker archive with daemonless pinned BuildKit"
  echo "BuildKit image: ${PINNED_BUILDKIT_IMAGE}"
  echo "archive output: ${archive_path}"
  docker run --rm --privileged --network host \
    --entrypoint buildctl-daemonless.sh \
    -e "BUILDKITD_FLAGS=--allow-insecure-entitlement network.host" \
    -e "SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH}" \
    -e "BUILDKIT_MULTI_PLATFORM=${BUILDKIT_MULTI_PLATFORM:-1}" \
    -e "BUILDX_NO_DEFAULT_ATTESTATIONS=${BUILDX_NO_DEFAULT_ATTESTATIONS:-1}" \
    -v "${RUN_DIR}:/workspace:ro" \
    -v "${archive_dir}:/out" \
    "${PINNED_BUILDKIT_IMAGE}" \
    build \
      --frontend dockerfile.v0 \
      --local context=/workspace \
      --local dockerfile=/workspace \
      --opt "filename=${dockerfile_rel}" \
      --opt platform=linux/amd64 \
      --opt force-network-mode=host \
      --opt "build-arg:SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH}" \
      --allow network.host \
      --output "type=docker,name=${DEPLOY_IMAGE},dest=/out/${archive_name},rewrite-timestamp=true" \
      --progress="${BUILDKIT_PROGRESS:-plain}"

  if [[ "${keep_archive}" == "1" ]]; then
    local archive_sha
    echo "reproducible docker archive: ${archive_path}"
    sha256sum "${archive_path}" | tee "${archive_path}.sha256"
    archive_sha="$(awk '{print $1}' "${archive_path}.sha256")"
    validate_repro_docker_archive_sha "${archive_path}" "${archive_sha}"
    if [[ "$(resolve_bool "${REPRO_DOCKER_LOAD_ARCHIVE:-1}")" == "1" && "$(resolve_bool "${BUILD_ONLY:-0}")" == "0" ]]; then
      echo "loading reproducible docker archive into active Docker daemon"
      docker load -i "${archive_path}"
      docker image inspect "${DEPLOY_IMAGE}" --format '{{.Id}} {{.Created}} {{json .RootFS.Layers}}'
    else
      echo "skipping docker archive load because BUILD_ONLY=1 or REPRO_DOCKER_LOAD_ARCHIVE=0"
    fi
  else
    if [[ "$(resolve_bool "${REPRO_DOCKER_LOAD_ARCHIVE:-1}")" == "1" && "$(resolve_bool "${BUILD_ONLY:-0}")" == "0" ]]; then
      echo "loading reproducible docker archive into active Docker daemon"
      docker load -i "${archive_path}"
      docker image inspect "${DEPLOY_IMAGE}" --format '{{.Id}} {{.Created}} {{json .RootFS.Layers}}'
    else
      echo "skipping temporary docker archive load because BUILD_ONLY=1 or REPRO_DOCKER_LOAD_ARCHIVE=0"
    fi
    rm -f "${archive_path}"
  fi
}

build_image_reproducibly_with_buildx() {
  ensure_pinned_buildx
  ensure_pinned_buildkit_builder
  probe_reproducible_buildx_exporter

  local output_spec
  if [[ "$(resolve_bool "${REPRO_EXPORT_DOCKER_ARCHIVE:-1}")" == "1" ]]; then
    mkdir -p "$(dirname "${REPRO_DOCKER_ARCHIVE_PATH}")"
    rm -f "${REPRO_DOCKER_ARCHIVE_PATH}" "${REPRO_DOCKER_ARCHIVE_PATH}.sha256"
    output_spec="type=docker,dest=${REPRO_DOCKER_ARCHIVE_PATH},rewrite-timestamp=true"
  else
    output_spec="type=docker,rewrite-timestamp=true"
  fi

  local build_args=(
    docker buildx build
  )
  if [[ -n "${REPRO_BUILDX_BUILDER:-}" ]]; then
    build_args+=(--builder "${REPRO_BUILDX_BUILDER}")
  fi
  build_args+=(
    --progress="${BUILDKIT_PROGRESS:-plain}"
    --network=host
    --provenance=false
    --sbom=false
    --platform linux/amd64
    --build-arg "SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH}"
    -t "${DEPLOY_IMAGE}"
    -f "${DEPLOY_DOCKERFILE_PATH}"
    --output "${output_spec}"
  )
  if [[ "$(resolve_bool "${FORCE_REBUILD:-0}")" == "1" ]]; then
    build_args+=(--no-cache)
  fi
  build_args+=("${RUN_DIR}")

  "${build_args[@]}"

  if [[ "$(resolve_bool "${REPRO_EXPORT_DOCKER_ARCHIVE:-1}")" == "1" ]]; then
    local archive_sha
    echo "reproducible docker archive: ${REPRO_DOCKER_ARCHIVE_PATH}"
    sha256sum "${REPRO_DOCKER_ARCHIVE_PATH}" | tee "${REPRO_DOCKER_ARCHIVE_PATH}.sha256"
    archive_sha="$(awk '{print $1}' "${REPRO_DOCKER_ARCHIVE_PATH}.sha256")"
    validate_repro_docker_archive_sha "${REPRO_DOCKER_ARCHIVE_PATH}" "${archive_sha}"
    if [[ "$(resolve_bool "${REPRO_DOCKER_LOAD_ARCHIVE:-1}")" == "1" && "$(resolve_bool "${BUILD_ONLY:-0}")" == "0" ]]; then
      echo "loading reproducible docker archive into active Docker daemon"
      docker load -i "${REPRO_DOCKER_ARCHIVE_PATH}"
      docker image inspect "${DEPLOY_IMAGE}" --format '{{.Id}} {{.Created}} {{json .RootFS.Layers}}'
    else
      echo "skipping docker archive load because BUILD_ONLY=1 or REPRO_DOCKER_LOAD_ARCHIVE=0"
    fi
  fi
}

build_image_reproducibly() {
  local export_mode="${DOCKER_REPRODUCIBLE_EXPORT_MODE:-buildctl-daemonless}"
  case "${export_mode}" in
    buildctl-daemonless|buildkit-daemonless|daemonless)
      build_image_reproducibly_with_buildctl_daemonless
      ;;
    buildx)
      build_image_reproducibly_with_buildx
      ;;
    *)
      echo "error: unsupported DOCKER_REPRODUCIBLE_EXPORT_MODE=${export_mode}; use buildctl-daemonless or buildx" >&2
      exit 1
      ;;
  esac
}

compose_build_seed_up() {
  if [[ "$(resolve_bool "${USE_PREBUILT_IMAGE:-0}")" == "1" ]]; then
    echo "prebuilt-image mode requested; using ${DEPLOY_IMAGE}"
    if [[ "$(resolve_bool "${PREBUILT_IMAGE_PULL:-1}")" == "1" ]]; then
      docker pull "${DEPLOY_IMAGE}"
    elif ! docker image inspect "${DEPLOY_IMAGE}" >/dev/null 2>&1; then
      echo "error: USE_PREBUILT_IMAGE=1 but image is not present locally and PREBUILT_IMAGE_PULL=0: ${DEPLOY_IMAGE}" >&2
      exit 1
    fi
    docker image inspect "${DEPLOY_IMAGE}" --format 'prebuilt image ready: {{.Id}} {{.Created}}'
  elif [[ "$(resolve_bool "${DOCKER_REPRODUCIBLE_BUILDX_EXPORT:-1}")" == "1" ]]; then
    build_image_reproducibly
  else
    local build_args=(build)
    if [[ "$(resolve_bool "${FORCE_REBUILD:-0}")" == "1" ]]; then
      build_args+=(--no-cache)
    fi
    "${COMPOSE_CMD[@]}" "${build_args[@]}"
  fi
  if [[ "$(resolve_bool "${BUILD_ONLY:-0}")" == "1" ]]; then
    echo "build-only mode requested; skipping Triton cache seed and container start"
    return 0
  fi
  seed_runtime_triton_cache_from_image
  "${COMPOSE_CMD[@]}" up -d --no-build
}

finish_build_only_if_requested() {
  if [[ "$(resolve_bool "${BUILD_ONLY:-0}")" != "1" ]]; then
    return 0
  fi

  if [[ "${DOCKER_TEMP_DATA_ROOT_ACTIVE:-0}" == "1" ]]; then
    restore_docker_data_root
    DOCKER_TEMP_DATA_ROOT_ACTIVE="0"
  fi

  DOCKER_ISOLATED_DEPLOY_SUCCEEDED=1
  echo "build-only complete"
  if [[ "$(resolve_bool "${REPRO_EXPORT_DOCKER_ARCHIVE:-1}")" == "1" ]]; then
    echo "archive: ${REPRO_DOCKER_ARCHIVE_PATH}"
    if [[ -f "${REPRO_DOCKER_ARCHIVE_PATH}.sha256" ]]; then
      cat "${REPRO_DOCKER_ARCHIVE_PATH}.sha256"
    fi
  fi
  exit 0
}

restore_docker_data_root() {
  if [[ "${DOCKER_TEMP_DATA_ROOT_ACTIVE}" != "1" ]]; then
    return 0
  fi

  local daemon_json='/etc/docker/daemon.json'
  if [[ "${DOCKER_DAEMON_JSON_EXISTED}" == "1" ]]; then
    if [[ -f "${DOCKER_DAEMON_JSON_BACKUP}" ]]; then
      run_as_root "restore original ${daemon_json} from ${DOCKER_DAEMON_JSON_BACKUP}" cp -a "${DOCKER_DAEMON_JSON_BACKUP}" "${daemon_json}"
    else
      echo "warning: docker daemon backup missing at ${DOCKER_DAEMON_JSON_BACKUP}; leaving /etc/docker/daemon.json untouched" >&2
      DOCKER_TEMP_DATA_ROOT_ACTIVE=0
      return 0
    fi
  else
    run_as_root "remove temporary ${daemon_json} because no Docker daemon config existed before this script" rm -f "${daemon_json}"
  fi

  if ! restart_docker_service; then
    echo "warning: docker daemon did not restart after restoring data-root; please restart docker manually" >&2
  fi
  DOCKER_TEMP_DATA_ROOT_ACTIVE=0
  if [[ "$(resolve_bool "${DOCKER_TEMP_DATA_ROOT_CLEANUP}")" == "1" ]]; then
    run_as_root "delete temporary Docker data-root ${DOCKER_TEMP_DATA_ROOT_DIR} after restoring system Docker" rm -rf "${DOCKER_TEMP_DATA_ROOT_DIR}"
  fi
}

cleanup_docker_data_root() {
  restore_docker_data_root
}

cleanup_isolated_docker_daemon_on_failure() {
  if [[ "$(resolve_bool "${DOCKER_ISOLATED_DAEMON_ENABLED:-0}")" != "1" ]]; then
    return 0
  fi
  if [[ "${DOCKER_ISOLATED_DEPLOY_SUCCEEDED:-0}" == "1" ]]; then
    return 0
  fi
  if [[ "$(resolve_bool "${DOCKER_ISOLATED_CLEANUP_ON_FAILURE:-1}")" != "1" ]]; then
    return 0
  fi

  local isolated_dir="${DOCKER_ISOLATED_DAEMON_DIR:-./.d}"
  isolated_dir="$(abs_path "${isolated_dir}" "${RUN_DIR}")"
  if [[ ! -e "${isolated_dir}" ]]; then
    return 0
  fi

  echo "deployment did not complete; cleaning failed private Docker/containerd state under ${isolated_dir}" >&2

  local pid_file pid pids=()
  for pid_file in \
    "${isolated_dir}/r/dockerd-daemon.pid" \
    "${isolated_dir}/r/dockerd-wrapper.pid" \
    "${isolated_dir}/r/containerd.pid"; do
    if [[ -f "${pid_file}" ]]; then
      pid="$(cat "${pid_file}" 2>/dev/null || true)"
      if [[ "${pid}" =~ ^[0-9]+$ ]]; then
        pids+=("${pid}")
      fi
    fi
  done

  if (( ${#pids[@]} > 0 )); then
    run_as_root "stop failed private Docker/containerd daemons started under ${isolated_dir}" kill "${pids[@]}" || true
    sleep 3
    run_as_root "force-stop failed private Docker/containerd daemons started under ${isolated_dir}" kill -9 "${pids[@]}" || true
  fi

  lazy_unmount_tree "${isolated_dir}"

  if [[ "$(resolve_bool "${DOCKER_ISOLATED_DELETE_ROOT_ON_FAILURE:-1}")" == "1" ]]; then
    run_as_root "delete failed private Docker/containerd root ${isolated_dir}" rm -rf "${isolated_dir}"
  fi
}

cleanup_deploy_resources() {
  local exit_code=$?
  set +e
  restore_docker_data_root
  cleanup_isolated_docker_daemon_on_failure
  return "${exit_code}"
}

apply_temp_docker_data_root() {
  DOCKER_TEMP_DATA_ROOT_DIR="$(abs_path "${DOCKER_TEMP_DATA_ROOT_DIR:-./.docker-data-root}" "${RUN_DIR}")"
  DOCKER_DAEMON_JSON_BACKUP="${RUN_DIR}/.deploy.daemon.json.backup"
  DOCKER_DAEMON_JSON_EXISTED="0"

  mkdir -p "${DOCKER_TEMP_DATA_ROOT_DIR}"

  check_free_space "temporary docker data-root" "${DOCKER_TEMP_DATA_ROOT_DIR}" "${MIN_FREE_GIB_TEMP_DOCKER_DATA_ROOT}"

  local daemon_json='/etc/docker/daemon.json'
  if [[ -f "${daemon_json}" ]]; then
    DOCKER_DAEMON_JSON_EXISTED="1"
    run_as_root "back up ${daemon_json} to ${DOCKER_DAEMON_JSON_BACKUP} before applying temporary Docker data-root" cp -a "${daemon_json}" "${DOCKER_DAEMON_JSON_BACKUP}"
  else
    run_as_root "ensure ${daemon_json} is absent before writing temporary Docker data-root config" rm -f "${daemon_json}"
    rm -f "${DOCKER_DAEMON_JSON_BACKUP}"
  fi

  local new_daemon_json
  new_daemon_json="$(mktemp)"
  if [[ -f "${daemon_json}" ]]; then
    write_docker_daemon_json "${daemon_json}" "${DOCKER_TEMP_DATA_ROOT_DIR}" "${new_daemon_json}"
  else
    write_docker_daemon_json '' "${DOCKER_TEMP_DATA_ROOT_DIR}" "${new_daemon_json}"
  fi
  run_as_root "install temporary ${daemon_json} pointing Docker data-root at ${DOCKER_TEMP_DATA_ROOT_DIR}" mv "${new_daemon_json}" "${daemon_json}"

  if ! restart_docker_service; then
    echo "error: failed to restart docker after applying temporary data-root" >&2
    rm -f "${new_daemon_json}"
    exit 1
  fi

  DOCKER_TEMP_DATA_ROOT_ACTIVE="1"
  DOCKER_TEMP_DATA_ROOT_PATCH_IMAGE_TAR="${RUN_DIR}/.${DEPLOY_IMAGE}.image.tar"
  echo "applied temporary docker data-root: ${DOCKER_TEMP_DATA_ROOT_DIR}"
}

check_docker_root_candidates() {
  local docker_root
  docker_root="$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || true)"
  if [[ -n "${docker_root}" ]]; then
    check_free_space "docker root" "${docker_root}" "${MIN_FREE_GIB_DOCKER_ROOT}"
  else
    echo "warning: could not read docker root from docker info; skipping docker-root free-space check" >&2
  fi
}

trap cleanup_deploy_resources EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
# Resolve bundle resources and generated artifact paths.
# Outputs (cache/runtime/patch/config/dockerfile/env) are rooted in the
# directory where this script was executed.
HF_CACHE_DIR="$(abs_path "${HF_CACHE_DIR}" "${RUN_DIR}")"
RUNTIME_ROOT="$(abs_path "${RUNTIME_ROOT}" "${RUN_DIR}")"
MOE_CONFIG_DIR="$(abs_path "${MOE_CONFIG_DIR}" "${RUN_DIR}")"
RUNTIME_PATCH_DIR="$(abs_path "${RUNTIME_PATCH_DIR}" "${RUN_DIR}")"
BUNDLED_HOTFIX_DIR="$(abs_path "${BUNDLED_HOTFIX_DIR}" "${RUN_DIR}")"
PATCH_SRC_PR43190_DIR="$(abs_path "${PATCH_SRC_PR43190_DIR}" "${RUN_DIR}")"
PATCH_SRC_FASTPATH_FUSED_MOE="$(abs_path "${PATCH_SRC_FASTPATH_FUSED_MOE}" "${RUN_DIR}")"
PATCH_SRC_PR41457="$(abs_path "${PATCH_SRC_PR41457}" "${RUN_DIR}")"
PATCH_SRC_UTILS="$(abs_path "${PATCH_SRC_UTILS}" "${RUN_DIR}")"
ROCM_PATCH_SRC_FILE="$(abs_path "${ROCM_PATCH_SRC_FILE}" "${RUN_DIR}")"
TRITON_ATTENTION_BACKEND_SRC_FILE="$(abs_path "${TRITON_ATTENTION_BACKEND_SRC_FILE}" "${RUN_DIR}")"
TRITON_ATTENTION_OP_SRC_FILE="$(abs_path "${TRITON_ATTENTION_OP_SRC_FILE}" "${RUN_DIR}")"
QWEN36_PYTHON_SRC_DIR="$(abs_path "${QWEN36_PYTHON_SRC_DIR}" "${RUN_DIR}")"
BUNDLE_PATCH_TARGET="${BUNDLE_PATCH_TARGET:-/opt/vllm_patch_bundle}"
RESTART_POLICY="${RESTART_POLICY:-no}"
BUILD_ONLY="${BUILD_ONLY:-0}"
USE_PREBUILT_IMAGE="${USE_PREBUILT_IMAGE:-0}"
PREBUILT_IMAGE_PULL="${PREBUILT_IMAGE_PULL:-1}"
STAGE_IMAGE="${STAGE_IMAGE:-python:3.12-slim}"
SKIP_DISK_SPACE_CHECK="${SKIP_DISK_SPACE_CHECK:-0}"
DOCKER_ISOLATED_DAEMON_ENABLED="${DOCKER_ISOLATED_DAEMON_ENABLED:-1}"
DOCKER_ISOLATED_DAEMON_DIR="${DOCKER_ISOLATED_DAEMON_DIR:-./.d}"
DOCKER_TEMP_DATA_ROOT_ENABLED="${DOCKER_TEMP_DATA_ROOT_ENABLED:-0}"
DOCKER_TEMP_DATA_ROOT_DIR="${DOCKER_TEMP_DATA_ROOT_DIR:-./.docker-data-root}"
DOCKER_TEMP_DATA_ROOT_CLEANUP="${DOCKER_TEMP_DATA_ROOT_CLEANUP:-1}"
MIN_FREE_GIB_HF_CACHE="${MIN_FREE_GIB_HF_CACHE:-120}"
MIN_FREE_GIB_RUNTIME_ROOT="${MIN_FREE_GIB_RUNTIME_ROOT:-120}"
MIN_FREE_GIB_BUILD_CONTEXT="${MIN_FREE_GIB_BUILD_CONTEXT:-20}"
MIN_FREE_GIB_DOCKER_ROOT="${MIN_FREE_GIB_DOCKER_ROOT:-200}"
MIN_FREE_GIB_TEMP_DOCKER_DATA_ROOT="${MIN_FREE_GIB_TEMP_DOCKER_DATA_ROOT:-200}"
REPRO_EXPORT_DOCKER_ARCHIVE="${REPRO_EXPORT_DOCKER_ARCHIVE:-1}"
REPRO_DOCKER_LOAD_ARCHIVE="${REPRO_DOCKER_LOAD_ARCHIVE:-1}"
REPRO_DOCKER_ARCHIVE_DIR="${REPRO_DOCKER_ARCHIVE_DIR:-./.repro-docker-archives}"
REPRO_DOCKER_ARCHIVE_NAME="${REPRO_DOCKER_ARCHIVE_NAME:-${DEPLOY_IMAGE//\//_}.docker.tar}"
REPRO_DOCKER_ARCHIVE_PATH="${REPRO_DOCKER_ARCHIVE_PATH:-}"
BYTE_FOR_BYTE_VALIDATION_MODE="${BYTE_FOR_BYTE_VALIDATION_MODE:-auto}"
EXPECTED_REPRO_DOCKER_ARCHIVE_SHA256="${EXPECTED_REPRO_DOCKER_ARCHIVE_SHA256:-}"
MIN_FREE_GIB_REPRO_DOCKER_ARCHIVE="${MIN_FREE_GIB_REPRO_DOCKER_ARCHIVE:-180}"
HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-auto}"
GPU_AUTO_SELECT="${GPU_AUTO_SELECT:-1}"
GPU_TARGET_ARCH="${GPU_TARGET_ARCH:-gfx906}"
GPU_TARGET_ARCH_ALIASES="${GPU_TARGET_ARCH_ALIASES:-gfx906,gfx9006}"
REQUIRED_GPU_COUNT="${REQUIRED_GPU_COUNT:-4}"
REQUIRED_GPU_VRAM_GIB="${REQUIRED_GPU_VRAM_GIB:-32}"
REQUIRED_GPU_FREE_VRAM_GIB="${REQUIRED_GPU_FREE_VRAM_GIB:-}"
SKIP_GPU_MEMORY_CHECK="${SKIP_GPU_MEMORY_CHECK:-0}"
SKIP_GPU_FREE_MEMORY_CHECK="${SKIP_GPU_FREE_MEMORY_CHECK:-0}"
AUTO_CONTINUE_GPU_CHECK="${AUTO_CONTINUE_GPU_CHECK:-0}"
CHECK_GPU_ONLY="${CHECK_GPU_ONLY:-0}"
WAIT_FOR_READY_TIMEOUT="${WAIT_FOR_READY_TIMEOUT:-3600}"
SKIP_READY_CHECK="${SKIP_READY_CHECK:-0}"
SKIP_TRITON_CACHE_SEED="${SKIP_TRITON_CACHE_SEED:-1}"
TRITON_CACHE_SEED_TIMEOUT_SECONDS="${TRITON_CACHE_SEED_TIMEOUT_SECONDS:-180}"
DOCKER_REPRODUCIBLE_EXPORT_MODE="${DOCKER_REPRODUCIBLE_EXPORT_MODE:-buildctl-daemonless}"
DEPLOY_DOCKERFILE_PATH="$(abs_path "${DOCKERFILE_PATH:-Dockerfile}" "${RUN_DIR}")"
DEPLOY_COMPOSE_PATH="$(abs_path "${COMPOSE_PATH:-docker-compose.deploy.yml}" "${RUN_DIR}")"
DEPLOY_ENTRYPOINT_PATH="$(abs_path "${ENTRYPOINT_PATH:-docker-entrypoint.sh}" "${RUN_DIR}")"
if [[ "${DEPLOY_DOCKERFILE_PATH}" == "${RUN_DIR}"/* ]]; then
  COMPOSE_DOCKERFILE_PATH="${DEPLOY_DOCKERFILE_PATH#"${RUN_DIR}/"}"
  if [[ -z "${COMPOSE_DOCKERFILE_PATH}" ]]; then
    COMPOSE_DOCKERFILE_PATH="Dockerfile"
  fi
  DOCKERFILE_PATH="${DOCKERFILE_PATH:-Dockerfile}"
else
  DOCKERFILE_PATH="${DOCKERFILE_PATH:-Dockerfile}"
  COMPOSE_DOCKERFILE_PATH="${DEPLOY_DOCKERFILE_PATH}"
fi
DOCKER_TEMP_DATA_ROOT_ACTIVE="0"
DOCKER_DAEMON_JSON_BACKUP=""
DOCKER_DAEMON_JSON_EXISTED="0"
DOCKER_TEMP_DATA_ROOT_PATCH_IMAGE_TAR=""
REPRO_DOCKER_ARCHIVE_DIR="$(abs_path "${REPRO_DOCKER_ARCHIVE_DIR}" "${RUN_DIR}")"
if [[ -z "${REPRO_DOCKER_ARCHIVE_PATH}" ]]; then
  REPRO_DOCKER_ARCHIVE_PATH="${REPRO_DOCKER_ARCHIVE_DIR}/${REPRO_DOCKER_ARCHIVE_NAME}"
else
  REPRO_DOCKER_ARCHIVE_PATH="$(abs_path "${REPRO_DOCKER_ARCHIVE_PATH}" "${RUN_DIR}")"
  REPRO_DOCKER_ARCHIVE_DIR="$(dirname "${REPRO_DOCKER_ARCHIVE_PATH}")"
fi

BYTE_FOR_BYTE_VALIDATION_MODE_NORMALIZED="${BYTE_FOR_BYTE_VALIDATION_MODE,,}"
BYTE_FOR_BYTE_VALIDATION_REQUIRES_ARCHIVE=0
if [[ "$(resolve_bool "${BYTE_FOR_BYTE_VALIDATION_MODE_NORMALIZED}")" == "1" ]]; then
  BYTE_FOR_BYTE_VALIDATION_REQUIRES_ARCHIVE=1
elif [[ "${BYTE_FOR_BYTE_VALIDATION_MODE_NORMALIZED}" == "auto" && -n "${EXPECTED_REPRO_DOCKER_ARCHIVE_SHA256}" ]]; then
  BYTE_FOR_BYTE_VALIDATION_REQUIRES_ARCHIVE=1
fi

if [[ "${BYTE_FOR_BYTE_VALIDATION_REQUIRES_ARCHIVE}" == "1" && "$(resolve_bool "${USE_PREBUILT_IMAGE}")" != "1" ]]; then
  if [[ "$(resolve_bool "${DOCKER_REPRODUCIBLE_BUILDX_EXPORT:-1}")" != "1" || "$(resolve_bool "${REPRO_EXPORT_DOCKER_ARCHIVE}")" != "1" ]]; then
    echo "error: byte-for-byte archive validation requires DOCKER_REPRODUCIBLE_BUILDX_EXPORT=1 and REPRO_EXPORT_DOCKER_ARCHIVE=1 for source builds" >&2
    echo "Set BYTE_FOR_BYTE_VALIDATION_MODE=0 only for non-canonical local deploys where release reproduction is not being claimed." >&2
    exit 1
  fi
fi

TRITON_CACHE_HOST="${RUNTIME_ROOT}/root/.triton"
TORCHINDUCTOR_CACHE_HOST="${RUNTIME_ROOT}/tmp/torchinductor_root"
VLLM_CACHE_HOST="${RUNTIME_ROOT}/root/.cache/vllm"

mkdir -p "${HF_CACHE_DIR}" "${TRITON_CACHE_HOST}" "${TORCHINDUCTOR_CACHE_HOST}" "${VLLM_CACHE_HOST}" "${RUNTIME_ROOT}" "${RUNTIME_PATCH_DIR}" "${MOE_CONFIG_DIR%/*}"
mkdir -p "${RUNTIME_ROOT}/vllm_tuned_moe_configs" "${RUNTIME_PATCH_DIR}"

if [[ "$(resolve_bool "${SKIP_DISK_SPACE_CHECK}")" == "0" ]]; then
  check_free_space "HF cache" "${HF_CACHE_DIR}" "${MIN_FREE_GIB_HF_CACHE}"
  check_free_space "runtime cache" "${RUNTIME_ROOT}" "${MIN_FREE_GIB_RUNTIME_ROOT}"
  check_free_space "build context" "${RUN_DIR}" "${MIN_FREE_GIB_BUILD_CONTEXT}"
  if [[ "$(resolve_bool "${REPRO_EXPORT_DOCKER_ARCHIVE}")" == "1" && "$(resolve_bool "${USE_PREBUILT_IMAGE}")" != "1" ]]; then
    mkdir -p "${REPRO_DOCKER_ARCHIVE_DIR}"
    check_free_space "reproducible docker archive output" "${REPRO_DOCKER_ARCHIVE_DIR}" "${MIN_FREE_GIB_REPRO_DOCKER_ARCHIVE}"
  fi
  if [[ "$(resolve_bool "${DOCKER_ISOLATED_DAEMON_ENABLED}")" == "1" ]]; then
    DOCKER_ISOLATED_DAEMON_DIR="$(abs_path "${DOCKER_ISOLATED_DAEMON_DIR:-./.d}" "${RUN_DIR}")"
    mkdir -p "${DOCKER_ISOLATED_DAEMON_DIR}"
    DOCKER_ISOLATED_DAEMON_DIR="$(cd "${DOCKER_ISOLATED_DAEMON_DIR}" && pwd -P)"
    check_free_space "isolated docker/containerd root" "${DOCKER_ISOLATED_DAEMON_DIR}" "${MIN_FREE_GIB_TEMP_DOCKER_DATA_ROOT}"
  elif [[ "$(resolve_bool "${DOCKER_TEMP_DATA_ROOT_ENABLED}")" == "1" ]]; then
    mkdir -p "${DOCKER_TEMP_DATA_ROOT_DIR}"
    check_free_space "temporary docker data-root" "${DOCKER_TEMP_DATA_ROOT_DIR}" "${MIN_FREE_GIB_TEMP_DOCKER_DATA_ROOT}"
  else
    check_docker_root_candidates
  fi
fi

if [[ "$(resolve_bool "${BUILD_ONLY}")" == "1" ]]; then
  echo "build-only mode requested; skipping runtime GPU total/free-memory gates"
else
  if ! auto_select_hip_visible_devices; then
    echo "error: GPU auto-selection failed before runtime preflight." >&2
    exit 1
  fi

  if ! check_gpu_vram_requirements; then
    if [[ "$(resolve_bool "${CHECK_GPU_ONLY}")" == "1" ]]; then
      echo "GPU preflight check-only mode detected; failing fast as requested." >&2
      exit 1
    fi

    prompt_continue_on_gpu_gate_failure "one or more GPUs do not meet the configured VRAM requirement"
  fi

  if ! check_gpu_free_memory_requirements; then
    if [[ "$(resolve_bool "${CHECK_GPU_ONLY}")" == "1" ]]; then
      echo "GPU preflight check-only mode detected; failing fast as requested." >&2
      exit 1
    fi

    prompt_continue_on_gpu_gate_failure "one or more GPUs do not have enough free VRAM for this runtime profile"
  fi
fi

if [[ "$(resolve_bool "${CHECK_GPU_ONLY}")" == "1" ]]; then
  echo "GPU preflight check-only mode complete."
  exit 0
fi

if [[ "$(resolve_bool "${DOCKER_ISOLATED_DAEMON_ENABLED}")" == "1" ]]; then
  start_isolated_docker_daemon
elif [[ "$(resolve_bool "${DOCKER_TEMP_DATA_ROOT_ENABLED}")" == "1" ]]; then
  apply_temp_docker_data_root
fi

write_embedded_dockerfile "${DEPLOY_DOCKERFILE_PATH}"
write_embedded_compose "${DEPLOY_COMPOSE_PATH}"
write_embedded_entrypoint "${DEPLOY_ENTRYPOINT_PATH}"
write_embedded_dockerignore "${RUN_DIR}/.dockerignore"

echo "Using deploy root: ${SCRIPT_DIR}"
echo "Using run directory: ${RUN_DIR}"
echo "Using HF cache:   ${HF_CACHE_DIR}"
echo "Using runtime dir: ${RUNTIME_ROOT}"

echo "Preparing patch bundle in ${RUNTIME_PATCH_DIR}"
rm -rf "${RUNTIME_PATCH_DIR}"
mkdir -p "${RUNTIME_PATCH_DIR}"

rm -rf "${BUNDLED_HOTFIX_DIR}"
write_embedded_hotfixes "${BUNDLED_HOTFIX_DIR}"

copy_patch "${BUNDLED_HOTFIX_DIR}/shared_expert_gate.py" "${RUNTIME_PATCH_DIR}/shared_expert_gate.py"
copy_patch "${BUNDLED_HOTFIX_DIR}/qwen3_moe.py" "${RUNTIME_PATCH_DIR}/qwen3_moe.py"
copy_patch "${BUNDLED_HOTFIX_DIR}/qwen2_moe.py" "${RUNTIME_PATCH_DIR}/qwen2_moe.py"
copy_patch "${BUNDLED_HOTFIX_DIR}/qwen3_5.py" "${RUNTIME_PATCH_DIR}/qwen3_5.py"
copy_patch "${BUNDLED_HOTFIX_DIR}/utils_llmm1_rpb2.py" "${RUNTIME_PATCH_DIR}/utils.py"
copy_patch "${BUNDLED_HOTFIX_DIR}/fused_moe_fastpath.py" "${RUNTIME_PATCH_DIR}/fused_moe.py"
copy_patch "${BUNDLED_HOTFIX_DIR}/fused_moe_pr39016.py" "${RUNTIME_PATCH_DIR}/fused_moe_pr39016.py"
copy_patch "${BUNDLED_HOTFIX_DIR}/rocm.py" "${RUNTIME_PATCH_DIR}/rocm.py"
copy_patch "${BUNDLED_HOTFIX_DIR}/triton_attn.py" "${RUNTIME_PATCH_DIR}/triton_attn.py"
copy_patch "${BUNDLED_HOTFIX_DIR}/triton_unified_attention.py" "${RUNTIME_PATCH_DIR}/triton_unified_attention.py"

if [[ -n "${PATCH_SRC_PR43190_DIR:-}" && -d "${PATCH_SRC_PR43190_DIR}" ]]; then
  copy_patch "${PATCH_SRC_PR43190_DIR}/shared_expert_gate.py" "${RUNTIME_PATCH_DIR}/shared_expert_gate.py"
  copy_patch "${PATCH_SRC_PR43190_DIR}/qwen3_moe.py" "${RUNTIME_PATCH_DIR}/qwen3_moe.py"
  copy_patch "${PATCH_SRC_PR43190_DIR}/qwen2_moe.py" "${RUNTIME_PATCH_DIR}/qwen2_moe.py"
fi

if [[ -n "${PATCH_SRC_FASTPATH_FUSED_MOE:-}" ]]; then
  copy_patch "${PATCH_SRC_FASTPATH_FUSED_MOE}" "${RUNTIME_PATCH_DIR}/fused_moe.py"
fi

if [[ -n "${PATCH_SRC_PR41457:-}" ]]; then
  if [[ -d "${PATCH_SRC_PR41457}" ]]; then
    copy_patch "${PATCH_SRC_PR41457}/qwen3_5.py" "${RUNTIME_PATCH_DIR}/qwen3_5.py"
  else
    copy_patch "${PATCH_SRC_PR41457}" "${RUNTIME_PATCH_DIR}/qwen3_5.py"
  fi
fi

if [[ -n "${PATCH_SRC_UTILS:-}" ]]; then
  if [[ -d "${PATCH_SRC_UTILS}" ]]; then
    copy_patch "${PATCH_SRC_UTILS}/utils_llmm1_rpb2.py" "${RUNTIME_PATCH_DIR}/utils.py"
  else
    copy_patch "${PATCH_SRC_UTILS}" "${RUNTIME_PATCH_DIR}/utils.py"
  fi
fi

copy_if_present "${ROCM_PATCH_SRC_FILE}" "${RUNTIME_PATCH_DIR}/rocm.py"
copy_if_present "${TRITON_ATTENTION_BACKEND_SRC_FILE}" "${RUNTIME_PATCH_DIR}/triton_attn.py"
copy_if_present "${TRITON_ATTENTION_OP_SRC_FILE}" "${RUNTIME_PATCH_DIR}/triton_unified_attention.py"

rm -rf "${RUNTIME_PATCH_DIR}/qwen36-python"
if [[ -n "${QWEN36_PYTHON_SRC_DIR:-}" ]]; then
  if [[ -d "${QWEN36_PYTHON_SRC_DIR}" ]]; then
    cp -a "${QWEN36_PYTHON_SRC_DIR}" "${RUNTIME_PATCH_DIR}/qwen36-python"
  else
    echo "warning: optional qwen36 python source is not a directory: ${QWEN36_PYTHON_SRC_DIR}" >&2
  fi
elif [[ -d "$(abs_path './qwen36-python' "${RUN_DIR}")" ]]; then
  # If an optional qwen36-python checkout was dropped into the bundle root, include it automatically.
  cp -a "$(abs_path './qwen36-python' "${RUN_DIR}")" "${RUNTIME_PATCH_DIR}/qwen36-python"
fi

for patch_file in shared_expert_gate.py qwen3_moe.py qwen2_moe.py qwen3_5.py fused_moe.py utils.py rocm.py triton_attn.py triton_unified_attention.py; do
  required_file "${RUNTIME_PATCH_DIR}/${patch_file}"
done

if [[ -d "${RUNTIME_PATCH_DIR}/qwen36-python" ]]; then
  QWEN36_PYTHON_DIR="${BUNDLE_PATCH_TARGET}/qwen36-python"
else
  QWEN36_PYTHON_DIR=''
  if [[ "${PREFER_REASONING_PARSER}" != "none" && "${REASONING_PARSER}" == "qwen3" ]]; then
    echo "info: qwen3 reasoning parser requested; qwen36-python sidecar will be built from pinned packages inside the image"
  fi
fi

if [[ -f "${SCRIPT_DIR}/files/${MOE_CONFIG_BASENAME}" ]]; then
  if [[ "${INSTALL_BUNDLED_MOE_CONFIG}" == "1" ]]; then
    src_moe="${SCRIPT_DIR}/files/${MOE_CONFIG_BASENAME}"
    dst_moe="${MOE_CONFIG_DIR}/${MOE_CONFIG_BASENAME}"
    if [[ ! -f "${dst_moe}" || "${REFRESH_MOE_CONFIG}" == "1" ]]; then
      mkdir -p "${MOE_CONFIG_DIR}"
      cp "${src_moe}" "${dst_moe}"
    fi
  fi
else
  if [[ ! -f "${MOE_CONFIG_DIR}/${MOE_CONFIG_BASENAME}" || "${REFRESH_MOE_CONFIG}" == "1" ]]; then
    mkdir -p "${MOE_CONFIG_DIR}"
    write_embedded_moe_config "${MOE_CONFIG_DIR}/${MOE_CONFIG_BASENAME}"
  fi
fi

if [[ ! -f "${MOE_CONFIG_DIR}/${MOE_CONFIG_BASENAME}" ]]; then
  echo "error: missing Moe config file: ${MOE_CONFIG_DIR}/${MOE_CONFIG_BASENAME}" >&2
  exit 1
fi

if [[ "${AUTO_STAGE_MODEL}" == "1" ]]; then
  if [[ -z "${MODEL_REPO_ID}" ]]; then
    echo "error: MODEL_REPO_ID must be set when AUTO_STAGE_MODEL=1" >&2
    exit 1
  fi
  MODEL_SAFE="${MODEL_REPO_ID//\//--}"
  MODEL_CACHE_DIR="${HF_CACHE_DIR}/hub/models--${MODEL_SAFE}"
  if [[ ! -d "${MODEL_CACHE_DIR}" || "$(resolve_bool "${AUTO_STAGE_MODEL_FORCE}")" == "1" || ! -f "${MODEL_CACHE_DIR}/config.json" ]]; then
    echo "staging model cache for ${MODEL_REPO_ID} into ${HF_CACHE_DIR}"
    docker run --rm \
      --network host \
      --entrypoint python3 \
      -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
      -e HF_HOME=/root/.cache/huggingface \
      -e HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub \
      -e DO_NOT_TRACK=1 \
      ${HF_TOKEN:+-e HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"} \
      "${STAGE_IMAGE}" - "${MODEL_REPO_ID}" <<'PY'
import sys
import subprocess

try:
  from huggingface_hub import snapshot_download
except ImportError:  # pragma: no cover
  subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', 'huggingface-hub'])
  from huggingface_hub import snapshot_download

if len(sys.argv) != 2:
  raise SystemExit('expected model id argument')

model_id = sys.argv[1]
print(f'staging {model_id}')
path = snapshot_download(repo_id=model_id)
print(f'staged {model_id} at {path}')
PY
  else
    echo "using existing model cache at ${MODEL_CACHE_DIR}"
  fi
fi

RUNTIME_ENV_FILE="${RUN_DIR}/.deploy.runtime.env"
cat <<EOF > "${RUNTIME_ENV_FILE}"
MODEL=${MODEL}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME}
PORT=${PORT}
TP_SIZE=${TP_SIZE}
HOST=${HOST}
MAX_MODEL_LEN=${MAX_MODEL_LEN}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}
TOOL_CALL_PARSER=${TOOL_CALL_PARSER}
REASONING_PARSER=${REASONING_PARSER}
REASONING_PARSER_PREFERRED=${PREFER_REASONING_PARSER}
VLLM_DTYPE=${VLLM_DTYPE}
VLLM_ENABLE_C1_TOPK8_MOE_FASTPATH=${VLLM_ENABLE_C1_TOPK8_MOE_FASTPATH}
OPT_LEVEL=${OPT_LEVEL}
EXTRA_VLLM_ARGS=${EXTRA_VLLM_ARGS}
DISABLE_ASYNC_SCHEDULING=${DISABLE_ASYNC_SCHEDULING}
NCCL_ALGO=${NCCL_ALGO}
NCCL_PROTO=${NCCL_PROTO}
NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE}
NCCL_MAX_NCHANNELS=${NCCL_MAX_NCHANNELS}
FLASH_ATTENTION_TRITON_AMD_ENABLE=${FLASH_ATTENTION_TRITON_AMD_ENABLE}
FLASH_ATTENTION_TRITON_AMD_REF=${FLASH_ATTENTION_TRITON_AMD_REF}
OMP_NUM_THREADS=${OMP_NUM_THREADS}
HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}
GPU_AUTO_SELECT=${GPU_AUTO_SELECT}
GPU_TARGET_ARCH=${GPU_TARGET_ARCH}
GPU_TARGET_ARCH_ALIASES=${GPU_TARGET_ARCH_ALIASES}
REQUIRED_GPU_COUNT=${REQUIRED_GPU_COUNT}
REQUIRED_GPU_VRAM_GIB=${REQUIRED_GPU_VRAM_GIB}
REQUIRED_GPU_FREE_VRAM_GIB=${REQUIRED_GPU_FREE_VRAM_GIB}

HF_CACHE_DIR=${HF_CACHE_DIR}
RUNTIME_ROOT=${RUNTIME_ROOT}
MOE_CONFIG_DIR=${MOE_CONFIG_DIR}
VLLM_TUNED_CONFIG_FOLDER=${VLLM_TUNED_CONFIG_FOLDER}
RUNTIME_PATCH_DIR=${RUNTIME_PATCH_DIR}
VLLM_PATCH_BUNDLE=${BUNDLE_PATCH_TARGET}
BUNDLE_PATCH_TARGET=${BUNDLE_PATCH_TARGET}
BUNDLED_HOTFIX_DIR=${BUNDLED_HOTFIX_DIR}
QWEN36_PYTHON_DIR=${QWEN36_PYTHON_DIR}
CONTAINER_NAME=${CONTAINER_NAME}
RESTART_POLICY=${RESTART_POLICY}
DEPLOY_IMAGE=${DEPLOY_IMAGE}
USE_PREBUILT_IMAGE=${USE_PREBUILT_IMAGE}
PREBUILT_IMAGE_PULL=${PREBUILT_IMAGE_PULL}
DOCKERFILE_PATH=${DEPLOY_DOCKERFILE_PATH}
DOCKER_ISOLATED_DAEMON_ENABLED=${DOCKER_ISOLATED_DAEMON_ENABLED}
DOCKER_ISOLATED_DAEMON_DIR=${DOCKER_ISOLATED_DAEMON_DIR}
DOCKER_ISOLATED_CLEANUP_ON_FAILURE=${DOCKER_ISOLATED_CLEANUP_ON_FAILURE}
DOCKER_ISOLATED_DELETE_ROOT_ON_FAILURE=${DOCKER_ISOLATED_DELETE_ROOT_ON_FAILURE}
DOCKER_TEMP_DATA_ROOT_ENABLED=${DOCKER_TEMP_DATA_ROOT_ENABLED}
WAIT_FOR_READY_TIMEOUT=${WAIT_FOR_READY_TIMEOUT}
SKIP_READY_CHECK=${SKIP_READY_CHECK}
SKIP_TRITON_CACHE_SEED=${SKIP_TRITON_CACHE_SEED}
TRITON_CACHE_SEED_TIMEOUT_SECONDS=${TRITON_CACHE_SEED_TIMEOUT_SECONDS}
FORCE_REBUILD=${FORCE_REBUILD}
BUILD_ONLY=${BUILD_ONLY}
DOCKER_REPRODUCIBLE_BUILDX_EXPORT=${DOCKER_REPRODUCIBLE_BUILDX_EXPORT}
DOCKER_REPRODUCIBLE_EXPORT_MODE=${DOCKER_REPRODUCIBLE_EXPORT_MODE}
DOCKER_REPRODUCIBLE_EXPORTER_PROBE=${DOCKER_REPRODUCIBLE_EXPORTER_PROBE}
REPRO_DOCKER_LOAD_ARCHIVE=${REPRO_DOCKER_LOAD_ARCHIVE}
REPRO_DOCKER_ARCHIVE_PATH=${REPRO_DOCKER_ARCHIVE_PATH}
BYTE_FOR_BYTE_VALIDATION_MODE=${BYTE_FOR_BYTE_VALIDATION_MODE}
EXPECTED_REPRO_DOCKER_ARCHIVE_SHA256=${EXPECTED_REPRO_DOCKER_ARCHIVE_SHA256}
EOF

export BUILDKIT_PROGRESS="${BUILDKIT_PROGRESS:-plain}"
export COMPOSE_PROGRESS="${COMPOSE_PROGRESS:-plain}"
export COMPOSE_ANSI="${COMPOSE_ANSI:-never}"
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-1764000000}"
export BUILDKIT_MULTI_PLATFORM="${BUILDKIT_MULTI_PLATFORM:-1}"
export BUILDX_NO_DEFAULT_ATTESTATIONS="${BUILDX_NO_DEFAULT_ATTESTATIONS:-1}"
export BUILDX_METADATA_PROVENANCE="${BUILDX_METADATA_PROVENANCE:-disabled}"

COMPOSE_CMD=(docker compose -f "${DEPLOY_COMPOSE_PATH}" --env-file "${RUNTIME_ENV_FILE}")
"${COMPOSE_CMD[@]}" down --remove-orphans >/dev/null 2>&1 || true

if [[ "$(resolve_bool "${DOCKER_ISOLATED_DAEMON_ENABLED}")" == "1" ]]; then
  compose_build_seed_up
  finish_build_only_if_requested
  if ! wait_for_container_running "${CONTAINER_NAME}" 180; then
    exit 1
  fi
elif [[ "$(resolve_bool "${DOCKER_TEMP_DATA_ROOT_ENABLED}")" == "1" ]]; then
  compose_build_seed_up
  finish_build_only_if_requested
  if [[ -f "${DOCKER_TEMP_DATA_ROOT_PATCH_IMAGE_TAR}" ]]; then
    rm -f "${DOCKER_TEMP_DATA_ROOT_PATCH_IMAGE_TAR}"
  fi
  docker save "${DEPLOY_IMAGE}" -o "${DOCKER_TEMP_DATA_ROOT_PATCH_IMAGE_TAR}"
  restore_docker_data_root
  DOCKER_TEMP_DATA_ROOT_ACTIVE="0"
  docker load -i "${DOCKER_TEMP_DATA_ROOT_PATCH_IMAGE_TAR}" >/dev/null
  "${COMPOSE_CMD[@]}" up -d --remove-orphans --no-build
  if ! wait_for_container_running "${CONTAINER_NAME}" 180; then
    exit 1
  fi
else
  compose_build_seed_up
  finish_build_only_if_requested
  if ! wait_for_container_running "${CONTAINER_NAME}" 180; then
    exit 1
  fi
fi

if ! wait_for_vllm_ready "${CONTAINER_NAME}" "${WAIT_FOR_READY_TIMEOUT}"; then
  exit 1
fi

DOCKER_ISOLATED_DEPLOY_SUCCEEDED=1
echo "deployment complete"
echo "container: ${CONTAINER_NAME}"
echo "listen: http://127.0.0.1:${PORT}/v1"
