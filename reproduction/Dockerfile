# The released decoder is Linux/x86_64-only. On Apple Silicon, build/run with
# --platform linux/amd64. This image targets checkpoint replay, not retraining.
FROM python:3.9-bookworm AS runtime-base

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    gcc-11 \
    g++-11 \
    git \
    ninja-build \
    zlib1g-dev \
 && rm -rf /var/lib/apt/lists/*

# GCC 12 repeatedly segfaults inside Docker's amd64 emulation on Apple Silicon
# while building the vendored OpenFST/glog sources. GCC 11 is compatible with
# the authors' documented GCC >=10.1 requirement and completes this legacy build.
ENV CC=gcc-11 \
    CXX=g++-11

WORKDIR /opt/speechbci
COPY NeuralDecoder ./NeuralDecoder
COPY reproduction/run_released_checkpoint.py ./reproduction/run_released_checkpoint.py

# Install a minimal, mutually compatible subset for the released CPU inference
# route. Upstream's full setup.py mixes TF 2.7 with a much newer NumPy pin.
RUN python -m pip install --upgrade "pip<24" "setuptools<69" wheel \
 && python -m pip install \
      "tensorflow==2.7.0" \
      "numpy==1.21.6" \
      "scipy==1.7.3" \
      "protobuf==3.19.6" \
      "tensorboard==2.7.0" \
      "wrapt==1.13.3" \
      "hydra-core==1.3.2" \
      "omegaconf==2.3.0" \
      "tqdm==4.66.5" \
      "edit_distance==1.0.6"

RUN python -m pip install --no-deps -e ./NeuralDecoder

COPY LanguageModelDecoder ./LanguageModelDecoder

FROM runtime-base AS final

RUN CMAKE_BUILD_PARALLEL_LEVEL=1 python -m pip install ./LanguageModelDecoder/runtime/server/x86 || \
    (find ./LanguageModelDecoder/runtime/server/x86 -name config.log -exec tail -n 200 {} \; && exit 1)

ENTRYPOINT ["python", "/opt/speechbci/reproduction/run_released_checkpoint.py"]
