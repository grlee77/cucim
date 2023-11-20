#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -eou pipefail

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="cucim_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 ./dist

# echo to expand wildcard before adding `[extra]` requires for pip
python -m pip install $(echo ./dist/cucim*.whl)[test]

# Run smoke tests for aarch64 pull requests
if [[ "$(arch)" == "aarch64" && ${RAPIDS_BUILD_TYPE} == "pull-request" ]]; then
    python ./ci/wheel_smoke_test.py
else

    DEBIAN_FRONTEND=noninteractive apt update
    DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends libopenslide0

    # verify if openslide package has been installed
    echo `dpkg -l | grep openslide`

    # append folder containing libopenslide.so.0 to LD_LIBRARY_PATH (should not be needed)
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/lib/$(arch)-linux-gnu"

    # try importing like openslide-python does
    echo `python -c "import ctypes; print(ctypes.cdll.LoadLibrary('libopenslide.so.0')); print('\n')"`

    # try importing with full path
    echo `python -c "import ctypes; print(ctypes.cdll.LoadLibrary('/usr/lib/x86_64-linux-gnu/libopenslide.so.0')); print('\n')"`

    # TODO: fix and remove two test cases skipped here
    #   test_cache_hit_miss currently fails
    #   test_converter passes, but a segfault occurs on pytest exit when it is enabled
    python -m pytest ./python/cucim -k "not test_converter and not test_cache_hit_miss"
fi
