#!/bin/bash
# Build the hklrs Python extension and place it at compare/_build/hklrs.abi3.so.
set -euo pipefail
PY=${HKLRS_PY:-/tmp/claude-1000/-home-stevek-work-hkl/48625d75-df46-440f-8894-add00a6c5fa0/scratchpad/envs/py2/bin/python}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PYO3_PYTHON="$PY" cargo build --release -p hklrs-py
mkdir -p compare/_build
cp target/release/libhklrs_py.so compare/_build/hklrs.abi3.so
