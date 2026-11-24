#!/bin/bash
# Build the hklrs Python extension and place it at compare/_build/hklrs.abi3.so.
set -euo pipefail
PY=${HKLRS_PY:-python3}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PYO3_PYTHON="$PY" cargo build --release -p hklrs-py
mkdir -p compare/_build
cp target/release/libhklrs_py.so compare/_build/hklrs.abi3.so
