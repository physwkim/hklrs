#!/bin/bash
# Builds bin_7be4f79_detectors.c against the binoculars 7be4f79 library the
# way ~/work/hkl/repro/run.sh builds `bin` drivers, then writes
# bin_7be4f79_detectors.json. ENVS is the conda env root (its bin/ holds glib),
# WORK the repro build tree (WORK/bin holds libhklbin.so).
set -eo pipefail
: "${ENVS:?}" "${WORK:?}"
D=$(cd "$(dirname "$0")" && pwd)
B=$WORK/bin
E=$ENVS/bin
mkdir -p "$D/../target"
gcc -g -I"$B" -I"$B/hkl" -I"$B/binoculars-ng/binoculars" -I"$B/third-party" -I"$E/include" \
	-I"$E/include/glib-2.0" -I"$E/lib/glib-2.0/include" -ftrack-macro-expansion=0 -w \
	"$D/bin_7be4f79_detectors.c" -o "$D/../target/bin_7be4f79_detectors" \
	-L"$B" -lhklbin -L"$E/lib" -lglib-2.0 -Wl,-rpath,"$B" -Wl,-rpath,"$E/lib" -lm
"$D/../target/bin_7be4f79_detectors" > "$D/bin_7be4f79_detectors.json"
