#!/bin/bash
# Builds hkl_7be4f79_forward.c against the 7be4f79 libhkl objects the way
# ~/work/hkl/repro/run.sh builds hkl drivers, then writes hkl_7be4f79_forward.json.
# The LISA engine object is left out of the link: the driver #includes that file.
set -eo pipefail
: "${ENVS:?}" "${WORK:?}"
D=$(cd "$(dirname "$0")" && pwd)
B=$WORK/hkl
export PATH=$ENVS/hkl/bin:$PATH PKG_CONFIG_PATH=$ENVS/hkl/lib/pkgconfig
CF="-O1 -g -std=gnu11 -w -ftrack-macro-expansion=0 -I$B -I$B/third-party $(pkg-config --cflags glib-2.0 gsl)"
OBJS=$(ls "$B"/obj/*.o | rg -v 'hkl_hkl-engine-petra3-p08-lisa\.c\.o')
x86_64-conda-linux-gnu-gcc $CF "$D/hkl_7be4f79_forward.c" $OBJS -o "$D/../target/hkl_7be4f79_forward" \
	$(pkg-config --libs glib-2.0 gobject-2.0 gsl) -lm -Wl,-rpath,"$ENVS/hkl/lib"
"$D/../target/hkl_7be4f79_forward" > "$D/hkl_7be4f79_forward.json"
