#!/bin/bash
# Builds bin_7be4f79_hdf5.c like bin_7be4f79_cube.sh (libhklbin links
# libhdf5), has binoculars save the merged cube to bin_7be4f79_cube.h5,
# has hklrs write the same cube to hklrs_7be4f79_cube.h5, then has
# binoculars load both into bin_7be4f79_hdf5.json. ENVS and WORK as in
# bin_7be4f79_cube.sh.
set -eo pipefail
: "${ENVS:?}" "${WORK:?}"
D=$(cd "$(dirname "$0")" && pwd)
B=$WORK/bin
E=$ENVS/bin
mkdir -p "$D/../target"
gcc -g -I"$B" -I"$B/hkl" -I"$B/binoculars-ng/binoculars" -I"$B/third-party" -I"$E/include" \
	-I"$E/include/glib-2.0" -I"$E/lib/glib-2.0/include" -ftrack-macro-expansion=0 -w \
	"$D/bin_7be4f79_hdf5.c" -o "$D/../target/bin_7be4f79_hdf5" \
	-L"$B" -lhklbin -L"$E/lib" -lglib-2.0 -Wl,-rpath,"$B" -Wl,-rpath,"$E/lib" -lm
"$D/../target/bin_7be4f79_hdf5" save "$D/bin_7be4f79_cube.h5"
(cd "$D/.." && HKLRS_BLESS=1 cargo nextest run -p hklrs --features hdf5 --test bin_cube \
	hklrs_writes_the_merged_cube)
"$D/../target/bin_7be4f79_hdf5" load "$D/bin_7be4f79_cube.h5" "$D/hklrs_7be4f79_cube.h5" \
	> "$D/bin_7be4f79_hdf5.json"
