#!/usr/bin/env bash
# Builds areaDetector's simDetector IOC with NDFileHDF5 over ADSupport's
# bundled HDF5 and runs it with st.cmd beside this script, for
# hklrs-bsrs/tests/rsm_ioc.rs.
#
#   simdetector-ioc.sh build   copy the modules into $AD_BUILD and build them
#   simdetector-ioc.sh run     run the built IOC
#   simdetector-ioc.sh         both
#   simdetector-ioc.sh test    run rsm_ioc.rs against the running IOC
#
# The modules are copied from $EPICS_MODULES and built in the copies only.
# The IOC serves CA on 127.0.0.1 at $HKLRS_IOC_CA_PORT, a port of its own,
# so a client finds it with
#   EPICS_CA_ADDR_LIST=127.0.0.1 EPICS_CA_AUTO_ADDR_LIST=NO
#   EPICS_CA_SERVER_PORT=$HKLRS_IOC_CA_PORT
# Its PVs are $HKLRS_IOC_PREFIX{cam1,HDF1}:..., the prefix rsm_ioc.rs reads
# from the same variable.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EPICS_BASE=${EPICS_BASE:-$HOME/work/epics-base}
EPICS_MODULES=${EPICS_MODULES:-$HOME/work/epics-modules}
AD_BUILD=${AD_BUILD:-$HOME/work/hkl/ad-build}
HKLRS_IOC_PREFIX=${HKLRS_IOC_PREFIX:-HKLRSRSM:}
HKLRS_IOC_CA_PORT=${HKLRS_IOC_CA_PORT:-5384}
# In build order; busy's record backs Acquire, Capture and FlushNow.
MODULES=(asyn busy ADSupport ADCore ADSimDetector)
IOC_TOP=$AD_BUILD/ADSimDetector/iocs/simDetectorIOC
EPICS_HOST_ARCH=${EPICS_HOST_ARCH:-$("$EPICS_BASE/startup/EpicsHostArch")}

build() {
    mkdir -p "$AD_BUILD/configure"
    for m in "${MODULES[@]}"; do
        [ -d "$AD_BUILD/$m" ] || cp -a "$EPICS_MODULES/$m" "$AD_BUILD/"
    done
    # asyn and busy read $(TOP)/../RELEASE.local, then their own
    # configure/RELEASE.local, which the copies bring pointing elsewhere.
    printf 'SUPPORT=%s\nEPICS_BASE=%s\n' "$AD_BUILD" "$EPICS_BASE" > "$AD_BUILD/RELEASE.local"
    printf 'EPICS_BASE=%s\n' "$EPICS_BASE" > "$AD_BUILD/asyn/configure/RELEASE.local"
    printf 'EPICS_BASE=%s\nSUPPORT=%s\nASYN=$(SUPPORT)/asyn\nAUTOSAVE=\nBUSY=$(SUPPORT)/busy\n' \
        "$EPICS_BASE" "$AD_BUILD" > "$AD_BUILD/busy/configure/RELEASE.local"
    # The areaDetector top the AD modules include.
    cat > "$AD_BUILD/configure/RELEASE_LIBS_INCLUDE" <<EOF
SUPPORT=$AD_BUILD
AREA_DETECTOR=\$(SUPPORT)
ASYN=\$(SUPPORT)/asyn
ADSUPPORT=\$(AREA_DETECTOR)/ADSupport
ADCORE=\$(AREA_DETECTOR)/ADCore
EPICS_BASE=$EPICS_BASE
EOF
    { cat "$AD_BUILD/configure/RELEASE_LIBS_INCLUDE"; echo 'BUSY=$(SUPPORT)/busy'; } \
        > "$AD_BUILD/configure/RELEASE_PRODS_INCLUDE"
    # ADSupport's bundled HDF5 and the compressions NDFileHDF5 offers.
    cat > "$AD_BUILD/configure/CONFIG_SITE" <<'EOF'
WITH_BOOST=NO
WITH_PVA=NO
WITH_PVXS=NO
WITH_QSRV=NO
WITH_BLOSC=YES
BLOSC_EXTERNAL=NO
WITH_BITSHUFFLE=YES
BITSHUFFLE_EXTERNAL=NO
WITH_GRAPHICSMAGICK=NO
WITH_HDF5=YES
HDF5_EXTERNAL=NO
WITH_JPEG=YES
JPEG_EXTERNAL=NO
WITH_NETCDF=NO
WITH_NEXUS=NO
WITH_OPENCV=NO
WITH_SZIP=YES
SZIP_EXTERNAL=NO
WITH_TIFF=NO
XML2_EXTERNAL=NO
WITH_ZLIB=YES
ZLIB_EXTERNAL=NO
WITH_CBF=NO
WITH_JSON=YES
BUILD_IOCS=YES
EOF
    for m in "${MODULES[@]}"; do
        log=$AD_BUILD/$m-build.log
        echo "building $m ($log)"
        if ! make -C "$AD_BUILD/$m" -j"$(nproc)" > "$log" 2>&1; then
            tail -n 40 "$log"
            echo "$m: build failed" >&2
            exit 1
        fi
    done
}

run() {
    local ioc=$IOC_TOP/bin/$EPICS_HOST_ARCH/simDetectorApp
    [ -x "$ioc" ] || { echo "$ioc is not built; run '$0 build'" >&2; exit 1; }
    export TOP=$IOC_TOP ADCORE=$AD_BUILD/ADCore ADSIMDETECTOR=$AD_BUILD/ADSimDetector
    export PREFIX=$HKLRS_IOC_PREFIX
    export EPICS_CA_SERVER_PORT=$HKLRS_IOC_CA_PORT
    export EPICS_CAS_INTF_ADDR_LIST=127.0.0.1
    export EPICS_CAS_AUTO_BEACON_ADDR_LIST=NO EPICS_CAS_BEACON_ADDR_LIST=127.0.0.1
    if [ -t 0 ]; then
        exec "$ioc" "$HERE/st.cmd"
    fi
    # The iocsh exits at the end of its input: give it a fifo the IOC holds
    # a writer of itself, so it runs until it is killed.
    local fifo
    fifo=$(mktemp -u "${TMPDIR:-/tmp}/simdetector-ioc.XXXXXX")
    mkfifo "$fifo"
    exec 3<> "$fifo"
    rm -f "$fifo"
    exec "$ioc" "$HERE/st.cmd" <&3
}

test_() {
    export EPICS_CA_ADDR_LIST=127.0.0.1 EPICS_CA_AUTO_ADDR_LIST=NO
    export EPICS_CA_SERVER_PORT=$HKLRS_IOC_CA_PORT HKLRS_IOC_PREFIX
    cd "$HERE/../../.."
    exec cargo nextest run -p hklrs-bsrs --features ioc --run-ignored only \
        --test rsm_ioc --no-capture "${@:2}"
}

case "${1:-all}" in
    build) build ;;
    run) run ;;
    all) build; run ;;
    test) test_ "$@" ;;
    *) echo "usage: $0 [build|run|test]" >&2; exit 2 ;;
esac
