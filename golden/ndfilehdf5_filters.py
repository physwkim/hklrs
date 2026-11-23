"""The compressions areaDetector's NDFileHDF5 (ADCore R3-14-182-g94e84778)
writes, each as a file it writes, for hklrs-bsrs's Hdf5FrameReader.

Writes ndfilehdf5_filters.h5 next to this script with h5py 3.16.0, libhdf5
2.2.0 and hdf5plugin 7.1.0. Each group /<name> holds the dataset `data` of
two uint16 frames, shape [2, 64, 48], maxshape [None, 64, 48], one chunk a
frame, in a file of the latest format as NDFileHDF5 writes it in SWMR mode.
Pixel (r, c) of frame f is (101 f + 37 (r // 4) + 11 (c // 3)) % 4096.

Filters and their cd_values as NDFileHDF5Plugin's setup gives them
(NDFileHDF5.cpp:3354-3459):
  nbit       H5Pset_nbit on a 12-bit type, offset 0 and offset 4, and on
             the full 16 bits, which the filter passes through
  szip       H5Pset_szip(NN, 16)
  deflate    H5Pset_deflate(6)
  blosc      32001, cds[4..7] level 5, shuffle, compressor: each of
             blosclz lz4 lz4hc snappy zlib zstd with byte shuffle, and lz4
             with no shuffle and with bit shuffle
  bshuf_lz4  32008, cds {0 (block size), 2 (lz4)}
  lz4        32004, cds {0, 0}
  jpeg       32019, cds {90, 48, 64, 0} on uint8 frames; no encoder here,
             so its chunks are placeholder bytes
and the filters NDFileHDF5 does not write that rust-hdf5 0.7.13 decodes,
so that each filter Hdf5FrameReader reads is tested: shuffle before deflate,
fletcher32, scaleoffset, lzf, zstd (32015), bitshuffle without compression,
szip with entropy coding, and bitgroom (32022) and bitround (32023) on
float32 frames, whose decoding is no change and whose chunks are the frames
as they are; and two it does not decode: bitshuffle with zstd (32008
cds[4] 3) and bzip2 (307).

Every compressed chunk is checked to be smaller than the frame, and every
blosc chunk to be compressed (not memcpyed) with its compressor, so each
codec's decoder is what a read exercises.
"""

import os

import h5py
import hdf5plugin
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SHAPE = (2, 64, 48)
FRAME = SHAPE[1] * SHAPE[2] * 2
# c-blosc 1 compressor codes (cds[6]) and the format each writes in its
# header flags (flags >> 5).
BLOSC = {"blosclz": (0, 0), "lz4": (1, 1), "lz4hc": (2, 1), "snappy": (3, 2), "zlib": (4, 3), "zstd": (5, 4)}


def pixels():
    f, r, c = np.indices(SHAPE)
    return ((101 * f + 37 * (r // 4) + 11 * (c // 3)) % 4096).astype(np.uint16)


def frames(h5, name, dtype=np.uint16, **filters):
    return h5.create_group(name).create_dataset(
        "data", shape=SHAPE, maxshape=(None,) + SHAPE[1:], chunks=(1,) + SHAPE[1:], dtype=dtype, **filters
    )


def plain(h5, name, **filters):
    ds = frames(h5, name, **filters)
    ds[...] = pixels()
    return ds


def unencoded(h5, name, filter_id, dtype):
    """Frames through a filter h5py has no encoder for; chunks as written."""
    return frames(h5, name, dtype, compression=filter_id, allow_unknown_filter=True)


def nbit(h5, name, offset, precision=12):
    tid = h5py.h5t.STD_U16LE.copy()
    tid.set_precision(precision)
    tid.set_offset(offset)
    space = h5py.h5s.create_simple(SHAPE, (h5py.h5s.UNLIMITED,) + SHAPE[1:])
    dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    dcpl.set_chunk((1,) + SHAPE[1:])
    # H5Pset_nbit, which h5py does not wrap, is this (H5Pdcpl.c:3076 in
    # ADSupport R1-10-48-gfe23754's libhdf5).
    dcpl.set_filter(h5py.h5z.FILTER_NBIT, h5py.h5z.FLAG_OPTIONAL)
    dsid = h5py.h5d.create(h5.create_group(name).id, b"data", tid, space, dcpl=dcpl)
    dsid.write(h5py.h5s.ALL, h5py.h5s.ALL, pixels())
    return h5[name]["data"]


def chunks(ds):
    return [ds.id.read_direct_chunk((i, 0, 0))[1] for i in range(SHAPE[0])]


def main():
    path = os.path.join(HERE, "ndfilehdf5_filters.h5")
    with h5py.File(path, "w", libver="latest") as h5:
        compressed = [nbit(h5, "nbit", 0), nbit(h5, "nbit_offset4", 4)]
        compressed.append(plain(h5, "szip", compression="szip", compression_opts=("nn", 16)))
        compressed.append(plain(h5, "deflate", compression="gzip", compression_opts=6))
        blosc = {}
        for cname in BLOSC:
            blosc[f"blosc_{cname}"] = (cname, hdf5plugin.Blosc.SHUFFLE)
        blosc["blosc_lz4_noshuffle"] = ("lz4", hdf5plugin.Blosc.NOSHUFFLE)
        blosc["blosc_lz4_bitshuffle"] = ("lz4", hdf5plugin.Blosc.BITSHUFFLE)
        for name, (cname, shuffle) in blosc.items():
            ds = plain(h5, name, **hdf5plugin.Blosc(cname=cname, clevel=5, shuffle=shuffle))
            assert ds.id.get_create_plist().get_filter(0)[2][4:7] == (5, shuffle, BLOSC[cname][0]), name
            for chunk in chunks(ds):
                flags = chunk[2]
                assert not flags & 0x02, f"{name}: memcpyed"
                assert flags >> 5 == BLOSC[cname][1], f"{name}: format {flags >> 5}"
            compressed.append(ds)
        compressed.append(plain(h5, "bshuf_lz4", **hdf5plugin.Bitshuffle(nelems=0, cname="lz4")))
        compressed.append(plain(h5, "lz4", **hdf5plugin.LZ4(nbytes=0)))
        compressed.append(plain(h5, "shuffle_deflate", shuffle=True, compression="gzip", compression_opts=6))
        compressed.append(plain(h5, "scaleoffset", scaleoffset=0))
        compressed.append(plain(h5, "lzf", compression="lzf"))
        compressed.append(plain(h5, "zstd", **hdf5plugin.Zstd(clevel=3)))
        compressed.append(plain(h5, "szip_ec", compression="szip", compression_opts=("ec", 16)))
        compressed.append(plain(h5, "bshuf_zstd", **hdf5plugin.Bitshuffle(nelems=0, cname="zstd")))
        compressed.append(plain(h5, "bzip2", **hdf5plugin.BZip2(blocksize=9)))
        for ds in compressed:
            assert (ds[...] == pixels()).all(), ds.name
            sizes = [len(c) for c in chunks(ds)]
            assert all(n < FRAME for n in sizes), (ds.name, sizes)
        whole = [nbit(h5, "nbit_full", 0, 16), plain(h5, "fletcher32", fletcher32=True)]
        whole.append(plain(h5, "bshuf_none", **hdf5plugin.Bitshuffle(nelems=0, cname="none")))
        for ds in whole:
            assert (ds[...] == pixels()).all(), ds.name
        for name, filter_id in [("bitgroom", 32022), ("bitround", 32023)]:
            ds = unencoded(h5, name, filter_id, np.float32)
            for i, frame in enumerate(pixels().astype(np.float32)):
                ds.id.write_direct_chunk((i, 0, 0), frame.tobytes())
            whole.append(ds)
        jpeg = frames(
            h5,
            "jpeg",
            np.uint8,
            compression=32019,
            compression_opts=(90, SHAPE[2], SHAPE[1], 0),
            allow_unknown_filter=True,
        )
        for i in range(SHAPE[0]):
            jpeg.id.write_direct_chunk((i, 0, 0), b"no JPEG encoder")
        for ds in compressed + whole + [jpeg]:
            pipeline = ds.id.get_create_plist()
            print(ds.name, [pipeline.get_filter(i)[0:3:2] for i in range(pipeline.get_nfilters())])


if __name__ == "__main__":
    main()
