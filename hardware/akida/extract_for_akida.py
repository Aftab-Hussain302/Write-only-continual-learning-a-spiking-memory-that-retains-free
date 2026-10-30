#!/usr/bin/env python3
"""Extract only the data the Akida measurement reads, straight from the uploaded archives, for a machine with little disk space.

    python3 extract_for_akida.py --src uploads/Akida --data_root data

Looks through --src (archives, nested archives inside them, and plain folders) and writes
    <data_root>/shd/shd_train.h5                      the whole file (the loader reads all of it)
    <data_root>/nmnist/Train/<digit>/<first N>.bin    the first N files of each digit in sorted order, N = 700 by default:
                                                      exactly the files the N-MNIST loader reads (500 train + 200 test per class)
Nested zip archives are read in place, never unpacked to disk, so the extra space needed is only what is written (a few hundred MB).
Nothing in --src is changed or deleted. Standard library only; run it again at any time (files already in place are kept)."""
import argparse, gzip, hashlib, os, shutil, sys, time, zipfile
from pathlib import PurePosixPath

JUNK = ("__MACOSX",)


def junk(path): return any(p in JUNK or p.startswith("._") for p in PurePosixPath(path).parts)


def is_nmnist_bin(path):
    """(digit, name) when path is .../Train/<digit>/<name>.bin, else None."""
    p = PurePosixPath(path)
    if p.suffix.lower() != ".bin" or len(p.parts) < 3: return None
    if p.parts[-2] in [str(c) for c in range(10)] and p.parts[-3].lower() == "train": return int(p.parts[-2]), p.name
    return None


def walk_zip(zf, label, out, depth=0):
    """Collect N-MNIST .bin entries and shd_train.h5(.gz) entries from a zip and the zips nested inside it (opened in place)."""
    for info in zf.infolist():
        n = info.filename
        if info.is_dir() or junk(n): continue
        low = n.lower()
        if low.endswith(".zip") and depth < 4:
            if PurePosixPath(low).name.startswith("test"): continue          # the official test split is not used
            try: inner = zipfile.ZipFile(zf.open(info))
            except zipfile.BadZipFile: print(f"   skipped {label}/{n}: not a readable zip", flush=True); continue
            walk_zip(inner, f"{label}/{n}", out, depth + 1); continue
        hit = is_nmnist_bin(n)
        if hit: out["bins"].append((hit[0], hit[1], ("zip", zf, info), f"{label}/{n}"))
        elif PurePosixPath(n).name in ("shd_train.h5", "shd_train.h5.gz"): out["shd"].append((("zip", zf, info), f"{label}/{n}", n.endswith(".gz")))


def collect(src):
    out = {"bins": [], "shd": []}
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if not junk(d)]
        for f in files:
            p = os.path.join(root, f)
            if junk(p): continue
            if f.lower().endswith(".zip"):
                try: walk_zip(zipfile.ZipFile(p), f, out)
                except zipfile.BadZipFile: print(f"   skipped {f}: not a readable zip (incomplete upload?)", flush=True)
                continue
            rel = os.path.relpath(p, src).replace(os.sep, "/")
            hit = is_nmnist_bin(rel)
            if hit: out["bins"].append((hit[0], hit[1], ("file", p), rel))
            elif f in ("shd_train.h5", "shd_train.h5.gz"): out["shd"].append((("file", p), rel, f.endswith(".gz")))
    return out


def open_src(ref):
    return open(ref[1], "rb") if ref[0] == "file" else ref[1].open(ref[2])


def size_of(ref): return os.path.getsize(ref[1]) if ref[0] == "file" else ref[2].file_size


def free_mb(path):
    p = os.path.abspath(path)
    while not os.path.exists(p): p = os.path.dirname(p)
    return shutil.disk_usage(p).free / 1e6


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True); ap.add_argument("--data_root", default="data"); ap.add_argument("--per_class", type=int, default=700)
    a = ap.parse_args(); src = os.path.expanduser(a.src); d = os.path.expanduser(a.data_root)
    t0 = time.time(); print(f"looking through {src} ...", flush=True)
    found = collect(src)
    # ---- SHD
    shd_dst = os.path.join(d, "shd", "shd_train.h5")
    if os.path.isfile(shd_dst) and open(shd_dst, "rb").read(8) == b"\x89HDF\r\n\x1a\n":
        print(f"SHD: already in place ({os.path.getsize(shd_dst) / 1e6:.0f} MB)")
    else:
        if not found["shd"]: sys.exit("SHD: no shd_train.h5 (or .h5.gz) found in the archives or folders under --src")
        ref, label, gz = found["shd"][0]
        need = size_of(ref) * (3 if gz else 1) / 1e6 + 50
        if free_mb(d) < need: sys.exit(f"SHD: about {need:.0f} MB needed, {free_mb(d):.0f} MB free")
        os.makedirs(os.path.dirname(shd_dst), exist_ok=True); tmp = shd_dst + ".part"
        with open_src(ref) as s, open(tmp, "wb") as o: shutil.copyfileobj(gzip.GzipFile(fileobj=s) if gz else s, o, 1 << 20)
        if open(tmp, "rb").read(8) != b"\x89HDF\r\n\x1a\n": os.remove(tmp); sys.exit(f"SHD: {label} is not an HDF5 file")
        os.replace(tmp, shd_dst); print(f"SHD: {label} -> {shd_dst} ({os.path.getsize(shd_dst) / 1e6:.0f} MB)", flush=True)
    # ---- N-MNIST: the loader's selection, sorted(glob(Train/<digit>/*.bin))[:N]
    by_digit = {c: {} for c in range(10)}
    for c, name, ref, label in found["bins"]: by_digit[c].setdefault(name, (ref, label))
    short = [c for c in range(10) if len(by_digit[c]) < a.per_class]
    if short: sys.exit(f"N-MNIST: too few Train/<digit>/*.bin files for digits {short} "
                       f"({[len(by_digit[c]) for c in range(10)]} found, {a.per_class} each needed)")
    pick = [(c, n, *by_digit[c][n]) for c in range(10) for n in sorted(by_digit[c])[:a.per_class]]
    h = hashlib.sha256()
    for c in range(10): h.update("\n".join(n for cc, n, *_ in pick if cc == c).encode())
    need = sum(size_of(r) for _, _, r, _ in pick) / 1e6 + 50
    if free_mb(d) < need: sys.exit(f"N-MNIST: about {need:.0f} MB needed, {free_mb(d):.0f} MB free")
    # read each archive front to back (nested zips are streams: going backwards would decompress them again)
    pick.sort(key=lambda x: (id(x[2][1]), x[2][2].header_offset if x[2][0] == "zip" else 0, x[1]))
    written = kept = 0
    for i, (c, name, ref, label) in enumerate(pick):
        dst = os.path.join(d, "nmnist", "Train", str(c), name)
        if os.path.isfile(dst) and os.path.getsize(dst) == size_of(ref): kept += 1; continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open_src(ref) as s, open(dst + ".part", "wb") as o: shutil.copyfileobj(s, o)
        os.replace(dst + ".part", dst); written += 1
        if (i + 1) % 1000 == 0: print(f"   N-MNIST {i + 1}/{len(pick)} files", flush=True)
    counts = [len(os.listdir(os.path.join(d, "nmnist", "Train", str(c)))) for c in range(10)]
    print(f"N-MNIST: {written} files written, {kept} already in place; per digit {counts}; file list sha256 {h.hexdigest()[:16]}")
    extra = [c for c in range(10) if counts[c] != a.per_class]
    if extra: print(f"   note: digits {extra} hold other files besides these {a.per_class}; the loader reads the first {a.per_class} in sorted order")
    print(f"done in {time.time() - t0:.0f} s; {free_mb(d):.0f} MB free.\nDATA READY under {os.path.abspath(d)}")


if __name__ == "__main__": main()
