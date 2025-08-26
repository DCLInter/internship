#!/usr/bin/env python3
"""
HDF5 Inspector: print a tree view of groups/datasets with shapes, dtypes,
compression, chunking, and optional attributes & lightweight stats.

- Import and call:  from h5_inspect import inspect_file
- CLI usage:       python h5_inspect.py path/to/file.h5 --show-attrs --stats
"""

import argparse
import sys
import math
from typing import Optional
import h5py
import numpy as np

BRANCH = "├─ "
LAST   = "└─ "
PIPE   = "│  "
SPACE  = "   "

def fmt_bytes(n: Optional[int]) -> str:
    if not n:
        return "?"
    if n < 1024:
        return f"{n} B"
    for unit in ["KB", "MB", "GB", "TB", "PB"]:
        n /= 1024.0
        if n < 1024.0:
            return f"{n:.2f} {unit}"
    return f"{n:.2f} EB"

def short(s: str, max_len: int) -> str:
    return s if len(s) <= max_len else s[: max_len - 1] + "…"

def dataset_summary(dset: h5py.Dataset) -> str:
    shape = dset.shape
    dtype = dset.dtype
    chunks = dset.chunks
    comp = dset.compression
    comp_opts = dset.compression_opts
    filters = []
    if comp is not None:
        filters.append(f"comp={comp}({comp_opts})")
    if chunks is not None:
        filters.append(f"chunks={chunks}")
    if getattr(dset, "shuffle", False):
        filters.append("shuffle")
    if getattr(dset, "fletcher32", False):
        filters.append("fletcher32")
    if getattr(dset, "scaleoffset", None) not in (None, False):
        filters.append(f"scaleoffset={dset.scaleoffset}")
    filtxt = ", ".join(filters) if filters else "no-filters"
    n = int(np.prod(shape)) if shape is not None else 0
    itemsize = dtype.itemsize if hasattr(dtype, "itemsize") else 0
    size_est = n * itemsize if n and itemsize else None
    return f"shape={shape}, dtype={dtype}, {filtxt}, est={fmt_bytes(size_est)}"

def safe_attr_repr(val, max_str: int) -> str:
    try:
        if isinstance(val, (bytes, bytearray)):
            try:
                s = val.decode("utf-8", errors="replace")
            except Exception:
                s = repr(val)
            return short(s, max_str)
        if isinstance(val, np.ndarray):
            if val.size <= 16:
                return short(np.array2string(val, threshold=16), max_str)
            return f"ndarray(shape={val.shape}, dtype={val.dtype})"
        return short(repr(val), max_str)
    except Exception as e:
        return f"<unprintable attr: {e!r}>"

def compute_light_stats(dset: h5py.Dataset, sample: int) -> Optional[str]:
    """
    Compute quick stats without reading whole dataset.
    Numeric dtypes only; samples up to `sample` elements.
    """
    try:
        if not np.issubdtype(dset.dtype, np.number):
            return None
        total = int(np.prod(dset.shape)) if dset.shape else 0
        if total == 0:
            return "empty"
        if total <= sample:
            arr = np.asarray(dset[...], dtype=np.float64)
        else:
            if len(dset.shape) >= 1 and dset.shape[0] > 0:
                k = min(sample, dset.shape[0])
                step = max(1, math.floor(dset.shape[0] / k))
                idx = np.arange(0, dset.shape[0], step)[:k]
                arr = dset[idx, ...].reshape(-1)
                if arr.size > sample:
                    arr = arr[:sample]
                arr = arr.astype(np.float64, copy=False)
            else:
                arr = np.asarray(dset[tuple(slice(0, 1) for _ in dset.shape)], dtype=np.float64)
        if arr.size == 0:
            return "empty"
        finite = np.isfinite(arr)
        if not finite.any():
            return "all non-finite"
        arr = arr[finite]
        return f"min={np.min(arr):.4g}, max={np.max(arr):.4g}, mean={np.mean(arr):.4g}, std={np.std(arr):.4g}, n={arr.size}"
    except Exception as e:
        return f"stats-error: {e}"

def print_tree(
    obj,
    name: str,
    prefix: str,
    is_last: bool,
    show_attrs: bool,
    stats: bool,
    sample: int,
    max_children: int,
    max_attrs: int,
    max_str: int,
) -> None:
    branch = LAST if is_last else BRANCH
    head = prefix + branch

    if isinstance(obj, h5py.Dataset):
        print(f"{head}[D] {name} :: {dataset_summary(obj)}")

        # NEW: dataset attributes (when requested)
        if show_attrs and len(obj.attrs) > 0:
            cnt = 0
            for k, v in obj.attrs.items():
                if cnt >= max_attrs:
                    print(prefix + (SPACE if is_last else PIPE) + f"   … {len(obj.attrs)-max_attrs} more attrs")
                    break
                print(prefix + (SPACE if is_last else PIPE) + f"   @ {k} = {safe_attr_repr(v, max_str)}")
                cnt += 1

        if stats:
            s = compute_light_stats(obj, sample)
            if s:
                print(prefix + (SPACE if is_last else PIPE) + f"   ↳ stats: {s}")
        return

    if isinstance(obj, h5py.Group):
        print(f"{head}[G] {name}")

        if show_attrs and len(obj.attrs) > 0:
            cnt = 0
            for k, v in obj.attrs.items():
                if cnt >= max_attrs:
                    print(prefix + (SPACE if is_last else PIPE) + f"   … {len(obj.attrs)-max_attrs} more attrs")
                    break
                print(prefix + (SPACE if is_last else PIPE) + f"   @ {k} = {safe_attr_repr(v, max_str)}")
                cnt += 1

        keys = list(obj.keys())
        clipped = False
        if len(keys) > max_children:
            keys = keys[:max_children]
            clipped = True

        for i, key in enumerate(keys):
            link = obj.get(key, getlink=True)
            child_is_last = (i == len(keys) - 1)
            new_prefix = prefix + (SPACE if is_last else PIPE) + "  "
            if isinstance(link, h5py.SoftLink):
                print(new_prefix + (LAST if child_is_last else BRANCH) + f"[L] {key} -> {link.path}")
            elif isinstance(link, h5py.ExternalLink):
                print(new_prefix + (LAST if child_is_last else BRANCH) + f"[X] {key} => {link.filename}:{link.path}")
            else:
                child = obj[key]
                print_tree(
                    child,
                    key,
                    prefix + (SPACE if is_last else PIPE) + "  ",
                    child_is_last,
                    show_attrs,
                    stats,
                    sample,
                    max_children,
                    max_attrs,
                    max_str,
                )

        if clipped:
            remaining = len(obj.keys()) - len(keys)
            print(prefix + (SPACE if is_last else PIPE) + f"  … {remaining} more items")
        return

    print(f"{head}[?] {name} (unknown type: {type(obj)})")

# ---------- IMPORTABLE FUNCTION ----------

def inspect_file(
    file_path: str,
    root: str = "/",
    show_attrs: bool = False,
    stats: bool = False,
    sample: int = 100000,
    max_children: int = 200,
    max_attrs: int = 50,
    max_str: int = 120,
) -> None:
    """Open an HDF5 file and print a tree view from `root`."""
    with h5py.File(file_path, "r") as f:
        print(f"\nFile: {file_path}")
        print(f"Driver: {getattr(f, 'driver', None)}, read-only: {f.mode == 'r'}")
        if show_attrs and len(f.attrs) > 0:
            print("\nFile attributes:")
            for i, (k, v) in enumerate(f.attrs.items()):
                if i >= max_attrs:
                    print(f"  … {len(f.attrs)-max_attrs} more attrs")
                    break
                print(f"  @ {k} = {safe_attr_repr(v, max_str)}")

        print("\nStructure:")
        root_obj = f[root] if root != "/" else f
        if root == "/":
            print_tree(root_obj, "/", "", True, show_attrs, stats, sample,
                       max_children, max_attrs, max_str)
        else:
            print_tree(root_obj, root, "", True, show_attrs, stats, sample,
                       max_children, max_attrs, max_str)
        print()

# ---------- CLI WRAPPER ----------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Inspect an HDF5 (.h5) file structure.")
    p.add_argument("file", help="Path to .h5 file")
    p.add_argument("--root", default="/", help="Root group to start from (default: '/')")
    p.add_argument("--show-attrs", action="store_true", help="Print attributes of groups/datasets (truncated)")
    p.add_argument("--stats", action="store_true", help="Compute quick numeric stats for datasets")
    p.add_argument("--sample", type=int, default=100000, help="Max elements to sample for stats (default: 100000)")
    p.add_argument("--max-children", type=int, default=200, help="Max children to list per group")
    p.add_argument("--max-attrs", type=int, default=50, help="Max attributes to print per object")
    p.add_argument("--max-str", type=int, default=120, help="Max characters for attribute string values")
    return p

def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        inspect_file(
            file_path=args.file,
            root=args.root,
            show_attrs=args.show_attrs,
            stats=args.stats,
            sample=args.sample,
            max_children=args.max_children,
            max_attrs=args.max_attrs,
            max_str=args.max_str,
        )
    except (OSError, IOError) as e:
        print(f"Error opening file: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyError as e:
        print(f"Invalid --root path: {e}", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()
