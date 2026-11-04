#!/usr/bin/env python3
"""Reproducible 146/187 source preparation for Stage 5S (2026-10-01 archive)."""
from pathlib import Path
import hashlib
import json
import shutil
import sys
import zipfile

EXPECTED = "d4089988750348e4b0a708f9a715b8bac1e314f415b034b142f342bc7466745c"
REL = "source/engine/yaneuraou-engine/yaneuraou-search.cpp"
OLD = b"constexpr int nonPawnWeight = 146;"
NEW = b"constexpr int nonPawnWeight = 187;"

def sha256(data):
    return hashlib.sha256(data).hexdigest()

def prepare(source):
    data = source.read_bytes()
    actual_sha = sha256(data)
    if actual_sha != EXPECTED:
        raise RuntimeError("Original ZIP SHA256 mismatch: " + actual_sha)
    with zipfile.ZipFile(source) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError("ZIP CRC failure: " + bad_member)
        items = archive.infolist()
        if len(items) != 220 or sum(i.filename == REL for i in items) != 1:
            raise RuntimeError("Unexpected archive structure")
        for label in ("146", "187"):
            out = Path("stage5s_build_" + label)
            if out.exists():
                shutil.rmtree(out)
            out.mkdir(parents=True)
            for info in items:
                name = info.filename
                if info.is_dir() or name.startswith("/") or "\\" in name or ".." in Path(name).parts:
                    raise RuntimeError("Unsafe archive entry: " + name)
                if not name.startswith("source/"):
                    raise RuntimeError("Unexpected source path: " + name)
                if name.lower().endswith(".exe"):
                    continue
                raw = archive.read(info)
                if name == REL:
                    if raw.count(OLD) != 1 or raw.count(NEW) != 0:
                        raise RuntimeError("Original nonPawnWeight=146 not unique")
                    if label == "187":
                        raw = raw.replace(OLD, NEW)
                dest = out / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(raw)
    def tree_hashes(root):
        return {
            p.relative_to(root).as_posix(): sha256(p.read_bytes())
            for p in root.rglob("*") if p.is_file()
        }
    a = tree_hashes(Path("stage5s_build_146"))
    b = tree_hashes(Path("stage5s_build_187"))
    if set(a) != set(b) or [p for p in a if a[p] != b[p]] != [REL]:
        raise RuntimeError("Unexpected A/B source differences")
    both_compat = []
    for label in ("146", "187"):
        path = Path("stage5s_build_" + label) / "source/numa.h"
        src = path.read_bytes()
        if b"#include <iterator>" not in src:
            lines = src.splitlines(keepends=True)
            positions = [i for i, line in enumerate(lines) if line.strip() == b"#include <algorithm>"]
            if len(positions) != 1:
                raise RuntimeError("Cannot apply identical Clang compatibility fix")
            idx = positions[0]
            eol = b"\r\n" if lines[idx].endswith(b"\r\n") else b"\n"
            lines.insert(idx + 1, b"#include <iterator>" + eol)
            src = b"".join(lines)
            path.write_bytes(src)
        both_compat.append(sha256(path.read_bytes()))
    if both_compat[0] != both_compat[1]:
        raise RuntimeError("Common compatibility fix mismatch")
    lock = {
        "original_zip_sha256": actual_sha,
        "original_zip_entries": 220,
        "source_code_entries_each": len(a),
        "single_ab_source_difference": REL,
        "only_ab_change": "nonPawnWeight = 146 -> 187",
        "source146_sha256": a[REL],
        "source187_sha256": b[REL],
        "common_compiler_fix": "#include <iterator> in source/numa.h for both variants",
        "common_numa_sha256": both_compat[0],
        "stale_prebuilt_exe": "excluded from both trees"
    }
    Path("STAGE5S_BUILD_SOURCE_LOCK.json").write_text(
        json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("STAGE5S_SOURCE_PAIR_PASS", json.dumps(lock, ensure_ascii=False))

if __name__ == "__main__":
    prepare(Path(sys.argv[1] if len(sys.argv) > 1 else "stage5s/source.zip"))
