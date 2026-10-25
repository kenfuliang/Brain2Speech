#!/usr/bin/env python3
"""Verify the pinned paper and Dryad artifacts without modifying them."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = Path(__file__).with_name("artifacts.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_file(path: Path, expected_hash: str, expected_size: Optional[int]) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "status": "missing"}
    actual_size = path.stat().st_size
    if expected_size is not None and actual_size != expected_size:
        return {
            "path": str(path),
            "status": "size-mismatch",
            "expected_size": expected_size,
            "actual_size": actual_size,
        }
    actual_hash = sha256(path)
    return {
        "path": str(path),
        "status": "ok" if actual_hash == expected_hash else "hash-mismatch",
        "size": actual_size,
        "sha256": actual_hash,
        "expected_sha256": expected_hash,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--artifact-dir", type=Path, default=REPO_ROOT / "artifacts" / "dryad")
    parser.add_argument("--skip-papers", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    results: list[dict[str, Any]] = []
    if not args.skip_papers:
        for item in manifest["paper"]["files"]:
            results.append(check_file(REPO_ROOT / item["path"], item["sha256"], None))
    for item in manifest["dataset"]["files"]:
        results.append(check_file(args.artifact_dir / item["name"], item["sha256"], item["size"]))

    print(json.dumps(results, indent=2))
    return 0 if all(result["status"] == "ok" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
