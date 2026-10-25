#!/usr/bin/env python3
"""Safely extract selected verified Dryad archives into a workspace."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path, PurePosixPath


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).with_name("artifacts.json")
CORE = {"derived.tar.gz", "languageModel.tar.gz"}


def validate_members(archive: tarfile.TarFile) -> None:
    for member in archive.getmembers():
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe path in archive: {member.name}")
        if member.issym() or member.islnk():
            raise ValueError(f"links are not allowed in archive: {member.name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=REPO_ROOT / "artifacts" / "dryad")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "workspace")
    parser.add_argument("--scope", choices=("core", "all"), default="core")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    selected = [
        item for item in manifest["dataset"]["files"]
        if args.scope == "all" or item["name"] in CORE
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for item in selected:
        source = args.artifact_dir / item["name"]
        if not source.is_file() or source.stat().st_size != item["size"]:
            raise FileNotFoundError(f"missing or incomplete verified archive: {source}")
        marker = args.output_dir / f".{item['name']}.complete"
        if marker.exists():
            print(f"already extracted: {item['name']}")
            continue
        print(f"extracting {item['name']} -> {args.output_dir}", flush=True)
        with tarfile.open(source, "r:gz") as archive:
            validate_members(archive)
            archive.extractall(args.output_dir)
        marker.write_text(item["sha256"] + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
