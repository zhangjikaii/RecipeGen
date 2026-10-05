#!/usr/bin/env python3
"""Install pinned, checksummed Neo4j + Java into this project on macOS ARM64.

No system software or existing Neo4j databases are modified. Network downloads
are bounded. Other platforms may use the existing Docker Compose configuration.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import platform
import shutil
import ssl
import tarfile
from urllib.request import urlopen

PACKAGES = (
    {"name": "neo4j-community-5.26.31-unix.tar.gz", "url": "https://dist.neo4j.org/neo4j-community-5.26.31-unix.tar.gz",
     "sha256": "f8fc23340561405f1ff10ca6ac2d317d095d3c74509a616883c45d7a61f5cfec", "max_bytes": 400 * 1024 * 1024},
    {"name": "zulu21.52.203-ca-jre21.0.12.1-macosx_aarch64.tar.gz", "url": "https://cdn.azul.com/zulu/bin/zulu21.52.203-ca-jre21.0.12.1-macosx_aarch64.tar.gz",
     "sha256": "611673ab332d58e6d3a61506eaafbfb13e9f9b0e01452067a9cc8b2f470289fb", "max_bytes": 150 * 1024 * 1024},
)

def main():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise ValueError("This local runtime is pinned for macOS ARM64; configure your platform's Neo4j explicitly")
    root = Path(__file__).resolve().parents[1] / ".runtime"
    root.mkdir(exist_ok=True)
    if shutil.disk_usage(root).free < 1024 * 1024 * 1024:
        raise ValueError("At least 1 GiB free disk space is required for this runtime")
    ctx = ssl.create_default_context(cafile="/etc/ssl/cert.pem")
    for package in PACKAGES:
        path = root / package["name"]
        if not path.exists():
            part = path.with_suffix(".part")
            h = hashlib.sha256()
            downloaded = 0
            with urlopen(package["url"], context=ctx, timeout=45) as response, part.open("wb") as out:
                while chunk := response.read(1024 * 1024):
                    downloaded += len(chunk)
                    if downloaded > package["max_bytes"]:
                        raise ValueError("Package exceeds download budget")
                    h.update(chunk)
                    out.write(chunk)
            if h.hexdigest() != package["sha256"]:
                raise ValueError("Official package checksum mismatch")
            part.replace(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != package["sha256"]:
            raise ValueError("Existing package checksum mismatch")
        expected_dir = root / package["name"].removesuffix("-unix.tar.gz").removesuffix(".tar.gz")
        if not expected_dir.is_dir():
            with tarfile.open(path) as archive:
                archive.extractall(root, filter="data")
        print(json.dumps({"package": package["name"], "sha256_verified": True, "size_bytes": path.stat().st_size}))
    (root / "runtime-packages.json").write_text(json.dumps(list(PACKAGES), indent=2) + "\n")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
