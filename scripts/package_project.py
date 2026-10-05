#!/usr/bin/env python3
"""Package publishable source files only; never bundle rebuilt data or runtime."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = {
    ".gitignore", ".env.example", "README.md", "LICENSE", "LICENSE.md",
    "pyproject.toml", "docker-compose.yml", "requirements.txt",
    "requirements-text.txt", "requirements-multimodal.txt",
}
DATA_FILES = {"data/README.md", "data/example_graph.json", "data/evaluation_cases.json"}
REPORT_FILES = {"reports/发布验收.md"}
SOURCE_SUFFIXES = {
    "recipegen": {".py"}, "scripts": {".py", ".cjs"}, "tests": {".py"},
    "static": {".html", ".css", ".js"}, "skills": {".md"},
    "docs": {".md", ".mmd", ".svg"}, ".github": {".md", ".yml", ".yaml"},
}
EXCLUDED_DIRS = {
    ".runtime", ".venv", ".venv-mm", ".venvs", "venv", ".git",
    "__pycache__", ".pytest_cache", "node_modules", "var",
}

def sha_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def publishable(path: Path) -> bool:
    """Use an explicit source allowlist, independent of Git state or cwd."""
    relative = path.relative_to(ROOT)
    name = relative.as_posix()
    if path.is_symlink() or not path.is_file():
        return False
    if any(part in EXCLUDED_DIRS or part.startswith(".venv")
           or part.endswith(".egg-info") for part in relative.parts):
        return False
    if any((ROOT.joinpath(*relative.parts[:i])).is_symlink()
           for i in range(1, len(relative.parts))):
        return False
    if path.name.startswith(".env") and name != ".env.example":
        return False
    if name in ROOT_FILES or name in DATA_FILES or name in REPORT_FILES or name == "docs/neo4j_mapping.example.json":
        return True
    return len(relative.parts) > 1 and path.suffix in SOURCE_SUFFIXES.get(relative.parts[0], set())


def project_files():
    candidates = [ROOT / name for name in ROOT_FILES | DATA_FILES | REPORT_FILES | {"docs/neo4j_mapping.example.json"}]
    for name in SOURCE_SUFFIXES:
        directory = ROOT / name
        if directory.is_dir() and not directory.is_symlink():
            candidates.extend(directory.rglob("*"))
    return sorted({path for path in candidates if publishable(path)})

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT.parent / "output/RecipeGen_测试集图谱项目.zip")
    args = parser.parse_args()
    target = args.output.resolve()
    if target.is_relative_to(ROOT):
        raise ValueError("Package output must be outside the project to prevent self-inclusion")
    files = project_files()
    hashes = {path.relative_to(ROOT).as_posix(): sha_file(path) for path in files}
    # Generate the source manifest inside the archive. Do not read or include
    # raw local reports, which can contain original records and personal paths.
    manifest_name = ROOT.name + "/reports/source-manifest.json"
    manifest = json.dumps(hashes, ensure_ascii=False, indent=2) + "\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(".zip.part")
    with zipfile.ZipFile(part, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, ROOT.name + "/" + path.relative_to(ROOT).as_posix())
        archive.writestr(manifest_name, manifest)
    with zipfile.ZipFile(part) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f"Archive CRC failed: {bad}")
        expected = {ROOT.name + "/" + name for name in hashes} | {manifest_name}
        if set(archive.namelist()) != expected:
            raise ValueError("Source allowlist differs from archive contents")
    part.replace(target)
    report = {"path": str(target), "size_bytes": target.stat().st_size, "sha256": sha_file(target),
              "file_count": len(files) + 1, "archive_crc_verified": True, "source_manifest_files": len(hashes),
              "scope": "publishable_source_allowlist",
              "excluded": ["runtime", "virtualenv", "database stores", "credentials", "logs", "raw data",
                           "original graph exports", "text vectors", "multimodal data", "raw local reports"]}
    target.with_suffix(".manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
