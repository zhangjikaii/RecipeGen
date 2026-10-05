#!/usr/bin/env python3
"""Control only RecipeGen's project-local, localhost-only Neo4j installation.

Runtime downloads are separate, pinned and checksummed. Credentials are generated
locally in an ignored file; they are never printed or included in project bundles.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / ".runtime"
HOME_DIR = RUNTIME / "neo4j-community-5.26.31"
CONFIG = RUNTIME / "local-neo4j.json"
ALIAS_DIR = Path(tempfile.gettempdir()) / ("recipegen-runtime-" + hashlib.sha256(str(ROOT).encode()).hexdigest()[:12])

def environment():
    candidates = list(RUNTIME.glob("zulu21*/zulu*.jre/Contents/Home")) + list(RUNTIME.glob("zulu21*/Contents/Home"))
    if not candidates:
        candidates = list(RUNTIME.glob("zulu21*"))
    java = next((p for p in candidates if (p / "bin/java").is_file()), None)
    if java is None or not (HOME_DIR / "bin/neo4j").is_file():
        raise ValueError("Project-local verified Java 21 / Neo4j 5.26.31 runtime is missing")
    # Neo4j 5's launcher escapes non-ASCII home paths in Java argument files.
    # Keep actual files in the project and launch via reproducible ASCII aliases.
    ALIAS_DIR.mkdir(exist_ok=True)
    for name, target in (("java", java), ("neo4j", HOME_DIR)):
        alias = ALIAS_DIR / name
        if alias.is_symlink() and alias.resolve() != target.resolve():
            raise ValueError("Runtime alias points to another installation")
        if not alias.exists():
            alias.symlink_to(target, target_is_directory=True)
    env = {**os.environ, "JAVA_HOME": str(ALIAS_DIR / "java"), "NEO4J_HOME": str(ALIAS_DIR / "neo4j"), "NEO4J_CONF": str(ALIAS_DIR / "neo4j/conf")}
    return env

def initialize(env):
    credentials = json.loads(CONFIG.read_text()) if CONFIG.exists() else {"uri": "bolt://127.0.0.1:8767", "user": "neo4j", "password": secrets.token_urlsafe(24),
                   "database": "neo4j", "browser_url": "http://127.0.0.1:8744", "home": str(HOME_DIR), "java_home": env["JAVA_HOME"]}
    conf = HOME_DIR / "conf/neo4j.conf"
    lines = conf.read_text().splitlines()
    updates = {"server.default_listen_address": "127.0.0.1", "server.default_advertised_address": "127.0.0.1",
               "server.bolt.listen_address": "127.0.0.1:8767", "server.bolt.advertised_address": "127.0.0.1:8767",
               "server.http.listen_address": "127.0.0.1:8744", "server.http.advertised_address": "127.0.0.1:8744",
               "server.https.enabled": "false", "server.memory.heap.initial_size": "256m", "server.memory.heap.max_size": "768m",
               "server.memory.pagecache.size": "256m", "db.tx_log.rotation.retention_policy": "1 files", "db.tx_log.rotation.size": "64M",
               "dbms.usage_report.enabled": "false"}
    lines = [line for line in lines if line.partition("=")[0].strip() not in updates]
    conf.write_text("\n".join(lines) + "\n\n# RecipeGen project-local settings\n" + "\n".join(f"{k}={v}" for k, v in updates.items()) + "\n")
    if (RUNTIME / "database-initialized").exists():
        return credentials
    result = subprocess.run([str(ALIAS_DIR / "neo4j/bin/neo4j-admin"), "dbms", "set-initial-password", credentials["password"]],
                            env=env, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError("Neo4j initialization failed: " + result.stderr.replace(credentials["password"], "[redacted]"))
    CONFIG.write_text(json.dumps(credentials, indent=2) + "\n")
    CONFIG.chmod(0o600)
    (RUNTIME / "database-initialized").write_text("Initialized via ASCII runtime aliases.\n")
    return credentials

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("start", "console", "stop", "status", "import-graph"))
    args = parser.parse_args()
    env = environment()
    if args.command in ("start", "console"):
        initialize(env)
    if args.command == "import-graph":
        if not CONFIG.exists():
            raise ValueError("Start the project-local database first")
        c = json.loads(CONFIG.read_text())
        env.update(NEO4J_URI=c["uri"], NEO4J_USER=c["user"], NEO4J_PASSWORD=c["password"], NEO4J_DATABASE=c["database"])
        return subprocess.call([sys.executable, str(ROOT / "scripts/import_test_graph_neo4j.py"),
                                "--nodes", str(ROOT / "data/test_graph/neo4j_import/nodes.csv"),
                                "--relationships", str(ROOT / "data/test_graph/neo4j_import/relationships.csv"),
                                "--report", str(ROOT / "reports/test-graph-neo4j.json")], cwd=ROOT, env=env)
    return subprocess.call([str(ALIAS_DIR / "neo4j/bin/neo4j"), args.command], env=env, cwd=ROOT)

if __name__ == "__main__":
    raise SystemExit(main())
