#!/usr/bin/env python3
"""启动、查看或停止本项目的本地网页服务；不启动媒体批处理。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / ".runtime"
STATE = RUNTIME / "recipegen-system-server.json"
LOG = RUNTIME / "recipegen-system.log"
PYTHON = ROOT / ".venv/bin/python"


def saved_state():
    try:
        return json.loads(STATE.read_text())
    except (FileNotFoundError, ValueError):
        return {}


def owned_process(state):
    """停止前核对 PID 和启动命令，避免误伤其他服务。"""
    pid = state.get("pid")
    if not isinstance(pid, int) or pid <= 1:
        return False
    result = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True)
    # macOS 的 Python.app 会改写 argv[0]；用服务参数和真实 cwd 共同核对归属。
    expected = f"-m recipegen serve --host 127.0.0.1 --port {state.get('port')}"
    if result.returncode != 0 or expected not in result.stdout:
        return False
    cwd = subprocess.run(["lsof", "-a", "-p", str(pid), "-d", "cwd", "-Fn"], capture_output=True, text=True)
    return cwd.returncode == 0 and f"n{ROOT}" in cwd.stdout.splitlines()


def healthy(url):
    try:
        with urlopen(url + "/health", timeout=1) as response:
            return json.load(response).get("status") == "ok"
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["start", "status", "stop"])
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    state = saved_state()
    owned = owned_process(state)
    if args.action == "status":
        print(json.dumps({**state, "running": owned, "healthy": owned and healthy(state["url"])}, ensure_ascii=False))
        return 0
    if args.action == "stop":
        if owned:
            os.kill(state["pid"], signal.SIGTERM)
            for _ in range(50):
                if not owned_process(state):
                    break
                time.sleep(0.1)
        state["running"] = owned_process(state)
        if STATE.exists():
            STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"stopped": not state["running"], "url": state.get("url")}, ensure_ascii=False))
        return 0 if not state["running"] else 1
    if owned:
        print(json.dumps({**state, "running": True, "healthy": healthy(state["url"])}, ensure_ascii=False))
        return 0 if healthy(state["url"]) else 1
    if not 1024 <= args.port <= 65535:
        parser.error("端口须在 1024～65535 范围")
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", args.port))
        except OSError:
            parser.exit(1, f"端口 {args.port} 已被占用，未停止其他服务。\n")
    RUNTIME.mkdir(exist_ok=True)
    os.umask(0o077)
    command = [str(PYTHON), "-m", "recipegen", "serve", "--host", "127.0.0.1", "--port", str(args.port)]
    with LOG.open("a") as log:
        os.chmod(LOG, 0o600)
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                   start_new_session=True)
    state = {"pid": process.pid, "port": args.port, "url": f"http://127.0.0.1:{args.port}",
             "project_root": str(ROOT), "log": str(LOG), "started_at": time.time()}
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    os.chmod(STATE, 0o600)
    for _ in range(100):
        if healthy(state["url"]):
            print(json.dumps({**state, "running": True, "healthy": True}, ensure_ascii=False))
            return 0
        if process.poll() is not None:
            break
        time.sleep(0.1)
    if process.poll() is None:
        process.terminate()
    parser.exit(1, f"服务未就绪，请查看 {LOG}\n")


if __name__ == "__main__":
    raise SystemExit(main())
