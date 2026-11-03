"""Workspace-scoped development process manager (Windows and Linux)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STATE = DATA / "processes.json"
SERVICES = {"identity": 8101, "media": 8102, "inspection": 8103,
            "analysis": 8104, "reporting": 8105, "gateway": 8000}


def ensure_workspace(path: Path) -> Path:
    path = path.resolve()
    if path != ROOT and ROOT not in path.parents:
        raise ValueError(f"Path is outside workspace: {path}")
    return path


def load_env() -> dict[str, str]:
    env = dict(os.environ)
    for key, value in {"TEMP": ROOT / ".cache/tmp", "TMP": ROOT / ".cache/tmp",
                       "PIP_CACHE_DIR": ROOT / ".cache/pip",
                       "npm_config_cache": ROOT / ".cache/npm",
                       "PLAYWRIGHT_BROWSERS_PATH": ROOT / ".cache/playwright",
                       "PYTHONPYCACHEPREFIX": ROOT / ".cache/pycache"}.items():
        ensure_workspace(value).mkdir(parents=True, exist_ok=True)
        env[key] = str(value)
    file = ROOT / ".env"
    if not file.exists():
        values = {"ROAD_ENV": "development", "ROAD_INTERNAL_SECRET": secrets.token_urlsafe(48),
                  "ROAD_BOOTSTRAP_PASSWORD": secrets.token_urlsafe(20)}
        values.update({f"{name.upper()}_DB_PASSWORD": secrets.token_urlsafe(24)
                       for name in SERVICES if name != "gateway"})
        file.write_text("\n".join(f"{k}={v}" for k, v in values.items()) + "\n", encoding="utf-8")
        try:
            file.chmod(0o600)
        except OSError:
            pass
    for line in file.read_text(encoding="utf-8-sig").splitlines():
        if line.strip() and not line.lstrip().startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip().strip("\"").strip("'")
    env["ROAD_DATA_DIR"] = str(DATA)
    env["PYTHONPATH"] = str(ROOT)
    env["ROAD_GATEWAY_URL"] = "http://127.0.0.1:8000"
    for name, port in SERVICES.items():
        env.setdefault(f"ROAD_SERVICE_{name.upper()}_URL", f"http://127.0.0.1:{port}")
    env.setdefault("ROAD_INSTANCE_ID", secrets.token_hex(16))
    return env


def read_state() -> dict:
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}


def owned(record: dict) -> bool:
    import psutil
    try:
        process = psutil.Process(record["pid"])
        return process.is_running() and abs(process.create_time() - record["created_at"]) < 0.1
    except (psutil.NoSuchProcess, psutil.AccessDenied, KeyError):
        return False


def request_health(port: int) -> dict | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
            return json.load(response) if response.status == 200 else None
    except (OSError, urllib.error.URLError, ValueError):
        return None


def stop() -> None:
    import psutil
    state = read_state()
    processes = []
    for name, record in state.items():
        if not owned(record):
            continue
        parent = psutil.Process(record["pid"])
        for child in parent.children(recursive=True):
            processes.append(child)
        processes.append(parent)
        print(f"Stopping owned {name} (PID {record['pid']})")
    for process in reversed(processes):
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(processes, timeout=8)
    for process in alive:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
    if STATE.exists():
        STATE.unlink()


def spawn(name: str, command: list[str], env: dict, state: dict) -> None:
    import psutil
    log_dir = ensure_workspace(DATA / "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    options = {"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    with (log_dir / f"{name}.log").open("ab") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, **options)
    state[name] = {"pid": process.pid, "created_at": psutil.Process(process.pid).create_time(),
                   "command": command, "log": f"data/logs/{name}.log"}
    STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def start(frontend: bool, timeout: float) -> None:
    state = read_state()
    if any(owned(record) for record in state.values()):
        raise SystemExit("Owned services are already running. Use status or stop first.")
    for name, port in SERVICES.items():
        if request_health(port) is not None:
            raise SystemExit(f"Port {port} already serves {name}; refusing to take over an unowned process.")
    env = load_env()
    DATA.mkdir(parents=True, exist_ok=True)
    python = ROOT / (".venv/Scripts/python.exe" if os.name == "nt" else ".venv/bin/python")
    if not python.exists():
        raise SystemExit("Create project .venv and install requirements.txt first.")
    state = {}
    try:
        for name, port in SERVICES.items():
            spawn(name, [str(python), "-m", "uvicorn", f"services.{name}.app:app",
                         "--host", "127.0.0.1", "--port", str(port), "--log-level", "info"], env, state)
        deadline = time.monotonic() + timeout
        pending = set(SERVICES)
        while pending and time.monotonic() < deadline:
            for name in list(pending):
                if not owned(state[name]):
                    raise RuntimeError(f"{name} exited. Inspect data/logs/{name}.log")
                if request_health(SERVICES[name]) is not None:
                    pending.remove(name)
                    print(f"Ready: {name}")
            if pending:
                time.sleep(0.3)
        if pending:
            raise RuntimeError("Readiness timeout: " + ", ".join(sorted(pending)))
        if frontend:
            command = (["cmd.exe", "/d", "/c", "npm.cmd", "--prefix", "frontend", "run", "dev", "--", "--port", "5173", "--strictPort"]
                       if os.name == "nt" else ["npm", "--prefix", "frontend", "run", "dev", "--", "--port", "5173", "--strictPort"])
            spawn("frontend", command, env, state)
        print("Gateway: http://127.0.0.1:8000")
        if frontend:
            print("Web: http://127.0.0.1:5173")
        print("Bootstrap login: demo / admin@road.local (password is in workspace .env)")
    except Exception:
        stop()
        raise


def status() -> None:
    state = read_state()
    for name, port in SERVICES.items():
        record = state.get(name)
        print(f"{name:12} port={port} owned={bool(record and owned(record))} healthy={request_health(port) is not None}")
    if "frontend" in state:
        print(f"frontend     port=5173 owned={owned(state['frontend'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["start", "stop", "status", "env"])
    parser.add_argument("--frontend", action="store_true")
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    if args.action == "start":
        start(args.frontend, args.timeout)
    elif args.action == "stop":
        stop()
    elif args.action == "status":
        status()
    else:
        load_env()
        print("Project .env is ready; values are not printed.")
