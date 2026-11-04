"""Bounded, isolated PostgreSQL MSA recovery/load probe; never touches live DBs.

Uses an already running workspace PostgreSQL cluster (or CI PostGIS container).
Creates new databases/roles, retains them for inspection, and stops only its own
six subprocesses. No Docker install, field/user acceptance, or detail backup.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import hashlib
import json
import os
import secrets
import shutil
import socket
import statistics
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import httpx
import psutil

PORTS = {"identity": 19101, "media": 19102, "inspection": 19103,
         "analysis": 19104, "reporting": 19105, "gateway": 19000}
PROXY_PORT = 19112


def percentile(values, fraction):
    return sorted(values)[max(0, min(len(values)-1, int(len(values)*fraction+.999999)-1))]


def checked(response, status=200):
    if response.status_code != status:
        raise AssertionError(f"unexpected_http_status:{response.status_code}:expected:{status}")
    return response.json()["data"]


def wait_ready(port, process):
    deadline = time.monotonic()+60
    with httpx.Client(trust_env=False, timeout=2) as client:
        while time.monotonic() < deadline and process.poll() is None:
            try:
                if client.get(f"http://127.0.0.1:{port}/health").status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(.2)
    raise RuntimeError("isolated_service_not_ready")


@contextmanager
def media_fault_proxy():
    """Only this probe's reporting service uses this deterministic fault point."""
    state = {"fail_delete": False, "injected_failures": 0}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if state["fail_delete"] and self.path == "/internal/privacy/delete":
                state["injected_failures"] += 1
                payload = b'{"error":{"code":"PROBE_DEPENDENCY_OUTAGE"},"meta":{}}'
                self.send_response(503)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers(); self.wfile.write(payload); return
            with httpx.Client(trust_env=False, timeout=10) as client:
                response = client.post(f"http://127.0.0.1:{PORTS['media']}"+self.path,
                                       content=body, headers={k:v for k,v in self.headers.items()
                                                            if k.lower() not in ("host", "content-length")})
            self.send_response(response.status_code)
            self.send_header("Content-Type", response.headers.get("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(response.content)))
            self.end_headers(); self.wfile.write(response.content)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", PROXY_PORT), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        yield state
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=3)


def config_roundtrip(folder):
    from scripts import config_backup
    archive = folder/"settings.zip"
    manifest = config_backup.create(archive)
    _, entries = config_backup.inspect(archive)
    sandbox = folder/"restored-project"; sandbox.mkdir()
    previous = config_backup.ROOT
    try:
        # Explicitly restore into a disposable empty workspace, never the app tree.
        config_backup.ROOT = sandbox
        copied = sandbox/"settings.zip"; copied.write_bytes(archive.read_bytes())
        result = config_backup.restore(copied, apply=True)
    finally:
        config_backup.ROOT = previous
    assert all((sandbox/name).read_bytes() == value for name,value in entries.items())
    assert result["detail_records_restored"] == 0
    return {"files_restored": len(entries), "all_hashes_match": True,
            "detail_records_restored": 0, "secrets_included": False,
            "backup_sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}


def run():
    import psycopg
    from psycopg import sql
    from scripts.manage import load_env
    suffix = uuid.uuid4().hex[:10]
    folder = ROOT/".cache"/"operational-probe"/suffix; folder.mkdir(parents=True)
    env = load_env()
    env.update(ROAD_DATA_DIR=str(folder/"data"), ROAD_POSTGIS_ENABLED="1",
               ROAD_INTERNAL_SECRET=secrets.token_urlsafe(48),
               ROAD_BOOTSTRAP_PASSWORD=secrets.token_urlsafe(24))
    env["ROAD_GATEWAY_URL"] = "http://127.0.0.1:19000"
    for name, port in PORTS.items():
        env[f"ROAD_SERVICE_{name.upper()}_URL"] = f"http://127.0.0.1:{port}"
    for port in (*PORTS.values(), PROXY_PORT):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))  # Fail rather than reuse/stop an existing process.
    pg = ROOT/".cache/postgres-probe"
    if os.name == "nt":
        pid = int((pg/"cluster/postmaster.pid").read_text().splitlines()[0])
        assert Path(psutil.Process(pid).exe()).resolve().is_relative_to((pg/"runtime").resolve())
    results = {"format":"roadwatch-operational-probe-v1", "synthetic_data_only":True,
               "docker_executable_available":bool(shutil.which("docker")),
               "docker_build_executed":False, "field_or_five_user_acceptance":False,
               "live_service_processes_stopped":0, "detail_database_backup_supported":False}
    databases = {}; roles = {}
    with psycopg.connect("host=127.0.0.1 port=55432 user=road_probe dbname=postgres", autocommit=True) as conn:
        results["postgresql_version"] = conn.execute("SHOW server_version").fetchone()[0]
        if os.name == "nt":
            assert Path(conn.execute("SHOW data_directory").fetchone()[0]).resolve() == (pg/"cluster").resolve()
        for name in PORTS:
            if name == "gateway": continue
            db = "road_ops_"+name+"_"+suffix
            role = "road_ops_"+name+"_"+suffix
            conn.execute(sql.SQL("CREATE ROLE {} LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE").format(sql.Identifier(role)))
            conn.execute(sql.SQL("CREATE DATABASE {} OWNER {}").format(sql.Identifier(db),sql.Identifier(role)))
            # PostGIS extension creation requires elevated rights; app role does not.
            if name == "inspection":
                with psycopg.connect(host="127.0.0.1",port=55432,user="road_probe",dbname=db,autocommit=True) as spatial:
                    spatial.execute("CREATE EXTENSION postgis")
            env[name.upper()+"_DATABASE_URL"] = f"postgresql+psycopg://{role}@127.0.0.1:55432/{db}"
            databases[name] = db; roles[name] = role
    # Isolated fixture credentials are not printed or saved in the public report.
    subprocess.run([sys.executable,"-m","services.identity.bootstrap","--org","demo","--login","admin@road.local"],
                   cwd=ROOT,env=env,stdout=subprocess.DEVNULL,check=True,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
    subprocess.run([sys.executable,"-m","services.identity.bootstrap","--org","isolation","--login","admin@road.local"],
                   cwd=ROOT,env=env,stdout=subprocess.DEVNULL,check=True,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
    processes = []; logs = []
    try:
        with media_fault_proxy() as fault:
            for name,port in PORTS.items():
                service_env = dict(env)
                if name == "reporting": service_env["ROAD_SERVICE_MEDIA_URL"] = f"http://127.0.0.1:{PROXY_PORT}"
                log = (folder/(name+".log")).open("wb"); logs.append(log)
                process = subprocess.Popen([sys.executable,"-m","uvicorn",f"services.{name}.app:app","--host","127.0.0.1","--port",str(port)],
                    cwd=ROOT,env=service_env,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
                processes.append(process); wait_ready(port,process)
            sys.path.insert(0,str(ROOT/"tests"))
            import test_http
            with httpx.Client(base_url=env["ROAD_GATEWAY_URL"],trust_env=False,timeout=30) as client:
                def login(org):
                    return {"Authorization":"Bearer "+checked(client.post("/api/v1/auth/login",json={"org_code":org,"login":"admin@road.local","password":env["ROAD_BOOTSTRAP_PASSWORD"]}))["access_token"]}
                auth = login("demo"); foreign = login("isolation")
                fixture = test_http.setup.__wrapped__(client,auth)
                observation = fixture["observation"]
                checked(client.post("/api/v1/observations",headers=fixture["dh"],json=observation),201)
                oid = observation["observation_id"]
                assert client.get("/api/v1/observations/"+oid,headers=foreign).status_code == 404
                for method,path,headers,status in [("GET","/api/v1/observations",{},401),
                    ("GET","/api/v1/users",fixture["dh"],403),
                    ("GET","/api/v1/observations/bad-uuid",auth,422),
                    ("DELETE","/api/v1/users",auth,405)]:
                    response=client.request(method,path,headers=headers)
                    assert response.status_code==status and response.json()["error"]["code"]
                    assert response.json()["meta"]["request_id"]==response.headers["X-Request-ID"]
                paths=["/api/v1/observations","/api/v1/summary","/api/v1/runs","/api/v1/audit-logs?limit=100"]
                def request(index):
                    started=time.monotonic(); response=client.get(paths[index%len(paths)],headers=auth)
                    assert response.status_code==200
                    return time.monotonic()-started
                started=time.monotonic()
                with ThreadPoolExecutor(max_workers=4) as executor: timings=list(executor.map(request,range(160)))
                results["bounded_load"]={"requests":160,"concurrency":4,"statuses":{"200":160},
                    "elapsed_seconds":time.monotonic()-started,"median_ms":statistics.median(timings)*1000,
                    "p95_ms":percentile(timings,.95)*1000,"max_ms":max(timings)*1000,
                    "synthetic_observations":1,"capacity_or_user_acceptance_claim":False}
                report=checked(client.get("/api/v1/reports?format=csv",headers=auth),202)
                wait_job(client,auth,report["job_id"],"succeeded")
                assert client.get("/api/v1/reports/"+report["job_id"]+"/download",headers=auth).status_code==200
                fault["fail_delete"]=True
                receipt=checked(client.post("/api/v1/privacy-requests",headers=auth,json={"request_type":"deletion",
                    "target_refs":[{"kind":"observation","id":oid}],"reason":"SYNTHETIC operational fault test"}),202)
                wait_job(client,auth,receipt["job_id"],"failed")
                assert client.get("/api/v1/observations/"+oid,headers=auth).status_code==410
                state=checked(client.get("/api/v1/privacy-requests/"+receipt["request_id"],headers=auth))
                fault["fail_delete"]=False
                checked(client.patch("/api/v1/privacy-requests/"+receipt["request_id"],headers=auth,json={
                    "expected_version":state["version"],"action":"retry","reason":"SYNTHETIC dependency restored"}))
                wait_job(client,auth,receipt["job_id"],"succeeded")
                assert client.get("/api/v1/observations/"+oid,headers=auth).status_code==410
                assert client.get("/api/v1/images/"+observation["image"]["file_id"]+"/content",headers=auth).status_code==410
                assert client.get("/api/v1/reports/"+report["job_id"]+"/download",headers=auth).status_code==409
                results["queue_recovery"]={"report_completed":True,"injected_dependency_failures":fault["injected_failures"],
                    "deletion_failed_then_manual_retry_completed":True,"blocked_during_failure":True,
                    "observation_image_and_report_unavailable_after_delete":True,"automatic_retry_claim":False}
                results["http_permissions"]={"anonymous":401,"device_admin_route":403,"cross_org":404,"invalid_uuid":422,"wrong_method":405,"error_envelopes_verified":True}
        for name,db in databases.items():
            with psycopg.connect(host="127.0.0.1",port=55432,user=roles[name],dbname=db) as conn:
                versions=[row[0] for row in conn.execute("SELECT version FROM schema_migrations ORDER BY version")]
                assert versions==([1,2,3,4] if name=="inspection" else [1,2])
                if name=="inspection": results["postgis_version"]=conn.execute("SELECT PostGIS_Version()").fetchone()[0]
        with psycopg.connect(host="127.0.0.1",port=55432,user=roles["inspection"],dbname=databases["identity"]) as conn:
            try: conn.execute("SELECT count(*) FROM aggregates")
            except psycopg.errors.InsufficientPrivilege: results["cross_service_table_read_denied"]=True
            else: raise AssertionError("cross_service_table_read_permitted")
        results["service_migrations_verified"]=5
        results["configuration_backup_restore"]=config_roundtrip(folder)
    finally:
        for process in reversed(processes):
            if process.poll() is None: process.terminate()
        for process in processes:
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=5)
        for log in logs: log.close()
    marker_count=sum(sum(marker in line for marker in ("Traceback", "ERROR:", "INTERNAL_ERROR"))
                     for path in folder.glob("*.log") for line in path.read_text(encoding="utf-8",errors="replace").splitlines())
    assert marker_count==0,"isolated_service_log_error"
    results["unexpected_service_log_errors"]=marker_count
    results["isolated_processes_stopped"]=len(processes)
    results["retained_synthetic_databases"]=len(databases)
    output=ROOT/"docs/reports/OPERATIONAL-PROBE-RESULTS.json"
    output.write_text(json.dumps(results,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(results,ensure_ascii=False,indent=2))


def wait_job(client,auth,identifier,expected):
    deadline=time.monotonic()+30
    while time.monotonic()<deadline:
        result=checked(client.get("/api/v1/jobs/"+identifier,headers=auth))
        if result["status"]==expected:return result
        if result["status"] in ("succeeded","failed"):raise AssertionError("unexpected_terminal_job_status")
        time.sleep(.2)
    raise AssertionError("job_deadline_exceeded")


if __name__=="__main__":run()
