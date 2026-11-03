"""Workspace-only development HTTPS gateway; never modifies the OS trust store."""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta, timezone
import ipaddress
import json
import os
from pathlib import Path
import ssl
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import manage

CERT=ROOT/"data/tls/localhost.pem"
KEY=ROOT/"data/tls/localhost-key.pem"
NAME="gateway_tls"


def initialize():
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    if CERT.exists() and KEY.exists():
        cert=x509.load_pem_x509_certificate(CERT.read_bytes())
        if cert.not_valid_after_utc<=datetime.now(timezone.utc):
            raise RuntimeError("Development TLS certificate expired; explicitly replace data/tls certificate files.")
        return
    CERT.parent.mkdir(parents=True,exist_ok=True)
    key=rsa.generate_private_key(public_exponent=65537,key_size=3072)
    subject=x509.Name([x509.NameAttribute(x509.NameOID.COMMON_NAME,"Road Insight localhost development")])
    moment=datetime.now(timezone.utc)
    cert=(x509.CertificateBuilder().subject_name(subject).issuer_name(subject).public_key(key.public_key())
          .serial_number(x509.random_serial_number()).not_valid_before(moment-timedelta(minutes=1))
          .not_valid_after(moment+timedelta(days=30))
          .add_extension(x509.SubjectAlternativeName([x509.DNSName("localhost"),x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]),critical=False)
          .add_extension(x509.BasicConstraints(ca=True,path_length=0),critical=True)
          .sign(key,hashes.SHA256()))
    KEY.write_bytes(key.private_bytes(serialization.Encoding.PEM,serialization.PrivateFormat.PKCS8,serialization.NoEncryption()))
    KEY.chmod(0o600)
    CERT.write_bytes(cert.public_bytes(serialization.Encoding.PEM))


def healthy(port):
    import httpx
    try:
        with httpx.Client(verify=ssl.create_default_context(cafile=str(CERT)),trust_env=False,timeout=2) as c:
            response=c.get(f"https://127.0.0.1:{port}/health")
            return response.json() if response.status_code==200 else None
    except (OSError,httpx.HTTPError,ValueError):return None


def stop():
    import psutil
    state=manage.read_state();record=state.get(NAME)
    if not record:return
    if manage.owned(record):
        process=psutil.Process(record["pid"])
        command=process.cmdline()
        if str(KEY) not in command or "uvicorn" not in command:
            raise RuntimeError("Refusing to stop a process with an unexpected command")
        processes=process.children(recursive=True)+[process]
        for p in processes:
            try:p.terminate()
            except psutil.NoSuchProcess:pass
        _,alive=psutil.wait_procs(processes,timeout=8)
        for p in alive:
            try:p.kill()
            except psutil.NoSuchProcess:pass
        psutil.wait_procs(alive,timeout=5)
    state.pop(NAME,None);manage.STATE.write_text(json.dumps(state,indent=2),encoding="utf-8")


def start(port=8443):
    import psutil
    initialize();state=manage.read_state()
    if state.get(NAME) and manage.owned(state[NAME]):raise RuntimeError("Owned HTTPS gateway already running")
    # Binding is checked by uvicorn; never take over a process on an occupied port.
    python=ROOT/(".venv/Scripts/python.exe" if os.name=="nt" else ".venv/bin/python")
    manage.spawn(NAME,[str(python),"-m","uvicorn","services.gateway.app:app","--host","127.0.0.1","--port",str(port),"--ssl-certfile",str(CERT),"--ssl-keyfile",str(KEY)],manage.load_env(),state)
    deadline=time.monotonic()+30
    while time.monotonic()<deadline:
        if not manage.owned(state[NAME]):raise RuntimeError("HTTPS gateway exited; inspect its scoped log")
        response=healthy(port)
        parent=psutil.Process(state[NAME]["pid"])
        owned_ids={parent.pid,*[p.pid for p in parent.children(recursive=True)]}
        if response and response.get("pid") in owned_ids:
            print(f"HTTPS gateway ready: https://127.0.0.1:{port}");return
        time.sleep(.2)
    stop();raise RuntimeError("HTTPS gateway readiness timeout")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["init","start","stop","status"])
    parser.add_argument("--port",type=int,default=8443)
    args=parser.parse_args()
    if not 1024<=args.port<=65535:parser.error("port must be in 1024..65535")
    if args.action=="init":initialize();print("Local certificate prepared; no system trust changes")
    elif args.action=="start":start(args.port)
    elif args.action=="stop":stop()
    else:print("healthy" if healthy(args.port) else "not running")
