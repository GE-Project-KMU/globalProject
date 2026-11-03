"""Real HTTPS transport and configuration-only recovery checks."""
from pathlib import Path
import json
import os
import ssl
import subprocess
import sys
import uuid
import zipfile
import httpx
import pytest
from scripts import config_backup
from scripts import tls

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def https_gateway():
    state=tls.manage.read_state()
    already=state.get(tls.NAME) and tls.manage.owned(state[tls.NAME])
    if not already:tls.start(8443)
    try:yield "https://127.0.0.1:8443"
    finally:
        if not already:tls.stop()


def test_real_https_checks_certificate_and_authentication(https_gateway):
    context=ssl.create_default_context(cafile=str(tls.CERT))
    with httpx.Client(verify=context,trust_env=False,timeout=10) as client:
        health=client.get(https_gateway+"/health")
        assert health.status_code==200 and health.json()["service"]=="gateway"
        assert client.get(https_gateway+"/api/v1/users").status_code==401
        from test_http import env_values
        login=client.post(https_gateway+"/api/v1/auth/login",json={"org_code":"demo","login":"admin@road.local","password":env_values()["ROAD_BOOTSTRAP_PASSWORD"]})
        assert login.status_code==200
        token=login.json()["data"]["access_token"]
        assert client.get(https_gateway+"/api/v1/me",headers={"Authorization":"Bearer "+token}).status_code==200
        assert client.post(https_gateway+"/api/v1/auth/logout",headers={"Authorization":"Bearer "+token}).status_code==204
    with httpx.Client(verify=True,trust_env=False,timeout=10) as client:
        with pytest.raises(httpx.ConnectError):client.get(https_gateway+"/health")


def test_certificate_has_localhost_san_and_short_expiration(https_gateway):
    from cryptography import x509
    from cryptography.x509.oid import ExtensionOID
    cert=x509.load_pem_x509_certificate(tls.CERT.read_bytes())
    names=cert.extensions.get_extension_for_oid(ExtensionOID.SUBJECT_ALTERNATIVE_NAME).value
    assert "localhost" in names.get_values_for_type(x509.DNSName)
    assert (cert.not_valid_after_utc-cert.not_valid_before_utc).days<=31


def test_backup_excludes_secrets_and_data_and_restore_does_not_resurrect_records():
    path=ROOT/".cache/backup-tests"/(uuid.uuid4().hex+".zip")
    manifest=config_backup.create(path)
    assert manifest["detail_data_included"] is False
    assert all(not name.startswith("data/") and not name.endswith((".db",".env",".pem")) for name in manifest["files"])
    verified=config_backup.restore(path,False)
    assert verified["verified"] and verified["detail_records_restored"]==0
    # Exact file payloads round-trip without touching any detail-data directory.
    _,entries=config_backup.inspect(path)
    assert entries["compose.yaml"]==(ROOT/"compose.yaml").read_bytes()


def test_backup_rejects_unlisted_archive_files_and_path_escape():
    folder=ROOT/".cache/backup-tests";folder.mkdir(parents=True,exist_ok=True)
    path=folder/(uuid.uuid4().hex+".zip")
    with zipfile.ZipFile(path,"w") as archive:
        archive.writestr("manifest.json",json.dumps({"schema_version":1,"policy":"configuration_only","detail_data_included":False,"files":{}}))
        archive.writestr("../../unexpected.txt","forbidden")
    with pytest.raises(ValueError,match="Forbidden"):config_backup.inspect(path)
    with pytest.raises(ValueError,match="outside"):config_backup.scoped(ROOT/".."/"outside.zip")


def test_inspection_retention_physically_removes_expired_payloads_and_adds_tombstones(monkeypatch):
    from common import runtime
    from sqlalchemy import select
    from services.inspection.privacy import purge
    folder=ROOT/".cache/retention-tests"/uuid.uuid4().hex;folder.mkdir(parents=True)
    monkeypatch.setattr(runtime,"DATA",folder)
    store=runtime.Store("retention-test")
    org="synthetic-retention";oid=str(uuid.uuid4());rid=str(uuid.uuid4());cid=uuid.uuid4().hex
    store.create(oid,org,"observation",{"payload":{"private_test_marker":"expired"}},runtime.now()+86400)
    store.create(rid,org,"run",{"path":[{"lat":37.5,"lon":127.0}]},runtime.now()+86400)
    store.create(cid,org,"correction",{"observation_id":oid,"value":"expired"},runtime.now()+86400)
    with store.tx() as session:
        for identifier in (oid,rid,cid):session.get(runtime.Aggregate,identifier).expires=runtime.now()-1
    keep=store.create(uuid.uuid4().hex,"other-org","observation",{"payload":{}},runtime.now()+86400)
    result=purge(store,org)
    assert result["deleted"]==3
    with store.tx() as session:
        assert session.get(runtime.Aggregate,oid) is None
        assert session.get(runtime.Aggregate,rid) is None
        assert session.get(runtime.Aggregate,cid) is None
    assert store.get(runtime.digest("observation:"+oid),org,"tombstone")["reason"]=="ttl_expiry"
    assert store.get(keep["id"],"other-org")["id"]==keep["id"]
    store.engine.dispose()
