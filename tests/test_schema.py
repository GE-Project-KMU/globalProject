"""Migrations are idempotent and existing evidence survives schema upgrades."""
from sqlalchemy import create_engine,text
from common.migrations import migrate
from common.runtime import Base

def test_sqlite_migrations_preserve_existing_rows():
    engine=create_engine("sqlite:///:memory:")
    migrate(engine,"inspection",Base.metadata)
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO aggregates(id,org,kind,payload,version,created) VALUES ('test','tenant','case','{}',1,0)"))
    migrate(engine,"inspection",Base.metadata)
    with engine.connect() as conn:
        assert list(conn.execute(text("SELECT version FROM schema_migrations ORDER BY version")).scalars())==[1,2]
        assert conn.execute(text("SELECT COUNT(*) FROM aggregates WHERE id='test'")).scalar_one()==1
        assert "ix_aggregates_org_kind_created" in [r[1] for r in conn.execute(text("PRAGMA index_list('aggregates')"))]

def test_legacy_registry_quarantine_preserves_payload_and_audit(monkeypatch):
    from pathlib import Path
    from uuid import uuid4
    from common import runtime
    from services.analysis.registry import quarantine_legacy,counts,assert_usable
    from fastapi import HTTPException
    import pytest
    folder=runtime.ROOT/".cache"/"registry-tests"/uuid4().hex;folder.mkdir(parents=True)
    monkeypatch.setattr(runtime,"DATA",folder)
    store=runtime.Store("registry-test");org=str(uuid4());identifier=str(uuid4());good_id=str(uuid4())
    calibration={"image_size":[16,16],"height_m":1.,"pitch_deg":0.,"camera_matrix":[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],"distortion":[0.,0.,0.,0.],"homography":[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],"roi_polygon_px":[[0.,0.],[15.,0.],[15.,15.]],"visible_distance_m":[1.,2.],"calibrated_at":runtime.iso()}
    payload={"name":42,"version":1,"camera_serial":"SYNTHETIC","lens_label":"synthetic","calibration":calibration,"config_hash":"a"*64}
    store.create(identifier,org,"profile",payload);store.create(good_id,org,"profile",{**payload,"name":"Valid synthetic"})
    assert quarantine_legacy(store)==1
    assert quarantine_legacy(store)==0
    assert store.get(identifier,org,"quarantined_profile",expired=True)["name"]==42
    assert store.get(good_id,org,"profile")["name"]=="Valid synthetic"
    assert counts(store,org,"profile")==1
    assert [x["action"] for x in store.list(org,"audit")]==["profile.quarantined"]
    with pytest.raises(HTTPException) as caught:assert_usable(store,identifier,org,"profile")
    assert caught.value.status_code==410
    with pytest.raises(HTTPException) as caught:store.get(identifier,"other-org","quarantined_profile")
    assert caught.value.status_code==404
    store.engine.dispose()

def test_response_validation_logs_only_fields_and_common_errors_have_envelopes(monkeypatch,caplog):
    from common import runtime
    from services.response_contracts import Envelope,User
    import asyncio,httpx
    monkeypatch.setenv("ROAD_INTERNAL_SECRET","synthetic-unit-secret-at-least-24-characters")
    app=runtime.app_for("schema-probe")
    marker="PRIVATE-SYNTHETIC-MARKER-MUST-NOT-BE-LOGGED"
    @app.get("/api/v1/broken",response_model=Envelope[User])
    def broken():return runtime.ok({"id":"synthetic","login":marker,"role":[]})
    headers={"X-Road-Principal":runtime.sign("schema-probe",{"sub":"synthetic","org_id":"synthetic","role":"admin"})}
    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://synthetic") as client:
            response=await client.get("/api/v1/broken",headers=headers)
            assert response.status_code==500 and response.json()["error"]["code"]=="INTERNAL_ERROR"
            assert response.json()["meta"]["request_id"]==response.headers["X-Request-ID"]
            assert marker not in caplog.text and "response schema failure" in caplog.text
            for path,method,status in (("/missing","GET",404),("/api/v1/broken","DELETE",405)):
                response=await client.request(method,path,headers=headers)
                assert response.status_code==status and "error" in response.json() and "meta" in response.json()
    asyncio.run(exercise())
