"""Probe isolated portable PostgreSQL/PostGIS without changing live SQLite services.

Requires PostgreSQL/PostGIS on loopback:55432 with a dedicated road_probe user.
On Windows the official runtime and cluster must be inside .cache/postgres-probe;
their PID/executable/data directory are verified before database writes. CI uses
an isolated PostGIS service container. Never installs a Windows service.
"""
from pathlib import Path
import json,os,subprocess,sys,time,uuid
import httpx,psutil

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def run():
    from scripts.manage import load_env
    os.environ.update(load_env())
    folder=ROOT/".cache"/"postgres-probe"
    if os.name=="nt":
        # A reused local port must never redirect this probe into another DB.
        pidfile=folder/"cluster"/"postmaster.pid"
        pid=int(pidfile.read_text(encoding="utf-8").splitlines()[0])
        if not Path(psutil.Process(pid).exe()).resolve().is_relative_to((folder/"runtime").resolve()):raise RuntimeError("Refusing a PostgreSQL process outside project probe runtime")
    data=folder/"service-data";data.mkdir(parents=True,exist_ok=True)
    os.environ["ROAD_DATA_DIR"]=str(data)
    os.environ["ROAD_POSTGIS_ENABLED"]="1"
    import psycopg
    from psycopg import sql
    from sqlalchemy import text
    from common.runtime import Store,uid,now,iso,sign
    suffix=uuid.uuid4().hex[:10];stores={};results={}
    with psycopg.connect("host=127.0.0.1 port=55432 user=road_probe dbname=postgres",autocommit=True) as conn:
        results["postgresql_version"]=conn.execute("SHOW server_version").fetchone()[0]
        if os.name=="nt":
            directory=Path(conn.execute("SHOW data_directory").fetchone()[0]).resolve()
            if directory!=(folder/"cluster").resolve():raise RuntimeError("Refusing a PostgreSQL data directory outside isolated project cluster")
        for name in ("identity","media","inspection","analysis","reporting"):
            database="road_probe_"+name+"_"+suffix
            conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
            url="postgresql+psycopg://road_probe@127.0.0.1:55432/"+database
            os.environ[name.upper()+"_DATABASE_URL"]=url
            store=Store(name);stores[name]=store
            with store.engine.connect() as c:
                versions=list(c.execute(text("SELECT version FROM schema_migrations ORDER BY version")).scalars())
                assert versions==([1,2,3,4] if name=="inspection" else [1,2])
                assert c.execute(text("SELECT data_type FROM information_schema.columns WHERE table_name='aggregates' AND column_name='payload'")).scalar_one()=="jsonb"
            results[name]={"database":database,"migrations":versions,"payload_type":"jsonb"}
    store=stores["inspection"];org=uid();case=uid();observation=uid()
    body={"schema_version":"0.4","observation_id":observation,"event_id":uid(),"run_id":uid(),"device_id":uid(),"type":"pothole","captured_at":iso(),"frame_id":1,
          "location":{"lat":37.5,"lon":127.0,"source":"vehicle_observation","quality":"valid","hdop":1.0,"age_ms":0,"accuracy_m":None},"profile_id":uid(),"model_version":"synthetic-pg-probe","config_hash":"a"*64,
          "pothole":{"bbox_xywh_px":[1.,1.,2.,2.],"confidence":.9,"track_observations":3,"candidate_state":"multi_frame","occluded":False},"lane":None,
          "quality":{"usable":True,"reasons":[]},"image":{"file_id":uid(),"sha256":"b"*64,"redacted":True,"reviewed_for_sharing":True},"review":{"status":"unreviewed","priority":"normal"},"demo":True}
    store.create(case,org,"case",{"type":"pothole","status":"unreviewed","priority":"normal","merged_into":None,"updated_at":iso()})
    store.create(observation,org,"observation",{"payload":body,"case_id":case,"request_hash":"c"*64},now()+3600)
    with store.engine.connect() as conn:
        results["postgis_version"]=conn.execute(text("SELECT PostGIS_Version()")).scalar_one()
        assert conn.execute(text("SELECT ST_AsText(observation_point::geometry) FROM aggregates WHERE id=:id"),{"id":observation}).scalar_one()=="POINT(127 37.5)"
        assert conn.execute(text("SELECT ST_Distance(observation_point,ST_SetSRID(ST_MakePoint(127,37.5),4326)::geography) FROM aggregates WHERE id=:id"),{"id":observation}).scalar_one()==0
        assert conn.execute(text("SELECT indexdef FROM pg_indexes WHERE indexname='ix_inspection_observation_point'")).scalar_one().find("gist")>=0
    process=None;logfile=(folder/"inspection-http.log").open("w",encoding="utf-8")
    try:
        process=subprocess.Popen([sys.executable,"-m","uvicorn","services.inspection.app:app","--host","127.0.0.1","--port","18103"],cwd=ROOT,env=os.environ.copy(),stdout=logfile,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        with httpx.Client(base_url="http://127.0.0.1:18103",timeout=5,trust_env=False) as client:
            deadline=time.monotonic()+30
            while True:
                try:
                    if client.get("/health").status_code==200:break
                except httpx.HTTPError:pass
                if process.poll() is not None or time.monotonic()>deadline:raise RuntimeError("PostgreSQL inspection service did not start")
                time.sleep(.2)
            headers={"X-Road-Principal":sign("inspection",{"sub":uid(),"org_id":org,"role":"admin"})}
            response=client.get("/api/v1/observations",params={"bbox":"126,37,128,38"},headers=headers)
            assert response.status_code==200,response.text[:500]
            assert [x["id"] for x in response.json()["data"]["items"]]==[observation]
            for bbox in ("126,37.5,128,38","-180,-90,180,90"):
                edge=client.get("/api/v1/observations",params={"bbox":bbox},headers=headers)
                assert edge.status_code==200 and [x["id"] for x in edge.json()["data"]["items"]]==[observation]
            assert client.get("/api/v1/observations",params={"bbox":"120,30,121,31"},headers=headers).json()["data"]["items"]==[]
            other={"X-Road-Principal":sign("inspection",{"sub":uid(),"org_id":uid(),"role":"admin"})}
            assert client.get("/api/v1/observations/"+observation,headers=other).status_code==404
        store.delete(observation,org)
        with store.engine.connect() as conn:assert conn.execute(text("SELECT count(*) FROM aggregates WHERE observation_point IS NOT NULL")).scalar_one()==0
        results["checks"]={"postgis_point":True,"distance_m":0,"gist_index":True,"http_bbox_inside":200,"http_bbox_outside_empty":True,"http_bbox_latitude_edge_and_world":True,"http_cross_org":404,"geometry_deleted_with_observation":True}
    finally:
        if process:
            try:
                parent=psutil.Process(process.pid);children=parent.children(recursive=True)
                for child in children:child.terminate()
                parent.terminate();_,alive=psutil.wait_procs(children+[parent],timeout=5)
                for child in alive:child.kill()
            except psutil.NoSuchProcess:pass
        logfile.close()
        for store in stores.values():store.engine.dispose()
    content=(folder/"inspection-http.log").read_text(encoding="utf-8",errors="replace")
    assert not any(marker in content for marker in ("Traceback","ERROR","INTERNAL_ERROR")),"Inspect project probe log"
    (ROOT/"data"/"postgres-probe-results.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
    print(json.dumps(results,indent=2))

if __name__=="__main__":run()
