
"""Create clearly marked synthetic records through real public API requests."""
import hashlib,io,json,sys
from pathlib import Path
from datetime import datetime,timedelta,timezone
from uuid import uuid4
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import httpx
from PIL import Image,ImageDraw
from scripts.manage import load_env
def seed():
    env=load_env();state=ROOT/"data"/"demo-seed.json"
    if state.exists():
        print("Synthetic seed already exists; retained existing records.");return
    def dt(seconds):return (datetime.now(timezone.utc)+timedelta(seconds=seconds)).isoformat().replace("+00:00","Z")
    with httpx.Client(base_url="http://127.0.0.1:8000",timeout=30) as client:
        def req(method,path,headers=None,**kwargs):
            r=client.request(method,"/api/v1"+path,headers=headers,**kwargs)
            if r.status_code>=400:raise RuntimeError(f"Seed failed: {method} {path} ({r.status_code}) {r.text[:250]}")
            return r.json()["data"]
        user=req("POST","/auth/login",json={"org_code":"demo","login":"admin@road.local","password":env["ROAD_BOOTSTRAP_PASSWORD"]})
        auth={"Authorization":"Bearer "+user["access_token"]}
        device=req("POST","/devices",auth,json={"name":"가상 시연 차량 · 실제 장비 아님"})
        dh={"Authorization":"Bearer "+device["device_token"]}
        profile=req("POST","/profiles",auth,json={"name":"SYNTHETIC-DEMO-"+uuid4().hex[:6],"version":1,"camera_serial":"SYNTHETIC-NOT-ZED","lens_label":"가상 데이터","calibration":{"image_size":[1280,720],"height_m":.7,"pitch_deg":20,"camera_matrix":[[800,0,640],[0,800,360],[0,0,1]],"distortion":[0,0,0,0,0],"homography":[[1,0,0],[0,1,0],[0,0,1]],"roi_polygon_px":[[0,0],[1280,0],[1280,720],[0,720]],"visible_distance_m":[1.5,6],"calibrated_at":dt(-3600)},"config_hash":"a"*64})
        version="synthetic-demo-"+uuid4().hex[:8]
        req("POST","/model-bundles",auth,json={"version":version,"weights_hashes":{"pothole":"b"*64,"lane":"c"*64,"redaction":"d"*64},"preprocess":{"pothole":{"input_size":[416,416]},"lane":{"long_edge":512},"redaction":{"input_size":[416,416]}},"thresholds":{"pothole_confidence":.35,"pothole_nms_iou":.5,"single_frame_confidence":.8,"lane_monitor":60,"lane_good":80,"min_valid_ratio":.7,"min_segment_support_ratio":.8},"config_hash":"e"*64,"code_commit":"SYNTHETIC-DEMO","environment_hash":"f"*64})
        run=str(uuid4())
        req("POST","/runs",dh,json={"run_id":run,"device_id":device["device_id"],"profile_id":profile["profile_id"],"started_at":dt(-3600),"manifest_hash":"1"*64,"precheck":{k:True for k in ("camera_preview_ok","mount_secure","lens_clear","profile_valid","gnss_device_connected","storage_ge_10gb","notice_present")}})
        created=[]
        for i in range(9):
            image=Image.new("RGB",(1280,720),(43,51,58));draw=ImageDraw.Draw(image)
            draw.rectangle((0,0,1279,90),fill=(24,34,42))
            draw.text((30,25),"SYNTHETIC DEMO - NOT MODEL PREDICTIONS / NOT REAL ROAD",fill=(255,194,80))
            draw.line((260,720,450,90),fill=(232,220,155),width=30);draw.line((1020,720,830,90),fill=(230,230,220),width=30)
            if i<4:draw.ellipse((480,380,750,530),fill=(20,24,27),outline=(115,105,86),width=12)
            else:
                for y in range(200,650,90):draw.rectangle((300,y,350,y+40),fill=(43,51,58))
            stream=io.BytesIO();image.save(stream,format="JPEG");blob=stream.getvalue();sha=hashlib.sha256(blob).hexdigest()
            fileid=str(uuid4());capt=dt(-300+i*5)
            approval=req("POST","/image-approvals",auth,json={"file_id":fileid,"device_id":device["device_id"],"sha256":sha,"size_bytes":len(blob),"mime":"image/jpeg","captured_at":capt,"reviewed_at":dt(-5),"redaction_model_version":"SYNTHETIC-NO-PEOPLE"})
            req("POST","/images",dh,data={"metadata":json.dumps({"file_id":fileid,"approval_id":approval["approval_id"],"sha256":sha,"captured_at":capt})},files={"file":("synthetic.jpg",blob,"image/jpeg")})
            lane=None
            if i>=4:
                r=[45,72,86,None,52][i-4]
                lane={"segment_id":str(uuid4()),"estimated_length_m":10 if r is not None else None,"retention_pct":r,"grade":"unassessable" if r is None else "worn_candidate" if r<60 else "monitor" if r<80 else "visible_good","color":"white" if r is not None else "unknown","line_type":"dashed" if r is not None else "unknown","valid_area_ratio":.9 if r is not None else None,"segment_support_ratio":.9 if r is not None else None,"outline_supported":r is not None,"patch_ids":[str(uuid4())] if r is not None else [],"unassessable_reason":"가상 판정 불가 예시" if r is None else None}
            obs={"schema_version":"0.4","observation_id":str(uuid4()),"event_id":str(uuid4()),"run_id":run,"device_id":device["device_id"],"type":"pothole" if i<4 else "lane_segment","captured_at":capt,"frame_id":i*100,"location":{"lat":37.501+i*.00015,"lon":127.033+i*.0002,"source":"vehicle_observation","quality":"valid","hdop":1.1,"age_ms":100,"accuracy_m":None},"profile_id":profile["profile_id"],"model_version":version,"config_hash":"e"*64,"pothole":{"bbox_xywh_px":[480,380,270,150],"confidence":.88+i*.01,"track_observations":5,"candidate_state":"multi_frame","occluded":False} if i<4 else None,"lane":lane,"quality":{"usable":True,"reasons":[]},"image":{"file_id":fileid,"sha256":sha,"redacted":True,"reviewed_for_sharing":True},"review":{"status":"unreviewed","priority":"normal"},"demo":True}
            result=req("POST","/observations",dh,json=obs);created.append(obs["observation_id"])
            if i in (1,2):
                req("PATCH",f"/observations/{obs['observation_id']}/review",auth,json={"expected_case_id":result["case_id"],"expected_version":1,"status":"confirmed","priority":"urgent" if i==1 else "normal","reason":"가상 시연 상태 예시"})
        state.write_text(json.dumps({"observation_ids":created,"demo":True}),encoding="utf-8")
        print(f"Created {len(created)} clearly marked synthetic observations through real HTTP APIs.")
if __name__=="__main__":seed()
