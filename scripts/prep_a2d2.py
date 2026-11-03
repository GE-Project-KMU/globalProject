"""Predeclared recording-heldout A2D2 solid/dashed pixel research subset.

The official per-file AWS mirror avoids downloading monolithic archives. Source
and derived masks stay private (CC BY-ND), with no deployment/privacy approval.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timedelta,timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import urllib.parse
import urllib.request
import uuid
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.dataset import validate_research_release

BASE=ROOT/"data/public/a2d2"
HOST="https://audi-autonomous-driving-dataset.s3.eu-central-1.amazonaws.com/"
SCENES={"20180807_145028":"train","20180810_142822":"train","20180925_101535":"train","20181008_095521":"train","20181108_084007":"validation","20181204_135952":"test"}
NS={"s":"http://s3.amazonaws.com/doc/2006-03-01/"}

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def rel(path):return path.relative_to(ROOT).as_posix()
def write(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
def get(url):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url,timeout=90) as response:return response.read()
        except Exception:
            if attempt==2:raise
            time.sleep(attempt+1)

def listing(prefix):
    result=[];token=None
    while True:
        params={"list-type":"2","prefix":prefix,"max-keys":"1000"}
        if token:params["continuation-token"]=token
        root=ET.fromstring(get(HOST+"?"+urllib.parse.urlencode(params)))
        for row in root.findall("s:Contents",NS):
            result.append({"key":row.findtext("s:Key",namespaces=NS),"bytes":int(row.findtext("s:Size",namespaces=NS)),"etag":row.findtext("s:ETag",namespaces=NS).strip('"')})
        token=root.findtext("s:NextContinuationToken",namespaces=NS)
        if not token:break
    return result

def acquire(item):
    path=(BASE/"original"/item["key"]).resolve()
    if not path.is_relative_to(BASE.resolve()):raise ValueError("source_path_outside_dataset:"+str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    raw=path.read_bytes() if path.exists() else get(HOST+urllib.parse.quote(item["key"],safe="/"))
    if len(raw)!=item["bytes"] or hashlib.md5(raw).hexdigest()!=item["etag"]:raise ValueError("publisher_object_size_or_etag_md5_mismatch")
    if not path.exists():path.write_bytes(raw)
    return path

def phash(image):
    a=np.asarray(image.convert("L").resize((32,32),Image.Resampling.LANCZOS),dtype=np.float64)
    basis=np.cos(np.pi*np.arange(8)[:,None]*(2*np.arange(32)[None,:]+1)/64);low=(basis@a@basis.T).ravel()
    return sum(int(b)<<i for i,b in enumerate(low>np.median(low[1:])))

def main():
    BASE.mkdir(parents=True,exist_ok=True)
    for name,key in (("class_list.json","camera_lidar_semantic/class_list.json"),("README-SemSeg.txt","camera_lidar_semantic/README-SemSeg.txt"),("LICENSE.txt","LICENSE.txt")):
        path=BASE/name
        if not path.exists():path.write_bytes(get(HOST+key))
    classes=json.loads((BASE/"class_list.json").read_text())
    if classes.get("#ffc125")!="Solid line" or classes.get("#8000ff")!="Dashed line":raise ValueError("foreground_class_mapping_changed")
    if "CC BY-ND 4.0" not in (BASE/"LICENSE.txt").read_text():raise ValueError("source_license_changed")
    evidence=BASE/"license-evidence.json"
    write(evidence,{"research_allowed":True,"license_id":"CC-BY-ND-4.0","source_url":"https://registry.opendata.aws/aev-a2d2/","license_url":"https://creativecommons.org/licenses/by-nd/4.0/",
        "dataset_title":"A2D2: Audi Autonomous Driving Dataset","authors":"Jakob Geyer, Yohannes Kassahun, Mentar Mahmudi et al., Audi AG","citation":"Geyer et al. A2D2: Audi Autonomous Driving Dataset, arXiv:2004.06320 (2020)",
        "official_homepage":"https://a2d2-dataset.github.io/","mirror":HOST,"class_mapping_sha256":sha(BASE/"class_list.json"),"source_license_sha256":sha(BASE/"LICENSE.txt"),
        "scope":"Private offline research, including private derived binary masks. CC BY-ND permits commercial use subject to terms but modified-image/mask redistribution and learned-artifact rights are not cleared here. No human privacy or project acceptance approval.",
        "modifications":"Original PNG unchanged. Union Solid line RGB255,193,37 and Dashed line RGB128,0,255 ->foreground1; other known classes->0; unknown, Blurred area and Rain dirt->ignore255. No expected outline, wear ratio, or synthetic labels."})
    acquisition=BASE/"acquisition.json"
    if not acquisition.exists():
        now=datetime.now(timezone.utc);write(acquisition,{"acquired_at":now.isoformat(),"expires_at":(now+timedelta(days=7)).isoformat()})
    expiry=json.loads(acquisition.read_text())["expires_at"]
    if datetime.fromisoformat(expiry)<=datetime.now(timezone.utc):raise ValueError("research_source_expired_no_ttl_extension")
    plan_path=BASE/"frozen-plan.json"
    if plan_path.exists():plan=json.loads(plan_path.read_text())
    else:
        items=[]
        for scene,split in SCENES.items():
            labels=[x for x in listing(f"camera_lidar_semantic/{scene}/label/cam_front_center/") if x["key"].endswith(".png")]
            cameras={x["key"]:x for x in listing(f"camera_lidar_semantic/{scene}/camera/cam_front_center/") if x["key"].endswith(".png")}
            labels=sorted(labels,key=lambda x:x["key"])
            indices=np.linspace(0,len(labels)-1,min(100,len(labels)),dtype=int)
            for index in indices:
                label=labels[int(index)];camera_key=label["key"].replace("/label/","/camera/").replace("_label_","_camera_")
                if camera_key not in cameras:raise ValueError("source_camera_label_pair_missing")
                items.append({"recording":scene,"split":split,"image":cameras[camera_key],"label":label})
            print("planned",scene,split,len(indices),flush=True)
        size=sum(item[k]["bytes"] for item in items for k in ("image","label"))
        if size>3_000_000_000:raise ValueError("predeclared_subset_exceeds_3GB")
        plan={"created_before_training":True,"selection":"100 uniformly spaced sorted front-center frames per predeclared recording, no labels or model predictions used for frame selection","recording_splits":SCENES,"pairs":items,"total_source_bytes":size,"limit_source_bytes":3_000_000_000,"source_codes":{"#ffc125":"Solid line","#8000ff":"Dashed line"},"independent_route_verified":False,"recording_heldout_verified":True,"project_acceptance_eligible":False}
        write(plan_path,plan)
    masks=BASE/"paint-masks";masks.mkdir(exist_ok=True)
    valid_colors=np.array([int(color[1:],16) for color in classes],dtype=np.uint32)
    ignore_colors=[int("60458f",16),int("352e52",16)]
    def prepare(item):
        path=acquire(item["image"]);label_path=acquire(item["label"])
        with Image.open(path) as image:
            image.load();size=image.size;visual=phash(image)
        with Image.open(label_path) as label:rgb=np.asarray(label.convert("RGB"),dtype=np.uint32)
        if rgb.shape[:2]!=(size[1],size[0]):raise ValueError("source_mask_image_size_mismatch")
        packed=rgb[:,:,0]*65536+rgb[:,:,1]*256+rgb[:,:,2]
        mask=np.full(packed.shape,255,dtype=np.uint8)
        mask[np.isin(packed,valid_colors)]=0
        mask[(packed==0xffc125)|(packed==0x8000ff)]=1
        mask[np.isin(packed,ignore_colors)]=255
        sid=str(uuid.uuid5(uuid.NAMESPACE_URL,HOST+item["image"]["key"]))
        mask_path=masks/(sid+".png");Image.fromarray(mask).save(mask_path)
        image_hash=sha(path)
        return {"sample_id":sid,"source_id":sid,"source_hash":image_hash,"leakage_group_id":"A2D2-recording:"+item["recording"],"annotation_version":"A2D2-publisher-semseg-solid-dashed-v1","path":rel(path),"sha256":image_hash,"mask_path":rel(mask_path),"mask_sha256":sha(mask_path),"publisher_mask_path":rel(label_path),"publisher_mask_sha256":sha(label_path),"split":item["split"],"recording":item["recording"],"privacy_status":"restricted_research","qc_status":"publisher_annotations_machine_checked","release_scope":"offline_research","human_privacy_reviewed":False,"not_for_deployment":True,"project_acceptance_eligible":False,"recording_heldout_verified":True,"independent_route_verified":False,"split_independence":"whole_recording_and_date_disjoint_route_unknown","allowed_use":"CC BY-ND 4.0 private offline model research; no derived image/mask or learned artifact redistribution approval; no human privacy acceptance","expires_at":expiry,"license_evidence_path":rel(evidence),"license_evidence_sha256":sha(evidence),"image":{"width":size[0],"height":size[1]},"positive_pixels":int(np.sum(mask==1)),"valid_pixels":int(np.sum(mask!=255)),"perceptual_phash":f"{visual:016x}","source_url":HOST+item["image"]["key"],"foreground_definition":"Publisher Solid line/Dashed line semantic pixels only. Fine wear holes, expected paint outline and retention ratio not separately annotated."}
    rows=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for index,row in enumerate(pool.map(prepare,plan["pairs"])):
            rows.append(row)
            if index%25==0:print("prepared",index+1,"of",len(plan["pairs"]),flush=True)
    parent=list(range(len(rows)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,row in enumerate(rows):
        for j in range(i):
            other=rows[j]
            if row["source_hash"]==other["source_hash"] or (int(row["perceptual_phash"],16)^int(other["perceptual_phash"],16)).bit_count()<=6:parent[find(i)]=find(j)
    cluster_splits=defaultdict(set)
    for i,row in enumerate(rows):cluster_splits[find(i)].add(row["split"])
    removed=[r["sample_id"] for i,r in enumerate(rows) if len(cluster_splits[find(i)])>1]
    rows=[r for i,r in enumerate(rows) if len(cluster_splits[find(i)])==1]
    state=validate_research_release(rows)
    if not state["valid"]:raise ValueError(json.dumps(state["errors"][:10]))
    manifest=BASE/"manifest.json";write(manifest,rows)
    report={"source_url":"https://registry.opendata.aws/aev-a2d2/","source_license":json.loads(evidence.read_text()),"predeclared_plan_path":rel(plan_path),"predeclared_plan_sha256":sha(plan_path),"source_bytes":plan["total_source_bytes"],"samples":len(rows),"split_counts":dict(Counter(r["split"] for r in rows)),"recording_counts":dict(Counter(r["recording"] for r in rows)),"positive_images_by_split":dict(Counter(r["split"] for r in rows if r["positive_pixels"]>0)),"positive_pixels_by_split":{split:sum(r["positive_pixels"] for r in rows if r["split"]==split) for split in ("train","validation","test")},"cross_split_duplicate_cluster_members_removed":removed,"duplicate_policy":"Union exact image SHA256 and grayscale32x32 DCT64 pHash Hamming<=6; remove ALL members of connected clusters crossing partitions. Shared routes or physical road objects cannot be ruled out by this check.","recording_heldout_verified":True,"independent_route_verified":False,"project_acceptance_eligible":False,"human_privacy_reviewed":False,"machine_validation":state,"manifest_path":rel(manifest),"manifest_sha256":sha(manifest),"script_sha256":sha(Path(__file__)),"integrity":"Every selected original image/label byte count and MD5 matches the official extracted S3 object listing ETag. Original full-tar checksums are not claimed.","limitation":"Publisher semantic solid/dashed line classes support pixel segmentation research, not measured remaining-paint wear, expected outline, R-MAE, reflectivity or road safety acceptance. Dates/recordings differ; GPS-disjoint routes are not verified. No synthetic outline labels and no RTK split/holdout changes."}
    write(BASE/"source-qc.json",report);folder=ROOT/"docs/reports";folder.mkdir(exist_ok=True);write(folder/"A2D2-SOURCE-QC.json",report)
    print(json.dumps({"manifest":rel(manifest),"samples":len(rows),"counts":report["split_counts"],"valid":state["valid"]}),flush=True)

if __name__=="__main__":main()
