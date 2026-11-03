"""Prepare publisher-labeled face/plate crops for private research, never release approval.

Uses archives acquired by research_detection_data.py fetch-redaction. All paths
remain in the workspace, and source TTL is inherited without extending it.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import random
import sys
import uuid
import zipfile

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agent.dataset import validate_research_release

SOURCE=ROOT/"data/research/redaction"
OUTPUT=ROOT/"data/training/research-redaction"
NAMESPACE=uuid.UUID("084483f8-1355-4544-bd42-6eb183998bae")

def digest(data):return hashlib.sha256(data).hexdigest()
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):h.update(block)
    return h.hexdigest()
def rel(path):return path.relative_to(ROOT).as_posix()
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
def ident(value):return str(uuid.uuid5(NAMESPACE,value))
def dhash(image):
    a=np.asarray(image.convert("L").resize((9,8),Image.Resampling.LANCZOS))
    return int.from_bytes(np.packbits(a[:,1:]>a[:,:-1]).tobytes(),"big")

def verify_archive(name):
    path=SOURCE/name;record=json.loads(path.with_suffix(path.suffix+".source.json").read_text())
    if path.stat().st_size!=record["bytes"] or sha(path)!=record["sha256"]:raise ValueError("archive_source_hash_mismatch:"+name)
    if datetime.fromisoformat(record["expires_at"])<=datetime.now(timezone.utc):raise ValueError("source_expired:"+name)
    return record

def annotations(text):
    lines=iter(text.splitlines());result=[]
    while True:
        try:name=next(lines).strip()
        except StopIteration:break
        if not name:continue
        count=int(next(lines));boxes=[]
        # WIDER's zero-face convention still contains one zero row.
        for _ in range(max(1,count)):
            values=list(map(int,next(lines).split()))
            if count:boxes.append(values)
        result.append((name,boxes))
    return result

def round_robin(groups):
    groups=[deque(sorted(values,key=lambda x:digest(x[0].encode()))) for _,values in sorted(groups.items())]
    while groups:
        live=[]
        for values in groups:
            yield values.popleft()
            if values:live.append(values)
        groups=live

def image_crop(image,box,margin):
    x,y,w,h=box
    left=max(0,int(np.floor(x-w*margin)));top=max(0,int(np.floor(y-h*margin)))
    right=min(image.width,int(np.ceil(x+w+w*margin)));bottom=min(image.height,int(np.ceil(y+h+h*margin)))
    if right-left<24 or bottom-top<24:raise ValueError("crop_too_small")
    return (left,top,right,bottom)

def face_crop(image,boxes):
    valid=[b for b in boxes if len(b)>=8 and b[7]==0 and min(b[2:4])>=24 and b[2]*b[3]>0]
    if not valid:raise ValueError("no_valid_sufficient_face")
    target=max(valid,key=lambda b:b[2]*b[3])
    crop=image_crop(image,target[:4],0.18)
    left,top,right,bottom=crop;objects=[]
    for b in boxes:
        if len(b)<8:raise ValueError("incomplete_publisher_face_attributes")
        x,y,w,h=b[:4];x1=max(left,x);y1=max(top,y);x2=min(right,x+w);y2=min(bottom,y+h)
        if x2<=x1 or y2<=y1:continue
        if b[7] or min(w,h)<=0:raise ValueError("ignored_face_intersects_crop")
        # Include every visible intersecting face, including truncated neighbors.
        objects.append({"class":"face","bbox_xywh_px":[float(x1-left),float(y1-top),float(x2-x1),float(y2-y1)],"certainty":"confirmed","publisher_bbox_clipped_to_crop":x1!=x or y1!=y or x2!=x+w or y2!=y+h})
    if not objects or len(objects)>120:raise ValueError("unsupported_face_count")
    return image.crop(crop),objects,crop

def plate_crop(image,name):
    fields=Path(name).stem.split("-")
    points=fields[2].split("_");x1,y1=map(int,points[0].split("&"));x2,y2=map(int,points[1].split("&"))
    if min(x1,y1)<0 or x2>image.width or y2>image.height or x2<=x1 or y2<=y1:raise ValueError("invalid_publisher_plate_bbox")
    crop=image_crop(image,(x1,y1,x2-x1,y2-y1),0.10)
    left,top,_,_=crop
    objects=[{"class":"license_plate","bbox_xywh_px":[float(x1-left),float(y1-top),float(x2-x1),float(y2-y1)],"certainty":"confirmed"}]
    return image.crop(crop),objects,crop

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-per-class",type=int,default=800)
    parser.add_argument("--holdout-per-class",type=int,default=200)
    args=parser.parse_args()
    if min(args.train_per_class,args.holdout_per_class)<1:raise ValueError("positive_sample_limits_required")
    OUTPUT.mkdir(parents=True,exist_ok=True);images=OUTPUT/"crops";images.mkdir(exist_ok=True)
    records={n:verify_archive(n) for n in ("wider_face_split.zip","WIDER_train.zip","WIDER_val.zip","CCPD2020.zip")}
    evidence={"face":SOURCE/"wider-license.json","license_plate":SOURCE/"ccpd-license.json"}
    for path in evidence.values():
        policy=json.loads(path.read_text())
        if policy.get("research_allowed") is not True:raise ValueError("research_license_not_established")
    archives={n:zipfile.ZipFile(SOURCE/n) for n in records}
    targets={"train":args.train_per_class,"validation":args.holdout_per_class,"test":args.holdout_per_class}
    source_rows=[]
    split_archive=archives["wider_face_split.zip"]
    for original_split in ("train","val"):
        for name,boxes in annotations(split_archive.read(f"wider_face_split/wider_face_{original_split}_bbx_gt.txt").decode()):
            source_rows.append((name,boxes,original_split))
    events=sorted({name.split("/")[0] for name,_,_ in source_rows});random.Random(2026).shuffle(events)
    event_split={event:("validation" if i<9 else "test" if i<18 else "train") for i,event in enumerate(events)}
    face_groups={s:defaultdict(list) for s in targets}
    for name,boxes,original_split in source_rows:
        face_groups[event_split[name.split("/")[0]]][name.split("/")[0]].append((name,boxes,original_split))
    ccpd=archives["CCPD2020.zip"];plate_groups=defaultdict(set);plate_names=[]
    for name in ccpd.namelist():
        if not name.lower().endswith(".jpg"):continue
        fields=Path(name).stem.split("-")
        if len(fields)!=7:continue
        original_split=Path(name).parts[-2]
        if original_split not in ("train","val","test"):continue
        group=digest(("CCPD2020:"+fields[4]).encode())
        plate_groups[group].add(original_split);plate_names.append((name,group,original_split))
    crossing_plate_groups={group for group,splits in plate_groups.items() if len(splits)>1}
    plate_by_split={s:defaultdict(list) for s in targets}
    for name,group,original_split in plate_names:
        if group in crossing_plate_groups:continue
        split="validation" if original_split=="val" else original_split
        plate_by_split[split][group].append((name,group,original_split))
    rejected=Counter();rows=[]
    def prepare(kind,name,original_split,group,archive,member,boxes=None):
        raw=archive.read(member);source_hash=digest(raw)
        with Image.open(io.BytesIO(raw)) as original:
            original=original.convert("RGB");visual=dhash(original)
            crop,objects,rectangle=face_crop(original,boxes) if kind=="face" else plate_crop(original,name)
            sample_id=ident(kind+":"+name+":"+str(rectangle));path=images/(sample_id+".png");crop.save(path)
            width,height=crop.size
        policy=evidence[kind]
        source_record=records["WIDER_"+original_split+".zip"] if kind=="face" else records["CCPD2020.zip"]
        return {"sample_id":sample_id,"source_id":ident(kind+":"+name),"parent_sample_id":ident(kind+":"+name),"source_hash":source_hash,
            "leakage_group_id":group,"annotation_version":"WIDER-FACE-publisher-bbox-crop-v1" if kind=="face" else "CCPD2020-publisher-filename-bbox-crop-v1",
            "path":rel(path),"sha256":sha(path),"image":{"width":width,"height":height},"publisher_split":original_split,
            "privacy_status":"restricted_research","qc_status":"publisher_annotations_machine_checked","release_scope":"offline_research",
            "human_privacy_reviewed":False,"not_for_deployment":True,"project_acceptance_eligible":False,
            "annotation_review":"Publisher bounding boxes; machine decode, bounds, original-source identity and crop-coordinate checks only. No independent human review.",
            "allowed_use":"Private non-commercial offline research only; WIDER CC-BY-NC-ND4 and CCPD MIT attribution retained. No source crops or learned artifact redistribution/deployment rights claimed.",
            "expires_at":min(source_record["expires_at"],records["wider_face_split.zip"]["expires_at"]) if kind=="face" else source_record["expires_at"],
            "source_url":"https://huggingface.co/datasets/CUHK-CSE/wider_face" if kind=="face" else "https://github.com/detectRecog/CCPD",
            "license_evidence_path":rel(policy),"license_evidence_sha256":sha(policy),"detection_labels_verified":True,"objects":objects,
            "source_dhash":f"{visual:016x}","crop_transform":{"xyxy_original":list(rectangle),"original_image_sha256":source_hash},
            "annotation_coverage":"tight_face_crop_all_intersecting_publisher_faces" if kind=="face" else "tight_single_plate_crop",
            "other_class_absence_human_verified":False,"evaluation_scope":"publisher-derived object crops only, not full-scene privacy recall or deployment acceptance"}
    for split,limit in targets.items():
        for kind,groups in (("face",face_groups[split]),("license_plate",plate_by_split[split])):
            count=0
            for entry in round_robin(groups):
                if count>=limit:break
                try:
                    if kind=="face":
                        name,boxes,original_split=entry;archive=archives["WIDER_"+original_split+".zip"]
                        row=prepare(kind,name,original_split,"WIDER-event:"+name.split("/")[0],archive,f"WIDER_{original_split}/images/{name}",boxes)
                    else:
                        name,group,original_split=entry;row=prepare(kind,name,original_split,"CCPD-plate-sha256:"+group,ccpd,name)
                    row["split"]=split;rows.append(row);count+=1
                    if count%100==0:print(kind,split,count,flush=True)
                except (ValueError,OSError,KeyError) as exc:rejected[type(exc).__name__+":"+str(exc)[:80]]+=1
            print("prepared",kind,split,count,flush=True)
    # Connected exact source/crop and near-duplicate source-image clusters.
    parent=list(range(len(rows)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,row in enumerate(rows):
        for j in range(i):
            other=rows[j]
            if row["source_hash"]==other["source_hash"] or row["sha256"]==other["sha256"] or (int(row["source_dhash"],16)^int(other["source_dhash"],16)).bit_count()<=3:
                parent[find(i)]=find(j)
    cluster_splits=defaultdict(set)
    for i,row in enumerate(rows):cluster_splits[find(i)].add(row["split"])
    removed=[row["sample_id"] for i,row in enumerate(rows) if len(cluster_splits[find(i)])>1]
    rows=[row for i,row in enumerate(rows) if len(cluster_splits[find(i)])==1]
    state=validate_research_release(rows)
    if not state["valid"]:raise ValueError(json.dumps(state["errors"][:10]))
    release=OUTPUT/"release.json";write(release,rows)
    report={"source_licenses":[json.loads(path.read_text()) for path in evidence.values()],
        "source_attributions":[
            {"dataset":"WIDER FACE","authors":["Shuo Yang","Ping Luo","Chen Change Loy","Xiaoou Tang"],"citation":"WIDER FACE: A Face Detection Benchmark, CVPR 2016","source_url":"https://huggingface.co/datasets/CUHK-CSE/wider_face","license_url":"https://creativecommons.org/licenses/by-nc-nd/4.0/","changes":"Private face crops; bbox coordinates clipped/translated; original source identity retained. No adapted images shared."},
            {"dataset":"CCPD2020 (CCPD-Green)","authors":["Zhenbo Xu","Wei Yang","Ajin Meng","Nanxue Lu","Huan Huang"],"citation":"Towards End-to-End License Plate Detection and Recognition: A Large Dataset and Baseline, ECCV 2018","source_url":"https://github.com/detectRecog/CCPD","license_url":"https://github.com/detectRecog/CCPD/blob/master/LICENSE","copyright":"Copyright (c) 2017 CCPD","changes":"Private plate bbox crops; translated detection labels; encoded plate identities hashed for grouping, not recognized or exported."}],
        "integrity_limit":"Local acquisition SHA256 and byte counts verified against downloader records; selected ZIP member CRC verified by zipfile. No independent publisher-provided full-archive checksum is claimed.",
        "source_archive_records":records,"count":len(rows),"counts":dict(Counter(row["split"]+":"+row["objects"][0]["class"] for row in rows)),
        "object_counts":dict(Counter(obj["class"] for row in rows for obj in row["objects"])),
        "face_event_split":event_split,"face_event_groups":len(events),"plate_identity_cross_split_groups_excluded":len(crossing_plate_groups),
        "cross_split_duplicate_cluster_members_removed":len(removed),"removed_crop_sample_ids":removed,"rejected":dict(rejected),
        "release_sha256":sha(release),"release_validation":state,"human_privacy_reviewed":False,"project_acceptance_eligible":False,
        "redistribution_policy":"WIDER-derived crops and learned artifacts remain private non-commercial research. No adapted image redistribution, artifact license clearance, privacy consent, B2G commercial permission or performance acceptance is claimed. CCPD MIT notices retained; third-party personality/privacy rights unaffected.",
        "coverage_limitation":"Crops reduce obvious unlabeled other-class negatives but neither dataset exhaustively labels both face and plate classes. Cropped test is not a full-scene masking recall evaluation; source or identity independence beyond publisher events/hashed plate identity and duplicate filters is unverified.",
        "split_policy":"Seed2026 whole WIDER event-category groups; CCPD original split with every cross-split hashed-plate-identity removed before sampling. Original source IDs preserved. Connected exactSHA and 64bit source-dHash Hamming<=3 crossing all selected splits remove ALL cluster members. Choices are frozen before training, no model outputs used."}
    write(OUTPUT/"source-qc.json",report)
    doc=ROOT/"docs/reports/REDACTION-SOURCE-QC.json";doc.parent.mkdir(exist_ok=True)
    write(doc,{**report,"release_path":rel(release),"script_sha256":sha(Path(__file__)),"source_files_not_committed":True})
    for archive in archives.values():archive.close()
    print(json.dumps({"release":rel(release),"counts":report["counts"],"removed":len(removed),"valid":state["valid"]}),flush=True)

if __name__=="__main__":main()
