"""Freeze an expanded original-D40 experiment without changing prior releases.

Filename blocks are conservative collection proxies, NOT verified physical routes.
The old Czech test is excluded from every new split and remains regression-only.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import tarfile
import uuid
import xml.etree.ElementTree as ET
from urllib.parse import urlsplit

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from agent.paths import workspace_path,atomic_json,sha256
from agent.dataset import validate_research_release

ARCHIVE_SHA="e06f74463b2d7410fa7e02f478818025be9b051087f80e0c51631abb32c0af55"
NAMESPACE=uuid.UUID("3c444c1d-42d3-4353-a708-71793ac7edd0")
HISTORY_SHA="3fc9525e28585b194574103f7b3d6d57b18ad42df17bdee3e5fbec0962348b3d"

def dhash(image):
    small=cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),(9,8))
    result=0
    for bit in (small[:,1:]>small[:,:-1]).flat:result=(result<<1)|int(bit)
    return result

def group(name):
    country=name.split("/")[1]
    number=int(re.search(r"(\d+)$",Path(name).stem).group(1))
    return f"RDD2020:{country}:filename-block20:{number//20}"

def historical_exclusions():
    prior_paths=[ROOT/"data/training/research-pothole-rdd2020/release.json",ROOT/"data/training/research-pothole/release.json",
                 ROOT/"data/training/research-pothole-rdd2020-v2/release.json"]
    previous=[r for path in prior_paths for r in json.loads(path.read_text())
              if path!=prior_paths[-1] or r["split"] in ("train","validation")]
    # Quarantined labels are NEVER read as annotations. Historical image
    # identity is still excluded, including that abandoned experiment's TRAIN.
    oldhash={r["sha256"] for r in previous};oldvisual=[int(r["perceptual_dhash"],16) for r in previous]
    oldgroups=set()
    for row in previous:
        name=row.get("original_member")
        if not name and row.get("country") in ("Japan","India","Czech"):
            name="train/"+row["country"]+"/images/"+Path(urlsplit(row["source_url"]).path).name
        if name:oldgroups.add(group(name))
    return {"purpose":"minimal historical image identities only; no labels/images/person UUID/source image paths",
        "historical_release_sha256":{p.relative_to(ROOT).as_posix():sha256(p) for p in prior_paths},
        "aborted_v2_filter":"TRAIN/validation only; test never evaluated",
        "sha256":sorted(oldhash),"dhash64":sorted({f"{v:016x}" for v in oldvisual}),"filename20_blocks":sorted(oldgroups)}

def freeze_history(destination):
    destination=workspace_path(destination);data=historical_exclusions()
    if destination.exists():
        if json.loads(destination.read_text())!=data:raise ValueError("existing_history_snapshot_differs")
        # Preserve the byte snapshot, including its line endings, on Linux too.
        print(sha256(destination),flush=True);return
    atomic_json(destination,data);print(sha256(destination),flush=True)

def prepare(output,history_snapshot):
    output=workspace_path(output)
    if output.exists():raise ValueError("new_output_directory_required_do_not_mutate_frozen_release")
    archive=ROOT/"data/research/rdd2020/train.tar.gz"
    if sha256(archive)!=ARCHIVE_SHA:raise ValueError("original_archive_hash_mismatch")
    snapshot=workspace_path(history_snapshot,exists=True)
    if HISTORY_SHA is None or sha256(snapshot)!=HISTORY_SHA:raise ValueError("historical_exclusion_snapshot_hash_mismatch")
    exclusions=json.loads(snapshot.read_text());oldhash=set(exclusions["sha256"])
    oldvisual=[int(v,16) for v in exclusions["dhash64"]];oldgroups=set(exclusions["filename20_blocks"])
    candidates={};counts=Counter()
    with tarfile.open(archive,"r|gz") as source:
        for member in source:
            if not member.isfile() or not member.name.endswith(".xml"):continue
            country=member.name.split("/")[1]
            if country not in ("India","Japan"):continue
            node=ET.fromstring(source.extractfile(member).read())
            filename=node.findtext("filename") or Path(member.name).with_suffix(".jpg").name
            if not filename.lower().endswith(".jpg"):filename+=".jpg"
            name=f"train/{country}/images/{filename}"
            width,height=int(node.findtext("size/width")),int(node.findtext("size/height"));objects=[]
            for obj in node.findall("object"):
                counts[country+":"+str(obj.findtext("name"))]+=1
                if obj.findtext("name")!="D40":continue
                box=obj.find("bndbox");a,b,c,d=[float(box.findtext(k)) for k in ("xmin","ymin","xmax","ymax")]
                x,y=max(0.,a-1),max(0.,b-1);right,bottom=min(float(width),c),min(float(height),d)
                if right<=x or bottom<=y:raise ValueError("invalid_original_D40")
                objects.append({"class":"pothole","bbox_xywh_px":[x,y,right-x,bottom-y],"certainty":"confirmed"})
            candidates[name]={"country":country,"objects":objects,"width":width,"height":height,"group":group(name),
                              "xml_sha256":hashlib.sha256(ET.tostring(node)).hexdigest()}
    # Select unused whole Japan filename blocks BEFORE looking at model outputs.
    groups={r["group"] for r in candidates.values() if r["country"]=="Japan"}
    fresh=sorted(groups-oldgroups);random.Random(20261104).shuffle(fresh)
    testgroups=set();positive_count=0
    for token in fresh:
        members=[r for r in candidates.values() if r["group"]==token]
        positives=sum(bool(r["objects"]) for r in members)
        if not positives:continue
        testgroups.add(token);positive_count+=positives
        if positive_count>=200:break
    if positive_count<30:raise ValueError("insufficient_never_used_grouped_positive_holdout")
    remaining=sorted(groups-testgroups);random.Random(20261105).shuffle(remaining)
    valgroups=set();valpositives=0
    for token in remaining:
        members=[r for r in candidates.values() if r["group"]==token]
        if not any(r["objects"] for r in members):continue
        valgroups.add(token);valpositives+=sum(bool(r["objects"]) for r in members)
        if valpositives>=200:break
    selected={};rng=random.Random(20261106)
    for split in ("train","validation","test"):
        eligible=[(n,r) for n,r in candidates.items() if
                  ("test" if r["group"] in testgroups else "validation" if r["group"] in valgroups else "train")==split]
        positives=[x for x in eligible if x[1]["objects"]];negatives=[x for x in eligible if not x[1]["objects"]]
        rng.shuffle(negatives)
        for name,info in positives+negatives[:len(positives)]:selected[name]=info|{"split":split}
    source_meta=json.loads((ROOT/"data/research/rdd2020/source.json").read_text());expiry=source_meta["expires_at"]
    evidence=ROOT/"data/research/rdd2020/license-evidence.json"
    images=output/"source";images.mkdir(parents=True)
    decoded={};removed=Counter()
    with tarfile.open(archive,"r|gz") as source:
        for member in source:
            if member.name not in selected:continue
            raw=source.extractfile(member).read();info=selected[member.name];digest=hashlib.sha256(raw).hexdigest()
            image=cv2.imdecode(np.frombuffer(raw,np.uint8),1)
            if image is None or image.shape[:2]!=(info["height"],info["width"]):raise ValueError("dimension_mismatch")
            visual=dhash(image)
            if info["split"]=="test" and (digest in oldhash or any((visual^v).bit_count()<=3 for v in oldvisual)):
                removed["new_test_matches_previous_used_source"]+=1;continue
            decoded[member.name]=(info,raw,digest,visual)
    # Protect the frozen test first, then validation. Remove cross-split near
    # duplicates rather than moving any heldout image into another partition.
    rows=[];seen=set();visuals=[]
    for split in ("test","validation","train"):
        for name,(info,raw,digest,visual) in sorted(decoded.items()):
            if info["split"]!=split:continue
            if digest in seen or any(s!=split and (visual^v).bit_count()<=3 for v,s in visuals):
                removed["cross_split_exact_or_dhash_le3"]+=1;continue
            seen.add(digest);visuals.append((visual,split))
            identifier=str(uuid.uuid5(NAMESPACE,"RDD2020:"+name));path=images/(identifier+".jpg");path.write_bytes(raw)
            rows.append({"sample_id":identifier,"source_id":str(uuid.uuid5(NAMESPACE,"source:"+digest)),"source_hash":digest,
                "leakage_group_id":info["group"],"country":info["country"],"split":split,"original_member":name,
                "annotation_version":"RDD2020-original-VOC-literal-D40-v2","xml_sha256":info["xml_sha256"],
                "path":path.relative_to(ROOT).as_posix(),"sha256":digest,"perceptual_dhash":f"{visual:016x}",
                "privacy_status":"restricted_research","qc_status":"publisher_annotations_machine_checked","release_scope":"offline_research",
                "human_privacy_reviewed":False,"not_for_deployment":True,"project_acceptance_eligible":False,"independent_route_verified":False,
                "annotation_review":"original publisher literal D40 XML; machine bbox/dimension checks; no human privacy approval",
                "allowed_use":"Private noncommercial research CC-BY-NC-3.0, attribution required; no redistribution/deployment approval",
                "expires_at":expiry,"source_expires_at":expiry,"source_url":"https://data.mendeley.com/datasets/5ty2wb6gvg/1",
                "license_evidence_path":evidence.relative_to(ROOT).as_posix(),"license_evidence_sha256":sha256(evidence),
                "detection_labels_verified":True,"objects":info["objects"]})
    result=validate_research_release(rows)
    if not result["valid"]:raise ValueError(str(result["errors"][:10]))
    atomic_json(output/"release.json",rows)
    report={"created_at":datetime.now(timezone.utc).isoformat(),"release_sha256":sha256(output/"release.json"),
        "archive_sha256":ARCHIVE_SHA,"source_expiry_inherited_unchanged":expiry,"counts":dict(Counter(r["split"] for r in rows)),
        "positive_counts":dict(Counter(r["split"] for r in rows if r["objects"])),"literal_XML_counts":dict(counts),
        "groups":{"test":sorted(testgroups),"validation":sorted(valgroups)},"removed":dict(removed),
        "old_release_sha256":exclusions["historical_release_sha256"]["data/training/research-pothole-rdd2020/release.json"],
        "historical_image_exclusion_releases":exclusions["historical_release_sha256"],
        "historical_exclusion_snapshot_sha256":HISTORY_SHA,
        "quarantined_annotations_never_used":True,
        "aborted_v2_history_filter":"TRAIN/validation blacklisted; unexamined v2 TEST not used by aborted model and eligible only if all other historical exclusions pass",
        "holdout_statistical_power":"small strict historical filename-block holdout; physical route independence unknown; report confidence interval",
        "old_Czech_test_excluded_all_new_splits":all(r["country"]!="Czech" for r in rows),
        "fresh_test_excludes_all_old_used_filename_blocks_SHA_dhash_le3":True,
        "independence":"disjoint filename20 blocks and cross-split exact/dHash<=3; same-country Japan holdout; actual route/recording independence unknown",
        "test_frozen_before_training":True,"research_only":True,"human_privacy_reviewed":False}
    atomic_json(output/"split-evidence.json",report)
    atomic_json(ROOT/"docs/reports/POTHOLE-IMPROVEMENT-SPLIT.json",report)
    print(json.dumps(report),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("output",nargs="?")
    parser.add_argument("--freeze-history");parser.add_argument("--history-snapshot",default="docs/reports/POTHOLE-HISTORY-EXCLUSIONS.json")
    args=parser.parse_args()
    if args.freeze_history:freeze_history(args.freeze_history)
    elif not args.output:parser.error("output required")
    else:prepare(args.output,args.history_snapshot)
