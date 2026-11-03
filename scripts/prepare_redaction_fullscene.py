"""Frozen held-out original images for class-conditional research evaluation.

Does not alter crop training data, its split, or its manifest. Unknown other
classes must not count as negatives or false positives in this challenge.
"""
from __future__ import annotations
from collections import Counter
import io
import json
from pathlib import Path
import sys
import zipfile

from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agent.dataset import validate_research_release
from scripts.prepare_redaction_research import SOURCE, OUTPUT, annotations, digest, ident, rel, sha, write


def clipped_box(box,width,height):
    x,y,w,h=map(float,box)
    if min(w,h)<=0:return None
    left=max(0.,x);top=max(0.,y);right=min(float(width),x+w);bottom=min(float(height),y+h)
    if right<=left or bottom<=top:return None
    return [left,top,right-left,bottom-top]


def main():
    release=OUTPUT/"release.json";frozen_sha=sha(release)
    test=[r for r in json.loads(release.read_text()) if r["split"]=="test"]
    originals=OUTPUT/"fullscene-test-originals";originals.mkdir(exist_ok=True)
    split_archive=zipfile.ZipFile(SOURCE/"wider_face_split.zip")
    lookup={}
    for source_split in ("train","val"):
        for name,boxes in annotations(split_archive.read(f"wider_face_split/wider_face_{source_split}_bbx_gt.txt").decode()):
            lookup[ident("face:"+name)]=(name,boxes,source_split)
    ccpd=zipfile.ZipFile(SOURCE/"CCPD2020.zip")
    for name in ccpd.namelist():
        if name.lower().endswith(".jpg"):lookup[ident("license_plate:"+name)]=(name,None,Path(name).parts[-2])
    archives={name:zipfile.ZipFile(SOURCE/name) for name in ("WIDER_train.zip","WIDER_val.zip")}
    rows=[]
    for crop in test:
        kind=crop["objects"][0]["class"]
        name,boxes,source_split=lookup[crop["source_id"]]
        raw=archives["WIDER_"+source_split+".zip"].read(f"WIDER_{source_split}/images/{name}") if kind=="face" else ccpd.read(name)
        if digest(raw)!=crop["source_hash"]:raise ValueError("original_does_not_match_frozen_crop_source")
        with Image.open(io.BytesIO(raw)) as image:width,height=image.size;image.load()
        objects=[];ignored=[]
        if kind=="face":
            for box in boxes:
                valid_box=clipped_box(box[:4],width,height)
                if not valid_box:continue
                if box[7]:ignored.append(valid_box)
                else:objects.append({"class":"face","bbox_xywh_px":valid_box,"certainty":"confirmed","publisher_bbox_clipped_to_image":valid_box!=list(map(float,box[:4]))})
        else:
            points=Path(name).stem.split("-")[2].split("_")
            left,top=map(float,points[0].split("&"));right,bottom=map(float,points[1].split("&"))
            valid_box=clipped_box([left,top,right-left,bottom-top],width,height)
            if not valid_box:raise ValueError("invalid_plate_box")
            objects=[{"class":"license_plate","bbox_xywh_px":valid_box,"certainty":"confirmed"}]
        if not objects:raise ValueError("heldout_original_has_no_known_class_gt")
        sample=ident("fullscene:"+crop["source_id"]);path=originals/(sample+".jpg")
        if not path.exists():path.write_bytes(raw)
        if sha(path)!=crop["source_hash"]:raise ValueError("saved_original_integrity_mismatch")
        rows.append({**{k:crop[k] for k in ("source_id","parent_sample_id","source_hash","leakage_group_id","publisher_split","privacy_status","qc_status","release_scope","human_privacy_reviewed","not_for_deployment","project_acceptance_eligible","allowed_use","expires_at","source_url","license_evidence_path","license_evidence_sha256","source_dhash")},
            "sample_id":sample,"path":rel(path),"sha256":sha(path),"split":"test",
            "annotation_version":"publisher-original-fullscene-known-class-v1",
            "image":{"width":width,"height":height},"objects":objects,"ignore_bboxes_xywh_px":ignored,
            "evaluation_class":kind,"evaluation_only":True,"training_eligible":False,
            "known_label_coverage":[kind],"unknown_other_classes_must_not_count_as_negative":True,
            "detection_labels_verified":True,
            "annotation_review":"All valid publisher known-class boxes retained; machine original-image hash/decode/box-bounds checks; ignored faces explicitly retained as ignore regions; no human privacy review.",
            "evaluation_scope":"Full original heldout image; known-class conditional detection only. Other class predictions are excluded from metrics, never false positives or negatives. No aggregate privacy recall/FPR or deployment claim.",
            "frozen_crop_source_sample_id":crop["sample_id"]})
    if sha(release)!=frozen_sha:raise ValueError("crop_release_changed_during_fullscene_preparation")
    result=validate_research_release(rows)
    if not result["valid"]:raise ValueError(json.dumps(result["errors"][:10]))
    output=OUTPUT/"fullscene-test.json";write(output,rows)
    report={"manifest_path":rel(output),"manifest_sha256":sha(output),"frozen_crop_release_sha256":frozen_sha,
        "count":len(rows),"images_by_known_class":dict(Counter(r["evaluation_class"] for r in rows)),
        "all_valid_publisher_objects_by_class":dict(Counter(o["class"] for r in rows for o in r["objects"])),
        "ignored_face_boxes":sum(len(r["ignore_bboxes_xywh_px"]) for r in rows),"validation":result,
        "selection":"Exactly the original source images underlying the already frozen crop test rows, without extra selection or split changes. No model outputs examined.",
        "evaluation_protocol":"Evaluate only evaluation_class per image; retain ALL valid publisher boxes including small and partly visible faces. Ignore publisher-invalid face regions; unknown other class never counts as negative or FP. Fullscene test once after crop-model selection; no threshold/model tuning on it.",
        "project_acceptance_eligible":False,"global_privacy_metrics_eligible":False,"human_privacy_reviewed":False,
        "redistribution":"Original images and plate-identity-bearing filenames stay in restricted local workspace only. No photos or original filenames in public report. Existing source-specific license and seven-day TTL apply.",
        "source_licenses":json.loads((OUTPUT/"source-qc.json").read_text())["source_licenses"]}
    write(OUTPUT/"fullscene-qc.json",report)
    write(ROOT/"docs/reports/REDACTION-FULLSCENE-QC.json",report)
    for archive in [*archives.values(),ccpd,split_archive]:archive.close()
    print(json.dumps({"manifest":rel(output),"images":report["images_by_known_class"],"objects":report["all_valid_publisher_objects_by_class"],"valid":result["valid"]}))


if __name__=="__main__":main()
