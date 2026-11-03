"""Original-frame research release with a NEW pre-training held-out source test.

Frozen crop train/validation source memberships are preserved, subject only to
publisher-invalid-label/max-label machine exclusions. Old tests remain diagnostic
and are never moved to training. Unknown other classes remain explicitly partial.
"""
from collections import Counter,defaultdict
import io
import json
from pathlib import Path
import sys
import zipfile
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.dataset import validate_research_release
from scripts.prepare_redaction_research import SOURCE,OUTPUT,annotations,digest,ident,rel,sha,write,dhash,round_robin
from scripts.prepare_redaction_fullscene import clipped_box

DEST=ROOT/"data/training/research-redaction-fullframe"

def main():
    DEST.mkdir(parents=True,exist_ok=True);originals=DEST/"originals";originals.mkdir(exist_ok=True)
    frozen=OUTPUT/"release.json";frozen_hash=sha(frozen);old=json.loads(frozen.read_text())
    old_qc=json.loads((OUTPUT/"source-qc.json").read_text());events=old_qc["face_event_split"]
    split_zip=zipfile.ZipFile(SOURCE/"wider_face_split.zip")
    archives={s:zipfile.ZipFile(SOURCE/("WIDER_"+s+".zip")) for s in ("train","val")}
    ccpd=zipfile.ZipFile(SOURCE/"CCPD2020.zip")
    face_lookup={};new_faces=defaultdict(list)
    for publisher_split in ("train","val"):
        for name,boxes in annotations(split_zip.read(f"wider_face_split/wider_face_{publisher_split}_bbx_gt.txt").decode()):
            sid=ident("face:"+name);face_lookup[sid]=(name,boxes,publisher_split)
            if events[name.split("/")[0]]=="test":new_faces[name.split("/")[0]].append((name,boxes,publisher_split))
    plate_lookup={};plate_partitions=defaultdict(set);all_plates=[]
    for name in ccpd.namelist():
        if not name.lower().endswith(".jpg"):continue
        fields=Path(name).stem.split("-")
        if len(fields)!=7:continue
        publisher_split=Path(name).parts[-2];group="CCPD-plate-sha256:"+digest(("CCPD2020:"+fields[4]).encode())
        plate_lookup[ident("license_plate:"+name)]=(name,group,publisher_split)
        plate_partitions[group].add(publisher_split);all_plates.append((name,group,publisher_split))
    old_ids={r["source_id"] for r in old};old_hashes={r["source_hash"] for r in old}
    old_plate_ids={r["leakage_group_id"] for r in old if r["objects"][0]["class"]=="license_plate"}
    old_visual=[int(r["source_dhash"],16) for r in old]
    old_by_id={r["source_id"]:r for r in old};rejected=Counter();rows=[]
    license_paths={"face":SOURCE/"wider-license.json","license_plate":SOURCE/"ccpd-license.json"}
    source_records={name:json.loads((SOURCE/name).with_suffix(".zip.source.json").read_text()) for name in ("WIDER_train.zip","WIDER_val.zip","CCPD2020.zip")}
    def prepare(kind,name,publisher_split,group,split,boxes=None,old_row=None):
        raw=archives[publisher_split].read(f"WIDER_{publisher_split}/images/{name}") if kind=="face" else ccpd.read(name)
        source_hash=digest(raw)
        if old_row and source_hash!=old_row["source_hash"]:raise ValueError("frozen_source_hash_mismatch")
        with Image.open(io.BytesIO(raw)) as im:im.load();width,height=im.size;visual=dhash(im)
        objects=[]
        if kind=="face":
            for b in boxes:
                bbox=clipped_box(b[:4],width,height)
                if not bbox:continue
                if b[7]:raise ValueError("publisher_invalid_face_image_omitted")
                objects.append({"class":"face","bbox_xywh_px":bbox,"certainty":"confirmed","publisher_bbox_clipped_to_image":bbox!=list(map(float,b[:4]))})
        else:
            a,b=Path(name).stem.split("-")[2].split("_");left,top=map(float,a.split("&"));right,bottom=map(float,b.split("&"))
            bbox=clipped_box([left,top,right-left,bottom-top],width,height)
            if bbox:objects=[{"class":"license_plate","bbox_xywh_px":bbox,"certainty":"confirmed"}]
        if not objects:raise ValueError("no_known_class_gt")
        if len(objects)>512:raise ValueError("gt_count_over512_image_omitted_not_truncated")
        source_id=ident(kind+":"+name);sample_id=ident("fullframe-v3:"+source_id)
        path=originals/(sample_id+".jpg")
        if split=="test":
            if source_id in old_ids or source_hash in old_hashes:raise ValueError("new_test_source_was_seen")
            if kind=="license_plate" and group in old_plate_ids:raise ValueError("new_test_plate_identity_was_seen")
            if any((visual^previous).bit_count()<=3 for previous in old_visual):raise ValueError("new_test_near_duplicate_of_seen_source")
        if not path.exists():path.write_bytes(raw)
        if sha(path)!=source_hash:raise ValueError("saved_original_hash_mismatch")
        source_record=source_records["WIDER_"+publisher_split+".zip"] if kind=="face" else source_records["CCPD2020.zip"]
        policy=license_paths[kind]
        return {"sample_id":sample_id,"source_id":source_id,"parent_sample_id":source_id,"source_hash":source_hash,"leakage_group_id":group,"annotation_version":"publisher-original-fullframe-known-class-v3","path":rel(path),"sha256":source_hash,"image":{"width":width,"height":height},"split":split,"publisher_split":publisher_split,"privacy_status":"restricted_research","qc_status":"publisher_annotations_machine_checked","release_scope":"offline_research","human_privacy_reviewed":False,"not_for_deployment":True,"project_acceptance_eligible":False,"allowed_use":"Private non-commercial offline research only; WIDER CC-BY-NC-ND4 and CCPD MIT retained. No source or learned-artifact redistribution/deployment clearance.","expires_at":old_row["expires_at"] if old_row else source_record["expires_at"],"source_url":"https://huggingface.co/datasets/CUHK-CSE/wider_face" if kind=="face" else "https://github.com/detectRecog/CCPD","license_evidence_path":rel(policy),"license_evidence_sha256":sha(policy),"detection_labels_verified":True,"objects":objects,"known_label_coverage":[kind],"evaluation_class":kind,"unknown_other_classes_must_not_count_as_negative":True,"source_dhash":f"{visual:016x}","annotation_review":"ALL valid publisher known-class boxes retained without max-label truncation. Any original with positive-area publisher-invalid face annotation is omitted. Machine checks only; no human privacy approval.","coverage_limitation":"Other class is unannotated. Native YOLOX class-agnostic objectness cannot fully mask unknown other-class false-negative training noise; no complete multiclass/privacy annotation claim.","evaluation_scope":"Original fullframe known-class conditional metrics; other-class predictions excluded from metrics. No global privacy recall/FPR, anonymization assurance or project acceptance.","frozen_crop_source_sample_id":old_row["sample_id"] if old_row else None,"test_status":"new_unseen_source_heldout_before_v3_training" if split=="test" else "same_frozen_train_validation_source_membership"}
    for row in old:
        if row["split"] not in ("train","validation"):continue
        kind=row["objects"][0]["class"]
        try:
            if kind=="face":
                name,boxes,publisher_split=face_lookup[row["source_id"]];new=prepare(kind,name,publisher_split,row["leakage_group_id"],row["split"],boxes,row)
            else:
                name,group,publisher_split=plate_lookup[row["source_id"]];new=prepare(kind,name,publisher_split,group,row["split"],old_row=row)
            rows.append(new)
        except ValueError as e:rejected[row["split"]+":"+kind+":"+str(e)]+=1
    print("train_validation_originals",dict(Counter(r["split"]+":"+r["evaluation_class"] for r in rows)),flush=True)
    new_plates=defaultdict(list)
    for name,group,publisher_split in all_plates:
        if publisher_split=="test" and len(plate_partitions[group])==1 and group not in old_plate_ids:new_plates[group].append((name,group,publisher_split))
    selected_source_hashes=set();selected_plate_groups=set()
    for kind,groups in (("face",new_faces),("license_plate",new_plates)):
        count=0
        for entry in round_robin(groups):
            if count>=200:break
            try:
                if kind=="face":
                    name,boxes,publisher_split=entry;group="WIDER-event:"+name.split("/")[0]
                    new=prepare(kind,name,publisher_split,group,"test",boxes)
                else:
                    name,group,publisher_split=entry
                    if group in selected_plate_groups:continue
                    new=prepare(kind,name,publisher_split,group,"test")
                if new["source_hash"] in selected_source_hashes:raise ValueError("new_test_exact_duplicate")
                selected_source_hashes.add(new["source_hash"])
                if kind=="license_plate":selected_plate_groups.add(group)
                rows.append(new);count+=1
            except ValueError as e:rejected["new_test:"+kind+":"+str(e)]+=1
        print("new_unseen_test",kind,count,flush=True)
    if sha(frozen)!=frozen_hash:raise ValueError("frozen_crop_release_changed")
    state=validate_research_release(rows)
    if not state["valid"]:raise ValueError(json.dumps(state["errors"][:10]))
    release=DEST/"release.json";write(release,rows)
    report={"manifest_path":rel(release),"manifest_sha256":sha(release),"script_sha256":sha(Path(__file__)),"frozen_crop_release_sha256":frozen_hash,"original_test_status":"development_diagnostic_seen; retained separately, never moved into training or represented as v3 final test","old_test_manifest_path":rel(OUTPUT/"fullscene-test.json"),"old_test_manifest_sha256":sha(OUTPUT/"fullscene-test.json"),"counts":dict(Counter(r["split"]+":"+r["evaluation_class"] for r in rows)),"objects":dict(Counter(r["split"]+":"+o["class"] for r in rows for o in r["objects"])),"max_known_objects_per_image":max(len(r["objects"]) for r in rows),"rejected":dict(rejected),"face_event_split_unchanged":events,"new_test_exclusion_policy":"Exclude ALL source UUIDs, original image SHA256s, hashed plate identities, and 64bit original-source dHash Hamming<=3 neighbors from every frozen crop train/validation/test image before selecting new original test. No seen source moved into train; original train/val only omit invalid publisher GT or >512-box images. New test drawn from same frozen held-out WIDER events/CCPD publisher test; plate identity unique per newtest image.","known_label_coverage":"Singleton per image. ALL valid known-class GT retained, other class unknown; no complete two-class annotation or global privacy claim. Unknown other-class objectness-negative training noise remains a documented limitation.","machine_validation":state,"human_privacy_reviewed":False,"project_acceptance_eligible":False,"independent_person_identity_verified":False,"source_licenses":old_qc["source_licenses"],"source_attributions":old_qc.get("source_attributions"),"redistribution_policy":old_qc["redistribution_policy"],"frozen_before_v3_training":True,"model_predictions_used_for_selection":False}
    write(DEST/"source-qc.json",report);write(ROOT/"docs/reports/REDACTION-FULLFRAME-V3-QC.json",report)
    for z in [*archives.values(),ccpd,split_zip]:z.close()
    print(json.dumps({"release":rel(release),"counts":report["counts"],"valid":state["valid"]}),flush=True)

if __name__=="__main__":main()
