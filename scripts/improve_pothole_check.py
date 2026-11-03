"""CPU augmentation/protocol contracts; not detector accuracy evidence."""
import json,argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"scripts"))
import improve_pothole_train as trainer
from agent.paths import atomic_json,sha256

def history_evidence(paths=None,snapshot_path=None):
    """Never shrink the frozen exclusion scope when original metadata is absent."""
    from improve_pothole_data import historical_exclusions,HISTORY_SHA,group
    from urllib.parse import urlsplit
    snapshot=ROOT/"docs/reports/POTHOLE-HISTORY-EXCLUSIONS.json" if snapshot_path is None else trainer.workspace_path(snapshot_path,exists=True)
    if sha256(snapshot)!=HISTORY_SHA:raise ValueError("historical_exclusion_snapshot_hash_mismatch")
    frozen=json.loads(snapshot.read_text(encoding="utf-8"))
    defaults=[ROOT/"data/training/research-pothole-rdd2020/release.json",ROOT/"data/training/research-pothole/release.json",
              ROOT/"data/training/research-pothole-rdd2020-v2/release.json"]
    paths=defaults if paths is None else paths
    available=[p for p in paths if p.is_file()]
    # Available identities must be covered even in a partial clean-clone audit.
    known_hash=set(frozen["sha256"]);known_visual=set(frozen["dhash64"]);known_groups=set(frozen["filename20_blocks"])
    for path in available:
        for row in json.loads(path.read_text(encoding="utf-8")):
            if path.parent.name=="research-pothole-rdd2020-v2" and row["split"]=="test":continue
            name=row.get("original_member")
            if not name and row.get("country") in ("India","Japan","Czech"):
                name="train/"+row["country"]+"/images/"+Path(urlsplit(row["source_url"]).path).name
            if row["sha256"] not in known_hash or row["perceptual_dhash"] not in known_visual or (name and group(name) not in known_groups):
                raise ValueError("available_historical_identity_not_covered_by_frozen_snapshot")
    complete=paths==defaults and len(available)==len(defaults)
    if complete and historical_exclusions()!=frozen:raise ValueError("historical_original_metadata_snapshot_mismatch")
    return frozen,{"performed":complete,"status":"direct_original_metadata_audit_performed" if complete else "not_performed_fixed_snapshot_substitute",
        "available_original_metadata_files":len(available),"required_original_metadata_files":len(defaults),
        "available_identity_subset_verified":True,"frozen_snapshot_sha256":HISTORY_SHA,
        "exclusion_scope_reduced":False}

def check(source,output):
    source=trainer.workspace_path(source,exists=True)
    rows=json.loads(source.read_text());previous=json.loads((ROOT/"data/training/research-pothole-rdd2020/release.json").read_text())
    oldtest=[r for r in previous if r["split"]=="test"];oldallhash={r["sha256"] for r in previous}
    oldtesthash={r["sha256"] for r in oldtest};oldtestvisual=[int(r["perceptual_dhash"],16) for r in oldtest]
    checks={}
    checks["old_test_sha_not_in_any_new_split"]=not any(r["sha256"] in oldtesthash for r in rows)
    checks["old_test_dhash_le3_not_in_new_train"]=not any((int(r["perceptual_dhash"],16)^v).bit_count()<=3 for r in rows if r["split"]=="train" for v in oldtestvisual)
    checks["fresh_test_no_prior_image_sha"]=not any(r["sha256"] in oldallhash for r in rows if r["split"]=="test")
    frozen,audit=history_evidence()
    checks["frozen_history_snapshot_hash_verified"]=audit["frozen_snapshot_sha256"]==sha256(ROOT/"docs/reports/POTHOLE-HISTORY-EXCLUSIONS.json")
    old_groups=set(frozen["filename20_blocks"])
    history_hash=set(frozen["sha256"]);history_visual=[int(v,16) for v in frozen["dhash64"]]
    checks["fresh_test_no_historical_used_block_including_quarantine_aborted"]=not any(r["leakage_group_id"] in old_groups for r in rows if r["split"]=="test")
    checks["fresh_test_no_historical_used_SHA_dhash_including_quarantine_aborted"]=not any(r["sha256"] in history_hash or any((int(r["perceptual_dhash"],16)^v).bit_count()<=3 for v in history_visual) for r in rows if r["split"]=="test")
    token={split:{r["leakage_group_id"] for r in rows if r["split"]==split} for split in ("train","validation","test")}
    checks["filename_blocks_disjoint"]=not(token["train"]&token["validation"] or token["train"]&token["test"] or token["validation"]&token["test"])
    source_expiry=json.loads((ROOT/"data/research/rdd2020/source.json").read_text())["expires_at"]
    checks["TTL_inherited_from_source_without_extension"]=all(r["expires_at"]==r["source_expires_at"]==source_expiry for r in rows)
    trainer.seed_all(20261104)
    training=[r for r in rows if r["split"]=="train"]
    cached=trainer.OriginalSamples(training[:2],640,cache_gib=.1);uncached=trainer.OriginalSamples(training[:2],640)
    checks["RAM_cache_preserves_exact_image_and_labels"]=all(trainer.np.array_equal(cached.pull_item(i)[0],uncached.pull_item(i)[0]) and trainer.np.array_equal(cached.pull_item(i)[1],uncached.pull_item(i)[1]) for i in range(2))
    before=cached.pull_item(0)[0];cached.pull_item(0)[0][:]=0
    checks["RAM_cache_copy_protects_future_epochs"]=trainer.np.array_equal(before,cached.pull_item(0)[0])
    wrapper=trainer.MosaicDetection(trainer.OriginalSamples(training,640),img_size=(640,640),
        preproc=trainer.TrainTransform(max_labels=120,hsv_prob=1.,flip_prob=.5),mosaic_prob=1.,enable_mixup=False)
    for index in range(8):
        image,label,_,_=wrapper[index]
        active=label[label[:,3]*label[:,4]>0]
        checks[f"native_mosaic_shape_finite_{index}"]=image.shape==(3,640,640) and label.shape==(120,5) and bool(trainer.np.isfinite(image).all() and trainer.np.isfinite(label).all())
        checks[f"native_mosaic_boxes_in_frame_{index}"]=bool((active[:,1:3]>=0).all() and (active[:,1:3]<=640).all() and (active[:,3:]<=640).all())
    # An exact GT and one high-score FP: validation calibration prefers the
    # lower score threshold that recovers GT, without consulting test rows.
    fixture={"truth":[{"sample_id":"synthetic","class":"pothole","bbox_xywh_px":[0,0,10,10]}],
        "predictions":[{"sample_id":"synthetic","class":"pothole","bbox_xywh_px":[0,0,10,10],"confidence":.2},
        {"sample_id":"synthetic","class":"pothole","bbox_xywh_px":[20,20,5,5],"confidence":.4}]}
    calibration=trainer.calibration(fixture)
    checks["F2_calibration_recovers_GT"]=calibration["selected_confidence"]<=.2 and max(s["f2"] for s in calibration["scores"])>0
    from improve_pothole_compare import grouped_interval,wilson,iou
    checks["verification_case_IoU_identical_and_disjoint"]=iou([0,0,10,10],[0,0,10,10])==1. and iou([0,0,10,10],[20,20,10,10])==0.
    checks["verification_case_IoU_partial_overlap"]=abs(iou([0,0,10,10],[5,0,10,10])-1/3)<1e-12
    fixture_rows=[{"sample_id":"a","leakage_group_id":"g1"},{"sample_id":"b","leakage_group_id":"g2"}]
    truth=[{"sample_id":r["sample_id"],"class":"pothole","bbox_xywh_px":[0,0,10,10]} for r in fixture_rows]
    perfect={"truth":truth,"predictions":[r|{"confidence":.9} for r in truth]};missed={"truth":truth,"predictions":[]}
    checks["paired_bootstrap_identical_models_delta_zero"]=grouped_interval(perfect,perfect,fixture_rows,20)["paired_AP50_change_interval95"]==[0.,0.]
    checks["paired_bootstrap_missed_GT_delta_negative_one"]=grouped_interval(perfect,missed,fixture_rows,20)["paired_AP50_change_interval95"]==[-1.,-1.]
    checks["Wilson_empty_and_boundary_contract"]=wilson(0,0) is None and wilson(0,10)[0]==0. and 0<=wilson(10,10)[0]<1
    from yolox.exp import get_exp
    experiment=get_exp(exp_file=str(ROOT/".cache/vendor/YOLOX/exps/default/yolox_nano.py"));experiment.num_classes=1
    model=experiment.get_model();optimizer=experiment.get_optimizer(16)
    parameters=[id(p) for group_ in optimizer.param_groups for p in group_["params"]]
    checks["native_SGD_covers_each_parameter_once"]=len(parameters)==len(set(parameters))==len(list(model.parameters()))
    no_decay={id(p) for group_ in optimizer.param_groups if group_["weight_decay"]==0 for p in group_["params"]}
    norm_and_bias={id(m.weight) for m in model.modules() if isinstance(m,trainer.torch.nn.BatchNorm2d)}
    norm_and_bias.update(id(m.bias) for m in model.modules() if getattr(m,"bias",None) is not None)
    checks["native_SGD_excludes_BN_and_bias_weight_decay"]=norm_and_bias<=no_decay
    scheduler=trainer.LRScheduler("yoloxwarmcos",.0025,100,60,warmup_epochs=3,warmup_lr_start=0.,no_aug_epochs=10,min_lr_ratio=.05)
    checks["native_warmup_and_final_LR_contract"]=abs(scheduler.update_lr(300)-.0025)<1e-12 and abs(scheduler.update_lr(6000)-.000125)<1e-12
    report={"checks":checks,"passed":all(checks.values()),"accuracy_evidence":False,
        "release_sha256":sha256(source),"current_future_CLI_sha256":sha256(ROOT/"scripts/improve_pothole_train.py"),
        "historical_originals_audit":audit,"old_source_expiry_unchanged":True,
        "purpose":"post-experiment future CLI checks; original executed 35-check evidence preserved separately",
        "existing_weights_or_threshold_changed":False}
    atomic_json(output,report)
    print(json.dumps(report),flush=True)
    if not report["passed"]:raise ValueError("contract_failed")

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("release")
    parser.add_argument("--output",default="docs/reports/POTHOLE-IMPROVEMENT-CONTRACT-FUTURE.json")
    args=parser.parse_args();check(args.release,args.output)
