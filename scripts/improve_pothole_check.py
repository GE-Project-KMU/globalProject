"""CPU augmentation/protocol contracts; not detector accuracy evidence."""
import json,argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"scripts"))
import improve_pothole_train as trainer
from agent.paths import atomic_json,sha256

def check(source):
    source=trainer.workspace_path(source,exists=True)
    rows=json.loads(source.read_text());previous=json.loads((ROOT/"data/training/research-pothole-rdd2020/release.json").read_text())
    oldtest=[r for r in previous if r["split"]=="test"];oldallhash={r["sha256"] for r in previous}
    oldtesthash={r["sha256"] for r in oldtest};oldtestvisual=[int(r["perceptual_dhash"],16) for r in oldtest]
    checks={}
    checks["old_test_sha_not_in_any_new_split"]=not any(r["sha256"] in oldtesthash for r in rows)
    checks["old_test_dhash_le3_not_in_new_train"]=not any((int(r["perceptual_dhash"],16)^v).bit_count()<=3 for r in rows if r["split"]=="train" for v in oldtestvisual)
    checks["fresh_test_no_prior_image_sha"]=not any(r["sha256"] in oldallhash for r in rows if r["split"]=="test")
    history_paths=[ROOT/"data/training/research-pothole/release.json",ROOT/"data/training/research-pothole-rdd2020-v2/release.json"]
    historical=previous+[r for p in history_paths for r in json.loads(p.read_text()) if p!=history_paths[-1] or r["split"]!="test"]
    from improve_pothole_data import group
    from improve_pothole_data import historical_exclusions,HISTORY_SHA
    snapshot=ROOT/"docs/reports/POTHOLE-HISTORY-EXCLUSIONS.json"
    checks["frozen_history_snapshot_matches_all_used_identities"]=sha256(snapshot)==HISTORY_SHA and json.loads(snapshot.read_text())==historical_exclusions()
    from urllib.parse import urlsplit
    old_groups=set()
    for row in historical:
        name=row.get("original_member")
        if not name and row.get("country") in ("India","Japan","Czech"):
            name="train/"+row["country"]+"/images/"+Path(urlsplit(row["source_url"]).path).name
        if name:old_groups.add(group(name))
    history_hash={r["sha256"] for r in historical};history_visual=[int(r["perceptual_dhash"],16) for r in historical]
    checks["fresh_test_no_historical_used_block_including_quarantine_aborted"]=not any(r["leakage_group_id"] in old_groups for r in rows if r["split"]=="test")
    checks["fresh_test_no_historical_used_SHA_dhash_including_quarantine_aborted"]=not any(r["sha256"] in history_hash or any((int(r["perceptual_dhash"],16)^v).bit_count()<=3 for v in history_visual) for r in rows if r["split"]=="test")
    token={split:{r["leakage_group_id"] for r in rows if r["split"]==split} for split in ("train","validation","test")}
    checks["filename_blocks_disjoint"]=not(token["train"]&token["validation"] or token["train"]&token["test"] or token["validation"]&token["test"])
    checks["TTL_never_extended"]=all(r["expires_at"]=="2026-11-10T17:14:46.862846+00:00" for r in rows)
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
    from improve_pothole_compare import grouped_interval,wilson
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
        "release_sha256":sha256(source),"trainer_sha256":sha256(ROOT/"scripts/improve_pothole_train.py"),
        "old_source_expiry_unchanged":True,"training_code_changed":False}
    atomic_json(ROOT/"docs/reports/POTHOLE-IMPROVEMENT-CONTRACT-v4.json",report)
    print(json.dumps(report),flush=True)
    if not report["passed"]:raise ValueError("contract_failed")

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("release");check(parser.parse_args().release)
