"""Final-only frozen baseline comparison, never a training/selection entry point."""
import argparse,json,sys,math,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"scripts"))
from agent.paths import workspace_path,atomic_json,sha256
from train_yolox_research import make_model,evaluate_model,operating_metrics,detection_metrics
import torch

def wilson(tp,total):
    if not total:return None
    z=1.959963984540054;p=tp/total;den=1+z*z/total
    center=(p+z*z/(2*total))/den;radius=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return [max(0.,center-radius),min(1.,center+radius)]

def grouped_interval(baseline,candidate,rows,repetitions=500):
    groups={}
    for row in rows:groups.setdefault(row["leakage_group_id"],set()).add(row["sample_id"])
    names=sorted(groups);rng=random.Random(20261104);scores=[]
    # Repeated sampled blocks get unique sample IDs so GT cannot match across
    # bootstrap copies. Paired draws use exactly the same images for both models.
    truth={k:[g for g in baseline["truth"] if g["sample_id"] in groups[k]] for k in names}
    predictions=[{k:[p for p in result["predictions"] if p["sample_id"] in groups[k]] for k in names} for result in (baseline,candidate)]
    for _ in range(repetitions):
        sampled=rng.choices(names,k=len(names));gt=[];preds=[[],[]]
        for index,key in enumerate(sampled):
            for row in truth[key]:gt.append(row|{"sample_id":str(index)+":"+row["sample_id"]})
            for model in (0,1):
                preds[model].extend(row|{"sample_id":str(index)+":"+row["sample_id"]} for row in predictions[model][key])
        ap=[detection_metrics(gt,p,["pothole"])["map50"] for p in preds]
        if all(value is not None for value in ap):scores.append((ap[0],ap[1],ap[1]-ap[0]))
    import numpy as np
    if not scores:raise ValueError("no_valid_bootstrap_sample")
    values=np.array(scores)
    return {"method":"paired filename20-block bootstrap percentile95; proxy groups, NOT verified independent routes",
        "repetitions":repetitions,"valid_repetitions":len(scores),"seed":20261104,"groups":len(groups),
        "baseline_AP50_interval95":np.percentile(values[:,0],[2.5,97.5]).tolist(),
        "candidate_AP50_interval95":np.percentile(values[:,1],[2.5,97.5]).tolist(),
        "paired_AP50_change_interval95":np.percentile(values[:,2],[2.5,97.5]).tolist()}

def compare(folder):
    folder=workspace_path(folder,exists=True)
    destination=folder/"baseline-new-heldout-predictions.json"
    if destination.exists():raise ValueError("baseline_final_comparison_already_performed_do_not_reselect")
    manifest=json.loads((folder/"model.json").read_text());config=json.loads((folder/"training-config.json").read_text())
    if manifest.get("research_only") is not True or manifest.get("test_split_used") is not False:raise ValueError("unfrozen_or_nonresearch_model")
    if config.get("test_used_for_selection") is not False:raise ValueError("test_selection_forbidden")
    heldout=json.loads((folder/"heldout-predictions.json").read_text());regression=json.loads((folder/"old-czech-regression-predictions.json").read_text())
    release=workspace_path(manifest["source_release_path"],exists=True)
    if sha256(release)!=config["release_sha256"]:raise ValueError("release_changed")
    all_rows=json.loads(release.read_text());rows=[r for r in all_rows if r["split"]=="test"]
    val={r["sample_id"]:r for r in all_rows if r["split"]=="validation" and r["objects"]}
    selected=json.loads((folder/"selected-validation-predictions.json").read_text())
    positives=[p for p in selected["predictions"] if p["sample_id"] in val and p["confidence"]>=manifest["confidence"]]
    if not positives:raise ValueError("no_positive_validation_case_at_frozen_confidence")
    positive=max(positives,key=lambda p:p["confidence"]);image_row=val[positive["sample_id"]]
    atomic_json(folder/"verification-case.json",{"image":image_row["path"],"sample_id":image_row["sample_id"],
        "selection":"validation positive GT and nonempty predictions at frozen confidence; numerical parity only, no heldout tuning",
        "confidence":manifest["confidence"],"top_prediction":positive,"expires_at":image_row["expires_at"],
        "checkpoint":(folder/"best.pt").relative_to(ROOT).as_posix(),"manifest":(folder/"model.json").relative_to(ROOT).as_posix(),
        "manifest_sha256":sha256(folder/"model.json")})
    model,_=make_model(["pothole"])
    checkpoint=ROOT/"data/training/pothole-nano-rdd2020-v1/best.pt"
    model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True)["model"]);model=model.cuda()
    baseline=evaluate_model(model,rows,["pothole"],size=640,batch_size=8)
    if baseline["truth"]!=heldout["truth"]:raise ValueError("baseline_candidate_truth_mismatch")
    baseline["operating_metrics"]=operating_metrics(baseline,["pothole"],.35)
    baseline["role"]="single final baseline comparison on same new frozen holdout AFTER candidate selection"
    baseline["baseline_checkpoint_sha256"]=sha256(checkpoint)
    atomic_json(destination,baseline)
    report={"research_only":True,"production_approved":False,"project_acceptance_eligible":False,
        "new_holdout":{"images":len(rows),"release_sha256":sha256(release),"baseline_AP50":baseline["metrics"]["map50"],
            "candidate_AP50":heldout["metrics"]["map50"],"AP50_change":heldout["metrics"]["map50"]-baseline["metrics"]["map50"],
            "baseline_fixed035":baseline["operating_metrics"],"candidate_fixed035":operating_metrics(heldout,["pothole"],.35),
            "candidate_validation_F2_selected_threshold":manifest["confidence"],"candidate_selected_operating":heldout["operating_metrics"]},
        "old_Czech_regression":{"role":"previously examined regression benchmark only; not fresh final acceptance",
            "candidate_AP50":regression["metrics"]["map50"],"candidate_fixed035":operating_metrics(regression,["pothole"],.35),
            "candidate_selected_operating":regression["operating_metrics"]},
        "manifest_sha256":sha256(folder/"model.json"),"weights_sha256":manifest["sha256"],"best_epoch":manifest["best_epoch"],
        "baseline_checkpoint_sha256":sha256(checkpoint),"executed_trainer_script_sha256":config["executed_trainer_script_sha256"],
        "protocol":{"candidate_count":1,"checkpoint":"new validation AP50 maximum","operating_threshold":"new validation F2 maximum only",
            "AP_confidence_floor":.01,"nms_iou":.5,"fixed_035_reference_declared_before_test":True,
            "baseline_old_test_never_reused_as_train":True,"new_final_test_never_used_for_selection":True},
        "limits":["Same-country Japan filename20 block holdout; physical route/recording independence unverified",
            "Small final holdout: uncertainty intervals are descriptive; source/route sampling bias is not removed by bootstrap",
            "Official publisher D40 annotation is retained without model-based relabeling; real-world Korean camera domain not validated",
            "CC-BY-NC-3.0 noncommercial research; human privacy not reviewed; no production acceptance"]}
    report["new_holdout"]["uncertainty"]=grouped_interval(baseline,heldout,rows)
    for key in ("baseline_fixed035","candidate_fixed035","candidate_selected_operating"):
        values=report["new_holdout"][key]["classes"]["pothole"]
        report["new_holdout"][key]["recall_Wilson95"]=wilson(values["tp"],values["gt_count"])
        report["new_holdout"][key]["Wilson_limit"]="Treats GT objects as independent; within-image/route correlation makes this an auxiliary descriptive interval only"
    atomic_json(ROOT/"docs/reports/POTHOLE-IMPROVEMENT-RESULTS.json",report)
    print(json.dumps(report),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("folder");compare(parser.parse_args().folder)
