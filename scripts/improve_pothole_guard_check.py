"""Post-run future-input guards; preserves the completed experiment's evidence."""
import copy,json,shutil,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"scripts"))
import improve_pothole_train as trainer
from improve_pothole_check import history_evidence
from agent.paths import atomic_json,sha256

def main():
    release=ROOT/"data/training/research-pothole-rdd2020-v4/release.json"
    checks={"canonical_v4_accepted":len(trainer.validate_original_release(release))==5840}
    cache=ROOT/".cache/pothole-guard-check";cache.mkdir(parents=True,exist_ok=True)
    temporary=Path(tempfile.mkdtemp(dir=cache))
    if not temporary.resolve().is_relative_to(cache.resolve()):raise ValueError("temporary_path_outside_workspace")
    try:
        def reject(name,rows,expected):
            path=temporary/(name+".json");atomic_json(path,rows)
            try:trainer.validate_original_release(path)
            except ValueError as error:return str(error)==expected
            return False
        canonical=json.loads(release.read_text());first=copy.deepcopy(next(r for r in canonical if r["objects"]))
        for label,field,value,expected in (
            ("unsupported_annotation","annotation_version","third-party-mapping","canonical_original_D40_annotation_version_required"),
            ("old_test_country","country","Czech","only_India_Japan_development_originals_allowed"),
            ("wrong_source","source_url","https://example.invalid/converted","canonical_RDD2020_source_required")):
            row=copy.deepcopy(first);row[field]=value;checks[label+"_rejected"]=reject(label,[row],expected)
        wrong=copy.deepcopy(first);wrong["objects"][0]["class"]="crack"
        checks["non_D40_class_rejected"]=reject("wrongclass",[wrong],"literal_D40_pothole_objects_only")
        checks["empty_release_rejected"]=reject("empty",[],"nonempty_original_D40_release_required")
        quarantined=temporary/"quarantined.json";atomic_json(quarantined,[]);atomic_json(quarantined.with_suffix(".quarantine.json"),{"do_not_use":True})
        try:trainer.validate_original_release(quarantined)
        except ValueError as error:checks["quarantine_sidecar_rejected"]=str(error)=="source_release_quarantined"
        else:checks["quarantine_sidecar_rejected"]=False
        try:trainer.validate_original_release(ROOT/"data/training/research-pothole-rdd2020-v2/release.json")
        except ValueError as error:checks["aborted_v2_history_overlap_rejected"]=str(error)=="fresh_test_overlaps_historical_used_image_or_block"
        else:checks["aborted_v2_history_overlap_rejected"]=False
        frozen,direct=history_evidence();substitute,fallback=history_evidence(paths=[])
        checks["direct_original_metadata_audit_performed"]=direct["performed"] is True
        checks["clean_clone_uses_identical_full_exclusion_scope"]=frozen==substitute
        checks["clean_clone_does_not_claim_original_audit"]=fallback["performed"] is False and fallback["status"]=="not_performed_fixed_snapshot_substitute"
        corrupt=temporary/"bad-snapshot.json";atomic_json(corrupt,{"sha256":[],"dhash64":[],"filename20_blocks":[]})
        try:history_evidence(paths=[],snapshot_path=corrupt)
        except ValueError as error:checks["reduced_or_changed_snapshot_rejected"]=str(error)=="historical_exclusion_snapshot_hash_mismatch"
        else:checks["reduced_or_changed_snapshot_rejected"]=False
        # Both audit routes enforce the very same test identity/block checks.
        def fresh(data):
            known_hash=set(data["sha256"]);known_group=set(data["filename20_blocks"]);known_visual=[int(v,16) for v in data["dhash64"]]
            return all(r["sha256"] not in known_hash and r["leakage_group_id"] not in known_group
                       and not any((int(r["perceptual_dhash"],16)^v).bit_count()<=3 for v in known_visual) for r in canonical if r["split"]=="test")
        checks["clean_clone_freshness_contract_equivalent"]=fresh(frozen) and fresh(substitute)
    finally:
        if not temporary.resolve().is_relative_to(cache.resolve()):raise ValueError("unsafe_cleanup_path")
        shutil.rmtree(temporary)
    folder=ROOT/"data/training/pothole-nano-rdd2020-v4"
    source_audit=json.loads((folder/"executed-source-audit.json").read_text())
    report={"checks":checks,"passed":all(checks.values()),"accuracy_evidence":False,
        "executed_trainer_sha256":source_audit["trainer_sha256"],"current_future_CLI_sha256":sha256(ROOT/"scripts/improve_pothole_train.py"),
        "initial_35_contract_sha256":sha256(folder/"initial-software-contract.executed.json"),
        "postrun_change":"input guards and optional direct-original audit only; frozen snapshot scope never reduced",
        "weights_or_operating_threshold_changed":False,"direct_original_audit":direct,"clean_clone_simulation_audit":fallback}
    atomic_json(ROOT/"docs/reports/POTHOLE-IMPROVEMENT-GUARD.json",report)
    atomic_json(folder/"future-cli-source-audit.json",report)
    print(json.dumps(report),flush=True)
    if not report["passed"]:raise ValueError("postrun_guard_check_failed")

if __name__=="__main__":main()
