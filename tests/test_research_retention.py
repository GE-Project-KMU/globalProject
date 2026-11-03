"""Synthetic exact-file deletion tests; never register or remove real research data."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,tempfile,uuid
import pytest
from scripts import research_retention as retention

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

@pytest.fixture
def case(monkeypatch):
    base=retention.ROOT/".cache/research-retention-tests";base.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=base) as directory:
        project=Path(directory);monkeypatch.setattr(retention,"ROOT",project);monkeypatch.setattr(retention,"STATE",project/"data/research-retention")
        folder=project/"data/research/redaction";folder.mkdir(parents=True)
        source=folder/"SYNTHETIC_PRIVATE_FILENAME.jpg";source.write_bytes(b"SYNTHETIC IMAGE BYTES")
        evidence=folder/"source.json";evidence.write_text(json.dumps({"downloaded_at":"2026-01-01T00:00:00Z","expires_at":"2026-01-08T00:00:00Z"}),encoding="utf-8")
        plan=folder/"plan.json";inventory=retention.STATE/("test-"+uuid.uuid4().hex+".json")
        def relative(path):return path.relative_to(retention.ROOT).as_posix()
        def save(path=source,kind="source_image"):
            value={"schema_version":1,"roots":["data/research/redaction"],"entries":[{"path":relative(path),"kind":kind,"sha256":sha(path),"evidence":[{"path":relative(evidence),"sha256":sha(evidence)}]}]}
            plan.write_text(json.dumps(value),encoding="utf-8");return value
        save()
        yield type("Case",(),{"folder":folder,"source":source,"evidence":evidence,"plan":plan,"inventory":inventory,"save":staticmethod(save),"relative":staticmethod(relative)})
        inventory.unlink(missing_ok=True)

def test_default_dry_run_does_not_delete_or_mutate_inventory(case):
    retention.register(case.relative(case.plan),case.relative(case.inventory))
    original=case.inventory.read_bytes()
    future=datetime(2026,1,9,tzinfo=timezone.utc)
    result=retention.purge(case.relative(case.inventory),now=future)
    assert result["dry_run"] and result["counts"]["would_delete"]==1
    assert case.source.exists() and case.inventory.read_bytes()==original
    assert "SYNTHETIC_PRIVATE_FILENAME" not in json.dumps(result)

def test_only_explicit_a2d2_extension_root_is_allowed(case):
    base=retention.ROOT/'data/public/a2d2-improvement-v2';base.mkdir(parents=True)
    source=base/'SYNTHETIC_NEW_FRAME.png';source.write_bytes(b'SYNTHETIC DATA')
    plan=case.save(source);plan['roots']=['data/public/a2d2-improvement-v2']
    case.plan.write_text(json.dumps(plan),encoding='utf-8')
    result=retention.register(case.relative(case.plan),case.relative(case.inventory))
    assert result['registered_files']==1 and source.exists()
    for parent in ('data','data/public','data/public/a2d2-improvement','data/public/a2d2-improvement-v2/../'):
        with pytest.raises(ValueError,match='unknown_research_root'):retention.roots([parent])

def test_apply_deletes_only_expired_registered_exact_file_and_minimizes_state(case):
    unregistered=case.folder/"unregistered.jpg";unregistered.write_bytes(b"PRESERVE")
    retention.register(case.relative(case.plan),case.relative(case.inventory))
    result=retention.purge(case.relative(case.inventory),apply=True,now=datetime(2026,1,9,tzinfo=timezone.utc))
    assert result["counts"]["deleted"]==1 and not case.source.exists()
    assert result["private_plan_counts"]["deleted"]==1 and not case.plan.exists()
    assert unregistered.exists() and case.folder.exists() and case.evidence.exists()
    value=case.inventory.read_text();assert "SYNTHETIC_PRIVATE_FILENAME" not in value
    assert next(iter(json.loads(value)["entries"].values()))["status"]=="deleted"
    assert retention.purge(case.relative(case.inventory),apply=True,now=datetime(2026,1,9,tzinfo=timezone.utc))["counts"]["retired"]==1

def test_not_due_and_modified_files_are_never_deleted(case):
    retention.register(case.relative(case.plan),case.relative(case.inventory))
    result=retention.purge(case.relative(case.inventory),apply=True,now=datetime(2026,1,2,tzinfo=timezone.utc))
    assert result["counts"]["not_due"]==1 and case.source.exists()
    case.source.write_bytes(b"CHANGED")
    result=retention.purge(case.relative(case.inventory),apply=True,now=datetime(2026,1,9,tzinfo=timezone.utc))
    assert result["counts"]["modified_refused"]==1 and case.source.read_bytes()==b"CHANGED"

def test_new_receipt_cannot_extend_fixed_deadline_or_replace_hash(case):
    retention.register(case.relative(case.plan),case.relative(case.inventory));original=case.inventory.read_bytes()
    case.evidence.write_text(json.dumps({"downloaded_at":"2026-01-02T00:00:00Z","expires_at":"2026-01-09T00:00:00Z"}),encoding="utf-8");case.save()
    with pytest.raises(ValueError,match="deadline_extension"):retention.register(case.relative(case.plan),case.relative(case.inventory))
    assert case.inventory.read_bytes()==original
    case.evidence.write_text(json.dumps({"expires_at":"2026-01-08T00:00:00Z"}),encoding="utf-8")
    case.source.write_bytes(b"CHANGED");case.save()
    with pytest.raises(ValueError,match="registered_file_changed"):retention.register(case.relative(case.plan),case.relative(case.inventory))
    assert case.inventory.read_bytes()==original

def test_known_expiry_required_no_current_time_or_mtime_fallback(case):
    case.evidence.write_text(json.dumps({"downloaded_at":"2026-01-01T00:00:00Z"}),encoding="utf-8");case.save()
    with pytest.raises(ValueError,match="fixed_source_expiry_required"):retention.register(case.relative(case.plan),case.relative(case.inventory))
    assert not case.inventory.exists()

@pytest.mark.parametrize("name",["model.pt","weights.onnx","checkpoint.pth.part"])
def test_weights_preserved_even_when_disguised_as_partial_archive(case,name):
    path=case.folder/name;path.write_bytes(b"WEIGHTS");case.save(path,"source_archive")
    with pytest.raises(ValueError):retention.register(case.relative(case.plan),case.relative(case.inventory))
    assert path.exists() and not case.inventory.exists()

@pytest.mark.parametrize("name",["train.tar.gz.download","archive.zip.part","archive.zip.assembled"])
def test_explicit_partial_downloads_can_be_registered(case,name):
    path=case.folder/name;path.write_bytes(b"PARTIAL DATA");case.save(path,"source_archive")
    assert retention.register(case.relative(case.plan),case.relative(case.inventory))["registered_files"]==1

def test_aggregate_and_minimal_provenance_are_preserved(case):
    for name,value in (("provenance.json",{"format":"training-provenance-v1","assets":[]}), ("summary.json",{"precision":.5,"recall":.6}), ("model.json",{"format":"onnx","weights":"model.onnx"})):
        path=case.folder/name;path.write_text(json.dumps(value),encoding="utf-8");case.save(path,"identifiable_manifest")
        with pytest.raises(ValueError):retention.register(case.relative(case.plan),case.relative(case.inventory))
        assert path.exists()
    private=case.folder/"release.json";private.write_text(json.dumps([{"source_member":"SYNTHETIC_PLATE_FILENAME"}]),encoding="utf-8");case.save(private,"identifiable_manifest")
    assert retention.register(case.relative(case.plan),case.relative(case.inventory))["registered_files"]==1


def test_per_image_numeric_diagnostics_inherit_source_expiry_but_aggregate_hashes_are_preserved(case):
    diagnostic=case.folder/"synthetic-numeric-diagnostic.json"
    diagnostic.write_text(json.dumps({"image_sha256":"a"*64,"residuals":[.0002]}),encoding="utf-8")
    case.save(diagnostic,"per_image_prediction")
    retention.register(case.relative(case.plan),case.relative(case.inventory))
    record=next(iter(json.loads(case.inventory.read_text())["entries"].values()))
    assert record["expires_at"]=="2026-01-08T00:00:00Z"
    result=retention.purge(case.relative(case.inventory),now=datetime(2026,1,9,tzinfo=timezone.utc))
    assert result["dry_run"] and result["counts"]["would_delete"]==1 and diagnostic.exists()
    aggregate=case.folder/"aggregate-metrics.json"
    aggregate.write_text(json.dumps({"dataset_release_sha256":"a"*64,"paint_iou":.5}),encoding="utf-8")
    with pytest.raises(ValueError,match="aggregate_or_unclassified"):
        retention.eligible(aggregate,"identifiable_manifest")

def test_path_escape_unknown_root_bad_hash_and_no_private_cli_errors(case,capsys,monkeypatch):
    plan=case.save();plan["roots"]=["data"];case.plan.write_text(json.dumps(plan),encoding="utf-8")
    with pytest.raises(ValueError,match="unknown_research_root"):retention.register(case.relative(case.plan),case.relative(case.inventory))
    plan=case.save();plan["entries"][0]["path"]="../outside.jpg";case.plan.write_text(json.dumps(plan),encoding="utf-8")
    with pytest.raises(ValueError):retention.register(case.relative(case.plan),case.relative(case.inventory))
    plan=case.save();plan["entries"][0]["sha256"]="0"*64;case.plan.write_text(json.dumps(plan),encoding="utf-8")
    monkeypatch.setattr("sys.argv",["research_retention.py","register",case.relative(case.plan),"--inventory",case.relative(case.inventory)])
    assert retention.main()==2
    assert "SYNTHETIC_PRIVATE_FILENAME" not in capsys.readouterr().out

def test_inventory_lock_refuses_parallel_writers_without_modifying_sources(case):
    with retention.inventory_lock(case.inventory):
        with pytest.raises(ValueError,match="inventory_busy"):retention.register(case.relative(case.plan),case.relative(case.inventory))
    assert case.source.exists() and not case.inventory.exists()

def test_readonly_coverage_reports_unregistered_partial_download_without_private_output(case):
    retention.register(case.relative(case.plan),case.relative(case.inventory))
    partial=case.folder/"SYNTHETIC_PARTIAL.tar.gz.download";partial.write_bytes(b"partial")
    weights=case.folder/"model.onnx";weights.write_bytes(b"weights")
    result=retention.coverage(case.relative(case.plan),case.relative(case.inventory))
    assert result["read_only"] and result["counts"]["uncovered_candidates"]==1
    assert result["counts"]["preserved_files"]>=2 and result["candidate_counts_by_kind"]["source_archive"]==1
    assert "SYNTHETIC" not in json.dumps(result) and partial.exists() and weights.exists()

def test_public_aggregate_csv_jsonl_remain_preserved(case):
    for name,content in (("metrics.csv","class,tp,fp,fn\npothole,1,2,3\n"),("metrics.jsonl",'{"class":"pothole","precision":0.5}\n')):
        path=case.folder/name;path.write_text(content,encoding="utf-8");case.save(path,"identifiable_manifest")
        with pytest.raises(ValueError,match="aggregate_or_unclassified"):retention.register(case.relative(case.plan),case.relative(case.inventory))
        assert path.exists()

def test_public_license_and_aggregate_logs_preserved_but_private_media_log_registered(case):
    license=case.folder/"PUBLISHER-LICENSE.txt";license.write_text("SYNTHETIC LICENSE",encoding="utf-8");case.save(license,"source_annotation")
    with pytest.raises(ValueError,match="preserved"):retention.register(case.relative(case.plan),case.relative(case.inventory))
    log=case.folder/"run.log";log.write_text('{"epoch":1,"loss":0.5}',encoding="utf-8");case.save(log,"identifiable_manifest")
    with pytest.raises(ValueError,match="preserved"):retention.register(case.relative(case.plan),case.relative(case.inventory))
    log.write_text('error loading "SYNTHETIC_PRIVATE_FILENAME.jpg"',encoding="utf-8");case.save(log,"identifiable_manifest")
    assert retention.register(case.relative(case.plan),case.relative(case.inventory))["registered_files"]==1


@pytest.mark.parametrize("content",[
    json.dumps({"image_sha256":"a"*64,"artifact_contract_verified":True},indent=2),
    'worker started\n'+json.dumps({"image_sha256":"a"*64})+'\nworker finished\n',
    str({"sample_id":"SYNTHETIC_PROBE","passed":False}),
])
def test_private_probe_stdout_log_inherits_expiry_without_media_filename(case,content):
    log=case.folder/"verification.log";log.write_text(content,encoding="utf-8")
    case.save(log,"identifiable_manifest")
    retention.register(case.relative(case.plan),case.relative(case.inventory))
    record=next(iter(json.loads(case.inventory.read_text())["entries"].values()))
    assert record["expires_at"]=="2026-01-08T00:00:00Z"
    result=retention.purge(case.relative(case.inventory),now=datetime(2026,1,9,tzinfo=timezone.utc))
    assert result["counts"]["would_delete"]==1 and log.exists()


def test_aggregate_stdout_hashes_and_private_field_mentions_remain_preserved(case):
    log=case.folder/"aggregate.log"
    log.write_text(json.dumps({"dataset_release_sha256":"a"*64,"checkpoint_sha256":"b"*64})+'\nimage_sha256 is omitted from this aggregate\n',encoding="utf-8")
    with pytest.raises(ValueError,match="aggregate_or_unclassified_log"):
        retention.eligible(log,"identifiable_manifest")


@pytest.mark.parametrize("suffix",[".json",".jsonl"])
def test_coverage_refuses_malformed_metadata_instead_of_counting_it_as_preserved(case,suffix):
    private=case.folder/("SYNTHETIC_BROKEN_PROBE"+suffix)
    private.write_text('{"image_sha256":',encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        retention.coverage(case.relative(case.plan),case.relative(case.inventory))
    assert private.exists() and not case.inventory.exists()


@pytest.mark.parametrize("error",[ValueError("metadata_too_large"),OSError("SYNTHETIC_PRIVATE_READ_ERROR")])
def test_coverage_does_not_treat_size_or_read_failures_as_preservation(case,monkeypatch,error):
    log=case.folder/"verification.log";log.write_text("synthetic",encoding="utf-8")
    original=retention.eligible
    def checked(path,kind):
        if path==log:raise error
        return original(path,kind)
    monkeypatch.setattr(retention,"eligible",checked)
    with pytest.raises(type(error)):
        retention.coverage(case.relative(case.plan),case.relative(case.inventory))
    assert log.exists() and not case.inventory.exists()
