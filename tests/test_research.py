"""Offline public-data experimentation never impersonates deployment approvals."""
from datetime import datetime,timezone,timedelta
from pathlib import Path
import uuid
from agent.dataset import validate_release,validate_research_release
from agent.paths import ROOT,atomic_json,sha256


def research_sample():
    folder=ROOT/".cache/research-tests"/uuid.uuid4().hex;folder.mkdir(parents=True)
    image=folder/"fixture.dat";image.write_bytes(b"explicit synthetic validator fixture")
    evidence=folder/"license.json"
    atomic_json(evidence,{"research_allowed":True,"license_id":"synthetic-test-only","source_url":"https://example.invalid/test","license_url":"https://example.invalid/test-license"})
    row={"sample_id":str(uuid.uuid4()),"source_id":"synthetic-test","source_hash":sha256(image),"leakage_group_id":"synthetic-group","path":str(image),"sha256":sha256(image),"annotation_version":"synthetic",
         "split":"train","privacy_status":"restricted_research","qc_status":"publisher_annotations_machine_checked","release_scope":"offline_research","human_privacy_reviewed":False,"not_for_deployment":True,
         "license_evidence_path":str(evidence),"license_evidence_sha256":sha256(evidence),"allowed_use":"validator fixture only","expires_at":(datetime.now(timezone.utc)+timedelta(days=1)).isoformat()}
    return row,evidence


def test_public_research_does_not_pass_deployment_approval_gate():
    row,_=research_sample()
    assert validate_research_release([row])["valid"]
    assert not validate_release([row])["valid"]


def test_research_requires_unchanged_license_evidence_and_explicit_nonapproval():
    row,evidence=research_sample();evidence.write_bytes(b"changed")
    assert not validate_research_release([row])["valid"]
    row,_=research_sample();row["human_privacy_reviewed"]=True
    assert not validate_research_release([row])["valid"]


def test_research_still_rejects_duplicate_source_across_holdouts():
    row,_=research_sample()
    other=row|{"sample_id":str(uuid.uuid4()),"split":"test"}
    assert any(error["code"]=="split_leakage" for error in validate_research_release([row,other])["errors"])


def test_calibrated_paint_threshold_changes_real_adapter_mask_without_changing_default_ties():
    import numpy as np
    from agent.models import OnnxModel
    class Net:
        def setInput(self,blob):self.blob=blob
        def forward(self):
            output=np.zeros((1,2,16,16),np.float32)
            output[0,1,:8]=np.log(.8/.2)
            return output
    model=OnnxModel.__new__(OnnxModel)
    model.manifest={"task":"lane","input_size":[16,16]};model.net=Net()
    image=np.zeros((16,16,3),np.uint8)
    assert int(model.predict(image).sum())==128
    model.manifest["paint_threshold"]=.9
    assert not model.predict(image).any()
