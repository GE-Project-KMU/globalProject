"""Small offline tests for new fixture boundaries; live workflow is a separate probe."""
import json
import uuid
import cv2
import numpy as np
import pytest
from agent.paths import ROOT,atomic_json
from agent.models import readiness,ModelNotReady
from scripts.probe_synthetic_workflow import fixtures,SyntheticOnnxModel,cleanup

@pytest.fixture
def assets():
    folder=ROOT/'.cache/synthetic-workflow-tests'/str(uuid.uuid4())
    video,bundle=fixtures(folder)
    return folder,video,bundle

def test_real_ort_graphs_contract_and_production_rejection(assets):
    folder,video,bundle=assets
    cap=cv2.VideoCapture(str(video));ok,image=cap.read();cap.release()
    assert ok and image.shape==(48,128,3)
    outputs={task:SyntheticOnnxModel(bundle[task]).predict(image[:,:64]) for task in ('pothole','lane','redaction')}
    assert outputs['pothole'][0]['class']=='pothole'
    assert outputs['redaction'][0]['class']=='face'
    assert outputs['lane'].shape==(48,64) and set(np.unique(outputs['lane']))<={0,1}
    for task in ('pothole','lane','redaction'):
        state=readiness(bundle[task])
        assert not state['ready'] and 'target_training_provenance_required' in state['reason']

@pytest.mark.parametrize('changes',[{'fixture_kind':'real-model'},{'target_trained':True},{'production_approved':True}])
def test_fixture_adapter_rejects_real_or_approved_model_claims(assets,changes):
    folder,_,bundle=assets
    path=ROOT/bundle['pothole'];manifest=json.loads(path.read_text());manifest.update(changes);atomic_json(path,manifest)
    with pytest.raises(ValueError,match='explicit_synthetic_fixture'):
        SyntheticOnnxModel(path)

def test_fixture_weights_cannot_be_replaced_by_external_weights(assets):
    folder,_,bundle=assets
    (folder/'pothole.onnx').write_bytes(b'not the frozen graph')
    with pytest.raises(ValueError,match='frozen_synthetic_graph'):
        SyntheticOnnxModel(bundle['pothole'])

@pytest.mark.parametrize('identity,expected',[
    ({'org_code':'demo'},'owned_synthetic_tenant'),
    ({'org_code':'workflow-test-pretend'},'owned_synthetic_directory'),
])
def test_cleanup_refuses_unowned_resources_before_any_http(assets,identity,expected):
    folder,_,_=assets
    receipt=folder/'receipt.json';atomic_json(receipt,{'identity':identity,'folder':folder.relative_to(ROOT).as_posix()})
    with pytest.raises(ValueError,match=expected):cleanup(receipt)
