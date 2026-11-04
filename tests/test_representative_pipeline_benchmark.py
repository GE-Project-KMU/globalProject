"""Synthetic CPU contracts only. Custom models never count as GPU evidence."""
from datetime import datetime,timedelta,timezone
from pathlib import Path
import hashlib,json,uuid
import cv2
import numpy as np
import pytest
from scripts import benchmark_representative_pipeline as b


@pytest.fixture
def folder():
    path=b.ROOT/'.cache/representative-benchmark-tests'/uuid.uuid4().hex;path.mkdir(parents=True);return path


def deadline():return (datetime.now(timezone.utc)+timedelta(days=1)).isoformat()


@pytest.mark.parametrize('approval',[False,None,1,'yes'])
def test_no_approval_means_no_metadata_or_image_read(monkeypatch,approval):
    monkeypatch.setattr(b,'validate_execution',lambda *a:pytest.fail('read metadata before approval'))
    monkeypatch.setattr(b,'prepare_frames',lambda *a:pytest.fail('read images before approval'))
    with pytest.raises(ValueError,match='approval_required'):b.run('not-read',approved=approval)


def test_missing_execution_digest_refused_before_metadata(monkeypatch):
    monkeypatch.setattr(b,'validate_execution',lambda *a:pytest.fail('metadata read without digest'))
    with pytest.raises(ValueError,match='pinned_execution'):b.run('not-read',approved=True)


def test_synthetic_decode_and_letterbox_bounded_without_changing_capture_date(folder,monkeypatch):
    monkeypatch.setattr(b.g,'public_case_path',b.workspace_path)
    rows=[];cap=deadline()
    for index in range(32):
        path=folder/f'{index}.png';assert cv2.imwrite(str(path),np.full((20,40,3),index,np.uint8))
        rows.append({'path':path,'sha256':b.sha256(path),'expires_at':cap})
    frames,details,seconds=b.prepare_frames(rows,cap)
    assert len(frames)==32 and all(frame.shape==(720,1280,3) for frame in frames)
    assert np.all(frames[1][:640]==1) and np.all(frames[1][640:]==114)
    assert sum(frame.nbytes for frame in frames)<=b.MAX_PREPARED_BYTES and seconds>0
    assert all('captured_at' not in item for item in details)


def test_hash_mismatch_and_expiry_precede_decode(folder,monkeypatch):
    monkeypatch.setattr(b.g,'public_case_path',b.workspace_path)
    path=folder/'invalid.png';path.write_bytes(b'not-an-image')
    row={'path':path,'sha256':'a'*64,'expires_at':deadline()}
    monkeypatch.setattr(b.cv2,'imdecode',lambda *a:pytest.fail('decoded changed source'))
    with pytest.raises(ValueError,match='bytes_changed'):b.prepare_frames([row]*32,deadline())
    with pytest.raises(ValueError,match='source_material_expired'):b.prepare_frames([row]*32,'2000-01-01T00:00:00Z')


def test_source_and_prepared_memory_bounds_are_fail_closed(folder,monkeypatch):
    monkeypatch.setattr(b.g,'public_case_path',b.workspace_path)
    path=folder/'bounded.png';assert cv2.imwrite(str(path),np.zeros((10,20,3),np.uint8))
    row={'path':path,'sha256':b.sha256(path),'expires_at':deadline()}
    monkeypatch.setattr(b,'MAX_SOURCE_PIXELS',1)
    with pytest.raises(ValueError,match='pixel_budget'):b.prepare_frames([row]*32,deadline())
    monkeypatch.setattr(b,'MAX_PREPARED_BYTES',1)
    with pytest.raises(ValueError,match='frame_budget'):b.prepare_frames([row]*32,deadline())


class ContractModel:
    def __init__(self,task,counts,fail=False,cleanup_fail=False):
        self.manifest={'task':task};self.counts=counts;self.task=task;self.closed=False;self.fail=fail;self.cleanup_fail=cleanup_fail
    def predict(self,image):
        self.counts[self.task]+=1
        if self.fail:raise ValueError('primary_inference_failure')
        if self.task=='lane':return np.ones(image.shape[:2],np.uint8)
        return [] if self.task=='redaction' else [{'class':'pothole','bbox_xywh_px':[4.,4.,8.,8.],'confidence':.9}]
    def close(self):
        self.closed=True
        if self.cleanup_fail:raise PermissionError('cleanup_failure')


def test_real_pipeline_abba32_cpu_doubles_preserve_bytes_order_quality_and_flush(folder,monkeypatch):
    frame=np.zeros((720,1280,3),np.uint8);frames=[frame]*32;counts={task:0 for task in ('pothole','lane','redaction')};made=[];kwargs=[]
    def load(*args,**unused):
        models={task:ContractModel(task,counts) for task in counts};made.extend(models.values());return models
    real=b.pipeline.analyze_video
    def observed(*args,**values):kwargs.append(values);return real(*args,**values)
    monkeypatch.setattr(b.pipeline,'analyze_video',observed)
    result=b.execute_abba(frames,folder/'abba',{task:task for task in counts},'cpu-test-double',deadline(),model_loader=load,require_cuda=False)
    assert result['opaque_png_bytes_same'] and result['observation_order_quality_tracking_lane_same']
    assert [row['io_profile'] for row in result['runs']]==['default','overlap','overlap','default']
    assert all(value==4*(32+5) for value in counts.values()) and all(model.closed for model in made)
    assert all(values['source_captured_at'] is None for values in kwargs)
    assert all(row['actual_native_cuda_required'] is False for row in result['runs'])


def test_inference_failure_is_preserved_and_all_models_close(folder):
    counts={task:0 for task in ('pothole','lane','redaction')};made=[]
    def load(*args,**unused):
        models={task:ContractModel(task,counts,fail=task=='lane',cleanup_fail=task=='pothole') for task in counts}
        made.extend(models.values());return models
    with pytest.raises(ValueError,match='primary_inference_failure') as error:
        b.execute_abba([np.zeros((720,1280,3),np.uint8)]*32,folder/'failure',{task:task for task in counts},'cpu-test-double',deadline(),model_loader=load,require_cuda=False)
    assert all(model.closed for model in made) and any('cleanup' in note.lower() for note in error.value.__notes__)


def test_execution_binding_source_environment_digest_and_one_output(folder,monkeypatch):
    monkeypatch.setattr(b,'PERF_ROOT',folder)
    profile={'path':'synthetic-profile','sha256':'a'*64}
    representative={'registered_profile':profile,'source_expires_at':deadline()}
    path=folder/'representative.json';path.write_text(json.dumps(representative))
    monkeypatch.setattr(b,'validated_representative',lambda *a:(representative,{},{}))
    monkeypatch.setattr(b,'model_source_cap',lambda *a:({},datetime.fromisoformat(representative['source_expires_at'])-timedelta(hours=1)))
    monkeypatch.setattr(b,'source_pins',lambda:{'synthetic_source':'a'*64})
    execution=folder/'execution.json';output=folder/'new-output'
    value=b.freeze_execution(path,execution,output)
    assert value['images_read']==0 and value['gpu_sessions_started']==0
    assert b.validate_execution(execution)[0]['source_captured_at'] is None
    assert datetime.fromisoformat(b.validate_execution(execution)[0]['source_expires_at'])<datetime.fromisoformat(representative['source_expires_at'])
    monkeypatch.setattr(b,'source_pins',lambda:{'synthetic_source':'b'*64})
    with pytest.raises(ValueError,match='source_or_environment'):b.validate_execution(execution)


def test_reused_output_refused_before_any_public_image_read(folder,monkeypatch):
    monkeypatch.setattr(b,'PERF_ROOT',folder)
    execution=folder/'binding.json';execution.write_text('{}');output=folder/'already-used';output.mkdir()
    monkeypatch.setattr(b,'validate_execution',lambda *a:({'output':str(output)},None,None,None))
    monkeypatch.setattr(b,'admitted_host',lambda:None)
    monkeypatch.setattr(b,'prepare_frames',lambda *a:pytest.fail('read images for reused output'))
    with pytest.raises(ValueError,match='already_used'):b.run(execution,execution_sha256=b.sha256(execution),approved=True)


def test_expiry_after_first_abba_run_stops_before_next_model_load(folder,monkeypatch):
    count=0;made=[];counts={task:0 for task in ('pothole','lane','redaction')};real=b.source_deadline
    def cap(value,**kwargs):
        nonlocal count
        count+=1
        if count>=34:raise ValueError('source_material_expired')
        return real(value,**kwargs)
    def load(*a,**k):
        models={task:ContractModel(task,counts) for task in counts};made.extend(models.values());return models
    monkeypatch.setattr(b,'source_deadline',cap)
    with pytest.raises(ValueError,match='source_material_expired'):
        b.execute_abba([np.zeros((720,1280,3),np.uint8)]*32,folder/'expired',{task:task for task in counts},'cpu-test-double',deadline(),model_loader=load,require_cuda=False)
    assert len(made)==3 and all(model.closed for model in made)


def test_original_model_provenance_caps_are_pinned_and_never_invented(folder,monkeypatch):
    monkeypatch.setattr(b,'blocked_identities',lambda:set())
    plan={'models':{}};cap=datetime.now(timezone.utc)+timedelta(hours=4)
    for task in ('pothole','lane','redaction'):
        proof=folder/(task+'-provenance.json');proof.write_text(json.dumps({'assets':[{'expires_at':cap.isoformat()}]}))
        manifest=folder/(task+'-model.json');manifest.write_text(json.dumps({'training_provenance':str(proof)}))
        plan['models'][task]={'manifest':b.g.pointer(manifest)}
    pointers,value=b.model_source_cap(plan)
    assert value==cap and set(pointers)=={'pothole','lane','redaction'}
    proof.write_text('{}')
    with pytest.raises(ValueError,match='nonempty_model_source_assets_required'):b.model_source_cap(plan)
    proof.write_text(json.dumps({'assets':[{}]}))
    with pytest.raises(ValueError,match='original_source_expiry_required'):b.model_source_cap(plan)


def test_model_revocation_blocks_before_image_preparation(folder,monkeypatch):
    monkeypatch.setattr(b,'blocked_identities',lambda:{'revoked-identity'})
    proof=folder/'revoked-provenance.json';proof.write_text(json.dumps({'assets':[{'identity_hashes':['revoked-identity'],'expires_at':deadline()}]}))
    manifest=folder/'revoked-model.json';manifest.write_text(json.dumps({'training_provenance':str(proof)}))
    with pytest.raises(ValueError,match='source_revoked'):b.model_source_cap({'models':{'pothole':{'manifest':b.g.pointer(manifest)}}})


def test_unsupported_host_and_actual_rss_limit_fail_closed(monkeypatch):
    monkeypatch.setattr(b.platform,'system',lambda:'UNSUPPORTED')
    with pytest.raises(ValueError,match='host_required'):b.admitted_host()
    monkeypatch.setattr(b,'process_rss',lambda pid:b.MAX_MAIN_RSS_BYTES+1)
    with pytest.raises(ValueError,match='rss_budget'):b.guarded_rss()
