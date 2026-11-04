"""Independent synthetic serial/process-roundtrip score and ordering contracts."""
from concurrent.futures import ThreadPoolExecutor
import json,os,shutil,subprocess,sys,uuid
import cv2,numpy as np,pytest
from agent.paths import ROOT,atomic_json
from scripts.compare_detector_models import score
from scripts.compare_privacy_parallel import assemble_results,private_output

@pytest.fixture
def folder():
    path=ROOT/'.cache/test-privacy-parallel'/uuid.uuid4().hex;path.mkdir(parents=True)
    yield path
    if not path.resolve().is_relative_to(ROOT.resolve()):raise ValueError('unsafe_cleanup')
    shutil.rmtree(path)

def fixture_data():
    truth=[{'sample_id':'A','class':'face','bbox_xywh_px':[0.,0.,10.,10.]},{'sample_id':'B','class':'license_plate','bbox_xywh_px':[2.,2.,10.,5.]}]
    baseline=[{'sample_id':'A','class':'face','bbox_xywh_px':[0.,0.,10.,10.],'confidence':.5},
        {'sample_id':'A','class':'face','bbox_xywh_px':[0.,0.,10.,10.],'confidence':.4},
        {'sample_id':'B','class':'license_plate','bbox_xywh_px':[2.,2.,10.,5.],'confidence':.9},
        {'sample_id':'C','class':'face','bbox_xywh_px':[1.0000001192092896,0.,10.,10.],'confidence':.5}]
    candidate=[dict(row) for row in baseline];candidate[0]=dict(candidate[0],bbox_xywh_px=[20.,20.,3.,3.]);candidate[1]=dict(candidate[1],bbox_xywh_px=[20.,20.,3.,3.])
    return truth,{'baseline':baseline,'candidate':candidate},['face','license_plate']

def test_serial_vs_two_subprocess_roundtrip_preserves_all_metrics_and_order(folder):
    truth,predictions,classes=fixture_data();metadata={name:{'operating_confidence':.35} for name in predictions};timings={name:[.1,.2,.3] for name in predictions}
    # This is the old serial scoring expression, independently assembled.
    expected={name:{**metadata[name],'ap_metrics':score(truth,p,classes,.01),'operating_metrics':score(truth,p,classes,.35),'fixed_035_reference':score(truth,p,classes,.35),
        'latency_seconds':{'median':float(np.median(timings[name])),'p95':float(np.percentile(timings[name],95))}} for name,p in predictions.items()}
    def child(name):
        source=folder/(name+'-input.json');output=folder/(name+'-result.json');atomic_json(source,{'predictions':predictions[name],'timings':timings[name]})
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
        process=subprocess.run([sys.executable,str(ROOT/'scripts/compare_privacy_parallel.py'),'contract-worker','--plan',str(source),'--output',str(output)],cwd=ROOT,env=env,capture_output=True,text=True)
        assert process.returncode==0,process.stderr
        return name,json.loads(output.read_text(encoding='utf-8'))
    # Submit reversed model order; per-model order must remain the serial order.
    with ThreadPoolExecutor(max_workers=2) as executor:results=dict(executor.map(child,['candidate','baseline']))
    assert all(result['synthetic_contract'] is True for result in results.values())
    assert {name:result['predictions'] for name,result in results.items()}==predictions
    actual=assemble_results(truth,{name:results[name]['predictions'] for name in metadata},classes,metadata,{name:results[name]['timings'] for name in metadata})
    assert actual==expected
    assert actual['baseline']['ap_metrics']['map50']==1.
    assert actual['candidate']['ap_metrics']['classes']['face']['tp']==0
    assert actual['baseline']['operating_metrics']['classes']['face']['fp']==2

@pytest.mark.parametrize('shape',[(123,341),(720,1280),(1080,1920)])
def test_resize_pixels_are_exact_across_worker_thread_counts(shape):
    rng=np.random.default_rng(23);image=rng.integers(0,256,(*shape,3),dtype=np.uint8);previous=cv2.getNumThreads()
    try:
        cv2.setNumThreads(1);serial=cv2.resize(image,(1920,1080))
        cv2.setNumThreads(2);parallel=cv2.resize(image,(1920,1080))
        assert np.array_equal(serial,parallel)
    finally:cv2.setNumThreads(previous)

@pytest.mark.parametrize('path',['docs/reports/predictions.json','data/elsewhere/predictions.json','data/training/refinement-evaluation/../../escape.json'])
def test_private_outputs_cannot_be_public(path):
    with pytest.raises(ValueError):private_output(path)

def test_json_float_roundtrip_does_not_move_half_iou_boundary():
    truth=[{'sample_id':'A','class':'face','bbox_xywh_px':[0.,0.,10.,10.]}]
    found=[{'sample_id':'A','class':'face','bbox_xywh_px':[0.,0.,20.,10.],'confidence':.3333333432674408}]
    before=score(truth,found,['face'],.01);after=score(json.loads(json.dumps(truth)),json.loads(json.dumps(found)),['face'],.01)
    assert before==after and before['classes']['face']['tp']==1
