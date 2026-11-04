"""CPU/synthetic-only architecture probe. No dataset, training run or TEST access."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import workspace_path,sha256,atomic_json

OUTPUT='data/training/redaction-rescue-cpu-prototype-v8'
CHECKPOINT='data/training/redaction-specialists-v4-efficient/best.pt'
CHECKPOINT_SHA='6c985f7e3afb0fa061cbb82d7ec7b304eb5f9fbfcd0cf9f775dbab28873c82ad'
BASE_MANIFEST='data/training/redaction-inference-options-v4/option-tiled/model.json'

def synthetic_image():
    import numpy as np
    y,x=np.indices((1920,1920),dtype=np.uint32)
    return np.stack(((x*3+y*5)%256,(x*7+y*11)%256,(x*13+y*17)%256),-1).astype(np.uint8)

def torch_probe():
    import torch,numpy as np
    from agent.redaction_tiled_network import build_redaction_tiled_network
    from agent.redaction_rescue_network import build_redaction_rescue_network,load_frozen_legacy,LEGACY_PROPOSALS
    torch.set_num_threads(2);torch.manual_seed(20261108)
    output=workspace_path(OUTPUT)
    if output.exists():raise ValueError('fresh_prototype_output_required')
    path=workspace_path(CHECKPOINT,exists=True)
    if sha256(path)!=CHECKPOINT_SHA:raise ValueError('frozen_legacy_checkpoint_changed')
    state=torch.load(path,map_location='cpu',weights_only=True)['model']
    baseline=build_redaction_tiled_network().cpu().eval();baseline.load_state_dict(state,strict=True)
    candidate=build_redaction_rescue_network().cpu();transplant=load_frozen_legacy(candidate,state);candidate.eval()
    blob=np.ascontiguousarray(synthetic_image().transpose(2,0,1)[None],dtype=np.float32);tensor=torch.from_numpy(blob)
    calls={'face_backbone':0,'plate_backbone':0}
    def count(name):
        def hook(*_):calls[name]+=1
        return hook
    handles=[candidate.face.backbone.backbone.register_forward_hook(count('face_backbone')),
             candidate.plate.backbone.backbone.register_forward_hook(count('plate_backbone'))]
    with torch.inference_mode():
        old=baseline(tensor);new=candidate(tensor)
    for handle in handles:handle.remove()
    prefix_exact=old.numpy().tobytes()==new[:,:LEGACY_PROPOSALS].numpy().tobytes()
    if not prefix_exact:raise ValueError('legacy_prefix_not_byte_exact')
    if new.shape!=(1,333300,7) or calls!={'face_backbone':5,'plate_backbone':1}:raise ValueError('unexpected_proposal_or_backbone_count')
    score=(new[:,LEGACY_PROPOSALS:,4]*new[:,LEGACY_PROPOSALS:,5]).max().item()
    if not score<.01:raise ValueError('initial_rescue_not_below_floor')
    initial={k:v.detach().clone() for k,v in candidate.state_dict().items()}
    candidate.train();labels=torch.tensor([[[0.,480.,480.,32.,32.]]],dtype=torch.float32)
    tile=tensor[:,:,:960,:960].contiguous();labels_before=labels.clone()
    optimizer=torch.optim.SGD([p for p in candidate.parameters() if p.requires_grad],lr=.001)
    losses=candidate.rescue_loss(tile,labels);loss=losses[0]
    if not torch.isfinite(loss):raise ValueError('non_finite_native_rescue_loss')
    optimizer.zero_grad();loss.backward()
    old_gradient_count=sum(p.grad is not None for name,p in candidate.named_parameters() if name.startswith(('face.','plate.')))
    if old_gradient_count:raise ValueError('legacy_received_gradients')
    new_finite=all(p.grad is None or torch.isfinite(p.grad).all() for p in candidate.parameters())
    optimizer.step()
    current=candidate.state_dict()
    legacy_unchanged=all(torch.equal(value,current[name]) for name,value in initial.items() if name.startswith(('face.','plate.')))
    changed=sum(not torch.equal(value,current[name]) for name,value in initial.items() if name.startswith(('rescue_projection.','rescue_head.')))
    if not legacy_unchanged or not changed or not new_finite or not torch.equal(labels,labels_before):raise ValueError('gradient_or_frozen_state_contract_failed')
    dense=torch.zeros((1,512,5));dense[:,:,1:]=torch.tensor([480.,480.,12.,12.]);dense_before=dense.clone();observed={}
    native=candidate.rescue_head.get_losses
    def observe(*args,**kwargs):
        observed['shape']=list(args[4].shape);observed['nonempty']=int((args[4].sum(-1)>0).sum());return native(*args,**kwargs)
    candidate.rescue_head.get_losses=observe
    with torch.no_grad():dense_loss=candidate.rescue_loss(tile,dense)[0]
    candidate.rescue_head.get_losses=native
    if observed!={'shape':[1,512,5],'nonempty':512} or not torch.equal(dense,dense_before) or not torch.isfinite(dense_loss):raise ValueError('512_native_labels_not_preserved')
    candidate.load_state_dict(initial,strict=True);candidate.eval()
    with torch.inference_mode():restored=candidate(tensor)
    if restored.numpy().tobytes()!=new.numpy().tobytes():raise ValueError('initial_state_restore_not_exact')
    durations={'baseline':[],'rescue':[]}
    with torch.inference_mode():
        for _ in range(2):baseline(tensor);candidate(tensor)
        for order in [('baseline','rescue'),('rescue','baseline'),('baseline','rescue')]:
            for name in order:
                start=time.perf_counter();(baseline if name=='baseline' else candidate)(tensor);durations[name].append(time.perf_counter()-start)
    output.mkdir();np.save(output/'synthetic-input.npy',blob,allow_pickle=False)
    np.save(output/'baseline-torch.npy',old.numpy(),allow_pickle=False);np.save(output/'candidate-torch.npy',new.numpy(),allow_pickle=False)
    torch.onnx.export(candidate,tensor,str(output/'prototype.onnx'),opset_version=17,input_names=['images'],output_names=['detections'],do_constant_folding=True,dynamo=False)
    if sha256(path)!=CHECKPOINT_SHA:raise ValueError('original_checkpoint_changed')
    report={'scope':'CPU synthetic-only untrained rescue architecture prototype','source_sha256':sha256(Path(__file__)),
            'factory_sha256':sha256(ROOT/'agent/redaction_rescue_network.py'),'legacy_checkpoint_sha256':CHECKPOINT_SHA,
            'legacy_prefix_byte_exact':prefix_exact,'legacy_prefix_proposals':LEGACY_PROPOSALS,'total_proposals':333300,
            'rescue_initial_max_confidence':score,'backbone_calls':calls,'transplant':transplant,'legacy_gradient_count':old_gradient_count,
            'legacy_parameters_and_BN_unchanged_after_synthetic_step':legacy_unchanged,'new_state_tensors_changed_after_synthetic_step':changed,
            'native_synthetic_loss_finite':True,'labels512_preserved_unchanged':True,'native_observed_GT':512,'initial_state_restored':True,
            'parameters':{'legacy':sum(p.numel() for n,p in candidate.named_parameters() if n.startswith(('face.','plate.'))),
                          'new':sum(p.numel() for n,p in candidate.named_parameters() if n.startswith(('rescue_projection.','rescue_head.')))},
            'cpu_seconds':{name:{'median':float(np.median(values)),'samples':len(values)} for name,values in durations.items()},
            'torch_version':torch.__version__,'CPU_threads':2,'GPU_used':False,'dataset_images_read':0,'TEST_images':0,'actual_training_run':False,
            'weights_or_original_source_modified':False,'production_approved':False,'prototype_onnx_sha256':sha256(output/'prototype.onnx')}
    atomic_json(output/'torch-contract.json',report);print(json.dumps(report))

def app_probe():
    import numpy as np,onnxruntime as ort
    from scripts.verify_trained_models import compare,postprocess_detection,compare_detections
    output=workspace_path(OUTPUT,exists=True);report=json.loads((output/'torch-contract.json').read_text(encoding='utf-8'))
    if sha256(output/'prototype.onnx')!=report['prototype_onnx_sha256']:raise ValueError('prototype_graph_changed')
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    options.add_session_config_entry('session.intra_op.allow_spinning','0');options.add_session_config_entry('session.inter_op.allow_spinning','0')
    session=ort.InferenceSession(str(output/'prototype.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    if session.get_providers()!=['CPUExecutionProvider']:raise ValueError('CPU_only_probe_required')
    blob=np.load(output/'synthetic-input.npy',allow_pickle=False);raw=session.run(None,{'images':blob})[0]
    expected=np.load(output/'candidate-torch.npy',allow_pickle=False);baseline=np.load(output/'baseline-torch.npy',allow_pickle=False)
    numerical=compare(raw,expected,.001,.0001)
    artifact=json.loads(workspace_path(BASE_MANIFEST,exists=True).read_text(encoding='utf-8'));image=synthetic_image()
    base_post=postprocess_detection(artifact,baseline,image);torch_post=postprocess_detection(artifact,expected,image);ort_post=postprocess_detection(artifact,raw,image)
    initial_exact=base_post==torch_post
    post=compare_detections(ort_post,torch_post)
    report.update(onnxruntime_version=ort.__version__,onnx_raw_parity=numerical,initial_legacy_NMS_exact=initial_exact,
                  app_postprocess_parity=post,all_contracts_passed=bool(numerical['passed'] and initial_exact and post['passed']),
                  timing_limits='Single synthetic input, shared CPU host during other GPU training; not camera or pipeline FPS.',
                  accuracy_claim=False,trained_rescue_precision_equivalent_claim=False)
    atomic_json(output/'aggregate.json',report);atomic_json(workspace_path('docs/reports/REDACTION-RESCUE-CPU-PROTOTYPE.json'),report)
    print(json.dumps(report));return 0 if report['all_contracts_passed'] else 2

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['torch','app']);args=parser.parse_args()
    try:raise SystemExit(torch_probe() if args.action=='torch' else app_probe())
    except Exception as error:
        print(json.dumps({'status':'CPU_prototype_stopped','error_type':type(error).__name__}),file=sys.stderr);raise SystemExit(1)
