"""Independent synthetic optimizer-step contract; not a dataset training run."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import workspace_path,sha256,atomic_json
from scripts.probe_redaction_rescue import OUTPUT,CHECKPOINT,CHECKPOINT_SHA,synthetic_image

def run():
    import torch,numpy as np
    from agent.redaction_rescue_network import build_redaction_rescue_network,load_frozen_legacy
    torch.set_num_threads(2);torch.manual_seed(20261108)
    path=workspace_path(CHECKPOINT,exists=True)
    if sha256(path)!=CHECKPOINT_SHA:raise ValueError('legacy_checkpoint_changed')
    model=build_redaction_rescue_network().cpu();load_frozen_legacy(model,torch.load(path,map_location='cpu',weights_only=True)['model']);model.train()
    legacy={k:v.detach().clone() for k,v in model.state_dict().items() if k.startswith(('face.','plate.'))}
    new={k:p.detach().clone() for k,p in model.named_parameters() if k.startswith(('rescue_projection.','rescue_head.'))}
    before_labels=torch.tensor([[[0.,480.,480.,32.,32.]]]);labels=before_labels.clone()
    image=synthetic_image()[:960,:960];pixels=torch.from_numpy(np.ascontiguousarray(image.transpose(2,0,1)[None],dtype=np.float32))
    optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.001)
    optimizer.zero_grad();loss=model.rescue_loss(pixels,labels)[0];loss.backward()
    finite=all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    legacy_grad=sum(p.grad is not None for name,p in model.named_parameters() if name.startswith(('face.','plate.')))
    nonzero=sum(p.grad is not None and bool(torch.count_nonzero(p.grad)) for name,p in model.named_parameters() if name in new)
    optimizer.step();state=model.state_dict()
    changed=sum(not torch.equal(value,state[name]) for name,value in new.items())
    preserved=all(torch.equal(value,state[name]) for name,value in legacy.items())
    modes=not model.face.training and not model.plate.training and all(not m.training for group in (model.face,model.plate) for m in group.modules())
    folder=workspace_path(OUTPUT,exists=True);baseline=np.load(folder/'baseline-torch.npy',allow_pickle=False)
    full=torch.from_numpy(np.ascontiguousarray(synthetic_image().transpose(2,0,1)[None],dtype=np.float32));model.eval()
    with torch.inference_mode():after=model(full).numpy()
    prefix_after_update=after[:,:102900].tobytes()==baseline.tobytes()
    passed=bool(torch.isfinite(loss) and finite and legacy_grad==0 and nonzero>0 and changed>0 and preserved and modes and torch.equal(labels,before_labels) and prefix_after_update)
    if sha256(path)!=CHECKPOINT_SHA:raise ValueError('original_checkpoint_modified')
    report={'passed':passed,'source_sha256':sha256(Path(__file__)),'factory_sha256':sha256(ROOT/'agent/redaction_rescue_network.py'),
            'legacy_checkpoint_sha256':CHECKPOINT_SHA,'legacy_gradient_tensors':legacy_grad,'legacy_parameters_and_BN_buffers_preserved':preserved,
            'legacy_eval_modes_preserved':modes,'new_trainable_parameter_tensors':len(new),'new_nonzero_gradient_parameter_tensors':nonzero,
            'new_parameter_tensors_changed':changed,'labels_unchanged':torch.equal(labels,before_labels),'synthetic_optimizer_steps':1,
            'legacy_prefix_after_new_parameter_update_byte_exact':prefix_after_update,
            'actual_dataset_training_run':False,'dataset_images_read':0,'TEST_images':0,'GPU_used':False}
    atomic_json(folder/'gradient-contract.json',report)
    aggregate=json.loads((folder/'aggregate.json').read_text(encoding='utf-8'));aggregate['new_parameter_update_contract']=report
    aggregate['all_contracts_passed']=aggregate['all_contracts_passed'] and passed
    atomic_json(folder/'aggregate.json',aggregate);atomic_json(workspace_path('docs/reports/REDACTION-RESCUE-CPU-PROTOTYPE.json'),aggregate)
    print(json.dumps(report));return 0 if passed else 2

if __name__=='__main__':raise SystemExit(run())
