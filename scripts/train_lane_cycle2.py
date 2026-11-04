"""Cycle2: recording-diverse full-context paint candidate; no TEST fitting."""
import argparse,copy,json,math,random,sys,time
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import workspace_path,atomic_json,sha256
from agent.dataset import validate_research_release
from agent.governance import begin_training,register_asset,link_model
from agent.lane_refinement_network import make_lane_refinement_network,load_legacy_warmstart,ARCHITECTURE
from agent.lane_network import freeze_batch_norm_statistics
from agent.models import SUPPORTED_CPU_PROFILE
from scripts.improve_lane import training_patch,validation_scores as original_validation_scores,MEAN,STD
from agent.train_lane import prepare_pair
BASE=ROOT/'data/training/lane-refinement-v4/lane_best.pt'
DEV=ROOT/'data/public/a2d2-cycle2-v5/development.json';RELEASE=DEV.parent/'release.json'
OUT=ROOT/'data/training/lane-cycle2-v5'

def paint_loss(logits,labels):
    import torch
    from torch.nn import functional as F
    valid=labels!=255
    if not valid.any():return logits.sum()*0
    all_ce=F.cross_entropy(logits,labels,weight=torch.tensor([1.,4.],device=logits.device),ignore_index=255,reduction='none')[valid]
    ce=torch.topk(all_ce,max(1,int(.25*all_ce.numel())),sorted=False).values.mean()
    prob=torch.softmax(logits.float(),1)[:,1];target=(labels==1).float()
    usable_prob=prob*valid
    dice=1-(2*(usable_prob*target).sum()+1)/(usable_prob.sum()+target.sum()+1)
    p=prob[:,None];t=target[:,None];inside=(F.avg_pool2d(valid[:,None].float(),3,1,1)==1).float()
    edge_p=(F.max_pool2d(p,3,1,1)+F.max_pool2d(-p,3,1,1))*inside
    edge_t=(F.max_pool2d(t,3,1,1)+F.max_pool2d(-t,3,1,1))*inside
    boundary=1-(2*(edge_p*edge_t).sum()+1)/(edge_p.sum()+edge_t.sum()+1)
    return .45*ce+.45*dice+.1*boundary

def full_context_sample(row):
    if random.random()<.5:
        pixels,labels=prepare_pair(row['path'],row['mask_path'],size=1024)
        rgb=pixels.transpose(1,2,0)*STD+MEAN
        rgb=np.clip((rgb-.5)*random.uniform(.8,1.2)+.5,0,1)
        rgb=np.clip(rgb*random.uniform(.85,1.15),0,1)
        pixels=((rgb-MEAN)/STD).transpose(2,0,1).copy()
        if random.random()<.5:pixels=pixels[:,:,::-1].copy();labels=labels[:,::-1].copy()
        return pixels,labels
    return training_patch(row,1024,1024)

def validation_scores(model,rows,size,torch):
    # Give the two development recording distributions equal selection weight.
    groups={name:[r for r in rows if r['recording']==name] for name in sorted({r['recording'] for r in rows})}
    records={name:original_validation_scores(model,items,size,torch)[1] for name,items in groups.items()}
    results=[]
    for index in range(len(next(iter(records.values())))):
        selected=[scores[index] for scores in records.values()]
        confusion={key:sum(v['confusion'][key] for v in selected) for key in selected[0]['confusion']}
        results.append({'paint_threshold':selected[0]['paint_threshold'],'paint_iou':float(np.mean([v['paint_iou'] for v in selected])),
            'selection_metric':'mean_recording_IoU','aggregate_paint_iou':confusion['tp']/(confusion['tp']+confusion['fp']+confusion['fn']) if confusion['tp']+confusion['fp']+confusion['fn'] else 0.,'confusion':confusion,'recording_IoUs':[v['paint_iou'] for v in selected]})
    return max(results,key=lambda v:(v['paint_iou'],-abs(v['paint_threshold']-.5))),results

def export_fp64_mean(model,weights,size=1024):
    import torch
    from torch import nn
    candidate=copy.deepcopy(model).cpu().eval();count=0
    class Mean(nn.Module):
        def forward(self,x):return x.double().mean((2,3),keepdim=True).to(x.dtype)
    def replace(module):
        nonlocal count
        for name,child in list(module.named_children()):
            if isinstance(child,nn.AdaptiveAvgPool2d) and child.output_size in (1,(1,1)):
                setattr(module,name,Mean());count+=1
            else:replace(child)
    replace(candidate)
    if count!=9:raise ValueError('expected_nine_global_pool_sites')
    for name,value in model.state_dict().items():
        if not torch.equal(value.cpu(),candidate.state_dict()[name]):raise ValueError('export_learned_state_changed')
    class Output(nn.Module):
        def __init__(self,network):super().__init__();self.network=network
        def forward(self,x):return self.network(x)['out']
    torch.onnx.export(Output(candidate),torch.zeros(1,3,size,size),str(weights),opset_version=17,input_names=['images'],output_names=['logits'],dynamo=False)
    return count

def plan():
    rows=json.loads(DEV.read_text(encoding='utf-8'));val=[r for r in rows if r['split']=='validation'];groups={r['recording'] for r in val}
    if len(groups)!=2 or min(sum(r['recording']==g for r in val) for g in groups)<20:raise ValueError('two_supported_development_recordings_required')
    return {'format':'lane-cycle2-plan-v1','candidate_count':1,'architecture':ARCHITECTURE,'additional_parameters':20114,'validation_count':len(val),'validation_recording_count':len(groups),'training_count':sum(r['split']=='train' for r in rows),
        'warmstart_checkpoint_sha256':sha256(BASE),'development_sha256':sha256(DEV),'release_sha256':sha256(RELEASE),
        'fresh_manifest_sha256':sha256(DEV.parent/'fresh-test.json'),'training_code_sha256':sha256(Path(__file__)),
        'data_preparation_source_sha256':sha256(ROOT/'scripts/cycle2_public_data.py'),'VAL_audit_sha256':sha256(ROOT/'docs/reports/NEXTCYCLE-VAL-ERROR-AUDIT.json'),'factory_sha256':sha256(ROOT/'agent/lane_refinement_network.py'),'shared_patch_metrics_source_sha256':sha256(ROOT/'scripts/improve_lane.py'),
        'seed':20261106,'max_epochs':20,'min_epochs':6,'patience':6,'min_delta':.001,'batch_size':2,'training_patch_size':1024,'input_size':1024,
        'optimizer':'AdamW wd1e-4 backbone1e-5 coarse_classifier1e-4 residual_decoder1e-3','scheduler':'2epoch linear warmup; cosine to10% over20 epochs',
        'loss':'.45 top25% valid weightedCE([1,4]) + .45 foregroundDice + .1 boundaryDice; ignore255 and invalid neighborhoods excluded',
        'augmentation':'50% exact fullframe1024 letterbox + photometric/hflip;50% context1024 crop scale.75..1.5 paint-centred75% + photometric/hflip','EMA':.99,'BN_running_statistics':'frozen, affine trained',
        'checkpoint_selection':'maximum mean-recording original-resolution FP32 VAL IoU (old100+new80), frozen14threshold grid; include warmstart epoch0; VAL only',
        'threshold_grid':[.1,.2,.3,.4,.5,.6,.7,.8,.85,.9,.925,.95,.975,.99],
        'fresh_evaluation':'new120 target images from two separately frozen unseen recordings after checkpoint/threshold/numeric gate frozen; baseline and candidate once',
        'regression':'old TEST cohorts only separately labelled regression; never fitting or selection','old_VAL_role':'reused development only, not independent final accuracy',
        'source_licenses':['CC-BY-ND-4.0'],'existing_expiries_extended':False,'research_only':True,'production_approved':False,'user_recordings_used':False,
        'time_budget_estimate_minutes':[50,80],'gpu_max_total_gib':9,'gpu_start_requires_root_slot':True}

def train():
    import torch
    if not torch.cuda.is_available():raise ValueError('cuda_training_required')
    if OUT.exists():raise ValueError('fresh_training_directory_required')
    rows=json.loads(DEV.read_text(encoding='utf-8'))
    if any(r['split'] not in ('train','validation') for r in rows):raise ValueError('test_in_development')
    if not validate_research_release(rows)['valid']:raise ValueError('development_invalid')
    for r in rows:
        if sha256(workspace_path(r['mask_path'],exists=True))!=r['mask_sha256']:raise ValueError('mask_changed')
    frozen=plan();public=ROOT/'docs/reports/LANE-CYCLE2-PLAN.json'
    if not public.exists() or json.loads(public.read_text(encoding='utf-8'))!=frozen:raise ValueError('public_pretraining_plan_must_match')
    torch.set_num_threads(3);cv2.setNumThreads(1);random.seed(frozen['seed']);np.random.seed(frozen['seed']);torch.manual_seed(frozen['seed']);torch.cuda.manual_seed_all(frozen['seed']);torch.backends.cudnn.benchmark=True
    training=[r for r in rows if r['split']=='train'];validation=[r for r in rows if r['split']=='validation']
    pp,p=begin_training(OUT,RELEASE,rows,'lane');p['research_only']=True;p['assets']=[register_asset(p,r,[]) for r in rows];atomic_json(pp,p)
    atomic_json(OUT/'training-config.json',frozen)
    atomic_json(OUT/'environment.json',{'torch':torch.__version__,'cuda':torch.version.cuda,'python':sys.version,'gpu':torch.cuda.get_device_name(),
        'cudnn_benchmark':True,'fixed_seeds_not_bitwise_GPU_determinism_guarantee':True})
    model=make_lane_refinement_network();state=torch.load(BASE,map_location='cpu',weights_only=True);model.load_state_dict(state,strict=True);model=model.cuda()
    ema=copy.deepcopy(model).eval()
    for param in ema.parameters():param.requires_grad_(False)
    optimizer=torch.optim.AdamW([{'params':model.base.backbone.parameters(),'lr':1e-5,'base_lr':1e-5},
        {'params':model.base.classifier.parameters(),'lr':1e-4,'base_lr':1e-4},
        {'params':[p for module in [model.low_projection,model.context_projection,model.refinement] for p in module.parameters()],'lr':1e-3,'base_lr':1e-3}],weight_decay=1e-4)
    scaler=torch.amp.GradScaler('cuda')
    class Patches(torch.utils.data.Dataset):
        def __len__(self):return len(training)
        def __getitem__(self,i):
            x,y=full_context_sample(training[i]);return torch.from_numpy(x),torch.from_numpy(y)
    loader=torch.utils.data.DataLoader(Patches(),batch_size=2,shuffle=True,drop_last=True,num_workers=0)
    selected,scores=validation_scores(ema,validation,1024,torch);best=selected['paint_iou'];patience_best=best;stale=0;best_epoch=0;history=[];start=time.monotonic();updates=0
    torch.save(ema.cpu().state_dict(),OUT/'lane_best.pt');ema.cuda();atomic_json(OUT/'best-validation.json',{'epoch':0,'selected':selected,'scores':scores,'selection':'VAL only zero-step reference'})
    for epoch in range(20):
        factor=min(1.,(epoch+1)/2)*(.1+.9*(1+math.cos(math.pi*max(0,epoch-1)/18))/2)
        for group in optimizer.param_groups:group['lr']=group['base_lr']*factor
        model.train();freeze_batch_norm_statistics(model);losses=[]
        for step,(pixels,labels) in enumerate(loader):
            optimizer.zero_grad(set_to_none=True);pixels,labels=pixels.cuda(),labels.cuda()
            with torch.autocast('cuda',dtype=torch.float16):loss=paint_loss(model(pixels)['out'],labels)
            if not torch.isfinite(loss):raise ValueError('nonfinite_loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),5.);before=scaler.get_scale();scaler.step(optimizer);scaler.update()
            free,total=torch.cuda.mem_get_info()
            if total-free>9*1024**3:raise ValueError('shared_GPU_memory_budget_9GiB_exceeded')
            if scaler.get_scale()>=before:
                updates+=1
                with torch.no_grad():
                    source=model.state_dict()
                    for key,value in ema.state_dict().items():
                        if value.is_floating_point():value.mul_(.99).add_(source[key],alpha=.01)
                        else:value.copy_(source[key])
            losses.append(float(loss.detach()))
            if step%100==0:print(json.dumps({'epoch':epoch+1,'step':step,'loss':losses[-1],'updates':updates}),flush=True)
        selected,scores=validation_scores(ema,validation,1024,torch);entry={'epoch':epoch+1,'loss':float(np.mean(losses)),'validation_paint_iou':selected['paint_iou'],'validation_threshold':selected['paint_threshold'],'updates':updates}
        history.append(entry);print(json.dumps(entry),flush=True)
        if selected['paint_iou']>best:
            best=selected['paint_iou'];best_epoch=epoch+1;torch.save(ema.cpu().state_dict(),OUT/'lane_best.pt');ema.cuda();atomic_json(OUT/'best-validation.json',{'epoch':best_epoch,'selected':selected,'scores':scores,'selection':'VAL only'})
        if selected['paint_iou']>patience_best+.001:patience_best=selected['paint_iou'];stale=0
        else:stale+=1
        atomic_json(OUT/'history.json',{'history':history,'best_epoch':best_epoch,'actual_updates':updates,'elapsed_seconds':time.monotonic()-start,'test_used':False,
            'peak_allocated_GiB':torch.cuda.max_memory_allocated()/1024**3,'peak_reserved_GiB':torch.cuda.max_memory_reserved()/1024**3})
        if epoch+1>=6 and stale>=6:break
    del model,optimizer;torch.cuda.empty_cache();ema.cpu();ema.load_state_dict(torch.load(OUT/'lane_best.pt',map_location='cpu',weights_only=True));ema.eval()
    bestdata=json.loads((OUT/'best-validation.json').read_text(encoding='utf-8'));weights=OUT/'lane.onnx';count=export_fp64_mean(ema,weights)
    manifest={'format':'onnx','task':'lane','architecture':ARCHITECTURE,'aspp_dilations':[6,12,18],'classes':['background','paint'],'weights':weights.relative_to(ROOT).as_posix(),'sha256':sha256(weights),
        'target_trained':True,'dataset_release':'manifest-sha256:'+sha256(RELEASE),'source_release_path':RELEASE.relative_to(ROOT).as_posix(),'training_development_release_path':DEV.relative_to(ROOT).as_posix(),
        'training_development_release_sha256':sha256(DEV),'training_provenance':pp.relative_to(ROOT).as_posix(),'input_size':[1024,1024],'mean':MEAN.tolist(),'std':STD.tolist(),
        'paint_threshold':bestdata['selected']['paint_threshold'],'threshold_calibration':{'selected_on':'validation','report_sha256':sha256(OUT/'best-validation.json'),'validation_samples':len(validation),'test_images_used':0},
        'required_runtime_profile':SUPPORTED_CPU_PROFILE,'export_transform':{'kind':'FP64 global mean then original dtype','sites':count,'learned_state_unchanged':True},
        'source_licenses':['CC-BY-ND-4.0'],'research_only':True,'production_approved':False,'project_acceptance_eligible':False,'independent_route_verified':False,'human_privacy_reviewed':False,
        'test_split_used':False,'test_split_used_scope':'training_and_selection','data_revoked':False,'needs_revalidation':False,'best_epoch':best_epoch,'validation_paint_iou':best,'validation_selection_metric':'mean_recording_IoU'}
    atomic_json(OUT/'lane.json',manifest);link_model(pp,OUT/'lane.json');print(json.dumps({'status':'trained','epochs':len(history),'best_epoch':best_epoch,'validation_iou':best}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze-plan','train']);a=p.parse_args()
    if a.action=='freeze-plan':atomic_json(ROOT/'docs/reports/LANE-CYCLE2-PLAN.json',plan())
    else:train()
