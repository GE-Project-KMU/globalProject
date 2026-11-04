"""COCO-S transfer with uniform fixed strata, frozen BN and accumulated batches."""
import argparse,json,math,sys,time
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'.cache/vendor/YOLOX'))
from agent.paths import workspace_path,atomic_json,sha256
from agent.dataset import validate_research_release
from agent.pothole_small_network import make_pothole_small,INITIAL_SHA
from scripts.train_pothole_localization import epoch_indices,QUOTAS
BASE='data/training/pothole-mwpd-v6';OUT='data/training/pothole-small-v9';INIT='.cache/training/pretrained/yolox_s_0.1.1rc0.pth'
DEPS=['agent/pothole_small_network.py','scripts/train_yolox_research.py','scripts/improve_pothole_train.py','scripts/train_pothole_localization.py','scripts/train_pothole_mwpd.py','scripts/train_pothole_tiny.py','.cache/vendor/YOLOX/exps/default/yolox_s.py']

def parameter_groups(model,body_lr=.0002,head_lr=.002):
    groups=[]
    for head in (False,True):
        for decay in (False,True):
            params=[p for name,p in model.named_parameters() if name.startswith('head.')==head and (p.ndim>1 and not name.endswith('.bias'))==decay]
            groups.append({'params':params,'lr':head_lr if head else body_lr,'base_lr':head_lr if head else body_lr,'weight_decay':.0005 if decay else 0.})
    return groups

def read(p):return json.loads(workspace_path(p,exists=True).read_text(encoding='utf-8'))

def prepare(output=OUT):
    output=workspace_path(output)
    if output.exists() or not output.is_relative_to(ROOT/'data/training'):raise ValueError('new_capacity_experiment_required')
    if sha256(ROOT/INIT)!=INITIAL_SHA:raise ValueError('pinned_COCO_initialization_required')
    rows=[r for r in read(BASE+'/release.json') if r['split']!='test']
    if not validate_research_release(rows)['valid'] or any(len(r['objects'])>30 for r in rows):raise ValueError('valid_development_GT_required')
    output.mkdir();atomic_json(output/'release.json',rows)
    plan={'format':'pothole-small-v9-plan-v1','candidate_count':1,'architecture':'YOLOX-S','source_sha256':sha256(Path(__file__)),'dependencies':{p:sha256(ROOT/p) for p in DEPS},
          'baseline_manifest_sha256':sha256(ROOT/(BASE+'/model.json')),'baseline_checkpoint_sha256':sha256(ROOT/(BASE+'/best.pt')),'development_sha256':sha256(output/'release.json'),
          'initial_checkpoint_sha256':INITIAL_SHA,'initial_source_url':'https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth',
          'maximum_epochs':28,'minimum_epochs':20,'patience':8,'batch_size':8,'gradient_accumulation':2,'effective_images_per_update':16,
          'accumulation_limit':'mean of two independently foreground-normalized native losses, not an assertion of exact full16 loss equivalence',
          'input_size':640,'seed':20261109,'body_learning_rate':.0002,'head_learning_rate':.002,'optimizer':'SGD Nesterov .9 wd.0005 excluding BN/bias; body/head separate LR',
          'scheduler':'1epoch linear warmup .1->1 then cosine to.1 by maximum28; LR stepped on optimizer attempts',
          'BN':'all running statistics frozen; affine trainable','body_frozen_first_epochs':2,'native_L1_enabled_after_epoch':6,
          'augmentation':'fullframe640 letterbox/HSV.3/flip.5; no mosaic/mixup','sampling':'uniform within exact4096 replacement draws, RDDpositive1024/RDDnegative1024/MWPDpositive2048; no dropped microbatches',
          'source_draw_counts':QUOTAS,'TRAIN_hardness_used':False,'TRAIN_images':7540,'VAL_images':652,'TEST_images_used_for_fitting':0,'US100_used_for_fitting':False,
          'selection':'max mean old/new VAL AP50 with original-domain drop<=3pp vs currentv6, baseline retained, earliest tie; preserve diagnostic best even if rejected',
          'confidence':.1,'AP_confidence_floor':.01,'nms_iou':.5,'own_peak_reserved_limit_GiB':5,
          'test_policy':'pothole family winner selected only on development before one US100 and historical regression; no TEST tuning',
          'source_expires_at':min(r['expires_at'] for r in rows),'source_expiry_extended':False,'research_only':True,'production_approved':False,'user_recordings_used':False}
    atomic_json(output/'plan.json',plan);atomic_json(ROOT/'docs/reports/POTHOLE-SMALL-V9-PLAN.json',plan);print(json.dumps({'plan_sha256':sha256(output/'plan.json'),'training_started':False}))

def preflight(output=OUT):
    import torch,cv2
    from scripts.improve_pothole_train import OriginalSamples
    from yolox.data.datasets.mosaicdetection import MosaicDetection
    from yolox.data.data_augment import TrainTransform
    torch.set_num_threads(2);cv2.setNumThreads(1);torch.manual_seed(20261109)
    if torch.cuda.is_initialized():raise ValueError('CPU_contract_only')
    output=workspace_path(output,exists=True);rows=read(output/'release.json');train=[r for r in rows if r['split']=='train'];selected=[max((r for r in train if r['development_domain']==d),key=lambda r:len(r['objects'])) for d in ('original','mwpd')]
    ds=MosaicDetection(OriginalSamples(selected,640,0),img_size=(640,640),mosaic=False,preproc=TrainTransform(max_labels=120,hsv_prob=0.,flip_prob=0.),enable_mixup=False)
    images,labels,_,_=next(iter(torch.utils.data.DataLoader(ds,batch_size=2,num_workers=0)))
    if (labels[:,:,3]>0).sum(1).tolist()!=[len(r['objects']) for r in selected]:raise ValueError('all_GT_preserved_required')
    model,audit=make_pothole_small(pretrained=INIT);model.train();model.head.use_l1=True
    for m in model.modules():
        if isinstance(m,torch.nn.modules.batchnorm._BatchNorm):m.eval()
    losses=model(images,labels);loss=losses['total_loss'];loss.backward();grads=[p.grad for p in model.parameters() if p.grad is not None]
    if not torch.isfinite(loss) or not grads or not all(torch.isfinite(g).all() for g in grads) or not any(torch.count_nonzero(g) for g in grads):raise ValueError('finite_nonzero_backward_required')
    model.eval()
    with torch.no_grad():raw=model(images[:1])
    if list(raw.shape)!=[1,8400,6] or not torch.isfinite(raw).all():raise ValueError('decoded_single_class_contract_required')
    result={'format':'pothole-small-v9-CPU-contract','source_sha256':sha256(Path(__file__)),'development_sha256':sha256(output/'release.json'),'initial_checkpoint_sha256':INITIAL_SHA,'transfer':audit,'parameters':sum(p.numel() for p in model.parameters()),'all_GT_preserved':True,'finite_loss':float(loss.detach()),'finite_nonzero_backward':True,'decoded_output_shape':list(raw.shape),'optimizer_updates':0,'GPU_initialized':torch.cuda.is_initialized(),'TEST_images_used':0}
    atomic_json(output/'preflight.json',result);atomic_json(ROOT/'docs/reports/POTHOLE-SMALL-V9-CONTRACT.json',result);print(json.dumps(result))

def train(output=OUT):
    import torch,cv2,numpy as np
    from scripts.train_yolox_research import make_model,evaluate_model,seed_all
    from scripts.train_pothole_mwpd import choose
    from scripts.train_pothole_tiny import learning_rate
    from scripts.improve_pothole_train import OriginalSamples
    from yolox.data.datasets.mosaicdetection import MosaicDetection
    from yolox.data.data_augment import TrainTransform
    from yolox.utils import ModelEMA
    from agent.governance import begin_training,register_asset,link_model
    output=workspace_path(output,exists=True);plan=read(output/'plan.json')
    if (output/'provenance.json').exists():raise ValueError('already_started_preserve_run')
    if sha256(Path(__file__))!=plan['source_sha256'] or any(sha256(ROOT/p)!=s for p,s in plan['dependencies'].items()):raise ValueError('frozen_sources_changed')
    for p,digest in [(output/'release.json',plan['development_sha256']),(ROOT/(BASE+'/model.json'),plan['baseline_manifest_sha256']),(ROOT/(BASE+'/best.pt'),plan['baseline_checkpoint_sha256']),(ROOT/INIT,INITIAL_SHA)]:
        if sha256(p)!=digest:raise ValueError('frozen_training_inputs_changed')
    if read(output/'preflight.json')['source_sha256']!=plan['source_sha256']:raise ValueError('matching_CPU_preflight_required')
    rows=read(output/'release.json')
    if any(r['split']=='test' for r in rows) or not validate_research_release(rows)['valid']:raise ValueError('development_only_valid_release_required')
    if not torch.cuda.is_available():raise RuntimeError('CUDA_required')
    seed_all(plan['seed']);torch.set_num_threads(2);cv2.setNumThreads(1);training=[r for r in rows if r['split']=='train'];validation={d:[r for r in rows if r['split']=='validation' and r['development_domain']==d] for d in ('original','mwpd')}
    pp,prov=begin_training(output,output/'release.json',rows,'pothole');prov['research_only']=True;prov['assets']=[register_asset(prov,r,[]) for r in rows];atomic_json(pp,prov);started=time.perf_counter()
    baseline,_=make_model(['pothole']);baseline.load_state_dict(torch.load(ROOT/(BASE+'/best.pt'),map_location='cpu',weights_only=True)['model'],strict=True);baseline.cuda();score={d:evaluate_model(baseline,v,['pothole'],size=640,batch_size=8)['metrics']['map50'] for d,v in validation.items()};del baseline;torch.cuda.empty_cache()
    history=[{'epoch':0,'old_validation_ap50':score['original'],'new_validation_ap50':score['mwpd'],'optimizer_updates':0,'mean_training_loss':None}]
    model,transfer=make_pothole_small(pretrained=INIT);model.cuda();groups=parameter_groups(model,plan['body_learning_rate'],plan['head_learning_rate'])
    optimizer=torch.optim.SGD(groups,momentum=.9,nesterov=True);ema=ModelEMA(model);scaler=torch.amp.GradScaler('cuda')
    ds=MosaicDetection(OriginalSamples(training,640,1),img_size=(640,640),mosaic=False,preproc=TrainTransform(max_labels=120,hsv_prob=.3,flip_prob=.5),enable_mixup=False)
    items=[{'stratum':('RDD' if r['development_domain']=='original' else 'MWPD')+('_positive' if r['objects'] else '_negative'),'weight':1.} for r in training]
    class UniformSampler:
        def __init__(self):self.epoch=0
        def __len__(self):return 4096
        def __iter__(self):indices=epoch_indices(items,plan['seed'],self.epoch);self.epoch+=1;return iter(indices)
    loader=torch.utils.data.DataLoader(ds,batch_size=8,sampler=UniformSampler(),num_workers=0,drop_last=False,pin_memory=True)
    if len(loader)%2:raise ValueError('exact_two_microbatch_updates_required')
    updates=0;attempts=0;last_best=0;diagnostic_best=-1.;atomic_json(output/'training-config.json',plan|{'transfer':transfer,'parameters':sum(p.numel() for p in model.parameters())})
    print(json.dumps(history[0]),flush=True)
    for epoch in range(1,plan['maximum_epochs']+1):
        if datetime.fromisoformat(plan['source_expires_at'].replace('Z','+00:00'))<=datetime.now(timezone.utc):raise ValueError('source_expired_stop_training')
        model.train();model.head.use_l1=epoch>6
        for name,p in model.named_parameters():p.requires_grad_(not name.startswith('backbone.') or epoch>2)
        for m in model.modules():
            if isinstance(m,torch.nn.modules.batchnorm._BatchNorm):m.eval()
        losses=[];optimizer.zero_grad(set_to_none=True)
        for micro,(images,labels,_,_) in enumerate(loader):
            if micro%2==0:
                for group in optimizer.param_groups:group['lr']=learning_rate(attempts,(len(loader)//2)*plan['maximum_epochs'],len(loader)//2,group['base_lr'])
            with torch.autocast('cuda',dtype=torch.float16):loss=model(images.cuda(non_blocking=True),labels.cuda(non_blocking=True))['total_loss']
            if not torch.isfinite(loss):raise RuntimeError('nonfinite_training_loss')
            scaler.scale(loss/2).backward();losses.append(float(loss.detach()))
            if micro%2==1:
                before=scaler.get_scale();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),10.);scaler.step(optimizer);scaler.update();attempts+=1
                if scaler.get_scale()>=before:updates+=1;ema.update(model)
                optimizer.zero_grad(set_to_none=True)
            if torch.cuda.max_memory_reserved()>5*2**30:raise RuntimeError('own_training_memory_budget_exceeded')
            if micro%100==0:print(json.dumps({'epoch':epoch,'microbatch':micro,'microbatches':len(loader),'optimizer_updates':updates,'loss':losses[-1]}),flush=True)
        metrics={d:evaluate_model(ema.ema,v,['pothole'],size=640,batch_size=8)['metrics']['map50'] for d,v in validation.items()};previous=choose(history)['epoch'];row={'epoch':epoch,'old_validation_ap50':metrics['original'],'new_validation_ap50':metrics['mwpd'],'optimizer_updates':updates,'mean_training_loss':float(np.mean(losses))};history.append(row);selected=choose(history);mean=sum(metrics.values())/2
        state={'model':ema.ema.state_dict(),'epoch':epoch,'classes':['pothole'],'architecture':'YOLOX-S'}
        if mean>diagnostic_best:diagnostic_best=mean;torch.save(state,output/'diagnostic-best.pt')
        if selected['epoch']==epoch and previous!=epoch:last_best=epoch;torch.save(state,output/'best.pt')
        atomic_json(output/'history.json',{'history':history,'selected':selected,'actual_updates':updates,'attempts':attempts,'elapsed_seconds':time.perf_counter()-started,'own_peak_reserved_GiB':torch.cuda.max_memory_reserved()/2**30});print(json.dumps(row|{'selected_epoch':selected['epoch'],'elapsed_seconds':time.perf_counter()-started}),flush=True)
        if epoch>=plan['minimum_epochs'] and epoch-last_best>=plan['patience']:break
    selected=choose(history)
    summary={'candidate_selected':selected['epoch']>0,'selected':selected,'baseline_validation':history[0],'optimizer_updates':updates,'epochs_completed':epoch,'diagnostic_checkpoint_sha256':sha256(output/'diagnostic-best.pt'),'TEST_inference_performed':False,'elapsed_seconds':time.perf_counter()-started}
    if selected['epoch']>0:
        model.load_state_dict(torch.load(output/'best.pt',map_location='cpu',weights_only=True)['model'],strict=True);model.cpu().eval();model.head.decode_in_inference=True
        torch.onnx.export(model,torch.zeros(1,3,640,640),str(output/'model.onnx'),input_names=['images'],output_names=['predictions'],opset_version=17,dynamo=False)
        manifest=read(BASE+'/model.json');manifest.update(architecture='YOLOX-S',weights=(output/'model.onnx').relative_to(ROOT).as_posix(),sha256=sha256(output/'model.onnx'),source_release_path=(output/'release.json').relative_to(ROOT).as_posix(),dataset_release='manifest-sha256:'+sha256(output/'release.json'),training_provenance=pp.relative_to(ROOT).as_posix(),best_epoch=selected['epoch'],trained_checkpoint_sha256=sha256(output/'best.pt'),refinement_plan_sha256=sha256(output/'plan.json'),exported_before_final_test=True,test_split_used=False)
        manifest['development_only_release']=True
        atomic_json(output/'model.json',manifest);link_model(pp,output/'model.json');summary.update(manifest_sha256=sha256(output/'model.json'),weights_sha256=manifest['sha256'],checkpoint_sha256=sha256(output/'best.pt'))
    atomic_json(output/'selection-freeze.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','preflight','train']);p.add_argument('--output',default=OUT);a=p.parse_args();{'prepare':prepare,'preflight':preflight,'train':train}[a.action](a.output)
