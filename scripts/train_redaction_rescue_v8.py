"""Frozen legacy + P2 rescue: TRAIN-only head learning, VAL-only selection.

No CUDA path runs without an explicit root-coordinated operator flag. Existing
P2 run/code and blind TEST pixels/predictions are never opened by this trainer.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,math,random,sys,time
from collections import Counter,defaultdict,OrderedDict
from datetime import datetime,timezone
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import workspace_path,sha256,atomic_json
from scripts.redaction_rescue_data import load_development,read_image,shared_view,tile_view,tile_targets,check_conservation,source_boxes,DEVELOPMENT

OUT=ROOT/'data/training/redaction-rescue-v8'
BASE=ROOT/'data/training/redaction-specialists-v4-efficient/best.pt'
BASE_SHA='6c985f7e3afb0fa061cbb82d7ec7b304eb5f9fbfcd0cf9f775dbab28873c82ad'
RELEASE=ROOT/'data/training/research-redaction-cycle2-v7/release.json'
PUBLIC=ROOT/'docs/reports/PRIVACY-RESCUE-V8-PLAN.json'
CLASSES=['face','license_plate'];GRID=[.01,.03,.05,.1,.2,.35,.5]
DEPS=['agent/redaction_rescue_network.py','agent/redaction_network.py','agent/redaction_tiled_network.py','scripts/redaction_rescue_data.py']

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def alive(expiry):
    date=datetime.fromisoformat(expiry.replace('Z','+00:00'))
    if date.tzinfo is None or date<=datetime.now(timezone.utc):raise ValueError('original_source_expired')

def grouped_match(result,label,confidence):
    from agent.pipeline import intersection_over_union
    truth=defaultdict(list);predictions=defaultdict(list)
    for item in result['truth']:
        if item['class']==label:truth[item['sample_id']].append(item)
    for item in result['predictions']:
        if item['class']==label and item['confidence']>=confidence:predictions[item['sample_id']].append(item)
    matched=[];tp=fp=0
    for sid in set(truth)|set(predictions):
        used=set();gt=truth[sid]
        for prediction in sorted(predictions[sid],key=lambda p:-p['confidence']):
            overlap,index=max(((intersection_over_union(prediction['bbox_xywh_px'],g['bbox_xywh_px']),i) for i,g in enumerate(gt) if i not in used),default=(0,None))
            if overlap>=.5:used.add(index);tp+=1
            else:fp+=1
        matched.extend((g,i in used) for i,g in enumerate(gt))
    count=sum(map(len,truth.values()))
    return {'gt':count,'tp':tp,'fp':fp,'fn':count-tp,'recall':tp/count if count else 0.,'precision':tp/(tp+fp) if tp+fp else 0.},matched

def large_recall(result):
    _,matched=grouped_match(result,'face',.1)
    large=[hit for g,hit in matched if math.sqrt(g['bbox_xywh_px'][2]*g['bbox_xywh_px'][3])>=64]
    if not large:raise ValueError('large_face_validation_support_required')
    return sum(large)/len(large)

def calibrate(result):
    options=[]
    for confidence in GRID:
        by={label:grouped_match(result,label,confidence)[0] for label in CLASSES}
        options.append({'confidence':confidence,'classes':by,'min_recall':min(v['recall'] for v in by.values()),
                        'macro_precision':sum(v['precision'] for v in by.values())/2,
                        'both_recall95':all(v['recall']>=.95 for v in by.values())})
    feasible=[r for r in options if r['both_recall95']]
    best=max(feasible,key=lambda r:(r['macro_precision'],r['min_recall'],r['confidence'])) if feasible else max(options,key=lambda r:(r['min_recall'],r['macro_precision'],r['confidence']))
    return {'selected_confidence':best['confidence'],'selected':best,'grid':options,'selection_source':'VAL508 only','TEST_images':0}

def family_winner(candidates,reference_large):
    if not any(item['name']=='legacy_v4' and item.get('training_complete') for item in candidates):raise ValueError('complete_legacy_baseline_required')
    eligible=[]
    for item in candidates:
        if not item.get('training_complete'):continue
        if large_recall(item['validation'])<reference_large-.01:continue
        choice=calibrate(item['validation'])['selected'];eligible.append((item,choice))
    if not eligible:raise ValueError('no_complete_largeface_eligible_family')
    feasible=[pair for pair in eligible if pair[1]['both_recall95']]
    def key(pair):
        item,point=pair
        metrics=(point['macro_precision'],point['min_recall']) if feasible else (point['min_recall'],point['macro_precision'])
        return (*metrics,item['name']=='legacy_v4',point['confidence'])
    return max(feasible or eligible,key=key)[0]['name']

def paired_tile_order(originals,seed,epoch):
    if type(originals) is not int or originals<=0 or originals%2:raise ValueError('even_original_count_for_exact_batch2_required')
    rng=random.Random(seed+epoch);order=list(range(originals));rng.shuffle(order);indices=[]
    for start in range(0,originals,2):
        left,right=order[start:start+2];a=list(range(4));b=list(range(4));rng.shuffle(a);rng.shuffle(b)
        for tile_a,tile_b in zip(a,b):indices.extend((left*4+tile_a,right*4+tile_b))
    return indices

def legacy_digest(bundle):
    result=hashlib.sha256()
    for name,value in sorted(bundle.state_dict().items()):
        if not name.startswith(('face.','plate.')):continue
        value=value.detach().cpu().contiguous();result.update(name.encode());result.update(str(value.dtype).encode());result.update(str(tuple(value.shape)).encode());result.update(value.numpy().tobytes())
    return result.hexdigest()

def construct():
    import torch
    from agent.redaction_rescue_network import build_redaction_rescue_network,load_frozen_legacy
    if sha256(BASE)!=BASE_SHA:raise ValueError('frozen_v4_checkpoint_changed')
    model=build_redaction_rescue_network();load_frozen_legacy(model,torch.load(BASE,map_location='cpu',weights_only=True)['model'])
    return model

def optimizer_for(bundle):
    import torch
    groups={'decay':[],'no_decay':[],'bias':[]}
    for module in bundle.modules():
        for name,p in module.named_parameters(recurse=False):
            if not p.requires_grad:continue
            groups['bias' if name=='bias' else 'no_decay' if isinstance(module,torch.nn.BatchNorm2d) else 'decay'].append(p)
    parameters=[p for p in bundle.parameters() if p.requires_grad]
    if {id(p) for group in groups.values() for p in group}!={id(p) for p in parameters}:raise ValueError('new_parameter_optimizer_coverage_failed')
    return torch.optim.SGD([{'params':group,'weight_decay':.0005 if name=='decay' else 0.,'base_lr':.005} for name,group in groups.items()],lr=.005,momentum=.9,nesterov=True),parameters

def cpu_contract():
    import torch
    from agent.redaction_tiled_network import build_redaction_tiled_network
    torch.set_num_threads(2);torch.manual_seed(20261109);cv2.setNumThreads(1)
    trainrows,valrows,devsha=load_development();total_gt=0
    for row in trainrows:
        boxes=source_boxes([o['bbox_xywh_px'] for o in row['objects'] if o['class']=='face'],row['image']['width'],row['image']['height'])
        total_gt+=len(boxes);scale=min(1920/row['image']['width'],1920/row['image']['height'])
        check_conservation(boxes*scale)
        flipped=boxes.copy();flipped[:,0]=row['image']['width']-flipped[:,0]-flipped[:,2];check_conservation(flipped*scale)
    row=min(trainrows,key=lambda r:(-len(r['objects']),r['sample_id']));image=read_image(row)
    boxes=[o['bbox_xywh_px'] for o in row['objects'] if o['class']=='face'];canvas,scaled,geometry=shared_view(image,boxes)
    conservation=check_conservation(scaled);tile=max(range(4),key=lambda i:len(tile_targets(scaled,i)[1]))
    pixels,packed,parents=tile_view(canvas,scaled,tile);labels=torch.from_numpy(packed[None]);labels_before=labels.clone()
    model=construct().cpu().eval();before=legacy_digest(model)
    baseline=build_redaction_tiled_network().cpu().eval();baseline.load_state_dict(torch.load(BASE,map_location='cpu',weights_only=True)['model'],strict=True)
    full=torch.from_numpy(np.ascontiguousarray(canvas.transpose(2,0,1)[None],dtype=np.float32))
    with torch.inference_mode():reference=baseline(full).numpy();initial=model(full).numpy()
    prefix_initial=reference.tobytes()==initial[:,:102900].tobytes()
    model.train();observed={};native=model.rescue_head.get_losses
    def observer(*args,**kwargs):
        observed['nonempty_GT']=int((args[4].sum(-1)>0).sum());observed['shape']=list(args[4].shape);return native(*args,**kwargs)
    model.rescue_head.get_losses=observer
    optimizer,new_parameters=optimizer_for(model);optimizer.zero_grad(set_to_none=True)
    start=time.perf_counter();loss=model.rescue_loss(torch.from_numpy(pixels[None]),labels)[0];loss.backward();elapsed=time.perf_counter()-start
    model.rescue_head.get_losses=native
    old_grad=sum(p.grad is not None for name,p in model.named_parameters() if name.startswith(('face.','plate.')))
    new_nonzero=sum(p.grad is not None and bool(torch.count_nonzero(p.grad)) for p in new_parameters)
    finite=bool(torch.isfinite(loss)) and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in new_parameters)
    model.eval()
    with torch.inference_mode():after=model(full).numpy()
    unchanged=before==legacy_digest(model);prefix_after=reference.tobytes()==after[:,:102900].tobytes()
    passed=prefix_initial and prefix_after and unchanged and old_grad==0 and new_nonzero>0 and finite and torch.equal(labels,labels_before) and observed=={'nonempty_GT':len(parents),'shape':[1,512,5]}
    if sha256(workspace_path(row['path'],exists=True))!=row['sha256'] or sha256(BASE)!=BASE_SHA:raise ValueError('original_assets_changed_during_CPU_contract')
    OUT.mkdir(exist_ok=True)
    private={'sample_id':row['sample_id'],'source_hash':row['sha256'],'tile':tile,'expires_at':row['expires_at'],'pixels_saved':False,'GT_coordinates_saved':False}
    atomic_json(OUT/'cpu-contract-case-private.json',private)
    report={'format':'privacy-rescue-v8-real-TRAIN-CPU-contract','passed':bool(passed),'trainer_source_sha256':sha256(Path(__file__)),
            'data_source_sha256':sha256(ROOT/'scripts/redaction_rescue_data.py'),'factory_sha256':sha256(ROOT/'agent/redaction_rescue_network.py'),
            'development_sha256':devsha,'checkpoint_sha256':BASE_SHA,'TRAIN_originals_geometry_checked':3000,'TRAIN_GT_objects_checked':total_gt,
            'all_four_tiles_and_flip_GT_coverage_and_area_conserved':True,'actual_TRAIN_images_read':1,'actual_original_GT':len(boxes),
            'actual_tile_GT':len(parents),'native_received_labels_shape':observed['shape'],'native_received_GT':observed['nonempty_GT'],
            'native_labels_unchanged':torch.equal(labels,labels_before),'legacy_prefix_initial_byte_exact':prefix_initial,
            'legacy_prefix_after_actual_GT_backward_byte_exact':prefix_after,'legacy_state_unchanged':unchanged,'legacy_gradient_tensors':old_grad,
            'new_nonzero_gradient_parameter_tensors':new_nonzero,'loss_and_gradients_finite':finite,'CPU_backward_seconds':elapsed,
            'source_expires_at':row['expires_at'],'raster_geometry':geometry,'CPU_threads':2,'cv2_threads':1,'GPU_used':False,
            'actual_optimizer_steps':0,'training_run_started':False,'VAL_images_inferred':0,'TEST_images':0,'user_recordings_used':False,'pixels_saved':False}
    atomic_json(OUT/'cpu-contract.json',report);atomic_json(ROOT/'docs/reports/REDACTION-RESCUE-V8-CPU-CONTRACT.json',report)
    print(json.dumps(report));return report

def config():
    from agent.redaction_rescue_network import ARCHITECTURE
    trainrows,valrows,devsha=load_development();proof=read(OUT/'cpu-contract.json')
    if not proof['passed'] or proof['trainer_source_sha256']!=sha256(Path(__file__)) or proof['data_source_sha256']!=sha256(ROOT/'scripts/redaction_rescue_data.py') or proof['factory_sha256']!=sha256(ROOT/'agent/redaction_rescue_network.py') or proof['development_sha256']!=devsha or proof['checkpoint_sha256']!=BASE_SHA or sha256(BASE)!=BASE_SHA:raise ValueError('current_real_TRAIN_CPU_contract_required')
    return {'format':'privacy-rescue-v8-plan-v1','training_source_sha256':sha256(Path(__file__)),
            'dependencies_sha256':{p:sha256(ROOT/p) for p in DEPS},'development_sha256':devsha,'release_sha256':sha256(RELEASE),
            'cpu_contract_sha256':sha256(OUT/'cpu-contract.json'),'warmstart_checkpoint_sha256':BASE_SHA,'architecture':ARCHITECTURE,
            'input_size':[1920,1920],'face_input_size':960,'plate_input_size':640,'legacy_prefix_proposals':102900,'rescue_proposals':230400,
            'output_shape':[1,333300,7],'rescue_score_scale':.5,'rescue_stride':4,'rescue_hidden_channels':32,
            'TRAIN_face_originals':3000,'tiles_per_epoch':12000,'VAL_face':400,'VAL_plate':108,'plate_TRAIN_used':False,'TEST_images':0,
            'batch_size':2,'epochs':12,'seed':20261109,'max_GT':512,'optimizer':'new parameters only; SGD Nesterov momentum.9 lr.005; decay.0005 except BN/bias',
            'scheduler':'1epoch linear .1to1 warmup; cosine to.1 byepoch12','AMP':True,'EMA_new_state_only':.99,'gradient_clip':10,
            'augmentation':'Original-image HSV h[-5,5]/s[.7,1.3]/v[.75,1.25] and flip.5; deterministic shared transform across four tiles per original/epoch; no mosaic/mixup',
            'sampler':'Shuffle3000 originals into1500 pairs; independently shuffle4 tiles per original; interleave each pair so batch2 contains distinct originals; all12000 tiles exactly once; bounded4-canvas cache',
            'view_geometry':'Exact app round1920 top-left114 letterbox then four960 crops; all positive-area GT intersections retained; nominal scale labels; no filtering/truncation',
            'legacy_parameters_and_BN':'All face/plate parameters frozen/eval; new projection/head BN train; no legacy EMA drift',
            'checkpoint_selection':'Max faceAP50 floor.01 on faceVAL400 among fixedconf.1 originalsqrtarea>=64 recall >=epoch0legacy-.01; epoch0 included; tieearlier',
            'operating_confidence_grid':GRID,'family_selection':'Complete families only; fixedconf.1 largefacegate baseline-.01; VAL508 bothclassRecall95 feasible→macroPrecision then minRecall; otherwise minRecall then macroPrecision; ties legacy baseline first',
            'P2_incomplete_run':'diagnostic only, excluded from family winner; existing run/source/state unchanged',
            'TEST_policy':'No TEST pixels/predictions/training/selection; fresh blind200 only once after complete family winner/threshold/numeric freeze by root',
            'source_expires_at':min(datetime.fromisoformat(r['expires_at'].replace('Z','+00:00')) for r in trainrows+valrows).isoformat(),'source_licenses':['CC-BY-NC-ND-4.0','MIT'],
            'known_class_conditional_only':True,'field_privacy_guarantee':False,'source_expiries_extended':False,'user_recordings_used':False,
            'cudnn_benchmark':False,'TF32':False,'recovery':'Atomic epoch-boundary model/EMA/optimizer/scaler/Python/NumPy/Torch/CUDA RNG/sampler/history snapshot; no auto resume, new retry plan/root authorization required',
            'research_only':True,'production_approved':False,'human_privacy_reviewed':False,'single_GPU_training_job_only':True,'GPU_start_requires_root_slot':True,'global_GPU_budget_GiB':9}

def freeze_plan():
    path=OUT/'plan.json'
    if path.exists() or PUBLIC.exists():raise ValueError('fresh_pretraining_plan_required')
    value=config();atomic_json(path,value);atomic_json(PUBLIC,value)
    print(json.dumps({'status':'rescue_plan_frozen_GPU_not_started','plan_sha256':sha256(path),'training_source_sha256':value['training_source_sha256']}))

def require_plan(gpu_authorized):
    if gpu_authorized is not True:raise ValueError('explicit_root_single_GPU_slot_authorization_required')
    value=config()
    if read(OUT/'plan.json')!=value or read(PUBLIC)!=value:raise ValueError('immutable_pretraining_plan_mismatch')
    alive(value['source_expires_at']);return value

def dataset(trainrows,seed):
    import torch
    class Tiles(torch.utils.data.Dataset):
        def __init__(self):self.epoch=0;self.cache=OrderedDict()
        def set_epoch(self,epoch):self.epoch=epoch;self.cache.clear()
        def __len__(self):return len(trainrows)*4
        def __getitem__(self,index):
            original,tile=divmod(index,4);row=trainrows[original];alive(row['expires_at'])
            if original not in self.cache:
                rng=random.Random(seed+self.epoch*len(trainrows)+original);image=read_image(row)
                boxes=[o['bbox_xywh_px'] for o in row['objects'] if o['class']=='face']
                hsv=(rng.uniform(-5,5),rng.uniform(.7,1.3),rng.uniform(.75,1.25));flip=rng.random()<.5
                self.cache[original]=shared_view(image,boxes,flip=flip,hsv=hsv)[:2]
                while len(self.cache)>4:self.cache.popitem(last=False)
            canvas,boxes=self.cache[original];pixels,labels,_=tile_view(canvas,boxes,tile)
            return torch.from_numpy(pixels),torch.from_numpy(labels)
    return Tiles()

def evaluate(model,rows):
    import torch
    from yolox.utils import postprocess
    from agent.evaluate import detection_metrics
    model.eval();device=next(model.parameters()).device;truth=[];predictions=[];start=time.perf_counter()
    with torch.no_grad():
        for row in rows:
            image=read_image(row);height,width=image.shape[:2];canvas,_,geometry=shared_view(image,[])
            tensor=torch.from_numpy(np.ascontiguousarray(canvas.transpose(2,0,1)[None],dtype=np.float32)).to(device)
            found=postprocess(model(tensor),2,conf_thre=.01,nms_thre=.5,class_agnostic=False)[0]
            truth.extend(dict(o,sample_id=row['sample_id']) for o in row['objects'] if o['class']==row['evaluation_class'])
            if found is None:continue
            for box in found.cpu().numpy():
                label=CLASSES[int(box[6])]
                if label!=row['evaluation_class']:continue
                x1,y1,x2,y2=box[:4]/geometry['nominal_scale'];x1,y1=max(0.,float(x1)),max(0.,float(y1));x2,y2=min(float(width),float(x2)),min(float(height),float(y2))
                if x2>x1 and y2>y1:predictions.append({'sample_id':row['sample_id'],'class':label,'bbox_xywh_px':[x1,y1,x2-x1,y2-y1],'confidence':float(box[4]*box[5])})
    return {'truth':truth,'predictions':predictions,'metrics':detection_metrics(truth,predictions,CLASSES),'images':len(rows),'confidence_floor':.01,'elapsed_seconds':time.perf_counter()-start,'TEST_images':0}

def budget(torch):
    free,total=torch.cuda.mem_get_info()
    if total-free>9*1024**3:raise ValueError('global_GPU_budget9GiB_exceeded')

def configure_GPU(torch):
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def recovery_save(model,ema,optimizer,scaler,generator,epoch,updates,history,best_epoch,reference):
    import torch,os
    def cpu(value):
        if isinstance(value,torch.Tensor):return value.detach().cpu()
        if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
        if isinstance(value,list):return [cpu(v) for v in value]
        if isinstance(value,tuple):return tuple(cpu(v) for v in value)
        return value
    state=np.random.get_state();partial=OUT/'epoch-recovery.part';target=OUT/'epoch-recovery.pt'
    value={'format':'rescue-epoch-recovery-v1','plan_sha256':sha256(OUT/'plan.json'),'completed_epoch':epoch,'updates':updates,'best_epoch':best_epoch,'reference_large_recall_conf01':reference,
           'model':cpu(model.state_dict()),'EMA':cpu(ema.state_dict()),'optimizer':cpu(optimizer.state_dict()),'scaler':scaler.state_dict(),
           'python_RNG':random.getstate(),'numpy_RNG':{'bit_generator':state[0],'keys':state[1].tolist(),'position':state[2],'has_gauss':state[3],'cached_gauss':state[4]},
           'torch_RNG':torch.get_rng_state(),'CUDA_RNG':torch.cuda.get_rng_state().cpu(),'sampler_RNG':generator.get_state(),'selection_history':history,
           'best_checkpoint_sha256':sha256(OUT/'best.pt'),'automatic_resume_allowed':False}
    torch.save(value,partial);os.replace(partial,target)
    atomic_json(OUT/'recovery-receipt.json',{'completed_epoch':epoch,'checkpoint_sha256':sha256(target),'plan_sha256':value['plan_sha256'],'automatic_resume_allowed':False})

def train(gpu_authorized=False):
    value=require_plan(gpu_authorized)
    preflight=read(OUT/'gpu-preflight.json')
    if preflight['plan_sha256']!=sha256(OUT/'plan.json') or not preflight['legacy_state_preserved']:raise ValueError('matching_authorized_GPU_preflight_required')
    import torch
    from agent.governance import begin_training,register_asset,link_model
    if not torch.cuda.is_available():raise ValueError('authorized_CUDA_required')
    if (OUT/'provenance.json').exists() or (OUT/'history.json').exists():raise ValueError('fresh_training_attempt_required')
    configure_GPU(torch);budget(torch);random.seed(value['seed']);np.random.seed(value['seed']);torch.manual_seed(value['seed']);torch.cuda.manual_seed_all(value['seed']);torch.set_num_threads(2);cv2.setNumThreads(1)
    trainrows,valrows,_=load_development();faceval=[r for r in valrows if r['evaluation_class']=='face']
    pp,provenance=begin_training(OUT,RELEASE,trainrows+valrows,'redaction');provenance['research_only']=True
    provenance['assets']=[register_asset(provenance,r,[]) for r in trainrows+valrows];atomic_json(pp,provenance);atomic_json(OUT/'training-config.json',value)
    model=construct().cuda();frozen=legacy_digest(model);ema=copy.deepcopy(model).eval()
    for p in ema.parameters():p.requires_grad_(False)
    optimizer,parameters=optimizer_for(model);scaler=torch.amp.GradScaler('cuda')
    source_new={k:v for k,v in model.state_dict().items() if k.startswith(('rescue_projection.','rescue_head.'))}
    ema_new={k:v for k,v in ema.state_dict().items() if k in source_new}
    data=dataset(trainrows,value['seed']);generator=torch.Generator().manual_seed(value['seed'])
    initial=evaluate(ema,faceval);reference=large_recall(initial);best=initial['metrics']['classes']['face']['ap50'];best_epoch=0
    def save(epoch):torch.save({'model':{k:v.detach().cpu() for k,v in ema.state_dict().items()},'epoch':epoch},OUT/'best.pt')
    save(0);atomic_json(OUT/'best-validation-face.json',initial)
    history=[{'epoch':0,'face_VAL_AP50':best,'large_face_recall_fixed01':reference,'large_face_gate_passed':True}];updates=0;started=time.monotonic()
    recovery_save(model,ema,optimizer,scaler,generator,0,updates,history,best_epoch,reference)
    try:
        for epoch in range(12):
            data.set_epoch(epoch);model.train();losses=[];alive(value['source_expires_at'])
            loader=torch.utils.data.DataLoader(data,batch_size=2,sampler=paired_tile_order(len(trainrows),value['seed'],epoch),generator=generator,num_workers=0,drop_last=False,pin_memory=True)
            for step,(images,labels) in enumerate(loader):
                alive(value['source_expires_at']);progress=epoch+step/len(loader);factor=.1+.9*progress if progress<1 else .1+.9*(1+math.cos(math.pi*(progress-1)/11))/2
                for group in optimizer.param_groups:group['lr']=group['base_lr']*factor
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16):loss=model.rescue_loss(images.cuda(non_blocking=True),labels.cuda(non_blocking=True))[0]
                if not torch.isfinite(loss):raise ValueError('nonfinite_native_rescue_loss')
                scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(parameters,10.)
                before=scaler.get_scale();scaler.step(optimizer);scaler.update();budget(torch)
                if scaler.get_scale()>=before:
                    updates+=1
                    with torch.no_grad():
                        for key,target in ema_new.items():
                            if target.is_floating_point():target.mul_(.99).add_(source_new[key],alpha=.01)
                            else:target.copy_(source_new[key])
                losses.append(float(loss.detach()))
                if step%200==0:print(json.dumps({'epoch':epoch+1,'tile_batch':step,'loss':losses[-1],'updates':updates}),flush=True)
            if legacy_digest(model)!=frozen or legacy_digest(ema)!=frozen:raise ValueError('frozen_legacy_state_changed')
            result=evaluate(ema,faceval);ap=result['metrics']['classes']['face']['ap50'];large=large_recall(result);passed=large>=reference-.01
            if passed and ap>best:best=ap;best_epoch=epoch+1;save(best_epoch);atomic_json(OUT/'best-validation-face.json',result)
            entry={'epoch':epoch+1,'face_VAL_AP50':ap,'large_face_recall_fixed01':large,'large_face_gate_passed':passed,'loss':float(np.mean(losses)),'updates':updates}
            history.append(entry);atomic_json(OUT/'history.json',{'history':history,'best_epoch':best_epoch,'elapsed_seconds':time.monotonic()-started,'TEST_images':0})
            recovery_save(model,ema,optimizer,scaler,generator,epoch+1,updates,history,best_epoch,reference);print(json.dumps(entry),flush=True)
    except BaseException as error:
        atomic_json(OUT/'training-status.json',{'status':'incomplete','error_type':type(error).__name__,'family_eligible':False,'TEST_images':0,'source_expires_at':value['source_expires_at']});raise
    if updates<1:raise ValueError('no_finite_optimizer_updates_training_not_complete')
    ema.load_state_dict(torch.load(OUT/'best.pt',map_location='cpu',weights_only=True)['model'],strict=True);ema.eval()
    selected=evaluate(ema,valrows);calibration=calibrate(selected)
    atomic_json(OUT/'selected-validation-predictions.json',selected);atomic_json(OUT/'confidence-calibration.json',calibration)
    if legacy_digest(ema)!=frozen or sha256(BASE)!=BASE_SHA:raise ValueError('original_legacy_changed_before_export')
    alive(value['source_expires_at']);final=copy.deepcopy(ema).cpu().eval();onnx=OUT/'model.onnx'
    torch.onnx.export(final,torch.zeros(1,3,1920,1920),str(onnx),opset_version=17,input_names=['images'],output_names=['detections'],dynamo=False)
    manifest={'format':'onnx','task':'redaction','architecture':value['architecture'],'architecture_factory_sha256':value['dependencies_sha256']['agent/redaction_rescue_network.py'],
              'classes':CLASSES,'weights':onnx.relative_to(ROOT).as_posix(),'sha256':sha256(onnx),'target_trained':True,'dataset_release':'manifest-sha256:'+sha256(RELEASE),'source_release_path':RELEASE.relative_to(ROOT).as_posix(),
              'training_provenance':pp.relative_to(ROOT).as_posix(),'input_size':[1920,1920],'shared_input_size':1920,'face_input_size':960,'plate_input_size':640,'face_tile_grid':2,
              'legacy_prefix_proposals':102900,'rescue_proposals':230400,'rescue_score_scale':.5,'rescue_stride':4,'rescue_hidden_channels':32,
              'output_layout':'yolox_decoded_xywh','confidence':calibration['selected_confidence'],'nms_iou':.5,'source_licenses':['CC-BY-NC-ND-4.0','MIT'],
              'research_only':True,'production_approved':False,'human_privacy_reviewed':False,'project_acceptance_eligible':False,'independent_route_verified':False,
              'test_split_used':False,'exported_before_final_test':True,'test_split_used_scope':'training_and_selection','data_revoked':False,'needs_revalidation':False,
              'legacy_state_sha256':frozen,'best_epoch':best_epoch,'training_complete':True,'optimizer_updates':updates,
              'rescue_head_selected_after_training':best_epoch>0,'selected_initial_legacy_fallback':best_epoch==0}
    atomic_json(OUT/'model.json',manifest);link_model(pp,OUT/'model.json')
    atomic_json(OUT/'candidate-freeze.json',{'manifest_sha256':sha256(OUT/'model.json'),'checkpoint_sha256':sha256(OUT/'best.pt'),'onnx_sha256':sha256(onnx),'confidence':manifest['confidence'],'TEST_images':0,'training_complete':True})
    atomic_json(OUT/'training-status.json',{'status':'completed','epochs':12,'best_epoch':best_epoch,'family_eligible':True,'TEST_images':0,'source_expires_at':value['source_expires_at']})
    print(json.dumps({'status':'rescue_training_complete_TEST_not_run','best_epoch':best_epoch,'face_VAL_AP50':best,'GPU_slot_release_required':True}),flush=True)

def gpu_preflight(gpu_authorized=False):
    value=require_plan(gpu_authorized)
    import torch
    if not torch.cuda.is_available():raise ValueError('authorized_CUDA_required')
    if (OUT/'gpu-preflight.json').exists():raise ValueError('GPU_preflight_already_recorded')
    configure_GPU(torch);budget(torch);torch.set_num_threads(2);cv2.setNumThreads(1);torch.manual_seed(value['seed']);trainrows,valrows,_=load_development()
    model=construct().cuda().train();frozen=legacy_digest(model);optimizer,parameters=optimizer_for(model);data=dataset(trainrows,value['seed']);phases=[];scaler=torch.amp.GradScaler('cuda')
    dense=max(range(len(trainrows)),key=lambda i:len(trainrows[i]['objects']));others=[i for i in range(len(trainrows)) if i!=dense]
    order=[]
    for a,b in [(dense,others[0]),(others[1],others[2])]:
        for tile in range(4):order.extend((a*4+tile,b*4+tile))
    for batch in range(8):
        begin=time.perf_counter();items=[data[order[batch*2+i]] for i in range(2)];images=torch.stack([x[0] for x in items]).cuda();labels=torch.stack([x[1] for x in items]).cuda();torch.cuda.synchronize();decoded=time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):loss=model.rescue_loss(images,labels)[0]
        if not torch.isfinite(loss):raise ValueError('nonfinite_preflight_native_loss')
        torch.cuda.synchronize();forward=time.perf_counter();scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(parameters,10.);scaler.step(optimizer);scaler.update();torch.cuda.synchronize();finish=time.perf_counter();budget(torch)
        phases.append({'decode_view_copy':decoded-begin,'forward_loss':forward-decoded,'backward_step':finish-forward,'total':finish-begin})
    if legacy_digest(model)!=frozen:raise ValueError('legacy_changed_during_preflight')
    model.eval();validation_probe=[valrows[i] for i in np.linspace(0,len(valrows)-1,8,dtype=int)];start=time.perf_counter();evaluate(model,validation_probe);validation_seconds=(time.perf_counter()-start)/8
    typical={key:float(np.mean([p[key] for p in phases])) for key in phases[0]}
    report={'format':'privacy-rescue-v8-authorized-GPU-resource-preflight','plan_sha256':sha256(OUT/'plan.json'),'training_source_sha256':sha256(Path(__file__)),
            'TRAIN_timing_batches':8,'batch_size':2,'phase_seconds_mean_including_cold_decode_and_kernel_startup':typical,'VAL_timing_originals':8,'VAL_seconds_per_original':validation_seconds,
            'train_originals_per_epoch':3000,'train_tiles_per_epoch':12000,'estimated_train_epoch_seconds':typical['total']*6000,
            'estimated_faceVAL400_seconds':validation_seconds*400,'estimated_12epoch_seconds_excluding_export':12*(typical['total']*6000+validation_seconds*400),
            'GPU_peak_allocated_GiB':torch.cuda.max_memory_allocated()/1024**3,'GPU_peak_reserved_GiB':torch.cuda.max_memory_reserved()/1024**3,
            'actual_training_experiment_started':False,'ephemeral_optimizer_steps':8,'TEST_images':0,'legacy_state_preserved':True,'source_expires_at':value['source_expires_at'],
            'limits':['Eight-batch resource timing only; validation metrics discarded and not used for tuning.','CPU decode, frozen feature/head loss and backward phases separated; no camera FPS claim.','Single GPU training job requires root coordination; no automatic next training.']}
    report['probe_selection']='One max-GT TRAIN original plus three first other TRAIN originals, all four tiles; conservative startup/density timing, not representative throughput.'
    atomic_json(OUT/'gpu-preflight.json',report);print(json.dumps(report))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['cpu-contract','freeze-plan','gpu-preflight','train']);parser.add_argument('--gpu-authorized',action='store_true');args=parser.parse_args()
    try:
        if args.action=='cpu-contract':cpu_contract()
        elif args.action=='freeze-plan':freeze_plan()
        elif args.action=='gpu-preflight':gpu_preflight(args.gpu_authorized)
        else:train(args.gpu_authorized)
    except Exception as error:
        if args.action=='train' and args.gpu_authorized and (OUT/'provenance.json').exists():
            status=read(OUT/'training-status.json') if (OUT/'training-status.json').exists() else {}
            if status.get('status')!='completed':atomic_json(OUT/'training-status.json',{'status':'incomplete','error_type':type(error).__name__,'family_eligible':False,'TEST_images':0,'source_expires_at':read(OUT/'plan.json')['source_expires_at']})
        print(json.dumps({'status':'rescue_stopped','error_type':type(error).__name__}),file=sys.stderr);raise SystemExit(1)
