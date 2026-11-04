"""Cycle2 P2 face640, explicit v4 transplant, frozen plate, development selection.

Public publisher annotations only. No TEST pixels or user recordings are loaded.
Torch imports remain inside entry points so CPU data/policy tests stay lightweight.
"""
import argparse,copy,hashlib,json,math,random,sys,time
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import workspace_path,atomic_json,sha256
DEV=ROOT/'data/training/research-redaction-cycle2-v7/development.json';RELEASE=DEV.parent/'release.json'
BASE=ROOT/'data/training/redaction-specialists-v4-efficient/best.pt'
BASE_MANIFEST=ROOT/'data/training/redaction-inference-options-v4/option-tiled/model.json'
OUT=ROOT/'data/training/redaction-p2-cycle2-v7';CLASSES=['face','license_plate']

def clip_boxes(boxes,left,top,width,height):
    result=[]
    for x,y,w,h in boxes:
        x1,y1=max(0.,x-left),max(0.,y-top);x2,y2=min(float(width),x+w-left),min(float(height),y+h-top)
        if x2>x1 and y2>y1:result.append([x1,y1,x2-x1,y2-y1])
    return np.asarray(result,np.float32).reshape(-1,4)

def view_sample(image,boxes,*,view='global',tile=0):
    import torch
    from torch.nn import functional as F
    boxes=np.asarray(boxes,np.float32).reshape(-1,4)
    if len(boxes)>512:raise ValueError('reject_original_gt_count_over512_no_truncation')
    h,w=image.shape[:2];scale=min(1920/w,1920/h)
    resized=cv2.resize(image,(max(1,round(w*scale)),max(1,round(h*scale))));canvas=np.full((1920,1920,3),114,np.uint8);canvas[:resized.shape[0],:resized.shape[1]]=resized
    labels=boxes*scale
    if view=='global':
        pixels=F.interpolate(torch.from_numpy(canvas.transpose(2,0,1).copy().astype(np.float32))[None],size=(640,640),mode='bilinear',align_corners=False)[0]
        labels=labels/3.
    elif view=='tile' and type(tile) is int and 0<=tile<4:
        left,top=(tile%2)*960,(tile//2)*960;pixels=torch.from_numpy(canvas[top:top+960,left:left+960].transpose(2,0,1).copy().astype(np.float32));labels=clip_boxes(labels,left,top,960,960)*(2./3.)
        pixels=F.interpolate(pixels[None],size=(640,640),mode='bilinear',align_corners=False)[0]
    else:raise ValueError('unsupported_frozen_view')
    packed=torch.zeros(512,5,dtype=torch.float32)
    for index,(x,y,w,h) in enumerate(labels):packed[index]=torch.tensor([0,x+w/2,y+h/2,w,h])
    return pixels,packed,len(labels)

def state_digest(state):
    result=hashlib.sha256()
    for key,value in sorted(state.items()):
        value=value.detach().cpu().contiguous();result.update(key.encode());result.update(str(value.dtype).encode());result.update(str(tuple(value.shape)).encode());result.update(value.numpy().tobytes())
    return result.hexdigest()

def plan():
    from agent.redaction_p2_network import ARCHITECTURE
    return {'format':'privacy-P2-cycle2-plan-v2','parent_plan_sha256':sha256(ROOT/'docs/reports/PRIVACY-P2-CYCLE2-PLAN.json'),'parent_source_sha256':'2335f562c36a64e5a36bff12708c3ab554b1717a93fe4fb8c74c4a8b8b352724','revision_reason':'pre-training shared-GPU resource revision: batch8to4/body-head LR halved/EMA sqrt(.99); same epochs/data/architecture/VAL selection/TEST; not mathematically equivalent training','candidate_count':1,'training_epochs':12,'seed':20261106,'batch_size':4,
        'warmstart_checkpoint_sha256':sha256(BASE),'baseline_manifest_sha256':sha256(BASE_MANIFEST),'development_sha256':sha256(DEV),'release_sha256':sha256(RELEASE),'fresh_manifest_sha256':sha256(DEV.parent/'fresh-test.json'),
        'training_code_sha256':sha256(Path(__file__)),'data_preparation_source_sha256':sha256(ROOT/'scripts/cycle2_public_data.py'),'factory_sha256':sha256(ROOT/'agent/redaction_p2_network.py'),'native_evaluation_source_sha256':sha256(ROOT/'scripts/train_yolox_research.py'),'slice_audit_source_sha256':sha256(ROOT/'scripts/nextcycle_val_audit.py'),
        'architecture':ARCHITECTURE,'shared_input_size':1920,'face_input_size':640,'face_strides':[4,8,16],'plate_input_size':640,'output_shape':[1,176400,7],
        'new_model_contract_not_precision_optimization':True,'baseline':'v4 tiled; v6 declined after fresh regression; never warmstart v6',
        'warmstart':'active backbone copied byte-exact fromv4; stride8/16 head correspondence copied; new P2 stem and neck initialized; every assignment audited; full-state-identical claim false',
        'plate_weights_and_BN_buffers_frozen':True,'max_gt_count':512,'training_views':'global1920->64050%; crop960->64050%; tile80% small-GT-bearing when available else any-GT-bearing,20%uniform; exact single bilinear resize',
        'training_sampler':'replacement weighted images 1+2*fraction GT with effective tile sqrtarea<16;3000 draws/epoch; keep all publisher GT, no silent truncation',
        'augmentation':'HSV and horizontal flip; no mosaic; native L1 localization loss enabled last3epochs','optimizer':'SGD Nesterov momentum.9 wd.0005, activebackbone lr.00005, newneck/head lr.0005','scheduler':'1epoch linear warmup.1to1 then cosine to.1 byepoch12','EMA':math.sqrt(.99),'gradient_clip':10,
        'BN':'pretrained activebackbone running stats frozen; affine trained; newneck/head BN trained','checkpoint_selection':'max face AP50 VAL400 atfloor.01 among largeface originalsqrtarea>=64 Recall >=baseline-.01; include epoch0; if none passes, select bestAP for explicit rejected research diagnostic only',
        'large_face_reference_VAL_recall':.9685534591194969,'large_face_max_recall_drop':.01,'smallface_VAL_slices':['<16','16..31','32..63','>=64'],'confidence_grid':[.01,.03,.05,.1,.2,.35,.5],'confidence_selection':'selected checkpoint fullVAL508 only; maximize minclass recall then precision;95% both ifachievable','nms_iou':.5,
        'fresh_evaluation':'newface100/plate100 frozen after allhistoricalTESTsource/hash/near/plateidentity exclusion; baseline+candidate once after selection/numeric gates','old_TEST':'regressiononly; never training/validation/selection','known_class_conditional_only':True,'physical_person_or_route_independence_verified':False,
        'source_licenses':['CC-BY-NC-ND-4.0','MIT'],'existing_expiries_extended':False,'user_recordings_used':False,'research_only':True,'production_approved':False,'exported_before_final_test':True,'time_budget_estimate_minutes':[90,120],'gpu_max_total_gib':9,'gpu_start_requires_root_slot':True}

def validation_slice(result,rows):
    from scripts.nextcycle_val_audit import match_groups
    stats,_=match_groups(result['truth'],result['predictions'],{r['sample_id']:r for r in rows})
    large=stats.get('>=64',{}).get('recall')
    return stats,large is not None and large>=.9685534591194969-.01


def checkpoint_is_better(metric,passes,best_eligible,best_overall):
    return (passes and metric>best_eligible) or (best_eligible<0 and metric>best_overall)

def training_image_weight(row):
    h,w=row['image']['height'],row['image']['width']
    if h<=0 or w<=0:raise ValueError('positive_image_dimensions_required')
    scale=min(1920/w,1920/h)*(2./3.)
    areas=[math.sqrt(o['bbox_xywh_px'][2]*o['bbox_xywh_px'][3])*scale for o in row['objects'] if o['class']=='face']
    if not areas:raise ValueError('face_training_labels_required')
    return 1+2*sum(a<16 for a in areas)/len(areas)

def train():
    import torch
    from agent.dataset import validate_research_release
    from agent.governance import begin_training,register_asset,link_model
    from agent.redaction_p2_network import build_redaction_p2_network,assemble_p2_network,load_p2_warmstart,ARCHITECTURE
    from scripts.train_yolox_research import seed_all,evaluate_model,calibrate
    if not torch.cuda.is_available():raise ValueError('cuda_training_required')
    if OUT.exists():raise ValueError('fresh_training_output_required')
    config=plan();public=ROOT/'docs/reports/PRIVACY-P2-CYCLE2-PLAN-V2.json'
    if not public.exists() or json.loads(public.read_text(encoding='utf-8'))!=config:raise ValueError('pretraining_public_plan_must_match')
    rows=json.loads(DEV.read_text(encoding='utf-8'))
    if any(r['split'] not in ('train','validation') for r in rows) or not validate_research_release(rows)['valid']:raise ValueError('invalid_development_only_release')
    seed_all(config['seed']);torch.set_num_threads(3);cv2.setNumThreads(1)
    trainrows=[r for r in rows if r['split']=='train' and r['evaluation_class']=='face'];valface=[r for r in rows if r['split']=='validation' and r['evaluation_class']=='face'];valall=[r for r in rows if r['split']=='validation']
    pp,provenance=begin_training(OUT,RELEASE,rows,'redaction');provenance['research_only']=True;provenance['assets']=[register_asset(provenance,r,[]) for r in rows];atomic_json(pp,provenance)
    atomic_json(OUT/'training-config.json',config)
    bundle=build_redaction_p2_network();transfer=load_p2_warmstart(bundle,torch.load(BASE,map_location='cpu',weights_only=True)['model']);atomic_json(OUT/'warmstart-transfer.json',transfer)
    plate_before=state_digest(bundle.plate.state_dict())
    for parameter in bundle.plate.parameters():parameter.requires_grad_(False)
    model=bundle.face.cuda();ema=copy.deepcopy(model).eval();plate=bundle.plate.cuda().eval()
    for parameter in ema.parameters():parameter.requires_grad_(False)
    evaluation=assemble_p2_network(ema,plate).cuda().eval()
    atomic_json(OUT/'environment.json',{'torch':torch.__version__,'CUDA':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'opencv':cv2.__version__,'python':sys.version,'plate_state_sha256_before':plate_before})
    class Faces(torch.utils.data.Dataset):
        def __len__(self):return len(trainrows)
        def __getitem__(self,index):
            row=trainrows[index];image=cv2.imread(str(workspace_path(row['path'],exists=True)))
            if image is None:raise ValueError('public_source_decode_failed')
            boxes=np.asarray([o['bbox_xywh_px'] for o in row['objects'] if o['class']=='face'],np.float32)
            hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV).astype(np.float32);hsv[:,:,0]=(hsv[:,:,0]+random.uniform(-5,5))%180;hsv[:,:,1]=np.clip(hsv[:,:,1]*random.uniform(.7,1.3),0,255);hsv[:,:,2]=np.clip(hsv[:,:,2]*random.uniform(.75,1.25),0,255)
            image=cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2BGR)
            view='tile' if random.random()<.5 else 'global';tile=random.randrange(4)
            if view=='tile' and len(boxes) and random.random()<.8:
                h,w=image.shape[:2];scale=min(1920/w,1920/h);scaled=boxes*scale;small=scaled[np.sqrt(scaled[:,2]*scaled[:,3])*(2./3.)<16];focus=small if len(small) else scaled;bearing=[i for i in range(4) if len(clip_boxes(focus,(i%2)*960,(i//2)*960,960,960))]
                tile=random.choice(bearing)
            pixels,labels,count=view_sample(image,boxes,view=view,tile=tile)
            if random.random()<.5:pixels=pixels.flip(-1);labels[:count,1]=640-labels[:count,1]
            return pixels,labels
    weights=[training_image_weight(row) for row in trainrows]
    sampler=torch.utils.data.WeightedRandomSampler(weights,len(trainrows),replacement=True)
    loader=torch.utils.data.DataLoader(Faces(),batch_size=4,sampler=sampler,num_workers=0,drop_last=True,pin_memory=True)
    body=list(model.backbone.backbone.parameters());bodyids={id(p) for p in body};other=[p for p in model.parameters() if id(p) not in bodyids]
    optimizer=torch.optim.SGD([{'params':body,'lr':.00005,'base_lr':.00005},{'params':other,'lr':.0005,'base_lr':.0005}],momentum=.9,nesterov=True,weight_decay=.0005);scaler=torch.amp.GradScaler('cuda')
    baseline=evaluate_model(evaluation,valface,CLASSES,size=1920,batch_size=4,fullscene=True);initial=baseline['metrics']['classes']['face']['ap50'];initial_slices,eligible=validation_slice(baseline,valface);best=initial if eligible else -1.;best_overall=initial;best_epoch=0;updates=0;history=[];start=time.monotonic()
    torch.save({'model':ema.cpu().state_dict(),'epoch':0},OUT/'face-best.pt');ema.cuda();atomic_json(OUT/'best-validation-face.json',baseline)
    for epoch in range(12):
        model.train();model.head.use_l1=epoch>=9
        from agent.lane_network import freeze_batch_norm_statistics
        freeze_batch_norm_statistics(model.backbone.backbone);losses=[]
        for step,(images,labels) in enumerate(loader):
            progress=epoch+step/len(loader);factor=.1+.9*progress if progress<1 else .1+.9*(1+math.cos(math.pi*(progress-1)/11))/2
            for group in optimizer.param_groups:group['lr']=group['base_lr']*factor
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):loss=model(images.cuda(non_blocking=True),labels.cuda(non_blocking=True))['total_loss']
            if not torch.isfinite(loss):raise ValueError('nonfinite_loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),10.);before=scaler.get_scale();scaler.step(optimizer);scaler.update()
            free,total=torch.cuda.mem_get_info()
            if total-free>9*1024**3:raise ValueError('shared_GPU_memory_budget_9GiB_exceeded')
            if scaler.get_scale()>=before:
                updates+=1
                with torch.no_grad():
                    source=model.state_dict()
                    for key,value in ema.state_dict().items():
                        if value.is_floating_point():value.mul_(math.sqrt(.99)).add_(source[key],alpha=1-math.sqrt(.99))
                        else:value.copy_(source[key])
            losses.append(float(loss.detach()))
            if step%200==0:print(json.dumps({'epoch':epoch+1,'step':step,'loss':losses[-1],'updates':updates}),flush=True)
        result=evaluate_model(evaluation,valface,CLASSES,size=1920,batch_size=4,fullscene=True);metric=result['metrics']['classes']['face']['ap50'];slices,passes=validation_slice(result,valface)
        entry={'epoch':epoch+1,'loss':float(np.mean(losses)),'validation_face_ap50':metric,'validation_face_size_slices':slices,'large_face_gate_passed':passes,'updates':updates};history.append(entry);print(json.dumps(entry),flush=True)
        if checkpoint_is_better(metric,passes,best,best_overall):
            if passes:best=metric
            best_overall=max(best_overall,metric);best_epoch=epoch+1;torch.save({'model':ema.cpu().state_dict(),'epoch':best_epoch},OUT/'face-best.pt');ema.cuda();atomic_json(OUT/'best-validation-face.json',result)
        atomic_json(OUT/'history.json',{'history':history,'best_epoch':best_epoch,'actual_updates':updates,'elapsed_seconds':time.monotonic()-start,'test_used':False,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/1024**3,'peak_reserved_GiB':torch.cuda.max_memory_reserved()/1024**3})
    del model,optimizer;torch.cuda.empty_cache();ema.load_state_dict(torch.load(OUT/'face-best.pt',map_location='cpu',weights_only=True)['model']);ema.eval()
    selected=evaluate_model(evaluation,valall,CLASSES,size=1920,batch_size=4,fullscene=True);atomic_json(OUT/'selected-validation-predictions.json',selected)
    calibration=calibrate(selected,CLASSES);atomic_json(OUT/'confidence-calibration.json',calibration)
    final=assemble_p2_network(ema.cpu(),plate.cpu()).eval();plate_after=state_digest(final.plate.state_dict())
    if plate_before!=plate_after:raise ValueError('frozen_plate_weights_changed')
    torch.save({'model':final.state_dict(),'best_epoch':best_epoch,'plate_state_unchanged':True,'development_quality_gate_passed':validation_slice(selected,valface)[1]},OUT/'best.pt')
    weights=OUT/'model.onnx';torch.onnx.export(final,torch.zeros(1,3,1920,1920),str(weights),opset_version=17,input_names=['images'],output_names=['detections'],dynamo=False)
    manifest={'format':'onnx','task':'redaction','architecture':config['architecture'],'shared_input_size':1920,'face_input_size':640,'plate_input_size':640,'face_tile_grid':2,
        'classes':CLASSES,'weights':weights.relative_to(ROOT).as_posix(),'sha256':sha256(weights),'target_trained':True,'dataset_release':'manifest-sha256:'+sha256(RELEASE),'source_release_path':RELEASE.relative_to(ROOT).as_posix(),
        'training_provenance':pp.relative_to(ROOT).as_posix(),'input_size':[1920,1920],'output_layout':'yolox_decoded_xywh','confidence':calibration['selected_confidence'],'nms_iou':.5,
        'source_licenses':['CC-BY-NC-ND-4.0','MIT'],'research_only':True,'production_approved':False,'project_acceptance_eligible':False,'human_privacy_reviewed':False,'independent_route_verified':False,
        'test_split_used':False,'exported_before_final_test':True,'test_split_used_scope':'training_and_selection','data_revoked':False,'needs_revalidation':False,'best_epoch':best_epoch,'plate_state_unchanged':True,'development_quality_gate_passed':validation_slice(selected,valface)[1]}
    atomic_json(OUT/'model.json',manifest);link_model(pp,OUT/'model.json');atomic_json(OUT/'candidate-freeze.json',{'model_manifest_sha256':sha256(OUT/'model.json'),'checkpoint_sha256':sha256(OUT/'best.pt'),'onnx_sha256':sha256(weights),'confidence':manifest['confidence'],'test_evaluated':False})
    print(json.dumps({'status':'trained','epochs':12,'best_epoch':best_epoch,'validation_face_ap50':selected['metrics']['classes']['face']['ap50'],'development_quality_gate_passed':validation_slice(selected,valface)[1],'plate_state_unchanged':True}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze-plan','train']);a=p.parse_args()
    if a.action=='freeze-plan':atomic_json(ROOT/'docs/reports/PRIVACY-P2-CYCLE2-PLAN-V2.json',plan())
    else:train()
