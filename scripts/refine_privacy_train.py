"""One face-only fine-tune, frozen plate branch, actual tiled VAL selection.

Public publisher annotations only. No TEST pixels or user recordings are loaded.
Torch imports remain inside entry points so CPU data/policy tests stay lightweight.
"""
import argparse,copy,hashlib,json,math,random,sys,time
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import workspace_path,atomic_json,sha256
DEV=ROOT/'data/training/research-redaction-v6/development.json';RELEASE=DEV.parent/'release.json'
BASE=ROOT/'data/training/redaction-specialists-v4-efficient/best.pt'
BASE_MANIFEST=ROOT/'data/training/redaction-inference-options-v4/option-tiled/model.json'
OUT=ROOT/'data/training/redaction-refinement-v6';CLASSES=['face','license_plate']

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
        pixels=F.interpolate(torch.from_numpy(canvas.transpose(2,0,1).copy().astype(np.float32))[None],size=(960,960),mode='bilinear',align_corners=False)[0]
        labels=labels*.5
    elif view=='tile' and type(tile) is int and 0<=tile<4:
        left,top=(tile%2)*960,(tile//2)*960;pixels=torch.from_numpy(canvas[top:top+960,left:left+960].transpose(2,0,1).copy().astype(np.float32));labels=clip_boxes(labels,left,top,960,960)
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
    return {'format':'privacy-refinement-plan-v1','candidate_count':1,'training_epochs':8,'include_zero_step_baseline_for_selection':True,'seed':20261105,'batch_size':4,
        'face_train_images':3000,'face_validation_images':400,'plate_weights_frozen':True,'full_validation_images':508,'max_gt_count':512,
        'warmstart_checkpoint_sha256':sha256(BASE),'baseline_manifest_sha256':sha256(BASE_MANIFEST),'development_sha256':sha256(DEV),'release_sha256':sha256(RELEASE),
        'fresh_manifest_sha256':sha256(DEV.parent/'fresh-test.json'),'training_code_sha256':sha256(Path(__file__)),
        'tiled_factory_sha256':sha256(ROOT/'agent/redaction_tiled_network.py'),'native_evaluation_source_sha256':sha256(ROOT/'scripts/train_yolox_research.py'),
        'architecture':'two YOLOX-Nano specialists global plus four tiles','inference_unchanged':'shared1920,face960 global+fourquadrants,plate640 once; same factory/output contract',
        'training_views':'50% exact global branch bilinear1920to960,50% exact960 quadrant; tile80% choose GT-bearing quadrants,20% uniform; no mosaic',
        'labels':'retain all positive-area clipped intersecting publisher face GT; never truncate over512','augmentation':'HSV then shared-letterbox; horizontal flip after exact view',
        'optimizer':'SGD nesterov momentum.9 wd.0005 lr.0001','scheduler':'1epoch linear warmup from10%, cosine to10% byepoch8','EMA':.99,'gradient_clip':10,
        'checkpoint_metric':'face AP50 on actual global+fourtile faceVAL400 at confidence_floor.01, tie earliest; include baseline epoch0',
        'confidence_selection':'selected checkpoint fullVAL508 only; fixedgrid maximize minimum class recall then precision; prefer95% both if achieved',
        'confidence_grid':[.01,.03,.05,.1,.2,.35,.5],'nms_iou':.5,'target_minimum_class_recall':.95,
        'fresh_evaluation':'new face100/plate100 after checkpoint/confidence/numeric gate frozen; baseline+candidate once; known-class conditional only',
        'old_VAL_role':'reused development only','all_old_TEST_role':'regression only, never training/validation/selection',
        'source_licenses':['CC-BY-NC-ND-4.0','MIT'],'existing_expiries_extended':False,'independent_person_or_route_verified':False,
        'research_only':True,'production_approved':False,'user_recordings_used':False,'time_budget_estimate_minutes':[35,50],'gpu_max_total_gib':9,'gpu_start_requires_root_slot':True}

def train():
    import torch
    from agent.dataset import validate_research_release
    from agent.governance import begin_training,register_asset,link_model
    from agent.redaction_tiled_network import build_redaction_tiled_network,assemble_tiled_network
    from scripts.train_yolox_research import seed_all,evaluate_model,calibrate
    if not torch.cuda.is_available():raise ValueError('cuda_training_required')
    if OUT.exists():raise ValueError('fresh_training_output_required')
    config=plan();public=ROOT/'docs/reports/PRIVACY-REFINEMENT-PLAN.json'
    if not public.exists() or json.loads(public.read_text(encoding='utf-8'))!=config:raise ValueError('pretraining_public_plan_must_match')
    rows=json.loads(DEV.read_text(encoding='utf-8'))
    if any(r['split'] not in ('train','validation') for r in rows) or not validate_research_release(rows)['valid']:raise ValueError('invalid_development_only_release')
    seed_all(config['seed']);torch.set_num_threads(3);cv2.setNumThreads(1)
    trainrows=[r for r in rows if r['split']=='train' and r['evaluation_class']=='face'];valface=[r for r in rows if r['split']=='validation' and r['evaluation_class']=='face'];valall=[r for r in rows if r['split']=='validation']
    pp,provenance=begin_training(OUT,RELEASE,rows,'redaction');provenance['research_only']=True;provenance['assets']=[register_asset(provenance,r,[]) for r in rows];atomic_json(pp,provenance)
    atomic_json(OUT/'training-config.json',config)
    bundle=build_redaction_tiled_network();bundle.load_state_dict(torch.load(BASE,map_location='cpu',weights_only=True)['model'],strict=True)
    plate_before=state_digest(bundle.plate.state_dict())
    for parameter in bundle.plate.parameters():parameter.requires_grad_(False)
    model=bundle.face.cuda();ema=copy.deepcopy(model).eval();plate=bundle.plate.cuda().eval()
    for parameter in ema.parameters():parameter.requires_grad_(False)
    evaluation=assemble_tiled_network(ema,plate).cuda().eval()
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
                h,w=image.shape[:2];scale=min(1920/w,1920/h);scaled=boxes*scale;bearing=[i for i in range(4) if len(clip_boxes(scaled,(i%2)*960,(i//2)*960,960,960))]
                tile=random.choice(bearing)
            pixels,labels,count=view_sample(image,boxes,view=view,tile=tile)
            if random.random()<.5:pixels=pixels.flip(-1);labels[:count,1]=960-labels[:count,1]
            return pixels,labels
    loader=torch.utils.data.DataLoader(Faces(),batch_size=4,shuffle=True,num_workers=0,drop_last=True,pin_memory=True)
    optimizer=torch.optim.SGD(model.parameters(),lr=.0001,momentum=.9,nesterov=True,weight_decay=.0005);scaler=torch.amp.GradScaler('cuda')
    baseline=evaluate_model(evaluation,valface,CLASSES,size=1920,batch_size=4,fullscene=True);best=baseline['metrics']['classes']['face']['ap50'];best_epoch=0;updates=0;history=[];start=time.monotonic()
    torch.save({'model':ema.cpu().state_dict(),'epoch':0},OUT/'face-best.pt');ema.cuda();atomic_json(OUT/'best-validation-face.json',baseline)
    for epoch in range(8):
        model.train();losses=[]
        for step,(images,labels) in enumerate(loader):
            progress=epoch+step/len(loader);factor=.1+.9*progress if progress<1 else .1+.9*(1+math.cos(math.pi*(progress-1)/7))/2
            for group in optimizer.param_groups:group['lr']=.0001*factor
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
                        if value.is_floating_point():value.mul_(.99).add_(source[key],alpha=.01)
                        else:value.copy_(source[key])
            losses.append(float(loss.detach()))
            if step%200==0:print(json.dumps({'epoch':epoch+1,'step':step,'loss':losses[-1],'updates':updates}),flush=True)
        result=evaluate_model(evaluation,valface,CLASSES,size=1920,batch_size=4,fullscene=True);metric=result['metrics']['classes']['face']['ap50']
        entry={'epoch':epoch+1,'loss':float(np.mean(losses)),'validation_face_ap50':metric,'updates':updates};history.append(entry);print(json.dumps(entry),flush=True)
        if metric>best:
            best=metric;best_epoch=epoch+1;torch.save({'model':ema.cpu().state_dict(),'epoch':best_epoch},OUT/'face-best.pt');ema.cuda();atomic_json(OUT/'best-validation-face.json',result)
        atomic_json(OUT/'history.json',{'history':history,'best_epoch':best_epoch,'actual_updates':updates,'elapsed_seconds':time.monotonic()-start,'test_used':False,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/1024**3,'peak_reserved_GiB':torch.cuda.max_memory_reserved()/1024**3})
    del model,optimizer;torch.cuda.empty_cache();ema.load_state_dict(torch.load(OUT/'face-best.pt',map_location='cpu',weights_only=True)['model']);ema.eval()
    selected=evaluate_model(evaluation,valall,CLASSES,size=1920,batch_size=4,fullscene=True);atomic_json(OUT/'selected-validation-predictions.json',selected)
    calibration=calibrate(selected,CLASSES);atomic_json(OUT/'confidence-calibration.json',calibration)
    final=assemble_tiled_network(ema.cpu(),plate.cpu()).eval();plate_after=state_digest(final.plate.state_dict())
    if plate_before!=plate_after:raise ValueError('frozen_plate_weights_changed')
    torch.save({'model':final.state_dict(),'best_epoch':best_epoch,'plate_state_unchanged':True},OUT/'best.pt')
    weights=OUT/'model.onnx';torch.onnx.export(final,torch.zeros(1,3,1920,1920),str(weights),opset_version=17,input_names=['images'],output_names=['detections'],dynamo=False)
    manifest={'format':'onnx','task':'redaction','architecture':config['architecture'],'shared_input_size':1920,'face_input_size':960,'plate_input_size':640,'face_tile_grid':2,
        'classes':CLASSES,'weights':weights.relative_to(ROOT).as_posix(),'sha256':sha256(weights),'target_trained':True,'dataset_release':'manifest-sha256:'+sha256(RELEASE),'source_release_path':RELEASE.relative_to(ROOT).as_posix(),
        'training_provenance':pp.relative_to(ROOT).as_posix(),'input_size':[1920,1920],'output_layout':'yolox_decoded_xywh','confidence':calibration['selected_confidence'],'nms_iou':.5,
        'source_licenses':['CC-BY-NC-ND-4.0','MIT'],'research_only':True,'production_approved':False,'project_acceptance_eligible':False,'human_privacy_reviewed':False,'independent_route_verified':False,
        'test_split_used':False,'test_evaluated':False,'test_split_used_scope':'training_and_selection','data_revoked':False,'needs_revalidation':False,'best_epoch':best_epoch,'plate_state_unchanged':True}
    atomic_json(OUT/'model.json',manifest);link_model(pp,OUT/'model.json');atomic_json(OUT/'candidate-freeze.json',{'model_manifest_sha256':sha256(OUT/'model.json'),'checkpoint_sha256':sha256(OUT/'best.pt'),'onnx_sha256':sha256(weights),'confidence':manifest['confidence'],'test_evaluated':False})
    print(json.dumps({'status':'trained','epochs':8,'best_epoch':best_epoch,'validation_face_ap50':best,'plate_state_unchanged':True}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze-plan','train']);a=p.parse_args()
    if a.action=='freeze-plan':atomic_json(ROOT/'docs/reports/PRIVACY-REFINEMENT-PLAN.json',plan())
    else:train()
