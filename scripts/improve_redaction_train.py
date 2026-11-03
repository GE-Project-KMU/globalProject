"""Frozen two-specialist Nano research recipe; no old-test fitting or activation."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'.cache/vendor/YOLOX'))
import cv2
import numpy as np
import torch
from torch import nn
from agent.paths import atomic_json,sha256,workspace_path
from agent.dataset import validate_research_release
from agent.governance import begin_training,register_asset,link_model
from agent.redaction_network import build_redaction_network
from scripts.train_yolox_research import seed_all,evaluate_model,calibrate,operating_metrics

RELEASE=ROOT/'data/training/research-redaction-v4/release.json'
OUTPUT=ROOT/'data/training/redaction-specialists-v4-efficient'
BASE=ROOT/'data/training/redaction-nano-v3'
CLASSES=['face','license_plate']

class Frames(torch.utils.data.Dataset):
    def __init__(self,rows,kind,augment=False):self.rows,self.kind,self.augment=rows,kind,augment
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index];image=cv2.imread(str(workspace_path(row['path'],exists=True)))
        if image is None:raise ValueError('source_decode_failed')
        boxes=np.asarray([o['bbox_xywh_px'] for o in row['objects'] if o['class']==self.kind],np.float32).reshape(-1,4)
        if len(boxes)>512:raise ValueError('labels_over512_reject_not_truncate')
        h,w=image.shape[:2]
        if self.augment and random.random()<.5:
            # GT-centred zoom counteracts tiny-object resolution loss. Retain ALL
            # intersecting source annotations; no cut-label or max-count truncation.
            weights=1/np.maximum(np.minimum(boxes[:,2],boxes[:,3]),1)
            selected=boxes[random.choices(range(len(boxes)),weights=weights,k=1)[0]]
            fraction=random.uniform(.5 if self.kind=='face' else .7,1.)
            cw,ch=max(32,int(w*fraction)),max(32,int(h*fraction))
            cx,cy=selected[0]+selected[2]/2,selected[1]+selected[3]/2
            left=int(np.clip(cx-cw*random.uniform(.3,.7),0,w-cw));top=int(np.clip(cy-ch*random.uniform(.3,.7),0,h-ch))
            image=image[top:top+ch,left:left+cw];updated=[]
            for x,y,bw,bh in boxes:
                x1,y1=max(0,x-left),max(0,y-top);x2,y2=min(cw,x+bw-left),min(ch,y+bh-top)
                if x2>x1 and y2>y1:updated.append([x1,y1,x2-x1,y2-y1])
            boxes=np.asarray(updated,np.float32).reshape(-1,4);h,w=image.shape[:2]
        scale=min(960/w,960/h);image=cv2.resize(image,(max(1,round(w*scale)),max(1,round(h*scale))))
        canvas=np.full((960,960,3),114,np.uint8);canvas[:image.shape[0],:image.shape[1]]=image
        labels=np.zeros((512,5),np.float32)
        for i,(x,y,bw,bh) in enumerate(boxes):labels[i]=[0,(x+bw/2)*scale,(y+bh/2)*scale,bw*scale,bh*scale]
        if self.augment and random.random()<.5:
            canvas=canvas[:,::-1].copy();labels[:len(boxes),1]=960-labels[:len(boxes),1]
        if self.augment:
            hsv=cv2.cvtColor(canvas,cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:,:,0]=(hsv[:,:,0]+random.uniform(-5,5))%180
            hsv[:,:,1]=np.clip(hsv[:,:,1]*random.uniform(.7,1.3),0,255)
            hsv[:,:,2]=np.clip(hsv[:,:,2]*random.uniform(.75,1.25),0,255)
            canvas=cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2BGR)
        return torch.from_numpy(canvas.transpose(2,0,1).copy().astype(np.float32)),torch.from_numpy(labels),index

class Branch(nn.Module):
    def __init__(self,core,kind):super().__init__();self.core,self.kind=core,kind
    def forward(self,images,labels=None):
        if self.kind=='license_plate':
            images=torch.nn.functional.interpolate(images,size=(640,640),mode='bilinear',align_corners=False)
            if labels is not None:labels=labels.clone();labels[:,:,1:]*=640/960
        out=self.core(images,labels) if labels is not None else self.core(images)
        if labels is None and self.kind=='license_plate':out=torch.cat((out[...,:4]*1.5,out[...,4:]),dim=-1)
        return out

def warmstart(bundle):
    old=torch.load(BASE/'best.pt',map_location='cpu',weights_only=True)['model']
    for index,kind in enumerate(CLASSES):
        net=bundle.face if kind=='face' else bundle.plate
        own=net.state_dict();state={}
        for key,value in old.items():
            if key.startswith('head.cls_preds.'):value=value[index:index+1]
            if key in own and own[key].shape==value.shape:state[key]=value
        if set(own)!=set(state):raise ValueError('incomplete_specialist_warmstart')
        net.load_state_dict(state,strict=True)

def frozen_plan(batch):
    return {'architecture':'two YOLOX-Nano specialists','shared_input_size':960,'face_input_size':960,'plate_input_size':640,
        'plate_chain':'original BGR -> top-left cv2 letterbox960 -> internal Torch bilinear640 align_corners=False -> decoded xywh*1.5; IDENTICAL during training/validation/export/app',
        'face_epochs':60,'plate_epochs':20,'batch_size':batch,'max_labels':512,'seed':20261104,'optimizer':'SGD nesterov momentum0.9 weight_decay0.0005','learning_rate':.001,
        'scheduler':'3epoch linear warmup from10%, then cosine to5%; global gradient clipping10','ema':'.9998*(1-exp(-updates/2000)); checkpoint EMA validation AP50 only',
        'augmentation':'50% GT-centred zoom; include every intersecting valid GT clipped to crop. Smaller face inverse-size sampling. HSV+hflip. Final5epochs no augmentation. No mosaic shrinking tiny faces.',
        'confidence_grid':[.01,.03,.05,.1,.2,.35,.5],'nms_iou':.5,'minimum_recall_goal':.95,'confidence_selection':'merged validation only; maximize min-class recall if95%not met, otherwise precision; unchanged target',
        'warmstart_manifest_sha256':sha256(BASE/'model.json'),'warmstart_checkpoint_sha256':sha256(BASE/'best.pt'),'release_sha256':sha256(RELEASE),'training_script_sha256':sha256(Path(__file__)),
        'factory_sha256':sha256(ROOT/'agent/redaction_network.py'),'test_policy':'New400source test frozen before fitting. All prior TEST used for regression only, never fitting/selection. Final candidate and baseline each evaluated once on same newtest after checkpoint+confidence frozen.',
        'research_only':True,'production_approved':False,'source_ttl_preserved':True,'no_external_weight_distribution_clearance':True}

def smoke(batch):
    seed_all(20261104);rows=json.loads(RELEASE.read_text());bundle=build_redaction_network();warmstart(bundle)
    samples=sorted([r for r in rows if r['split']=='train' and r['evaluation_class']=='face'],key=lambda r:len(r['objects']),reverse=True)[:batch]
    data=Frames(samples,'face');images=torch.stack([data[i][0] for i in range(len(data))]).cuda();labels=torch.stack([data[i][1] for i in range(len(data))]).cuda()
    net=Branch(bundle.face,'face').cuda().train();optimizer=torch.optim.SGD(net.parameters(),lr=.001,momentum=.9,nesterov=True)
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    with torch.autocast('cuda',dtype=torch.float16):loss=net(images,labels)['total_loss']
    loss.backward();optimizer.step();torch.cuda.synchronize()
    report={'batch_size':batch,'input_size':960,'dense_gt_counts':[len(r['objects']) for r in samples],'loss':float(loss.detach()),'seconds':time.perf_counter()-start,'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,'peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2,'training_started':False}
    atomic_json(ROOT/f'docs/reports/REDACTION-V4-SMOKE-B{batch}.json',report);print(json.dumps(report),flush=True)

def train_branch(core,kind,rows,epochs,batch):
    train_rows=[r for r in rows if r['split']=='train' and r['evaluation_class']==kind];val=[r for r in rows if r['split']=='validation' and r['evaluation_class']==kind]
    model=Branch(core,kind).cuda();ema=deepcopy(model).eval()
    for p in ema.parameters():p.requires_grad_(False)
    optimizer=torch.optim.SGD(model.parameters(),lr=.001,momentum=.9,nesterov=True,weight_decay=.0005)
    scaler=torch.amp.GradScaler('cuda');data=Frames(train_rows,kind,True)
    loader=torch.utils.data.DataLoader(data,batch_size=batch,shuffle=True,num_workers=0,drop_last=True,pin_memory=True)
    history=[];best=-1;updates=0;start=time.perf_counter()
    for epoch in range(epochs):
        model.train();data.augment=epoch<epochs-5;totals=[]
        for step,(images,labels,_) in enumerate(loader):
            progress=epoch+step/len(loader)
            factor=.1+.9*progress/3 if progress<3 else .05+.95*(1+math.cos(math.pi*(progress-3)/(epochs-3)))/2
            lr=.001*factor
            for group in optimizer.param_groups:group['lr']=lr
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):loss=model(images.cuda(non_blocking=True),labels.cuda(non_blocking=True))['total_loss']
            if not torch.isfinite(loss):raise RuntimeError('nonfinite_loss_abort_no_fallback')
            scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
            previous=scaler.get_scale();scaler.step(optimizer);scaler.update()
            if scaler.get_scale()>=previous:
                updates+=1;decay=.9998*(1-math.exp(-updates/2000))
                with torch.no_grad():
                    source_state=model.state_dict()
                    for key,value in ema.state_dict().items():
                        source=source_state[key]
                        if value.dtype.is_floating_point:value.mul_(decay).add_(source.detach(),alpha=1-decay)
                        else:value.copy_(source)
            totals.append(float(loss.detach()))
            if step%100==0:print(json.dumps({'branch':kind,'epoch':epoch+1,'step':step,'loss':totals[-1],'lr':lr,'elapsed_seconds':time.perf_counter()-start}),flush=True)
        result=evaluate_model(ema,val,[kind],size=960,batch_size=batch,fullscene=True)
        metric=result['metrics']['map50'];item={'epoch':epoch+1,'loss':float(np.mean(totals)),'validation_ap50':metric,'optimizer_updates':updates}
        history.append(item);print(json.dumps({'branch':kind,**item}),flush=True)
        if metric>best:
            best=metric;torch.save({'model':ema.core.state_dict(),'epoch':epoch+1,'class':kind},OUTPUT/(kind+'-best.pt'))
        atomic_json(OUTPUT/(kind+'-history.json'),{'history':history,'elapsed_seconds':time.perf_counter()-start,'best_validation_ap50':best,'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2})
    state=torch.load(OUTPUT/(kind+'-best.pt'),map_location='cpu',weights_only=True);core.load_state_dict(state['model']);core.cpu()
    del model,ema,optimizer;torch.cuda.empty_cache()
    return {'best_epoch':state['epoch'],'best_validation_ap50':best,'updates':updates}

def train(batch):
    seed_all(20261104)
    if (OUTPUT/'training-config.json').exists():raise ValueError('fresh_output_required_no_overwrite_or_resume')
    rows=json.loads(RELEASE.read_text());validation=validate_research_release(rows)
    if not validation['valid']:raise ValueError(str(validation['errors'][:10]))
    OUTPUT.mkdir(parents=True,exist_ok=True);plan=frozen_plan(batch);atomic_json(OUTPUT/'training-config.json',plan)
    selected=[r for r in rows if r['split']!='test'];pp,provenance=begin_training(OUTPUT,RELEASE,selected,'redaction')
    provenance['research_only']=True;provenance['assets']=[register_asset(provenance,r,[]) for r in selected];atomic_json(pp,provenance)
    bundle=build_redaction_network();warmstart(bundle);stats={}
    for kind,core,epochs in [('face',bundle.face,60),('license_plate',bundle.plate,20)]:stats[kind]=train_branch(core,kind,rows,epochs,batch)
    bundle=bundle.cuda().eval();val=[r for r in rows if r['split']=='validation']
    validation_result=evaluate_model(bundle,val,CLASSES,size=960,batch_size=batch,fullscene=True)
    atomic_json(OUTPUT/'selected-validation-predictions.json',validation_result);calibration=calibrate(validation_result,CLASSES);atomic_json(OUTPUT/'confidence-calibration.json',calibration)
    # Model and operating point frozen before accessing any final test pixels.
    bundle.cpu().eval();torch.save({'model':bundle.state_dict(),'components':stats,'architecture':plan['architecture']},OUTPUT/'best.pt')
    for head in (bundle.face.head,bundle.plate.head):head.decode_in_inference=True
    torch.onnx.export(bundle,torch.zeros(1,3,960,960),str(OUTPUT/'model.onnx'),input_names=['images'],output_names=['predictions'],opset_version=17,dynamo=False)
    manifest={'format':'onnx','task':'redaction','classes':CLASSES,'architecture':plan['architecture'],'face_input_size':960,'plate_input_size':640,'weights':(OUTPUT/'model.onnx').relative_to(ROOT).as_posix(),'sha256':sha256(OUTPUT/'model.onnx'),'target_trained':True,'dataset_release':'manifest-sha256:'+sha256(RELEASE),'source_release_path':RELEASE.relative_to(ROOT).as_posix(),'training_provenance':pp.relative_to(ROOT).as_posix(),'input_size':[960,960],'output_layout':'yolox_decoded_xywh','confidence':calibration['selected_confidence'],'nms_iou':.5,'production_approved':False,'research_only':True,'human_privacy_reviewed':False,'project_acceptance_eligible':False,'source_licenses':['CC-BY-NC-ND-4.0','MIT'],'independent_route_verified':False,'test_split_used':False,'test_evaluated':False,'test_split_used_scope':'training_and_selection','data_revoked':False,'needs_revalidation':False,'architecture_factory_sha256':plan['factory_sha256'],'compute_cost':'Two separate genuine Nano networks, native960 face + native640 plate; approximately3.25x input pixels of one640Nano, not a throughput guarantee.'}
    atomic_json(OUTPUT/'model.json',manifest);link_model(pp,OUTPUT/'model.json')
    frozen={'checkpoint_sha256':sha256(OUTPUT/'best.pt'),'onnx_sha256':sha256(OUTPUT/'model.onnx'),'model_manifest_sha256':sha256(OUTPUT/'model.json'),'calibration_sha256':sha256(OUTPUT/'confidence-calibration.json'),'frozen_at':datetime.now(timezone.utc).isoformat(),'test_evaluated':False}
    atomic_json(OUTPUT/'candidate-freeze.json',frozen)
    # Equal-distribution comparison: each frozen model evaluated once, separately.
    test=[r for r in rows if r['split']=='test'];bundle.cuda()
    heldout=evaluate_model(bundle,test,CLASSES,size=960,batch_size=batch,fullscene=True);heldout['operating_metrics']=operating_metrics(heldout,CLASSES,manifest['confidence']);atomic_json(OUTPUT/'heldout-predictions.json',heldout)
    regression_rows=[r for r in json.loads((ROOT/'data/training/research-redaction-fullframe/release.json').read_text()) if r['split']=='test']
    regression=evaluate_model(bundle,regression_rows,CLASSES,size=960,batch_size=batch,fullscene=True);regression['operating_metrics']=operating_metrics(regression,CLASSES,manifest['confidence']);regression['scope']='Previously examined v3 TEST: regression ONLY, never used in fitting, confidence or checkpoint selection';atomic_json(OUTPUT/'old-v3-regression-predictions.json',regression)
    bundle.cpu();del bundle;torch.cuda.empty_cache()
    from scripts.train_yolox_research import make_model
    baseline,_=make_model(CLASSES);baseline.load_state_dict(torch.load(BASE/'best.pt',map_location='cpu',weights_only=True)['model']);baseline.cuda()
    old=evaluate_model(baseline,test,CLASSES,size=640,batch_size=batch,fullscene=True);old['operating_metrics']=operating_metrics(old,CLASSES,json.loads((BASE/'model.json').read_text())['confidence']);atomic_json(OUTPUT/'baseline-same-newtest-predictions.json',old)
    manifest['test_evaluated']=True;manifest['heldout_map50']=heldout['metrics']['map50'];atomic_json(OUTPUT/'model.json',manifest)
    summary={'architecture':plan['architecture'],'components':stats,'genuine_gradient_training':True,'research_only':True,'production_approved':False,'validation_metrics':validation_result['metrics'],'confidence_calibration':calibration,'fresh_heldout_metrics':heldout['metrics'],'fresh_operating_metrics':heldout['operating_metrics'],'old_v3_regression_metrics':regression['metrics'],'old_v3_regression_operating_metrics':regression['operating_metrics'],'baseline_same_fresh_holdout_metrics':old['metrics'],'baseline_same_fresh_operating_metrics':old['operating_metrics'],'source_release_sha256':sha256(RELEASE),'candidate_freeze':frozen,'limits':'Conditional known-class source labels only. No exhaustive privacy detection/field acceptance. New test unused in fitting/selection. Old failure benchmarks preserved, no automatic activation. Weights remain private research, redistribution rights not cleared.'}
    atomic_json(OUTPUT/'summary.json',summary);atomic_json(ROOT/'docs/reports/REDACTION-SPECIALISTS-V4-TRAINING.json',summary)
    print(json.dumps(summary),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['smoke','train']);parser.add_argument('--batch-size',type=int,default=4);args=parser.parse_args()
    if args.action=='smoke':smoke(args.batch_size)
    else:train(args.batch_size)
