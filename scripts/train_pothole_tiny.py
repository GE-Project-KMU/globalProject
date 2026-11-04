"""Predeclared capacity/augmentation candidate; development VAL only selects."""
import argparse,json,math,sys,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'.cache/vendor/YOLOX'))
from agent.paths import workspace_path,atomic_json,sha256
from agent.dataset import validate_research_release
BASE='data/training/pothole-mwpd-v6'
DEST='data/training/pothole-tiny-v7'
PRETRAIN='.cache/training/pretrained/yolox_tiny_0.1.1rc0.pth'
PRETRAIN_SHA='9de513de589ac98bb92d3bca53b5af7b9acfa9b0bacb831f7999d0f7afaee8f0'
DEPS=['scripts/train_yolox_research.py','scripts/improve_pothole_train.py',
      '.cache/vendor/YOLOX/exps/default/yolox_tiny.py',
      '.cache/vendor/YOLOX/yolox/data/datasets/mosaicdetection.py',
      '.cache/vendor/YOLOX/yolox/data/data_augment.py',
      '.cache/vendor/YOLOX/yolox/models/yolo_head.py']

def read(path):return json.loads(workspace_path(path,exists=True).read_text(encoding='utf-8'))

def choose(history):
    if not history or history[0]['epoch']!=0:raise ValueError('baseline_validation_required')
    for i,row in enumerate(history):
        if row['epoch']!=i or any(type(row[k]) not in (float,int) or not math.isfinite(row[k]) or not 0<=row[k]<=1 for k in ('old_validation_ap50','new_validation_ap50')):raise ValueError('ordered_finite_validation_history_required')
    eligible=[r for r in history if r['old_validation_ap50']>=history[0]['old_validation_ap50']-.03]
    return max(eligible,key=lambda r:((r['old_validation_ap50']+r['new_validation_ap50'])/2,-r['epoch']))

def learning_rate(step,total,warmup,maximum):
    if not 0<=step<total or not 0<warmup<total:raise ValueError('invalid_scheduler_step')
    if step<warmup:return maximum*(.1+.9*(step+1)/warmup)
    return maximum*(.1+.9*(1+math.cos(math.pi*(step-warmup)/(total-warmup)))/2)

def make_tiny(pretrained=None):
    import torch
    from yolox.exp import get_exp
    exp=get_exp(exp_file=str(ROOT/'.cache/vendor/YOLOX/exps/default/yolox_tiny.py'));exp.num_classes=1
    model=exp.get_model();transferred=0;missing=[]
    if pretrained:
        path=workspace_path(pretrained,exists=True)
        if sha256(path)!=PRETRAIN_SHA:raise ValueError('official_initialization_hash_mismatch')
        state=torch.load(path,map_location='cpu',weights_only=True)['model'];own=model.state_dict()
        compatible={k:v for k,v in state.items() if k in own and own[k].shape==v.shape}
        missing=sorted(set(own)-set(compatible))
        expected=sorted(f'head.cls_preds.{i}.{kind}' for i in range(3) for kind in ('weight','bias'))
        if missing!=expected:raise ValueError('only_COCO_classifiers_may_be_reinitialized')
        model.load_state_dict(compatible,strict=False);transferred=len(compatible)
    return model,exp,{'transferred_tensors':transferred,'reinitialized_keys':missing}

def prepare(output=DEST):
    output=workspace_path(output)
    if output.exists() or not output.is_relative_to(ROOT/'data/training'):raise ValueError('new_training_directory_required')
    rows=[r for r in read(BASE+'/release.json') if r['split'] in ('train','validation')]
    if not validate_research_release(rows)['valid']:raise ValueError('valid_development_only_release_required')
    if any(len(r['objects'])>30 for r in rows):raise ValueError('source_target_cap_exceeded')
    if sha256(ROOT/PRETRAIN)!=PRETRAIN_SHA:raise ValueError('official_initialization_hash_mismatch')
    output.mkdir(parents=True);atomic_json(output/'release.json',rows)
    plan={'format':'pothole-tiny-development-plan-v1','candidate_count':1,'architecture':'YOLOX-Tiny',
          'baseline_manifest':BASE+'/model.json','baseline_checkpoint':BASE+'/best.pt',
          'baseline_manifest_sha256':sha256(ROOT/(BASE+'/model.json')),'baseline_checkpoint_sha256':sha256(ROOT/(BASE+'/best.pt')),
          'parent_release_sha256':sha256(ROOT/(BASE+'/release.json')),'development_release_sha256':sha256(output/'release.json'),
          'pretrained_path':PRETRAIN,'pretrained_sha256':PRETRAIN_SHA,
          'pretrained_url':'https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.pth',
          'trainer_sha256':sha256(Path(__file__)),'dependencies':{p:sha256(ROOT/p) for p in DEPS},
          'seed':20261105,'maximum_epochs':36,'minimum_epochs':18,'patience_epochs':10,'batch_size':16,'input_size':640,
          'learning_rate':.0025,'optimizer':'native SGD momentum.9 Nesterov wd.0005, BN/bias no decay',
          'scheduler':'one epoch 10%-100% linear warmup, cosine to10% maximum at maximum epoch36',
          'batchnorm_statistics':'trainable, transferred COCO statistics; native EMA includes buffers',
          'sampling':'outer replacement equal expected RDD/MWPD; 4376 images/epoch; mosaic inner donors uniform in eligible TRAIN',
          'augmentation':'native mosaic probability.5 scale.75..1.25 degrees0 shear0 translate.1; HSV.5 flip.5; no mixup; mosaic disabled last5 maximum epochs',
          'target_policy':'source max30, four-image mosaic upper bound120 and padded120; native clipped/subpixel filtering, no source target truncation',
          'selection':'max mean(RDD VAL AP50,MWPD VAL AP50), RDD drop<=3pp vs current baseline; epoch0 current baseline retained; ties earliest',
          'early_stop':'after minimum18, stop at10 epochs without a new eligible best; never uses TEST',
          'confidence':.1,'confidence_floor_for_AP':.01,'nms_iou':.5,
          'test_policy':'all prior TEST remain regression-only, never selected or trained; fresh external source protocol frozen separately before model inference',
          'source_expiry':min(r['expires_at'] for r in rows),'source_expiry_extended':False,
          'research_only':True,'production_approved':False,'user_recordings_used':False,'weights_uploaded':False}
    atomic_json(output/'plan.json',plan)
    atomic_json(ROOT/'docs/reports/POTHOLE-TINY-V7-PLAN.json',{k:v for k,v in plan.items() if k not in ('pretrained_path','baseline_manifest','baseline_checkpoint')})
    print(json.dumps({'plan_sha256':sha256(output/'plan.json'),'training_images':sum(r['split']=='train' for r in rows),'validation_images':sum(r['split']=='validation' for r in rows),'TEST_images_in_development':0}),flush=True)

def preflight(output=DEST):
    import cv2,torch
    from scripts.improve_pothole_train import OriginalSamples
    from yolox.data.datasets.mosaicdetection import MosaicDetection
    from yolox.data.data_augment import TrainTransform
    torch.set_num_threads(2);cv2.setNumThreads(1);torch.manual_seed(20261105)
    if torch.cuda.is_initialized():raise ValueError('CPU_contract_must_not_initialize_GPU')
    output=workspace_path(output,exists=True);rows=read(output/'release.json');train=[r for r in rows if r['split']=='train']
    selected=[next(r for r in train if r['development_domain']==d and r['objects']) for d in ('original','mwpd')]
    dataset=MosaicDetection(OriginalSamples(selected,640,0),img_size=(640,640),mosaic=False,preproc=TrainTransform(max_labels=120,hsv_prob=0.,flip_prob=0.),enable_mixup=False)
    images,labels,_,_=next(iter(torch.utils.data.DataLoader(dataset,batch_size=2,num_workers=0)))
    if (labels[:,:,3]>0).sum(1).tolist()!=[len(r['objects']) for r in selected]:raise ValueError('all_source_GT_must_be_preserved')
    model,_,transfer=make_tiny(PRETRAIN);model.train();loss=model(images,labels)['total_loss']
    loss.backward();grads=[p.grad for p in model.parameters() if p.grad is not None]
    if not torch.isfinite(loss) or not grads or not all(torch.isfinite(g).all() for g in grads) or not any(torch.count_nonzero(g) for g in grads):raise ValueError('finite_nonzero_backward_required')
    model.eval();model.head.decode_in_inference=True
    with torch.no_grad():prediction=model(images[:1])
    if list(prediction.shape)!=[1,8400,6] or not torch.isfinite(prediction).all():raise ValueError('decoded_single_class_contract_failed')
    report={'format':'pothole-tiny-contract-v1','trainer_sha256':sha256(Path(__file__)),'release_sha256':sha256(output/'release.json'),
            'pretrained_sha256':PRETRAIN_SHA,'transfer':transfer,'batch_shape':list(images.shape),'targets_shape':list(labels.shape),'decoded_output_shape':list(prediction.shape),
            'finite_loss':float(loss.detach()),'finite_nonzero_gradients':True,'all_GT_preserved':True,'TEST_images_used':0,'optimizer_updates':0,'GPU_initialized':torch.cuda.is_initialized()}
    atomic_json(output/'preflight.json',report);atomic_json(ROOT/'docs/reports/POTHOLE-TINY-V7-CONTRACT.json',report);print(json.dumps(report),flush=True)

def train(output=DEST):
    import cv2,numpy as np,torch
    from scripts.train_yolox_research import make_model,evaluate_model,seed_all
    from scripts.improve_pothole_train import OriginalSamples
    from agent.governance import begin_training,register_asset,link_model
    from yolox.data.datasets.mosaicdetection import MosaicDetection
    from yolox.data.data_augment import TrainTransform
    from yolox.utils import ModelEMA
    output=workspace_path(output,exists=True);plan=read(output/'plan.json')
    if sha256(Path(__file__))!=plan['trainer_sha256'] or any(sha256(ROOT/p)!=s for p,s in plan['dependencies'].items()):raise ValueError('frozen_training_code_changed')
    for path,digest in [(output/'release.json',plan['development_release_sha256']),(plan['baseline_manifest'],plan['baseline_manifest_sha256']),(plan['baseline_checkpoint'],plan['baseline_checkpoint_sha256'])]:
        if sha256(workspace_path(path,exists=True))!=digest:raise ValueError('frozen_training_input_changed')
    if (output/'provenance.json').exists():raise ValueError('training_already_started_preserve_original_run')
    proof=read(output/'preflight.json')
    if proof['trainer_sha256']!=plan['trainer_sha256'] or proof['release_sha256']!=plan['development_release_sha256'] or proof['finite_nonzero_gradients'] is not True:raise ValueError('matching_CPU_contract_required')
    if not torch.cuda.is_available():raise RuntimeError('CUDA_required')
    seed_all(plan['seed']);torch.set_num_threads(2);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rows=read(output/'release.json')
    if not validate_research_release(rows)['valid'] or any(r['split']=='test' for r in rows):raise ValueError('valid_development_only_release_required')
    training=[r for r in rows if r['split']=='train'];validation={d:[r for r in rows if r['split']=='validation' and r['development_domain']==d] for d in ('original','mwpd')}
    old_n=sum(r['development_domain']=='original' for r in training);new_n=len(training)-old_n
    provenance_path,provenance=begin_training(output,output/'release.json',rows,'pothole');provenance['research_only']=True
    provenance['assets']=[register_asset(provenance,r,[]) for r in rows];atomic_json(provenance_path,provenance)
    started=time.perf_counter();history=[];updates=0;attempts=0
    baseline,_=make_model(['pothole']);baseline.load_state_dict(torch.load(ROOT/plan['baseline_checkpoint'],map_location='cpu',weights_only=True)['model'],strict=True);baseline.cuda()
    scores={d:evaluate_model(baseline,v,['pothole'],size=640,batch_size=16)['metrics']['map50'] for d,v in validation.items()}
    del baseline;torch.cuda.empty_cache()
    history.append({'epoch':0,'old_validation_ap50':scores['original'],'new_validation_ap50':scores['mwpd'],'optimizer_updates':0,'mean_training_loss':None})
    model,exp,transfer=make_tiny(plan['pretrained_path']);model.cuda();exp.model=model;optimizer=exp.get_optimizer(16)
    dataset=MosaicDetection(OriginalSamples(training,640,2),img_size=(640,640),mosaic=True,mosaic_prob=.5,mosaic_scale=(.75,1.25),degrees=0.,shear=0.,translate=.1,preproc=TrainTransform(max_labels=120,hsv_prob=.5,flip_prob=.5),enable_mixup=False)
    sampler=torch.utils.data.WeightedRandomSampler([1/(old_n if r['development_domain']=='original' else new_n) for r in training],2*new_n,replacement=True,generator=torch.Generator().manual_seed(plan['seed']))
    loader=torch.utils.data.DataLoader(dataset,batch_size=16,sampler=sampler,drop_last=True,num_workers=0,pin_memory=True)
    ema=ModelEMA(model);scaler=torch.amp.GradScaler('cuda');total=len(loader)*plan['maximum_epochs'];last_best=0
    config=plan|{'transfer':transfer,'parameters':sum(p.numel() for p in model.parameters()),'batches_per_epoch':len(loader),'torch':torch.__version__,'device':torch.cuda.get_device_name(),'GPU_peak_policy_GiB':9,'image_cache_bytes':dataset._dataset.cache_bytes}
    atomic_json(output/'training-config.json',config);print(json.dumps({'epoch':0,**history[0],'parameters':config['parameters']}),flush=True)
    for epoch in range(1,plan['maximum_epochs']+1):
        if datetime.fromisoformat(plan['source_expiry'].replace('Z','+00:00'))<=datetime.now(timezone.utc):raise ValueError('source_expired_stop_training')
        model.train();dataset.enable_mosaic=epoch<=plan['maximum_epochs']-5;losses=[]
        for step,(images,labels,_,_) in enumerate(loader):
            lr=learning_rate(attempts,total,len(loader),plan['learning_rate'])
            for group in optimizer.param_groups:group['lr']=lr
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):loss=model(images.cuda(non_blocking=True),labels.cuda(non_blocking=True))['total_loss']
            if not torch.isfinite(loss):raise RuntimeError('nonfinite_training_loss')
            before=scaler.get_scale();scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
            scaler.step(optimizer);scaler.update();attempts+=1
            if scaler.get_scale()>=before:updates+=1;ema.update(model)
            losses.append(float(loss.detach()))
            if torch.cuda.max_memory_reserved()>9*2**30:raise RuntimeError('training_GPU_memory_policy_exceeded')
            if step%100==0:print(json.dumps({'epoch':epoch,'step':step,'steps':len(loader),'actual_updates':updates,'loss':losses[-1],'lr':lr}),flush=True)
        previous=choose(history)['epoch'];scores={d:evaluate_model(ema.ema,v,['pothole'],size=640,batch_size=16)['metrics']['map50'] for d,v in validation.items()}
        row={'epoch':epoch,'old_validation_ap50':scores['original'],'new_validation_ap50':scores['mwpd'],'optimizer_updates':updates,'mean_training_loss':float(np.mean(losses))};history.append(row);selected=choose(history)
        if selected['epoch']==epoch and epoch!=previous:
            last_best=epoch;torch.save({'model':ema.ema.state_dict(),'epoch':epoch,'classes':['pothole'],'architecture':'YOLOX-Tiny'},output/'best.pt')
        atomic_json(output/'history.json',{'history':history,'selected':selected,'actual_updates':updates,'attempts':attempts,'elapsed_seconds':time.perf_counter()-started,'peak_reserved_GiB':torch.cuda.max_memory_reserved()/2**30})
        print(json.dumps(row|{'selected_epoch':selected['epoch'],'elapsed_seconds':time.perf_counter()-started}),flush=True)
        if epoch>=plan['minimum_epochs'] and epoch-last_best>=plan['patience_epochs']:break
    selected=choose(history)
    if selected['epoch']==0:
        atomic_json(output/'selection-freeze.json',{'candidate_selected':False,'reason':'No eligible Tiny candidate beat development baseline; retain baseline','optimizer_updates':updates,'epochs':epoch,'test_used':False});return
    model.load_state_dict(torch.load(output/'best.pt',map_location='cpu',weights_only=True)['model'],strict=True);model.cpu().eval();model.head.decode_in_inference=True
    torch.onnx.export(model,torch.zeros(1,3,640,640),str(output/'model.onnx'),input_names=['images'],output_names=['predictions'],opset_version=17,dynamo=False)
    manifest=read(plan['baseline_manifest']);manifest.update(architecture='YOLOX-Tiny',weights=(output/'model.onnx').relative_to(ROOT).as_posix(),sha256=sha256(output/'model.onnx'),source_release_path=(output/'release.json').relative_to(ROOT).as_posix(),dataset_release='manifest-sha256:'+sha256(output/'release.json'),training_provenance=provenance_path.relative_to(ROOT).as_posix(),best_epoch=selected['epoch'],refinement_plan_sha256=sha256(output/'plan.json'),trained_checkpoint_sha256=sha256(output/'best.pt'),confidence=.1,exported_before_final_test=True,test_split_used=False)
    atomic_json(output/'model.json',manifest);link_model(provenance_path,output/'model.json')
    atomic_json(output/'selection-freeze.json',{'candidate_selected':True,'plan_sha256':sha256(output/'plan.json'),'manifest_sha256':sha256(output/'model.json'),'weights_sha256':manifest['sha256'],'checkpoint_sha256':sha256(output/'best.pt'),'selected':selected,'baseline_validation':history[0],'optimizer_updates':updates,'epochs_completed':epoch,'TEST_inference_performed':False,'elapsed_seconds':time.perf_counter()-started})
    print(json.dumps({'status':'completed_before_regression_or_fresh_TEST','epochs':epoch,'selected_epoch':selected['epoch'],'optimizer_updates':updates}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','preflight','train']);p.add_argument('--output',default=DEST);a=p.parse_args()
    {'prepare':prepare,'preflight':preflight,'train':train}[a.action](a.output)
