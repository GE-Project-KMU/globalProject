"""One frozen MWPD transfer candidate with old-domain replay; TEST never selects."""
import argparse,json,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'.cache/vendor/YOLOX'))
from agent.paths import workspace_path,atomic_json,sha256
from agent.dataset import validate_research_release
OLD='data/training/research-pothole-rdd2020-v4/release.json'
NEW='data/training/research-pothole-mwpd-v6/release.json'
BASE='data/training/pothole-nano-rdd2020-v4'
DEST='data/training/pothole-mwpd-v6'


def choose(history):
    if not history or history[0]['epoch']!=0:raise ValueError('initial_baseline_required')
    reference=history[0]['old_validation_ap50']
    for i,row in enumerate(history):
        if row['epoch']!=i or any(not isinstance(row[k],(int,float)) or not math.isfinite(row[k]) or not 0<=row[k]<=1 for k in ('old_validation_ap50','new_validation_ap50')):
            raise ValueError('ordered_finite_validation_history_required')
    eligible=[r for r in history if r['old_validation_ap50']>=reference-.03]
    return max(eligible,key=lambda r:((r['old_validation_ap50']+r['new_validation_ap50'])/2,-r['epoch']))


def prepare(output=DEST):
    output=workspace_path(output)
    if not output.is_relative_to(ROOT/'data/training') or output.exists():raise ValueError('new_training_output_required')
    old=json.loads(workspace_path(OLD,exists=True).read_text());new=json.loads(workspace_path(NEW,exists=True).read_text())
    rows=[r|{'development_domain':'original'} for r in old if r['split']!='test']+[r|{'development_domain':'mwpd'} for r in new]
    if any(len(r['objects'])>30 for r in rows if r['split']=='train'):
        raise ValueError('original_samples_30_target_limit_exceeded_before_training')
    check=validate_research_release(rows)
    if not check['valid']:raise ValueError(str(check['errors'][:3]))
    output.mkdir(parents=True);atomic_json(output/'release.json',rows)
    plan={'format':'pothole-mwpd-transfer-plan-v1','candidate_count':1,'baseline_model':BASE+'/model.json','baseline_checkpoint':BASE+'/best.pt',
          'baseline_model_sha256':sha256(ROOT/(BASE+'/model.json')),'baseline_checkpoint_sha256':sha256(ROOT/(BASE+'/best.pt')),
          'old_release_sha256':sha256(ROOT/OLD),'new_release_sha256':sha256(ROOT/NEW),'combined_release_sha256':sha256(output/'release.json'),
          'script_sha256':sha256(Path(__file__)),'dependency_sha256':{p:sha256(ROOT/p) for p in ['scripts/train_yolox_research.py','scripts/improve_pothole_train.py']},
          'epochs':12,'batch_size':16,'input_size':640,'learning_rate':.0001,'seed':20261106,
          'optimizer':'native YOLOX SGD momentum.9 Nesterov decay.0005 excluding BN/bias','batchnorm_running_statistics':'frozen',
          'sampling':'Weighted replacement, equal expected new MWPD TRAIN and original RDD2020 TRAIN; 2*new_train_count samples/epoch',
          'augmentation':'native HSV probability.3 and horizontal flip.5; no mosaic, no mixup, no L1','ema':True,
          'selection':'max mean(old VAL AP50,new VAL AP50), old VAL drop <=.03 absolute vs epoch0; initial baseline included, ties earliest epoch',
          'confidence':.1,'confidence_selection':'unchanged original operating point, no new threshold sweep',
          'test_policy':'MWPD TEST untouched until checkpoint and model SHA frozen, evaluate baseline/candidate once with actual app; old TEST only historical regression',
          'research_only':True,'production_approved':False,'public_source_only':True,'user_recordings_used':False}
    atomic_json(output/'plan.json',plan)
    print(json.dumps({'plan_sha256':sha256(output/'plan.json'),'train':sum(r['split']=='train' for r in rows),'new_test':sum(r['split']=='test' for r in rows)}),flush=True)


def revise_plan(output=DEST):
    output=workspace_path(output,exists=True);old=output/'plan.json';new=output/'plan-v2.json'
    if (output/'provenance.json').exists() or new.exists():raise ValueError('fresh_pretraining_plan_revision_required')
    plan=json.loads(old.read_text(encoding='utf-8'))
    plan.update(parent_plan_sha256=sha256(old),parent_training_code_sha256=plan['script_sha256'],script_sha256=sha256(Path(__file__)),
                plan_revision_reason='Metadata-only before any training: export records exported_before_final_test, immutable external evaluation reports record later TEST execution. Training/selection unchanged.')
    atomic_json(new,plan);print(json.dumps({'revised_plan_sha256':sha256(new)}),flush=True)


def train(output=DEST,plan_file=None):
    output=workspace_path(output,exists=True);plan_path=workspace_path(plan_file,exists=True) if plan_file else output/'plan.json';plan=json.loads(plan_path.read_text())
    if sha256(Path(__file__))!=plan['script_sha256'] or any(sha256(ROOT/p)!=h for p,h in plan['dependency_sha256'].items()):raise ValueError('frozen_training_source_changed')
    for path,expected in [(plan['baseline_model'],plan['baseline_model_sha256']),(plan['baseline_checkpoint'],plan['baseline_checkpoint_sha256']),(output/'release.json',plan['combined_release_sha256'])]:
        if sha256(workspace_path(path,exists=True))!=expected:raise ValueError('frozen_training_input_changed')
    if (output/'provenance.json').exists():raise ValueError('training_already_started_preserve_run')
    import cv2,numpy as np,torch
    from scripts.train_yolox_research import make_model,evaluate_model,seed_all
    from scripts.improve_pothole_train import OriginalSamples
    from yolox.data.datasets.mosaicdetection import MosaicDetection
    from yolox.data.data_augment import TrainTransform
    from yolox.exp import get_exp
    from yolox.utils import ModelEMA
    from agent.governance import begin_training,register_asset,link_model
    seed_all(plan['seed']);cv2.setNumThreads(1);torch.set_num_threads(2)
    if not torch.cuda.is_available():raise RuntimeError('CUDA_required')
    rows=json.loads((output/'release.json').read_text());checked=validate_research_release(rows)
    if not checked['valid']:raise ValueError('invalid_or_expired_research_release')
    training=[r for r in rows if r['split']=='train'];validation={domain:[r for r in rows if r['split']=='validation' and r['development_domain']==domain] for domain in ('original','mwpd')}
    old_n=sum(r['development_domain']=='original' for r in training);new_n=len(training)-old_n
    if min(old_n,new_n,len(validation['original']),len(validation['mwpd']))<2:raise ValueError('both_development_domains_required')
    provenance_path,provenance=begin_training(output,output/'release.json',training+validation['original']+validation['mwpd'],'pothole')
    provenance['research_only']=True;provenance['assets']=[register_asset(provenance,r,[]) for r in training+validation['original']+validation['mwpd']];atomic_json(provenance_path,provenance)
    model,_=make_model(['pothole']);model.load_state_dict(torch.load(ROOT/plan['baseline_checkpoint'],map_location='cpu',weights_only=True)['model'],strict=True);model.cuda()
    exp=get_exp(exp_file=str(ROOT/'.cache/vendor/YOLOX/exps/default/yolox_nano.py'));exp.model=model;optimizer=exp.get_optimizer(plan['batch_size'])
    for group in optimizer.param_groups:group['lr']=plan['learning_rate']
    dataset=MosaicDetection(OriginalSamples(training,640,2),img_size=(640,640),mosaic=False,preproc=TrainTransform(max_labels=120,hsv_prob=.3,flip_prob=.5),enable_mixup=False)
    sampler=torch.utils.data.WeightedRandomSampler([1/(old_n if r['development_domain']=='original' else new_n) for r in training],2*new_n,replacement=True,generator=torch.Generator().manual_seed(plan['seed']))
    loader=torch.utils.data.DataLoader(dataset,batch_size=16,sampler=sampler,num_workers=0,drop_last=True,pin_memory=True)
    ema=ModelEMA(model);scaler=torch.amp.GradScaler('cuda');history=[];updates=0;started=time.perf_counter()
    def validate(epoch,loss=None):
        metrics={k:evaluate_model(ema.ema,v,['pothole'],size=640,batch_size=16)['metrics']['map50'] for k,v in validation.items()}
        row={'epoch':epoch,'old_validation_ap50':metrics['original'],'new_validation_ap50':metrics['mwpd'],'optimizer_updates':updates,'mean_training_loss':loss}
        previous=choose(history)['epoch'] if history else None;history.append(row);selected=choose(history)
        if selected['epoch']==epoch and selected['epoch']!=previous:torch.save({'model':ema.ema.state_dict(),'epoch':epoch,'classes':['pothole']},output/'best.pt')
        atomic_json(output/'history.json',{'history':history,'selected':selected,'elapsed_seconds':time.perf_counter()-started})
        print(json.dumps(row|{'selected_epoch':selected['epoch']}),flush=True)
    validate(0)
    for epoch in range(1,plan['epochs']+1):
        model.train()
        for module in model.modules():
            if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.eval()
        losses=[]
        for images,labels,_,_ in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):loss=model(images.cuda(non_blocking=True),labels.cuda(non_blocking=True))['total_loss']
            if not torch.isfinite(loss):raise RuntimeError('non_finite_training_loss')
            before=scaler.get_scale();scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
            scaler.step(optimizer);scaler.update()
            if scaler.get_scale()>=before:updates+=1;ema.update(model)
            losses.append(float(loss.detach()))
        validate(epoch,float(np.mean(losses)))
    if updates==0:raise RuntimeError('no_optimizer_updates')
    selected=choose(history);model.load_state_dict(torch.load(output/'best.pt',map_location='cpu',weights_only=True)['model']);model.cpu().eval();model.head.decode_in_inference=True
    torch.onnx.export(model,torch.zeros(1,3,640,640),str(output/'model.onnx'),input_names=['images'],output_names=['predictions'],opset_version=17,dynamo=False)
    manifest=json.loads((ROOT/plan['baseline_model']).read_text());manifest.pop('heldout_map50',None);manifest.pop('test_evaluated',None)
    manifest.update(weights=(output/'model.onnx').relative_to(ROOT).as_posix(),sha256=sha256(output/'model.onnx'),source_release_path=(output/'release.json').relative_to(ROOT).as_posix(),dataset_release='manifest-sha256:'+sha256(output/'release.json'),training_provenance=provenance_path.relative_to(ROOT).as_posix(),source_licenses=['CC-BY-4.0','CC-BY-NC-3.0'],best_epoch=selected['epoch'],exported_before_final_test=True,test_split_used=False,confidence=.1,refinement_plan_sha256=sha256(plan_path),trained_checkpoint_sha256=sha256(output/'best.pt'))
    atomic_json(output/'model.json',manifest);link_model(provenance_path,output/'model.json')
    atomic_json(output/'selection-freeze.json',{'plan_sha256':sha256(plan_path),'manifest_sha256':sha256(output/'model.json'),'onnx_sha256':manifest['sha256'],'checkpoint_sha256':sha256(output/'best.pt'),'selected_epoch':selected['epoch'],'selected_validation':selected,'baseline_validation':history[0],'optimizer_updates':updates,'test_inference_performed':False,'elapsed_seconds':time.perf_counter()-started})
    print(json.dumps({'status':'training_completed_test_unseen','selected_epoch':selected['epoch'],'optimizer_updates':updates}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','revise-plan','train']);p.add_argument('--output',default=DEST);p.add_argument('--plan');a=p.parse_args()
    if a.action=='train':train(a.output,a.plan)
    elif a.action=='prepare':prepare(a.output)
    else:revise_plan(a.output)
