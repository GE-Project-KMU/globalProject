"""One VAL-only L1 refinement; every previously evaluated TEST is regression-only."""
import argparse,json,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from agent.paths import atomic_json,sha256,workspace_path
BASE='data/training/pothole-nano-rdd2020-v4'
RELEASE='data/training/research-pothole-rdd2020-v4/release.json'
DEST='data/training/pothole-refinement-v5'
PINS={'release':'6c5618898d96dd032ae5d9579c8c79aa676a063d6557c1c2f361a37fe5a8388c','model':'28efe86e7c9eacdf90d11caf0eb942dbb19b0a4d6830229069b3277cc8a45a6a','checkpoint':'2fb2c082964e81b918339a2a26f1a37fc2cd655aa78e9d98a2a38ed04fd89375'}

def choose(history):
    if not history or history[0].get('epoch')!=0:raise ValueError('baseline_epoch_zero_required')
    for i,row in enumerate(history):
        if type(row.get('epoch')) is not int or row.get('epoch')!=i or type(row.get('validation_ap50')) not in (int,float) or not math.isfinite(row['validation_ap50']) or not 0<=row['validation_ap50']<=1:raise ValueError('finite_ordered_validation_history_required')
    return max(history,key=lambda r:(r['validation_ap50'],-r['epoch']))

def prepare(output=DEST):
    output=workspace_path(output)
    if not output.is_relative_to(ROOT/'data/training') or output==ROOT/'data/training' or output.exists():raise ValueError('new_training_output_required')
    inputs={'release':workspace_path(RELEASE,exists=True),'model':workspace_path(BASE+'/model.json',exists=True),'checkpoint':workspace_path(BASE+'/best.pt',exists=True)}
    if any(sha256(p)!=PINS[k] for k,p in inputs.items()):raise ValueError('original_v4_inputs_changed')
    plan={'format':'pothole-val-refinement-plan-v1','inputs':{k:{'path':p.relative_to(ROOT).as_posix(),'sha256':PINS[k]} for k,p in inputs.items()},'script_sha256':sha256(Path(__file__)),'dependency_sha256':{p:sha256(ROOT/p) for p in ['scripts/improve_pothole_train.py','scripts/train_yolox_research.py','agent/models.py']},'candidate_count':1,'epochs':10,'batch_size':16,'input_size':640,'learning_rate':.00025,'seed':20261105,'optimizer':'native SGD momentum.9/Nesterov/weight_decay.0005 excluding BN and bias','augmentation':'none; native L1 box regression enabled for every update','ema':True,'selection':'Best same previously used VAL402 AP50, including initial v4 checkpoint at epoch0; exact ties prefer earlier epoch. No TEST-based choice.','confidence_selection':'same VAL F2 frozen grid as v4','test_policy':'All v4 Japan86 and prior Czech231 are previously evaluated regression benchmarks only; no new independent holdout acquired. No TEST inference in trainer.','reason':'v4 plateau stopped before its scheduled no-augmentation/L1 stage. Bounded single follow-up, not a promise of improvement. Additional official-source acquisition currently HTTP403.','research_only':True,'production_approved':False}
    output.mkdir(parents=True);atomic_json(output/'plan.json',plan)
    print(json.dumps({'plan_sha256':sha256(output/'plan.json'),'candidate_count':1,'epochs':10,'test_inference':False}),flush=True)

def train(output=DEST):
    output=workspace_path(output,exists=True);pp=output/'plan.json';plan=json.loads(pp.read_text())
    if plan['script_sha256']!=sha256(Path(__file__)) or any(sha256(ROOT/p)!=h for p,h in plan['dependency_sha256'].items()):raise ValueError('planned_source_changed')
    if any(sha256(workspace_path(v['path'],exists=True))!=v['sha256'] for v in plan['inputs'].values()):raise ValueError('planned_input_changed')
    if (output/'provenance.json').exists():raise ValueError('training_already_started_preserve_original')
    import torch,cv2,numpy as np
    from improve_pothole_train import OriginalSamples,validate_original_release,calibration
    from train_yolox_research import make_model,seed_all,evaluate_model
    from yolox.data.datasets.mosaicdetection import MosaicDetection
    from yolox.data.data_augment import TrainTransform
    from yolox.utils import ModelEMA
    from yolox.exp import get_exp
    from agent.governance import begin_training,register_asset,link_model
    seed_all(plan['seed']);cv2.setNumThreads(1);torch.set_num_threads(2)
    if not torch.cuda.is_available():raise RuntimeError('CUDA_required')
    release=workspace_path(plan['inputs']['release']['path'],exists=True);rows=validate_original_release(release)
    training=[r for r in rows if r['split']=='train'];validation=[r for r in rows if r['split']=='validation']
    assert len(training)==5352 and len(validation)==402
    provenance_path,provenance=begin_training(output,release,training+validation,'pothole')
    provenance['research_only']=True;provenance['assets']=[register_asset(provenance,r,[]) for r in training+validation];atomic_json(provenance_path,provenance)
    model,_=make_model(['pothole']);state=torch.load(workspace_path(plan['inputs']['checkpoint']['path']),map_location='cpu',weights_only=True)
    model.load_state_dict(state['model'],strict=True);model.cuda();model.head.use_l1=True
    exp=get_exp(exp_file=str(ROOT/'.cache/vendor/YOLOX/exps/default/yolox_nano.py'));exp.model=model
    optimizer=exp.get_optimizer(plan['batch_size'])
    for group in optimizer.param_groups:group['lr']=plan['learning_rate']
    dataset=MosaicDetection(OriginalSamples(training,640,4),img_size=(640,640),mosaic=False,preproc=TrainTransform(max_labels=120,hsv_prob=0.,flip_prob=0.),enable_mixup=False)
    loader=torch.utils.data.DataLoader(dataset,batch_size=16,shuffle=True,num_workers=0,drop_last=True,pin_memory=True)
    ema=ModelEMA(model);scaler=torch.amp.GradScaler('cuda');history=[];updates=0;started=time.perf_counter()
    def validation_step(epoch,loss=None):
        result=evaluate_model(ema.ema,validation,['pothole'],size=640,batch_size=16)
        item={'epoch':epoch,'validation_ap50':result['metrics']['map50'],'mean_training_loss':loss,'optimizer_updates':updates}
        previous=choose(history)['validation_ap50'] if history else -1
        history.append(item)
        if item['validation_ap50']>previous:
            torch.save({'model':ema.ema.state_dict(),'epoch':epoch,'classes':['pothole']},output/'best.pt')
        atomic_json(output/'training-history.json',{'history':history,'selected_epoch':choose(history)['epoch'],'elapsed_seconds':time.perf_counter()-started})
        print(json.dumps(item),flush=True)
    validation_step(0)
    for epoch in range(1,11):
        model.train();losses=[]
        for images,labels,_,_ in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):loss=model(images.cuda(non_blocking=True),labels.cuda(non_blocking=True))['total_loss']
            if not torch.isfinite(loss):raise RuntimeError('non_finite_loss')
            before=scaler.get_scale();scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
            scaler.step(optimizer);scaler.update()
            if scaler.get_scale()>=before:updates+=1;ema.update(model)
            losses.append(float(loss.detach()))
        validation_step(epoch,float(np.mean(losses)))
    if updates==0:raise RuntimeError('no_actual_optimizer_updates')
    selected=choose(history);model.load_state_dict(torch.load(output/'best.pt',map_location='cpu',weights_only=True)['model']);model.eval()
    predictions=evaluate_model(model,validation,['pothole'],size=640,batch_size=16);atomic_json(output/'selected-validation-predictions.json',predictions)
    calibrated=calibration(predictions);calibrated['selection_data']='Same previously used VAL402; development refinement only, not fresh holdout';atomic_json(output/'confidence-calibration.json',calibrated)
    model.cpu();model.head.decode_in_inference=True
    torch.onnx.export(model,torch.zeros(1,3,640,640),str(output/'model.onnx'),input_names=['images'],output_names=['predictions'],opset_version=17,dynamo=False)
    manifest=json.loads(workspace_path(plan['inputs']['model']['path']).read_text());manifest.pop('heldout_map50',None)
    manifest.update(weights=(output/'model.onnx').relative_to(ROOT).as_posix(),sha256=sha256(output/'model.onnx'),training_provenance=provenance_path.relative_to(ROOT).as_posix(),confidence=calibrated['selected_confidence'],best_epoch=selected['epoch'],test_evaluated=False,independent_holdout_available=False,refinement_plan_sha256=sha256(pp),trained_checkpoint_sha256=sha256(output/'best.pt'))
    atomic_json(output/'model.json',manifest);link_model(provenance_path,output/'model.json')
    atomic_json(output/'selection-freeze.json',{'plan_sha256':sha256(pp),'model_manifest_sha256':sha256(output/'model.json'),'onnx_sha256':manifest['sha256'],'checkpoint_sha256':sha256(output/'best.pt'),'selected_epoch':selected['epoch'],'selected_validation_ap50':selected['validation_ap50'],'baseline_validation_ap50':history[0]['validation_ap50'],'baseline_retained':selected['epoch']==0,'confidence':manifest['confidence'],'optimizer_updates':updates,'test_inference_performed':False,'research_only':True,'production_approved':False})
    print(json.dumps({'status':'completed_validation_only','selected_epoch':selected['epoch'],'validation_ap50':selected['validation_ap50'],'baseline_retained':selected['epoch']==0}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','train']);parser.add_argument('--output',default=DEST);args=parser.parse_args()
    (prepare if args.action=='prepare' else train)(args.output)
