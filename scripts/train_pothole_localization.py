"""One v6 warmstart candidate: TRAIN-only hardness and native localization L1."""
import argparse,json,math,random,sys,time
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'.cache/vendor/YOLOX'))
from agent.paths import workspace_path,atomic_json,sha256
from agent.dataset import validate_research_release
BASE='data/training/pothole-mwpd-v6';DEST='data/training/pothole-localization-v8';HARD='data/training/pothole-train-hardness-v8'
QUOTAS={'RDD_positive':1024,'RDD_negative':1024,'MWPD_positive':2048}
DEPS=['scripts/train_yolox_research.py','scripts/improve_pothole_train.py','scripts/train_pothole_mwpd.py','scripts/train_pothole_tiny.py','scripts/mine_pothole_training_hardness.py']

def read(path):return json.loads(workspace_path(path,exists=True).read_text(encoding='utf-8'))

def validate_hardness(rows,weights):
    samples=weights['samples']
    if len(samples)!=7540 or len({r['sample_id'] for r in samples})!=len(samples):raise ValueError('complete_unique_TRAIN_weights_required')
    expected={r['sample_id']:r for r in rows if r['split']=='train'}
    if set(expected)!={r['sample_id'] for r in samples}:raise ValueError('TRAIN_membership_changed')
    for item in samples:
        row=expected[item['sample_id']];domain=row['development_domain'];stratum=('RDD' if domain=='original' else 'MWPD')+('_positive' if row['objects'] else '_negative')
        if item['sha256']!=row['sha256'] or item['stratum']!=stratum:raise ValueError('TRAIN_weight_source_or_stratum_changed')
        value=item['weight']
        if type(value) not in (int,float) or not math.isfinite(value) or not 1<=value<=3:raise ValueError('bounded_finite_TRAIN_weight_required')
    if Counter(r['stratum'] for r in samples)!={'RDD_positive':2676,'RDD_negative':2676,'MWPD_positive':2188}:raise ValueError('fixed_TRAIN_strata_required')
    return {r['sample_id']:r for r in samples}

def epoch_indices(items,seed,epoch):
    """Exact source counts, local RNG independent of augmentation RNG."""
    rng=random.Random(seed+epoch);chosen=[]
    for stratum,count in QUOTAS.items():
        indices=[i for i,r in enumerate(items) if r['stratum']==stratum]
        if not indices:raise ValueError('nonempty_training_strata_required')
        chosen.extend(rng.choices(indices,weights=[items[i]['weight'] for i in indices],k=count))
    rng.shuffle(chosen);return chosen

def bind_completed_hardness(weights,rows):
    mining=read(HARD+'/plan.json');receipt=read(HARD+'/run-receipt.json');aggregate=read(HARD+'/aggregate.json');baseline=read(BASE+'/model.json')
    pin=sha256(ROOT/(HARD+'/plan.json'))
    if receipt.get('status')!='completed' or receipt.get('samples')!=7540 or receipt.get('plan_sha256')!=pin or aggregate.get('status')!='completed':raise ValueError('completed_frozen_TRAIN_mining_required')
    if aggregate.get('training_weights_file_sha256')!=sha256(ROOT/(HARD+'/training-weights.json')):raise ValueError('mining_result_digest_changed')
    if weights.get('plan_sha256')!=pin or weights.get('source_release_sha256')!=mining['source_release_sha256'] or weights.get('training_membership_sha256')!=mining['training_membership_sha256']:raise ValueError('TRAIN_mining_evidence_binding_failed')
    if weights.get('manifest_sha256')!=sha256(ROOT/(BASE+'/model.json')) or weights.get('weights_sha256')!=baseline['sha256']:raise ValueError('TRAIN_mining_teacher_changed')
    if (weights.get('train_only') is not True or weights.get('val_images')!=0 or weights.get('test_images')!=0 or weights.get('US100_used') is not False or weights.get('source_quotas_changed') is not False):raise ValueError('TRAIN_only_mining_proof_required')
    expiry=datetime.fromisoformat(weights['expires_at'].replace('Z','+00:00'))
    if expiry.tzinfo is None or expiry<=datetime.now(timezone.utc) or expiry!=min(datetime.fromisoformat(r['expires_at'].replace('Z','+00:00')) for r in rows):raise ValueError('original_TRAIN_expiry_required')

def prepare(output=DEST):
    output=workspace_path(output)
    if output.exists() or not output.is_relative_to(ROOT/'data/training'):raise ValueError('new_localization_experiment_required')
    rows=[r for r in read(BASE+'/release.json') if r['split']!='test'];weights=read(HARD+'/training-weights.json');validate_hardness(rows,weights);bind_completed_hardness(weights,rows)
    if not validate_research_release(rows)['valid']:raise ValueError('valid_development_only_release_required')
    if weights.get('train_only') is not True:raise ValueError('completed_TRAIN_only_mining_required')
    output.mkdir();atomic_json(output/'release.json',rows)
    plan={'format':'pothole-localization-v8-plan-v1','candidate_count':1,'architecture':'YOLOX-Nano','input_size':640,'batch_size':16,
          'baseline_manifest':BASE+'/model.json','baseline_checkpoint':BASE+'/best.pt','baseline_manifest_sha256':sha256(ROOT/(BASE+'/model.json')),'baseline_checkpoint_sha256':sha256(ROOT/(BASE+'/best.pt')),
          'development_sha256':sha256(output/'release.json'),'TRAIN_hardness_sha256':sha256(ROOT/(HARD+'/training-weights.json')),'TRAIN_mining_plan_sha256':sha256(ROOT/(HARD+'/plan.json')),
          'trainer_sha256':sha256(Path(__file__)),'dependencies':{p:sha256(ROOT/p) for p in DEPS},'seed':20261108,
          'maximum_epochs':18,'minimum_epochs':8,'patience':6,'learning_rate':.0001,'warmup_epochs':1,'minimum_lr_fraction':.1,
          'optimizer':'native SGD momentum.9 Nesterov wd.0005, BN/bias decay excluded','BN':'running statistics frozen; learned affine trainable',
          'augmentation':'full frame640 native letterbox/HSV.3/flip.5; mosaic0/mixup0','native_L1_localization_enabled_after_epoch':2,
          'sampling':'4096 replacement draws, exact per epoch RDDpositive1024/RDDnegative1024/MWPDpositive2048, hardness1..3 within each stratum; locally seeded shuffle; no dropped batches',
          'source_draw_counts':QUOTAS,'TRAIN_images':7540,'VAL_images':652,'TEST_images_in_development':0,'US100_used_for_learning':False,
          'selection':'mean(old VAL AP50,new VAL AP50) with old VAL drop<=3pp vs v6 baseline; initial warmstart retained, ties earliest; no TEST selection',
          'confidence':.1,'AP_confidence_floor':.01,'nms_iou':.5,'test_policy':'Select best family member on development only before one fresh US100 and historical regression evaluation; never tune on either',
          'source_expires_at':min(r['expires_at'] for r in rows),'research_only':True,'production_approved':False,'source_expiry_extended':False,'user_recordings_used':False}
    atomic_json(output/'plan.json',plan);atomic_json(ROOT/'docs/reports/POTHOLE-LOCALIZATION-V8-PLAN.json',plan)
    print(json.dumps({'plan_sha256':sha256(output/'plan.json'),'train_images':7540,'validation_images':652,'model_training_started':False}))

def preflight(output=DEST):
    import cv2,torch
    from scripts.train_yolox_research import make_model
    from scripts.improve_pothole_train import OriginalSamples
    from yolox.data.datasets.mosaicdetection import MosaicDetection
    from yolox.data.data_augment import TrainTransform
    cv2.setNumThreads(1);torch.set_num_threads(2);torch.manual_seed(20261108)
    if torch.cuda.is_initialized():raise ValueError('CPU_contract_only')
    output=workspace_path(output,exists=True);rows=read(output/'release.json');weights=validate_hardness(rows,read(HARD+'/training-weights.json'))
    train=[r for r in rows if r['split']=='train'];chosen=[max((r for r in train if r['development_domain']==d),key=lambda r:len(r['objects'])) for d in ('original','mwpd')]
    dataset=MosaicDetection(OriginalSamples(chosen,640,0),img_size=(640,640),mosaic=False,preproc=TrainTransform(max_labels=120,hsv_prob=0.,flip_prob=0.),enable_mixup=False)
    images,labels,_,_=next(iter(torch.utils.data.DataLoader(dataset,batch_size=2,num_workers=0)))
    if (labels[:,:,3]>0).sum(1).tolist()!=[len(r['objects']) for r in chosen]:raise ValueError('complete_source_GT_required')
    model,_=make_model(['pothole']);model.load_state_dict(torch.load(ROOT/(BASE+'/best.pt'),map_location='cpu',weights_only=True)['model'],strict=True);model.train();model.head.use_l1=True
    for m in model.modules():
        if isinstance(m,torch.nn.modules.batchnorm._BatchNorm):m.eval()
    losses=model(images,labels);loss=losses['total_loss'];loss.backward();grads=[p.grad for p in model.parameters() if p.grad is not None]
    if not torch.isfinite(loss) or not all(torch.isfinite(g).all() for g in grads) or not any(torch.count_nonzero(g) for g in grads) or not float(losses['l1_loss'])>0:raise ValueError('finite_real_L1_backward_required')
    items=[weights[r['sample_id']] for r in train];draws=epoch_indices(items,20261108,0)
    if Counter(items[i]['stratum'] for i in draws)!=QUOTAS or len(draws)%16:raise ValueError('exact_source_quota_without_drop_required')
    result={'format':'pothole-localization-v8-CPU-contract','source_sha256':sha256(Path(__file__)),'development_sha256':sha256(output/'release.json'),'hardness_sha256':sha256(ROOT/(HARD+'/training-weights.json')),'batch_shape':list(images.shape),'all_GT_preserved':True,'finite_loss':float(loss.detach()),'native_L1_loss':float(losses['l1_loss'].detach()),'finite_nonzero_backward':True,'source_draw_counts':dict(Counter(items[i]['stratum'] for i in draws)),'optimizer_updates':0,'GPU_initialized':torch.cuda.is_initialized(),'TEST_images_used':0}
    atomic_json(output/'preflight.json',result);atomic_json(ROOT/'docs/reports/POTHOLE-LOCALIZATION-V8-CONTRACT.json',result);print(json.dumps(result))

def train(output=DEST):
    import cv2,numpy as np,torch
    from scripts.train_yolox_research import make_model,evaluate_model,seed_all
    from scripts.train_pothole_mwpd import choose
    from scripts.train_pothole_tiny import learning_rate
    from scripts.improve_pothole_train import OriginalSamples
    from yolox.data.datasets.mosaicdetection import MosaicDetection
    from yolox.data.data_augment import TrainTransform
    from yolox.exp import get_exp
    from yolox.utils import ModelEMA
    from agent.governance import begin_training,register_asset,link_model
    output=workspace_path(output,exists=True);plan=read(output/'plan.json')
    if (output/'provenance.json').exists():raise ValueError('already_started_preserve_experiment')
    if sha256(Path(__file__))!=plan['trainer_sha256'] or any(sha256(ROOT/p)!=s for p,s in plan['dependencies'].items()):raise ValueError('frozen_localization_sources_changed')
    checks=[(output/'release.json',plan['development_sha256']),(ROOT/(BASE+'/model.json'),plan['baseline_manifest_sha256']),(ROOT/(BASE+'/best.pt'),plan['baseline_checkpoint_sha256']),(ROOT/(HARD+'/training-weights.json'),plan['TRAIN_hardness_sha256']),(ROOT/(HARD+'/plan.json'),plan['TRAIN_mining_plan_sha256'])]
    if any(sha256(p)!=s for p,s in checks):raise ValueError('frozen_localization_inputs_changed')
    proof=read(output/'preflight.json')
    if proof['source_sha256']!=plan['trainer_sha256'] or proof['finite_nonzero_backward'] is not True:raise ValueError('matching_real_CPU_contract_required')
    rows=read(output/'release.json');hardness=read(HARD+'/training-weights.json');weights=validate_hardness(rows,hardness);bind_completed_hardness(hardness,rows)
    if not validate_research_release(rows)['valid'] or any(r['split']=='test' for r in rows):raise ValueError('valid_development_only_release_required')
    if not torch.cuda.is_available():raise RuntimeError('CUDA_required')
    seed_all(plan['seed']);torch.set_num_threads(2);cv2.setNumThreads(1);training=[r for r in rows if r['split']=='train'];validation={d:[r for r in rows if r['split']=='validation' and r['development_domain']==d] for d in ('original','mwpd')}
    pp,prov=begin_training(output,output/'release.json',rows,'pothole');prov['research_only']=True;prov['assets']=[register_asset(prov,r,[]) for r in rows];atomic_json(pp,prov)
    model,_=make_model(['pothole']);model.load_state_dict(torch.load(ROOT/(BASE+'/best.pt'),map_location='cpu',weights_only=True)['model'],strict=True);model.cuda()
    exp=get_exp(exp_file=str(ROOT/'.cache/vendor/YOLOX/exps/default/yolox_nano.py'));exp.model=model;optimizer=exp.get_optimizer(16);ema=ModelEMA(model);scaler=torch.amp.GradScaler('cuda')
    dataset=MosaicDetection(OriginalSamples(training,640,2),img_size=(640,640),mosaic=False,preproc=TrainTransform(max_labels=120,hsv_prob=.3,flip_prob=.5),enable_mixup=False)
    items=[weights[r['sample_id']] for r in training]
    class ExactSampler:
        def __init__(self):self.epoch=0
        def __len__(self):return 4096
        def __iter__(self):indices=epoch_indices(items,plan['seed'],self.epoch);self.epoch+=1;return iter(indices)
    loader=torch.utils.data.DataLoader(dataset,batch_size=16,sampler=ExactSampler(),num_workers=0,drop_last=False,pin_memory=True)
    history=[];updates=0;attempts=0;started=time.perf_counter();last_best=0
    def validate(epoch,loss=None):
        nonlocal last_best
        scores={d:evaluate_model(ema.ema,v,['pothole'],size=640,batch_size=16)['metrics']['map50'] for d,v in validation.items()};row={'epoch':epoch,'old_validation_ap50':scores['original'],'new_validation_ap50':scores['mwpd'],'optimizer_updates':updates,'mean_training_loss':loss}
        previous=choose(history)['epoch'] if history else None;history.append(row);selected=choose(history)
        if selected['epoch']==epoch and previous!=epoch:
            last_best=epoch;torch.save({'model':ema.ema.state_dict(),'epoch':epoch,'classes':['pothole']},output/'best.pt')
        atomic_json(output/'history.json',{'history':history,'selected':selected,'actual_updates':updates,'attempts':attempts,'elapsed_seconds':time.perf_counter()-started,'peak_reserved_GiB':torch.cuda.max_memory_reserved()/2**30});print(json.dumps(row|{'selected_epoch':selected['epoch'],'elapsed_seconds':time.perf_counter()-started}),flush=True)
    atomic_json(output/'training-config.json',plan);validate(0)
    for epoch in range(1,plan['maximum_epochs']+1):
        if datetime.fromisoformat(plan['source_expires_at'].replace('Z','+00:00'))<=datetime.now(timezone.utc):raise ValueError('source_expired_stop_training')
        model.train();model.head.use_l1=epoch>2
        for m in model.modules():
            if isinstance(m,torch.nn.modules.batchnorm._BatchNorm):m.eval()
        losses=[]
        for step,(images,labels,_,_) in enumerate(loader):
            lr=learning_rate(attempts,len(loader)*plan['maximum_epochs'],len(loader),plan['learning_rate'])
            for group in optimizer.param_groups:group['lr']=lr
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):loss=model(images.cuda(non_blocking=True),labels.cuda(non_blocking=True))['total_loss']
            if not torch.isfinite(loss):raise RuntimeError('nonfinite_localization_loss')
            before=scaler.get_scale();scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),10.);scaler.step(optimizer);scaler.update();attempts+=1
            if scaler.get_scale()>=before:updates+=1;ema.update(model)
            losses.append(float(loss.detach()))
            if torch.cuda.max_memory_reserved()>9*2**30:raise RuntimeError('own_training_memory_budget_exceeded')
            if step%100==0:print(json.dumps({'epoch':epoch,'step':step,'updates':updates,'loss':losses[-1],'native_L1_enabled':model.head.use_l1}),flush=True)
        validate(epoch,float(np.mean(losses)))
        if epoch>=plan['minimum_epochs'] and epoch-last_best>=plan['patience']:break
    selected=choose(history);model.load_state_dict(torch.load(output/'best.pt',map_location='cpu',weights_only=True)['model'],strict=True);model.cpu().eval();model.head.decode_in_inference=True
    torch.onnx.export(model,torch.zeros(1,3,640,640),str(output/'model.onnx'),input_names=['images'],output_names=['predictions'],opset_version=17,dynamo=False)
    manifest=read(BASE+'/model.json');manifest.update(weights=(output/'model.onnx').relative_to(ROOT).as_posix(),sha256=sha256(output/'model.onnx'),source_release_path=(output/'release.json').relative_to(ROOT).as_posix(),dataset_release='manifest-sha256:'+sha256(output/'release.json'),training_provenance=pp.relative_to(ROOT).as_posix(),best_epoch=selected['epoch'],trained_checkpoint_sha256=sha256(output/'best.pt'),refinement_plan_sha256=sha256(output/'plan.json'),exported_before_final_test=True,test_split_used=False)
    atomic_json(output/'model.json',manifest);link_model(pp,output/'model.json');atomic_json(output/'selection-freeze.json',{'selected':selected,'baseline_validation':history[0],'candidate_improved_on_VAL':selected['epoch']>0,'manifest_sha256':sha256(output/'model.json'),'weights_sha256':manifest['sha256'],'checkpoint_sha256':sha256(output/'best.pt'),'plan_sha256':sha256(output/'plan.json'),'epochs_completed':epoch,'optimizer_updates':updates,'TEST_inference_performed':False,'elapsed_seconds':time.perf_counter()-started})
    print(json.dumps({'status':'completed_before_TEST','selected_epoch':selected['epoch'],'optimizer_updates':updates}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','preflight','train']);p.add_argument('--output',default=DEST);a=p.parse_args();{'prepare':prepare,'preflight':preflight,'train':train}[a.action](a.output)
