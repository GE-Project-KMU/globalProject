"""Two optimizer0 lane memory batches; development-only, no images/weights saved."""
import copy,json,random,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import atomic_json,sha256,workspace_path
from scripts.cycle2_device_memory import DeviceMemoryObserver
_ACTIVE=None

def probe():
 global _ACTIVE
 import torch,numpy as np,cv2
 from agent.dataset import validate_research_release
 from agent.lane_refinement_network import make_lane_refinement_network
 from agent.lane_network import freeze_batch_norm_statistics
 from agent.train_lane import prepare_pair
 from scripts.train_lane_cycle2 import DEV,BASE,full_context_sample,paint_loss
 protocol=ROOT/'docs/reports/LANE-CYCLE2-GPU-PREFLIGHT-PLAN.json';spec=json.loads(protocol.read_text());out=ROOT/'data/training/public-nextcycle/lane-shared-memory-probe.json'
 if out.exists():raise ValueError('one_probe_existing_evidence_preserved')
 if spec['probe_source_sha256']!=sha256(Path(__file__)) or spec['observer_source_sha256']!=sha256(ROOT/'scripts/cycle2_device_memory.py') or spec['learning_source_sha256']!=sha256(ROOT/'scripts/train_lane_cycle2.py'):raise ValueError('frozen_lane_probe_sources_required')
 rows=json.loads(DEV.read_text());training=sorted((r for r in rows if r['split']=='train'),key=lambda r:(-r.get('positive_pixels',0)/max(1,r.get('valid_pixels',1)),r['source_id']))[:2];validation=sorted((r for r in rows if r['split']=='validation' and r.get('positive_pixels',0)>0),key=lambda r:r['source_id'])[:1];chosen=training+validation
 if len(training)!=2 or len(validation)!=1 or not validate_research_release(chosen)['valid']:raise ValueError('current_licensed_hashed_development_before_GPU_required')
 if any(sha256(workspace_path(r['mask_path'],exists=True))!=r['mask_sha256'] for r in chosen):raise ValueError('development_mask_hash_changed_before_GPU')
 before={path:sha256(workspace_path(path,exists=True)) for r in chosen for path in (r['path'],r['mask_path'])};base_sha=sha256(BASE)
 if not torch.cuda.is_available():raise ValueError('cuda_required')
 torch.set_num_threads(2);cv2.setNumThreads(1);torch.manual_seed(20261106);random.seed(20261106);np.random.seed(20261106)
 device=DeviceMemoryObserver(ROOT/'.cache/cycle2-lane-probe-device-observer.log').start();_ACTIVE=device;initial=device.sample_rows()[-1]['used_bytes'];torch.cuda.init();torch.cuda.synchronize();oom_start=torch.cuda.memory_stats().get('num_ooms',0);torch.cuda.reset_peak_memory_stats()
 report={'format':'lane-shared-GPU-memory-probe-v1','status':'started','optimizer_steps':0,'training_updates':0,'max_batches':2,'TEST_images_loaded':0,'user_recordings_used':False,'model_weights_saved':False,'source_expiries_extended':False,'probe_source_sha256':sha256(Path(__file__)),'plan_sha256':sha256(protocol),'learning_source_sha256':sha256(ROOT/'scripts/train_lane_cycle2.py'),'device_before_cuda_init_bytes':initial,'batches':[],'OOM_counter_scope':'probe_process_only','global_OOM_zero_claimed':False,'limits':['Only2 pre-training memory batches, not a universal future peak','Device-reported WDDM memory sampled; no hard quota or instantaneous-excess guarantee','No optimizer update, accuracy or speed acceptance evidence']}
 def guard():
  torch.cuda.synchronize();device.raise_if_unsafe()
  if torch.cuda.memory_stats().get('num_ooms',0)!=oom_start:raise ValueError('own_OOM_counter_changed')
 try:
  model=make_lane_refinement_network();model.load_state_dict(torch.load(BASE,map_location='cpu',weights_only=True),strict=True);model.cuda();ema=copy.deepcopy(model).eval()
  for p in ema.parameters():p.requires_grad_(False)
  guard();values=[full_context_sample(r) for r in training];images=torch.from_numpy(np.stack([v[0] for v in values])).cuda();targets=torch.from_numpy(np.stack([v[1] for v in values])).cuda();model.train();freeze_batch_norm_statistics(model)
  with torch.autocast('cuda',dtype=torch.float16):loss=paint_loss(model(images)['out'],targets)
  if not torch.isfinite(loss):raise ValueError('nonfinite_lane_probe_loss')
  loss.backward()
  if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('nonfinite_lane_probe_gradient')
  guard();report['batches'].append({'kind':'AMP_train_forward_backward_optimizer0','shape':list(images.shape),'loss':float(loss.detach()),'valid_GT_pixels':int((targets!=255).sum()),'paint_GT_pixels':int((targets==1).sum())});model.zero_grad(set_to_none=True);del images,targets,loss
  x,_=prepare_pair(validation[0]['path'],validation[0]['mask_path'],size=1024)
  with torch.no_grad():raw=ema(torch.from_numpy(x[None]).cuda())['out']
  if tuple(raw.shape)!=(1,2,1024,1024) or not torch.isfinite(raw).all():raise ValueError('lane_VAL_output_contract_failed')
  guard();report['batches'].append({'kind':'FP32_VAL_forward','shape':[1,3,1024,1024],'output_shape':list(raw.shape)});report['status']='passed'
 except BaseException as error:report['status']='failed';report['failure_type']=type(error).__name__
 finally:
  try:device.wait_additional_samples(2)
  except ValueError:report['status']='failed';report['device_guard_failed']=True
  report['device_after_batches_with_models_resident_bytes']=device.sample_rows()[-1]['used_bytes'];device.close();report['device_memory']=device.summary();report['observer_child_stopped']=device.owned.process.poll() is not None;report['observer_thread_stopped']=not device.thread.is_alive();report['own_peak_allocated_bytes']=torch.cuda.max_memory_allocated();report['own_peak_reserved_bytes']=torch.cuda.max_memory_reserved();report['own_OOM_counter_delta']=torch.cuda.memory_stats().get('num_ooms',0)-oom_start;report['source_checksums_unchanged']=sha256(BASE)==base_sha and all(sha256(workspace_path(p,exists=True))==v for p,v in before.items())
  if not report['observer_child_stopped'] or not report['observer_thread_stopped'] or report['own_OOM_counter_delta'] or not report['source_checksums_unchanged']:report['status']='failed'
  atomic_json(out.with_name(out.stem+'-device-samples.json'),{'samples':device.sample_rows()});atomic_json(out,report);atomic_json(ROOT/'docs/reports/LANE-CYCLE2-GPU-PREFLIGHT.json',report)
 print(json.dumps(report),flush=True)
 if report['status']!='passed':raise SystemExit(1)
if __name__=='__main__':
 try:probe()
 finally:
  if _ACTIVE is not None:_ACTIVE.close()
