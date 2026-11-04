"""At most 3 ephemeral GPU batches, optimizer0, development-only shared-slot probe.

Never trains/saves weights, reads TEST/user recordings, kills another process or
changes device/global settings. Real observed sampling intervals are reported.
"""
import argparse,copy,json,sys,time,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import atomic_json,workspace_path,sha256
LIMIT=9*1024**3

def memory_allowed(used,oom_before,oom_after):return type(used) is int and 0<=used<=LIMIT and oom_after==oom_before

def interval_summary(samples):
 import statistics
 gaps=[(b['elapsed_ns']-a['elapsed_ns'])/1e6 for a,b in zip(samples,samples[1:])]
 if any(g<0 for g in gaps):raise ValueError('monotonic_sampling_timestamps_required')
 return {'sample_count':len(samples),'requested_sleep_ms':20.,'observed_interval_ms':{'min':min(gaps),'median':statistics.median(gaps),'max':max(gaps)} if gaps else None,'sleep_not_guaranteed_sampling_period':True,'max_observed_global_bytes':max((v['used_bytes'] for v in samples),default=None)}

def check_probe_release(rows,*,now=None):
 from agent.dataset import validate_research_release
 state=validate_research_release(rows,now=now)
 if not state['valid']:raise ValueError('invalid_or_expired_research_development_before_GPU_or_pixel_decode')
 return state

def probe():
 import torch,cv2,numpy as np
 from agent.redaction_p2_network import build_redaction_p2_network,assemble_p2_network,load_p2_warmstart
 from agent.lane_network import freeze_batch_norm_statistics
 from scripts.train_privacy_p2_cycle2 import DEV,BASE,plan,view_sample
 frozen=plan();public=ROOT/'docs/reports/PRIVACY-P2-CYCLE2-PLAN-V2.json'
 if json.loads(public.read_text())!=frozen or frozen['batch_size']!=4:raise ValueError('frozen_batch4_pretraining_plan_required')
 out=ROOT/'data/training/public-nextcycle/P2-shared-memory-probe.json'
 if out.exists():raise ValueError('one_probe_only_existing_result_must_be_preserved')
 rows=json.loads(DEV.read_text());training=sorted((r for r in rows if r['split']=='train' and r['evaluation_class']=='face'),key=lambda r:(-len(r['objects']),r['source_id']))[:4]
 validation=sorted((r for r in rows if r['split']=='validation'),key=lambda r:(-len(r['objects']),r['source_id']))[:4]
 if len(training)!=4 or len(validation)!=4:raise ValueError('four_development_rows_required')
 check_probe_release(training+validation)
 if not torch.cuda.is_available():raise ValueError('cuda_probe_required')
 source_before={r['path']:sha256(workspace_path(r['path'],exists=True)) for r in training+validation}
 if any(source_before[r['path']]!=r['sha256'] for r in training+validation):raise ValueError('development_source_hash_changed')
 base_sha=sha256(BASE);torch.set_num_threads(2);cv2.setNumThreads(1);torch.manual_seed(20261106)
 torch.cuda.init();torch.cuda.synchronize();oom_before=torch.cuda.memory_stats().get('num_ooms',0);torch.cuda.reset_peak_memory_stats()
 samples=[];stop=threading.Event();violated=threading.Event();start=time.perf_counter_ns();sampling_errors=[]
 def sample():
  while not stop.is_set():
   try:
    free,total=torch.cuda.mem_get_info();used=total-free;samples.append({'elapsed_ns':time.perf_counter_ns()-start,'used_bytes':used})
    if used>LIMIT:violated.set()
   except BaseException as error:sampling_errors.append(type(error).__name__);violated.set();return
   stop.wait(.02)
 def guard():
  torch.cuda.synchronize();free,total=torch.cuda.mem_get_info()
  if violated.is_set() or not memory_allowed(total-free,oom_before,torch.cuda.memory_stats().get('num_ooms',0)):raise ValueError('global_9GiB_or_OOM_guard_failed')
 thread=threading.Thread(target=sample,daemon=True);thread.start();report={'format':'P2-shared-GPU-memory-probe-v1','status':'started','optimizer_steps':0,'training_updates':0,'OOM_counter_scope':'probe_process_only','global_OOM_zero_claimed':False,'ephemeral_model_BN_changes_discarded':True,'model_weights_saved':False,'TEST_images_loaded':0,'user_recordings_used':False,'max_batches':3,'batches':[],'limits':['Three pre-training memory batches; not a universal maximum for future learned assignments','Observed sampling includes scheduling/query time; in-flight CUDA operation is not cancelled','No optimizer update or speed/accuracy acceptance evidence'],'gpu_budget_bytes':LIMIT,'source_expiries_extended':False,'plan_sha256':sha256(public),'probe_source_sha256':sha256(Path(__file__)),'trainer_source_sha256':sha256(ROOT/'scripts/train_privacy_p2_cycle2.py'),'factory_sha256':sha256(ROOT/'agent/redaction_p2_network.py')}
 try:
  guard();bundle=build_redaction_p2_network();load_p2_warmstart(bundle,torch.load(BASE,map_location='cpu',weights_only=True)['model']);model=bundle.face.cuda();ema=copy.deepcopy(model).eval();plate=bundle.plate.cuda().eval()
  for p in ema.parameters():p.requires_grad_(False)
  for p in plate.parameters():p.requires_grad_(False)
  evaluation=assemble_p2_network(ema,plate).eval();guard()
  for batch in range(2):
   pixels=[];targets=[];counts=[]
   for index,row in enumerate(training):
    image=cv2.imread(str(workspace_path(row['path'],exists=True)));boxes=np.asarray([o['bbox_xywh_px'] for o in row['objects']],np.float32)
    x,y,n=view_sample(image,boxes,view='global' if batch==0 else 'tile',tile=index);pixels.append(x);targets.append(y);counts.append(n)
   guard();model.zero_grad(set_to_none=True);model.train();freeze_batch_norm_statistics(model.backbone.backbone)
   with torch.autocast('cuda',dtype=torch.float16):loss=model(torch.stack(pixels).cuda(),torch.stack(targets).cuda())['total_loss']
   if not torch.isfinite(loss):raise ValueError('nonfinite_probe_loss')
   loss.backward()
   if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):raise ValueError('nonfinite_probe_gradient')
   guard();report['batches'].append({'kind':'ephemeral_AMP_train_forward_backward_optimizer0','batch_size':4,'original_GT_counts':[len(r['objects']) for r in training],'view_GT_counts':counts,'loss':float(loss.detach())});model.zero_grad(set_to_none=True);del loss
  full=[]
  for row in validation:
   im=cv2.imread(str(workspace_path(row['path'],exists=True)));h,w=im.shape[:2];scale=min(1920/w,1920/h);canvas=np.full((1920,1920,3),114,np.uint8);rw,rh=max(1,round(w*scale)),max(1,round(h*scale));canvas[:rh,:rw]=cv2.resize(im,(rw,rh));full.append(torch.from_numpy(canvas.transpose(2,0,1).copy().astype(np.float32)))
  guard()
  with torch.no_grad():raw=evaluation(torch.stack(full).cuda())
  if tuple(raw.shape)!=(4,176400,7) or not torch.isfinite(raw).all():raise ValueError('VAL_probe_output_contract_failed')
  guard();report['batches'].append({'kind':'FP32_full_shared1920_VAL_forward','batch_size':4,'output_shape':list(raw.shape)});report['status']='passed'
 except BaseException as error:
  report['status']='failed';report['failure_type']=type(error).__name__;report['failure_reason']=str(error) if isinstance(error,ValueError) else 'runtime_failure_no_other_process_touched'
 finally:
  stop.set();thread.join(timeout=10);report.update(interval_summary(samples));report['sampling_errors']=sampling_errors;report['sampling_thread_stopped']=not thread.is_alive();report['own_peak_allocated_bytes']=torch.cuda.max_memory_allocated();report['own_peak_reserved_bytes']=torch.cuda.max_memory_reserved();report['OOM_counter_delta']=torch.cuda.memory_stats().get('num_ooms',0)-oom_before;report['guard_violation_observed']=violated.is_set();report['source_checksums_unchanged']=sha256(BASE)==base_sha and all(sha256(workspace_path(path,exists=True))==value for path,value in source_before.items())
  if report['status']=='passed' and (not report['source_checksums_unchanged'] or violated.is_set() or sampling_errors or report['OOM_counter_delta']!=0 or thread.is_alive()):report['status']='failed'
  atomic_json(out.with_name(out.stem+'-memory-samples.json'),{'samples':samples});atomic_json(out,report);atomic_json(ROOT/'docs/reports/PRIVACY-P2-CYCLE2-GPU-PREFLIGHT.json',report)
 print(json.dumps(report),flush=True)
 if report['status']!='passed':raise SystemExit(1)
if __name__=='__main__':probe()
