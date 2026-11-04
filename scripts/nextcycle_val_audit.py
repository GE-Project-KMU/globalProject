"""Development-VAL-only error audit; no historical TEST predictions are loaded."""
import argparse,json,sys,math
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import workspace_path,atomic_json,sha256

def overlap(a,b):
 x=max(a[0],b[0]);y=max(a[1],b[1]);w=max(0,min(a[0]+a[2],b[0]+b[2])-x);h=max(0,min(a[1]+a[3],b[1]+b[3])-y)
 return w*h/(a[2]*a[3]+b[2]*b[3]-w*h) if a[2]*a[3]+b[2]*b[3]>w*h else 0.
def face_size(box):
 s=math.sqrt(box[2]*box[3]);return '<16' if s<16 else '16..31' if s<32 else '32..63' if s<64 else '>=64'
def match_groups(truth,preds,rows):
 by=defaultdict(list)
 for p in preds:
  if p['class']=='face' and p['confidence']>=.01:by[p['sample_id']].append(p)
 stats=defaultdict(lambda:{'GT':0,'TP':0,'FN':0});detail=[]
 for sid,row in rows.items():
  gt=[t for t in truth if t['sample_id']==sid and t['class']=='face'];used=set()
  for p in sorted(by[sid],key=lambda r:-r['confidence']):
   value,idx=max(((overlap(p['bbox_xywh_px'],t['bbox_xywh_px']),i) for i,t in enumerate(gt) if i not in used),default=(0,None))
   if value>=.5:used.add(idx)
  for i,t in enumerate(gt):
   key=face_size(t['bbox_xywh_px']);stats[key]['GT']+=1;stats[key]['TP']+=i in used;stats[key]['FN']+=i not in used
  detail.append({'sample_id':sid,'GT':len(gt),'FN':len(gt)-len(used)})
 for v in stats.values():v['recall']=v['TP']/v['GT'] if v['GT'] else None
 return dict(stats),detail

def run():
 out=ROOT/'data/training/public-nextcycle/val-audit';out.mkdir(parents=True,exist_ok=True)
 if (out/'aggregate.json').exists():raise ValueError('immutable_audit_already_exists')
 devpath=ROOT/'data/training/research-redaction-v6/development.json';dev=json.loads(devpath.read_text());val={r['sample_id']:r for r in dev if r['split']=='validation' and r['evaluation_class']=='face'}
 old=json.loads((ROOT/'data/training/redaction-inference-options-v4/validation-details.json').read_text());truth=old['truth']
 if {r['sample_id'] for r in truth if r['class']=='face'}!=set(val):raise ValueError('exact_face_VAL_binding_required')
 if any(p['sample_id'] not in {r['sample_id'] for r in dev if r['split']=='validation'} for p in old['predictions']['tiled']):raise ValueError('non_validation_prediction')
 stats,detail=match_groups(truth,old['predictions']['tiled'],val);atomic_json(out/'privacy-per-image.json',{'rows':detail,'expires_at':min(r['expires_at'] for r in val.values())})
 import cv2,numpy as np
 from agent.models import OnnxModel
 lane_dev=ROOT/'data/public/a2d2-refinement-v3/development.json';lrows=[r for r in json.loads(lane_dev.read_text()) if r['split']=='validation'];model=OnnxModel(ROOT/'data/training/lane-refinement-v4/lane.json');cv2.setNumThreads(1)
 byrec=defaultdict(lambda:{'images':0,'tp':0,'fp':0,'fn':0,'valid':0,'paint':0});details=[]
 for index,r in enumerate(lrows):
  im=cv2.imread(str(workspace_path(r['path'],exists=True)));gt=cv2.imread(str(workspace_path(r['mask_path'],exists=True)),0);pred=model.predict(im).astype(bool);valid=gt!=255;target=gt==1
  v={'tp':int((pred&target&valid).sum()),'fp':int((pred&~target&valid).sum()),'fn':int((~pred&target&valid).sum()),'valid':int(valid.sum()),'paint':int((target&valid).sum())};rec=byrec[r['recording']];rec['images']+=1
  for k,n in v.items():rec[k]+=n
  details.append(dict(sample_id=r['sample_id'],**v))
  if index%20==0:print('lane_VAL',index,flush=True)
 for v in byrec.values():v['IoU']=v['tp']/(v['tp']+v['fp']+v['fn']) if v['tp']+v['fp']+v['fn'] else None;v['paint_fraction']=v['paint']/v['valid']
 atomic_json(out/'lane-per-image.json',{'rows':details,'expires_at':min(r['expires_at'] for r in lrows)})
 # Public recording labels are replaced by ordinal groups.
 report={'format':'public-nextcycle-VAL-error-audit-v1','source_role':'reused development VAL only','TEST_predictions_loaded':0,'user_recordings_used':False,'privacy_baseline':'v4_tiled','privacy_face_original_pixel_sqrt_area':stats,'privacy_VAL_images':len(val),'lane_VAL_images':len(lrows),'lane_recording_groups':[dict(group_index=i,**v) for i,v in enumerate(byrec.values())],'lane_threshold':model.manifest['paint_threshold'],'source_expiries_extended':False,'limits':['Original-pixel face bins differ with source resolution; not standardized WIDER difficulty subsets','Development diagnosis only; no independent test accuracy']}
 atomic_json(out/'aggregate.json',report);atomic_json(ROOT/'docs/reports/NEXTCYCLE-VAL-ERROR-AUDIT.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':run()
