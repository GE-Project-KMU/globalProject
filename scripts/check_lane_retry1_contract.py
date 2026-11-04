"""CPU-only retry1 backend/learning-function consistency and real GT smoke."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import ast,hashlib,json,random,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import atomic_json,sha256,workspace_path

def functions(path):
 tree=ast.parse(path.read_text(encoding='utf-8-sig'));wanted={'paint_loss','full_context_sample','validation_scores','export_fp64_mean'}
 return {n.name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest() for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in wanted}
def benchmark_assignments(path):
 tree=ast.parse(path.read_text(encoding='utf-8-sig'));values=[]
 for node in ast.walk(tree):
  if isinstance(node,ast.Assign) and any(isinstance(v,ast.Attribute) and ast.unparse(v)=='torch.backends.cudnn.benchmark' for v in node.targets):values.append(ast.literal_eval(node.value))
 return values

def run():
 import cv2,numpy as np,torch
 from scripts import train_lane_cycle2 as old,train_lane_cycle2_retry1 as new
 from agent.dataset import validate_research_release
 from agent.lane_refinement_network import make_lane_refinement_network
 from agent.lane_network import freeze_batch_norm_statistics
 original=ROOT/'scripts/train_lane_cycle2.py';source=ROOT/'scripts/train_lane_cycle2_retry1.py';oldplan=json.loads((ROOT/'docs/reports/LANE-CYCLE2-PLAN.json').read_text());newplan=new.plan()
 if sha256(original)!='401fab5253cfa29492ca4747d9a3eee1ee57a17304cbd414731b32200f513a84':raise ValueError('original_source_changed')
 if newplan!=json.loads((ROOT/'docs/reports/LANE-CYCLE2-RETRY1-PLAN.json').read_text()):raise ValueError('frozen_retry_plan_required')
 digests=functions(original)
 if digests!=functions(source) or len(digests)!=4:raise ValueError('learning_math_or_geometry_changed')
 if benchmark_assignments(original)!=[True] or benchmark_assignments(source)!=[False] or newplan['cudnn_benchmark'] is not False:raise ValueError('runtime_backend_and_plan_mismatch')
 for key in ['development_sha256','release_sha256','fresh_manifest_sha256','seed','max_epochs','min_epochs','patience','min_delta','batch_size','training_patch_size','input_size','optimizer','scheduler','loss','augmentation','EMA','threshold_grid']:
  if oldplan[key]!=newplan[key]:raise ValueError('unchanged_recipe_required:'+key)
 if new.BASE!=old.BASE or new.DEV!=old.DEV or new.RELEASE!=old.RELEASE or new.OUT==old.OUT:raise ValueError('fresh_output_same_original_warmstart_required')
 torch.set_num_threads(2);cv2.setNumThreads(1);torch.manual_seed(20261106);random.seed(20261106);np.random.seed(20261106);torch.backends.cudnn.benchmark=False
 rows=[r for r in json.loads(new.DEV.read_text()) if r['split']=='train'];chosen=max(rows,key=lambda r:r.get('positive_pixels',0)/max(1,r.get('valid_pixels',1)))
 if not validate_research_release([chosen])['valid'] or sha256(workspace_path(chosen['mask_path'],exists=True))!=chosen['mask_sha256']:raise ValueError('current_licensed_hashed_development_required')
 x,y=new.full_context_sample(chosen);model=make_lane_refinement_network();model.load_state_dict(torch.load(new.BASE,map_location='cpu',weights_only=True),strict=True);model.train();freeze_batch_norm_statistics(model);out=model(torch.from_numpy(x[None]))['out'];loss=new.paint_loss(out,torch.from_numpy(y[None]));loss.backward()
 if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('real_development_loss_backward_nonfinite')
 synthetic=torch.randn(1,2,8,8,requires_grad=True);ignored=torch.full((1,8,8),255,dtype=torch.long);zero=new.paint_loss(synthetic,ignored);zero.backward()
 if float(zero)!=0 or torch.count_nonzero(synthetic.grad)!=0:raise ValueError('ignore_gradient_contract_failed')
 report={'format':'lane-retry1-CPU-contract-v1','passed':True,'GPU_operations':0,'optimizer_updates':0,'weights_saved':False,'TEST_images_loaded':0,'user_recordings_used':False,'original_source_sha256':sha256(original),'retry_source_sha256':sha256(source),'retry_plan_sha256':sha256(ROOT/'docs/reports/LANE-CYCLE2-RETRY1-PLAN.json'),'contract_source_sha256':sha256(Path(__file__)),'learning_math_and_geometry_AST_identical':True,'AST_digests':digests,'learning_recipe_fields_identical':True,'old_cudnn_benchmark':True,'retry_cudnn_benchmark':torch.backends.cudnn.benchmark,'bitwise_training_equivalence_claimed':False,'full_restart_not_resume':True,'same_v4_warmstart':True,'real_TRAIN_loss':float(loss.detach()),'tensor_shape':list(x[None].shape),'valid_GT_pixels':int((y!=255).sum()),'positive_GT_pixels':int((y==1).sum()),'ignore_loss_gradient_zero':True,'source_expiries_extended':False,'limits':['Runtime algorithm-selection policy changed; no bitwise learning equivalence','CPU/backward smoke, not accuracy or GPU memory acceptance']}
 atomic_json(ROOT/'data/training/public-nextcycle/lane-retry1-CPU-contract.json',report);atomic_json(ROOT/'docs/reports/LANE-CYCLE2-RETRY1-CONTRACT.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':run()
