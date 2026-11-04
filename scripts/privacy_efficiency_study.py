"""Fixed-weight, validation-only selective-tiling CPU study; never auto-deployed."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.select_redaction_inference import (atomic_json,sha,json_snapshot,scoped,load_validation,
    decode_validation_image,operating_metrics,percentiles,iou)

OUTPUT=ROOT/'data/training/privacy-efficiency-v5'
OPTIONS=ROOT/'data/training/redaction-inference-options-v4'
VALIDATION=ROOT/'data/training/research-redaction-v4/validation-only.json'
VAL_SHA='754958da4b13d1a80abcd40a64467342eb229f7df1b5e6e0e74132dc2963f7f8'
TILED_SHA='bf54a5ad4211e945e71c84502a75a2d32e03de397883d70df8c64881c515aa3a'
CHECKPOINT_SHA='6c985f7e3afb0fa061cbb82d7ec7b304eb5f9fbfcd0cf9f775dbab28873c82ad'
NAMES=('standard','tiled','small_evidence_top2','density_top2','tile_consensus')
FACE_ROWS=18900


def freeze():
    if OUTPUT.exists():raise ValueError('study_output_already_exists')
    standard=OPTIONS/'option-standard/model.json';tiled=OPTIONS/'option-tiled/model.json'
    sm,smsha=json_snapshot(standard);tm,tmsha=json_snapshot(tiled)
    if tm['sha256']!=TILED_SHA or tm['trained_checkpoint_sha256']!=CHECKPOINT_SHA or sm['trained_checkpoint_sha256']!=CHECKPOINT_SHA:raise ValueError('frozen_artifact_binding_failed')
    if sha(ROOT/tm['weights'])!=TILED_SHA or sha(VALIDATION)!=VAL_SHA:raise ValueError('frozen_input_changed')
    OUTPUT.mkdir()
    value={'format':'privacy-efficiency-development-study-v1','created_at':datetime.now(timezone.utc).isoformat(),
        'validation_manifest_sha256':VAL_SHA,'validation_counts':{'face':400,'license_plate':108},
        'checkpoint_sha256':CHECKPOINT_SHA,'standard_manifest_sha256':smsha,'tiled_manifest_sha256':tmsha,
        'tiled_onnx_sha256':TILED_SHA,'confidence':.01,'nms_iou':.5,
        'small_evidence_max_canvas_min_side':64.,'density_confidence':.03,'max_tiles':2,
        'tile_order':'top-left,top-right,bottom-left,bottom-right',
        'ranking':'descending count then summed confidence; ascending tile index for exact ties',
        'empty_evidence_tiles':0,'consensus_strong_confidence':.1,'consensus_iou':.3,
        'consensus_rule':'Low-confidence tile face requires global or another tile face IoU>=.3; preserve global,plate,strong tile rows.',
        'options':list(NAMES),'maximum_validation_images':508,'maximum_prediction_calls_per_image':5,
        'selection_policy':'Report all Pareto points; no TEST access or deployment selection.',
        'known_validation_previously_examined':True,'independent_final_accuracy_claim':False,
        'test_access_permitted':False,'no_new_gradient_steps':True,'gpu_operations':0,
        'expiry_extension_permitted':False,'runtime_default_changed':False,
        'fresh_holdout_requirement':'Before independent acceptance, freeze new sources excluding every historical v4 TRAIN/VAL/TEST/aborted source identity, SHA and near-duplicate; preserve original acquisition deadlines. Physical identity/route unknown must remain explicit.'}
    atomic_json(OUTPUT/'plan.json',value)
    print(json.dumps({'plan_sha256':sha(OUTPUT/'plan.json'),'options':list(NAMES)}))


def select_tiles(global_rows,mode):
    import numpy as np
    scores=(global_rows[:,4]*global_rows[:,5]).astype(np.float64)
    valid=scores >= (.01 if mode=='small_evidence_top2' else .03)
    if mode=='small_evidence_top2':valid &= np.minimum(global_rows[:,2],global_rows[:,3])<=64
    elif mode!='density_top2':raise ValueError('invalid_tile_selection_mode')
    valid &= (global_rows[:,2]>0)&(global_rows[:,3]>0)&(global_rows[:,0]>=0)&(global_rows[:,0]<1920)&(global_rows[:,1]>=0)&(global_rows[:,1]<1920)
    indices=np.flatnonzero(valid);stats=[]
    quadrants=(global_rows[indices,0]>=960).astype(int)+2*(global_rows[indices,1]>=960).astype(int)
    for tile in range(4):
        selected=indices[quadrants==tile]
        if len(selected):stats.append((tile,len(selected),float(scores[selected].sum())))
    return [row[0] for row in sorted(stats,key=lambda row:(-row[1],-row[2],row[0]))[:2]]


def consensus(parts):
    import numpy as np
    evidence=[]
    for branch,rows in enumerate(parts[:5]):
        scores=(rows[:,4]*rows[:,5]).astype(np.float64)
        indices=np.flatnonzero((scores>=.01)&(rows[:,2]>0)&(rows[:,3]>0))
        for index in indices:
            cx,cy,w,h=map(float,rows[index,:4]);evidence.append((branch,index,[cx-w/2,cy-h/2,w,h],scores[index]))
    kept=[parts[0]]
    all_origins=np.array([item[0] for item in evidence])
    all_boxes=np.array([item[2] for item in evidence],dtype=np.float64).reshape(-1,4)
    for branch in range(1,5):
        rows=parts[branch];mask=np.zeros(len(rows),dtype=bool)
        for origin,index,box,score in evidence:
            if origin!=branch:continue
            if score>=.1:mask[index]=True;continue
            others=all_boxes[all_origins!=branch]
            if not len(others):continue
            x,y,w,h=box
            overlap=np.maximum(0,np.minimum(x+w,others[:,0]+others[:,2])-np.maximum(x,others[:,0]))*np.maximum(0,np.minimum(y+h,others[:,1]+others[:,3])-np.maximum(y,others[:,1]))
            union=w*h+others[:,2]*others[:,3]-overlap
            if np.any(np.divide(overlap,union,out=np.zeros_like(overlap),where=union>0)>=.3):mask[index]=True
        kept.append(rows[mask])
    kept.append(parts[5]);return np.concatenate(kept)


def extract_branches():
    """Called in dedicated Python with ONNX installed; CPU metadata surgery only."""
    import onnx
    path=OPTIONS/'option-tiled/model.json';m,_=json_snapshot(path)
    source=ROOT/m['weights']
    if sha(source)!=TILED_SHA:raise ValueError('tiled_onnx_changed')
    model=onnx.load(source,load_external_data=False)
    if any(initializer.data_location==onnx.TensorProto.EXTERNAL for initializer in model.graph.initializer):raise ValueError('external_weights_forbidden')
    output=model.graph.output[0].name
    node=next(n for n in model.graph.node if output in n.output)
    if node.op_type!='Concat' or len(node.input)!=6:raise ValueError('six_branch_concat_required')
    branches=list(node.input);extracted=[]
    for index,name in enumerate(branches):
        rows=FACE_ROWS if index<5 else 8400
        model.graph.value_info.append(onnx.helper.make_tensor_value_info(name,onnx.TensorProto.FLOAT,[1,rows,7]))
    extractor=onnx.utils.Extractor(model)
    for index,name in enumerate(branches):
        graph=extractor.extract_model([model.graph.input[0].name],[name])
        target=OUTPUT/f'branch-{index}.onnx'
        if target.exists():raise ValueError('branch_output_exists')
        onnx.save(graph,target);extracted.append(sha(target))
    atomic_json(OUTPUT/'branch-receipt.json',{'source_onnx_sha256':TILED_SHA,'branch_sha256':extracted,
        'method':'dependency subgraph extraction of existing final concat inputs; weights/Resize/offset unchanged'})


def canvas(frame,size):
    import numpy as np,cv2
    scale=min(size/frame.shape[1],size/frame.shape[0])
    resized=cv2.resize(frame,(max(1,round(frame.shape[1]*scale)),max(1,round(frame.shape[0]*scale))))
    image=np.full((size,size,3),114,np.uint8);image[:resized.shape[0],:resized.shape[1]]=resized
    return image.transpose(2,0,1)[None].astype(np.float32),scale


def postprocess(raw,scale,shape):
    import numpy as np,cv2
    from agent.models import detection_candidates
    boxes,scores,labels=detection_candidates(raw,scale,.01);result=[]
    for label in set(labels):
        candidates=[i for i,value in enumerate(labels) if value==label]
        kept=cv2.dnn.NMSBoxes([boxes[i] for i in candidates],[scores[i] for i in candidates],.01,.5)
        for selected in np.asarray(kept).reshape(-1):
            i=candidates[int(selected)];x,y,w,h=boxes[i]
            left,top=max(0.,x),max(0.,y);right,bottom=min(shape[1],x+w),min(shape[0],y+h)
            if right>left and bottom>top:result.append({'bbox_xywh_px':[left,top,right-left,bottom-top],
                'confidence':scores[i],'class':('face','license_plate')[label]})
    return result


def execute(plan_sha):
    import numpy as np
    from agent.models import OnnxModel,OrtNetwork
    plan,_=json_snapshot(OUTPUT/'plan.json',plan_sha)
    required={'validation_manifest_sha256':VAL_SHA,'options':list(NAMES),'confidence':.01,'nms_iou':.5,
        'small_evidence_max_canvas_min_side':64.,'density_confidence':.03,'max_tiles':2,
        'empty_evidence_tiles':0,'consensus_strong_confidence':.1,'consensus_iou':.3,
        'test_access_permitted':False,'no_new_gradient_steps':True,'runtime_default_changed':False}
    if any(plan.get(key)!=value for key,value in required.items()):raise ValueError('invalid_study_plan')
    if (OUTPUT/'aggregate.json').exists() or (OUTPUT/'running.lock').exists():raise ValueError('study_already_started')
    lock=OUTPUT/'running.lock'
    with lock.open('x') as stream:stream.write(plan_sha)
    atomic_json(OUTPUT/'execution-receipt.json',{'plan_sha256':plan_sha,'study_script_sha256':sha(Path(__file__)),
        'adapter_source_sha256':sha(ROOT/'agent/models.py'),'started_at':datetime.now(timezone.utc).isoformat(),
        'validation_images_read_at_receipt_creation':0,'test_images_read':0})
    for name in ('standard','tiled'):
        if sha(OPTIONS/f'option-{name}/model.json')!=plan[f'{name}_manifest_sha256']:raise ValueError('frozen_manifest_changed')
    receipt,_=json_snapshot(OUTPUT/'branch-receipt.json')
    if receipt['source_onnx_sha256']!=TILED_SHA:raise ValueError('branch_binding_failed')
    for index,digest in enumerate(receipt['branch_sha256']):
        if sha(OUTPUT/f'branch-{index}.onnx')!=digest:raise ValueError('branch_changed')
    models={name:OnnxModel(OPTIONS/f'option-{name}/model.json') for name in ('standard','tiled')}
    branches=[OrtNetwork(OUTPUT/f'branch-{index}.onnx') for index in range(6)]
    def run_branch(index,blob):
        branches[index].setInput(blob);return branches[index].forward()[0]
    # Fixed synthetic full-graph-vs-extracted proof precedes all source decoding.
    rng=np.random.default_rng(20261104);dummy=rng.integers(0,256,size=(1,3,1920,1920),dtype=np.uint8).astype(np.float32)
    models['tiled'].net.setInput(dummy);reference=models['tiled'].net.forward()[0]
    reconstructed=np.concatenate([run_branch(i,dummy) for i in range(6)])
    parity=bool(np.allclose(reference,reconstructed,rtol=.001,atol=.0001))
    if not parity:raise ValueError('extracted_branch_numeric_parity_failed')
    rows,counts,expiry=load_validation(VALIDATION,VAL_SHA)
    truth=[];predictions={name:[] for name in NAMES};timings={name:[] for name in NAMES};tile_counts={name:[] for name in NAMES[2:4]}
    for number,row in enumerate(rows):
        frame=decode_validation_image(row);known=row['evaluation_class']
        truth.extend({'sample_id':row['sample_id'],'class':known,'bbox_xywh_px':item['bbox_xywh_px']} for item in row['objects'])
        for name in NAMES:
            started=time.perf_counter()
            if name in ('standard','tiled'):
                result=models[name].predict(frame)
            else:
                blob,scale=canvas(frame,1920)
                if name=='tile_consensus':raw=consensus([run_branch(i,blob) for i in range(6)])
                else:
                    global_rows=run_branch(0,blob);plate=run_branch(5,blob);tiles=select_tiles(global_rows,name)
                    tile_counts[name].append(len(tiles))
                    parts=[global_rows,*[run_branch(i+1,blob) for i in sorted(tiles)],plate]
                    raw=np.concatenate(parts)
                result=postprocess(raw,scale,frame.shape)
            timings[name].append(time.perf_counter()-started)
            predictions[name].extend({'sample_id':row['sample_id'],**item} for item in result if item['class']==known)
        if (number+1)%50==0:print(json.dumps({'validation_completed':number+1,'total':508}),flush=True)
    aggregate={'format':'privacy-efficiency-development-result-v1','plan_sha256':plan_sha,'validation_counts':counts,
        'source_expiry_min':expiry,'source_expiries_extended':False,'test_images_read':0,'gpu_operations':0,
        'independent_final_accuracy_claim':False,'runtime_default_changed':False,'numeric_tolerance':{'rtol':.001,'atol':.0001},
        'extracted_graph_synthetic_parity_passed':parity,'raw_max_absolute_error':float(np.max(np.abs(reference-reconstructed))),
        'scores':{name:operating_metrics(truth,predictions[name],.01) for name in NAMES},
        'predict_seconds':{name:percentiles(timings[name]) for name in NAMES},
        'selected_tile_counts':{name:{str(i):values.count(i) for i in range(3)} for name,values in tile_counts.items()},
        'limits':['Previously examined VAL is development evidence only; all old TEST is regression only.',
            'Known-class conditional annotations exclude unknown-class FP; no deployment/privacy acceptance.',
            'Global-only evidence can miss small faces and suppress all tile recovery for those images.',
            'Timing is actual preprocess/ORT/postprocess CPU cost, excluding source decoding and full vehicle pipeline.']}
    atomic_json(OUTPUT/'private-predictions.json',{'truth':truth,'predictions':predictions,'source_expiry_min':expiry})
    atomic_json(OUTPUT/'aggregate.json',aggregate)
    print(json.dumps(aggregate,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['freeze','extract','run']);parser.add_argument('--plan-sha256')
    args=parser.parse_args()
    if args.mode=='freeze':freeze()
    elif args.mode=='extract':extract_branches()
    else:execute(args.plan_sha256)
