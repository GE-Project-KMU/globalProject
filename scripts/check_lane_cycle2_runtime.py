"""Reused-session all-VAL raw/default-tolerance and exact-mask comparison.

App ORT1.31.1 / isolated ORT1.31.1 / original FP32 Torch checkpoint. No TEST.
Working arrays are bounded to three files and removed on completion/failure.
"""
import argparse,json,math,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import workspace_path,atomic_json,sha256
def read(p):return json.loads(p.read_text(encoding='utf-8'))

def worker(manifest,checkpoint):
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch,onnxruntime as ort,numpy as np,cv2
    from scripts.verify_trained_models import build_torch_reference,compare
    info=read(workspace_path(manifest,exists=True));torch.set_num_threads(2);cv2.setNumThreads(2)
    if ort.__version__!='1.31.1':raise ValueError('isolated_reference_ORT1311_required')
    model=build_torch_reference(info,ROOT/'.cache/vendor/YOLOX').eval().cpu();state=torch.load(workspace_path(checkpoint,exists=True),map_location='cpu',weights_only=True);model.load_state_dict(state.get('model',state),strict=True)
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(workspace_path(info['weights'],exists=True)),sess_options=options,providers=['CPUExecutionProvider'])
    print(json.dumps({'status':'ready','torch':torch.__version__,'ort':ort.__version__,'providers':session.get_providers(),'GPU_initialized':torch.cuda.is_initialized()}),flush=True)
    for line in sys.stdin:
        item=json.loads(line)
        if item.get('stop'):break
        blob=np.load(workspace_path(item['blob'],exists=True),allow_pickle=False);app=np.load(workspace_path(item['raw'],exists=True),allow_pickle=False);mask=np.load(workspace_path(item['mask'],exists=True),allow_pickle=False)
        with torch.inference_mode():native=model(torch.from_numpy(blob))['out'].numpy()
        reference=session.run(None,{session.get_inputs()[0].name:blob})[0]
        raw={'app_vs_original_FP32_torch':compare(app,native,.001,.0001),'app_vs_reference_ORT131':compare(app,reference,.001,.0001),'reference_ORT131_vs_original_FP32_torch':compare(reference,native,.001,.0001)}
        h,w=item['original_size'];rh,rw=item['resized_size'];post={}
        for name,value in [('app',app),('torch',native),('ort',reference)]:
            small=((value[0,1]-value[0,0])>math.log(info['paint_threshold']/(1-info['paint_threshold']))).astype(np.uint8)
            small=small[:max(1,round(rh*small.shape[0]/info['input_size'][1])),:max(1,round(rw*small.shape[1]/info['input_size'][0]))]
            predicted=cv2.resize(small,(w,h),interpolation=cv2.INTER_NEAREST);post[name]={'exact_match':bool(np.array_equal(mask,predicted)),'mismatching_pixels':int((mask!=predicted).sum())}
        print(json.dumps({'index':item['index'],'raw':raw,'post':post,'raw_passed':all(v['passed'] for v in raw.values()),'post_passed':all(v['exact_match'] for v in post.values())}),flush=True)

def validate(manifest,checkpoint,training_python,output):
    import cv2,numpy as np
    from agent.models import OnnxModel
    from scripts.setup_training import environment
    manifest=workspace_path(manifest,exists=True);checkpoint=workspace_path(checkpoint,exists=True);output=workspace_path(output)
    if not output.is_relative_to(ROOT/'data/training') or output.exists():raise ValueError('fresh_private_training_output_required')
    info=read(manifest);dev=workspace_path(info['training_development_release_path'],exists=True)
    if sha256(dev)!=info['training_development_release_sha256']:raise ValueError('development_changed')
    rows=sorted((r for r in read(dev) if r['split']=='validation'),key=lambda r:r['source_id'])
    expected=read(ROOT/'docs/reports/LANE-CYCLE2-PLAN.json')['validation_count']
    if len(rows)!=expected:raise ValueError('complete_predeclared_VAL_required')
    plan=output.with_name(output.stem+'-plan.json');atomic_json(plan,{'manifest_sha256':sha256(manifest),'checkpoint_sha256':sha256(checkpoint),'development_sha256':sha256(dev),
        'cases':[{'path':r['path'],'sha256':r['sha256'],'expires_at':r['expires_at']} for r in rows],'rtol':.001,'atol':.0001,'test_accessed':False})
    working=output.parent/'numeric-working';working.mkdir();env=environment();env['CUDA_VISIBLE_DEVICES']=''
    log=output.with_suffix('.worker.log').open('w',encoding='utf-8')
    process=subprocess.Popen([str(workspace_path(training_python,exists=True)),str(Path(__file__)),'worker',str(manifest),'--checkpoint',str(checkpoint)],cwd=ROOT,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1)
    result={'manifest_sha256':sha256(manifest),'weights_sha256':info['sha256'],'checkpoint_sha256':sha256(checkpoint),'plan_sha256':sha256(plan),'rtol':.001,'atol':.0001,'test_accessed':False,
        'source_expires_at':min(r['expires_at'] for r in rows),'worker_source_sha256':sha256(Path(__file__)),'cases':[],'complete':False}
    try:
        handshake=json.loads(process.stdout.readline());result['reference_runtime']=handshake
        if handshake.get('status')!='ready' or handshake.get('GPU_initialized') is not False or handshake.get('providers')!=['CPUExecutionProvider']:raise ValueError('CPU_worker_not_ready')
        adapter=OnnxModel(manifest);result['app_runtime']={'engine':adapter.backend,'version':adapter.backend_version}
        class Capture:
            def __init__(self,net):self.net=net
            def setInput(self,value):self.blob=value.copy();self.net.setInput(value)
            def forward(self):self.raw=self.net.forward();return self.raw
        capture=Capture(adapter.net);adapter.net=capture
        for index,row in enumerate(rows):
            image=workspace_path(row['path'],exists=True)
            if sha256(image)!=row['sha256']:raise ValueError('VAL_image_changed')
            pixels=cv2.imread(str(image));mask=adapter.predict(pixels)
            for name,value in [('blob',capture.blob),('raw',capture.raw),('mask',mask)]:np.save(working/(name+'.npy'),value,allow_pickle=False)
            h,w=pixels.shape[:2];scale=min(info['input_size'][0]/w,info['input_size'][1]/h)
            request={name:str(working/(name+'.npy')) for name in ['blob','raw','mask']};request.update(index=index,original_size=[h,w],resized_size=[round(h*scale),round(w*scale)])
            process.stdin.write(json.dumps(request)+'\n');process.stdin.flush();value=json.loads(process.stdout.readline());value['image_sha256']=row['sha256'];result['cases'].append(value);atomic_json(output,result)
            print(json.dumps({'completed_VAL':index+1,'raw_failures':sum(not r['raw_passed'] for r in result['cases']),'post_failures':sum(not r['post_passed'] for r in result['cases'])}),flush=True)
        result['complete']=True;result['all_raw_passed']=all(r['raw_passed'] for r in result['cases']);result['all_post_passed']=all(r['post_passed'] for r in result['cases'])
        if sha256(manifest)!=result['manifest_sha256'] or sha256(checkpoint)!=result['checkpoint_sha256'] or sha256(workspace_path(info['weights']))!=result['weights_sha256']:raise ValueError('frozen_artifact_changed')
        atomic_json(output,result)
    finally:
        try:process.stdin.write('{"stop":true}\n');process.stdin.flush()
        except (BrokenPipeError,OSError):pass
        process.stdin.close()
        try:process.wait(timeout=30)
        except subprocess.TimeoutExpired:process.terminate();process.wait(timeout=30)
        process.stdout.close();log.close()
        for name in ['blob','raw','mask']:workspace_path(working/(name+'.npy')).unlink(missing_ok=True)
    return result['all_raw_passed'] and result['all_post_passed']

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['worker','validate']);p.add_argument('manifest');p.add_argument('--checkpoint',required=True);p.add_argument('--training-python',default='.cache/lane-eval-311-ort1311/python.exe');p.add_argument('--output');a=p.parse_args()
    if a.action=='worker':worker(a.manifest,a.checkpoint)
    else:raise SystemExit(0 if validate(a.manifest,a.checkpoint,a.training_python,a.output) else 2)
