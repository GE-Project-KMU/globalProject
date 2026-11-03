"""Actual CUDA/architecture/dependency smoke; synthetic tensors measure no accuracy."""
from pathlib import Path
import gc,json,platform,sys,time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from setup_training import environment
import os
os.environ.update(environment())
import cv2,numpy as np,onnx,onnxruntime,torch,torchvision

def exercise(name,model,images,labels=None):
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    model=model.cuda().train();optimizer=torch.optim.SGD(model.parameters(),lr=.001)
    began=time.perf_counter()
    if name=="yolox_nano":loss=model(images,labels)["total_loss"]
    else:loss=torch.nn.functional.cross_entropy(model(images)["out"],labels)
    loss.backward();optimizer.step();torch.cuda.synchronize()
    result={"loss_finite":bool(torch.isfinite(loss).item()),"peak_allocated_MiB":round(torch.cuda.max_memory_allocated()/2**20,2),"peak_reserved_MiB":round(torch.cuda.max_memory_reserved()/2**20,2),"seconds":round(time.perf_counter()-began,3)}
    assert result["loss_finite"]
    del model,optimizer,loss;gc.collect();torch.cuda.empty_cache()
    return result

def main():
    assert torch.cuda.is_available(),"CUDA unavailable; do not claim GPU training"
    torch.set_num_threads(4)
    result={"python":platform.python_version(),"executable":str(Path(sys.executable).resolve()),"torch":torch.__version__,"torchvision":torchvision.__version__,"cuda_runtime":torch.version.cuda,"cudnn":torch.backends.cudnn.version(),"gpu":torch.cuda.get_device_name(0),"capability":list(torch.cuda.get_device_capability(0)),"compiled_architectures":torch.cuda.get_arch_list(),"numpy":np.__version__,"opencv":cv2.__version__,"onnx":onnx.__version__,"onnxruntime":onnxruntime.__version__,"synthetic_smoke_only":True}
    assert result["capability"]==[7,5] and "sm_75" in result["compiled_architectures"]
    x=torch.randn(512,512,device="cuda");y=x@x.T;torch.cuda.synchronize();assert torch.isfinite(y).all();del x,y
    kept=torchvision.ops.nms(torch.tensor([[0.,0.,10.,10.],[1.,1.,9.,9.]],device="cuda"),torch.tensor([.9,.8],device="cuda"),.5);assert kept.tolist()==[0]
    result["cuda_matmul_and_torchvision_nms"]=True
    lane=torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(weights=None,weights_backbone=None,num_classes=2,aux_loss=False)
    result["deeplabv3_mobilenetv3"]=exercise("lane",lane,torch.randn(2,3,512,512,device="cuda"),torch.randint(0,2,(2,512,512),device="cuda"))
    del lane;gc.collect();torch.cuda.empty_cache()
    source=ROOT/".cache"/"vendor"/"YOLOX";sys.path.insert(0,str(source))
    from yolox.exp import get_exp
    exp=get_exp(exp_file=str(source/"exps"/"default"/"yolox_nano.py"));exp.num_classes=2
    model=exp.get_model();labels=torch.zeros((2,50,5),device="cuda");labels[:,0,:]=torch.tensor([0,208,208,64,64],device="cuda")
    result["yolox_nano"]=exercise("yolox_nano",model,torch.randn(2,3,416,416,device="cuda"),labels)
    del model,labels;gc.collect();torch.cuda.empty_cache()
    out=ROOT/"data"/"training"/"environment";out.mkdir(parents=True,exist_ok=True)
    conv=torch.nn.Conv2d(3,2,1).eval();sample=torch.randn(1,3,16,16)
    torch.onnx.export(conv,sample,str(out/"synthetic-smoke.onnx"),opset_version=17,input_names=["images"],output_names=["output"],dynamo=False)
    onnx.checker.check_model(onnx.load(str(out/"synthetic-smoke.onnx")))
    session=onnxruntime.InferenceSession(str(out/"synthetic-smoke.onnx"),providers=["CPUExecutionProvider"])
    expected=conv(sample).detach().numpy();actual=session.run(None,{"images":sample.numpy()})[0]
    np.testing.assert_allclose(expected,actual,rtol=1e-4,atol=1e-5)
    net=cv2.dnn.readNetFromONNX(str(out/"synthetic-smoke.onnx"));net.setInput(sample.numpy());np.testing.assert_allclose(expected,net.forward(),rtol=1e-4,atol=1e-5)
    result["onnx_export_checker_onnxruntime_opencv_parity"]=True
    (out/"environment-report.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))

if __name__=="__main__":main()
