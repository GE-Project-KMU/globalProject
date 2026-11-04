"""Private CUDA process protocol: fixed model, bounded shared-memory tensors."""
import argparse,hashlib,json,os,sys,uuid
from pathlib import Path
from multiprocessing import shared_memory
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def run(weights,digest):
    weights=Path(weights).resolve()
    if not weights.is_relative_to(ROOT) or not weights.is_file():raise ValueError('worker_weight_path_outside_workspace')
    raw=weights.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('worker_weight_digest_mismatch')
    handle=os.add_dll_directory(str(ROOT/'.venv-training/Lib/site-packages/torch/lib'))
    sys.path.append(str(ROOT/'.venv-training/Lib/site-packages'))
    import torch
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    import numpy as np,onnxruntime as ort
    from agent.gpu_worker_client import shape_bytes
    if ort.__version__!='1.20.2':raise ValueError('worker_ort_version_mismatch')
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1;options.log_severity_level=3
    profile_directory=os.getenv('ROAD_GPU_PROFILE_DIRECTORY')
    if profile_directory:
        directory=Path(profile_directory).resolve()
        if not directory.is_relative_to(ROOT):raise ValueError('worker_profile_directory_outside_workspace')
        directory.mkdir(parents=True,exist_ok=True);options.enable_profiling=True
        options.profile_file_prefix=str(directory/('cuda-'+uuid.uuid4().hex))
    session=ort.InferenceSession(raw,sess_options=options,providers=[('CUDAExecutionProvider',{'use_tf32':0,
        'gpu_mem_limit':2*1024**3,'cudnn_conv_algo_search':'HEURISTIC'}),'CPUExecutionProvider'])
    providers=session.get_providers();provider_options=session.get_provider_options()
    if providers!=['CUDAExecutionProvider','CPUExecutionProvider'] or provider_options['CUDAExecutionProvider'].get('use_tf32')!='0':raise ValueError('worker_cuda_fallback_or_tf32')
    inputs=session.get_inputs();outputs=session.get_outputs()
    if len(inputs)!=1 or len(outputs)!=1 or inputs[0].type!='tensor(float)' or outputs[0].type!='tensor(float)':raise ValueError('worker_tensor_contract')
    sizes=[shape_bytes(value.shape) for value in (inputs[0],outputs[0])]
    print(json.dumps({'status':'ready','backend':'onnxruntime-cuda-worker','version':ort.__version__,
        'providers':providers,'provider_options':provider_options,'use_tf32':provider_options['CUDAExecutionProvider']['use_tf32'],
        'input_shape':inputs[0].shape,'output_shape':outputs[0].shape,'input_type':inputs[0].type,'output_type':outputs[0].type,
        'device':torch.cuda.get_device_name(0),'device_capability':list(torch.cuda.get_device_capability(0)),
        'python':sys.version.split()[0],'cuda_runtime':torch.version.cuda,'cudnn':torch.backends.cudnn.version()}),flush=True)
    buffers=[];arrays=[];previous_request=0
    try:
        for line in sys.stdin:
            if len(line)>65536:raise ValueError('worker_message_too_large')
            request=json.loads(line)
            if request.get('mode')=='stop':return
            if type(request.get('request_id')) is not int or request['request_id']!=previous_request+1:raise ValueError('worker_request_order')
            previous_request=request['request_id']
            if request.get('mode')=='bind':
                if buffers:raise ValueError('worker_already_bound')
                for key,size,shape in zip(('input','output'),sizes,(inputs[0].shape,outputs[0].shape)):
                    name=request[key]
                    if not isinstance(name,str) or not name.startswith('roadwatch_gpu_') or len(name)>100:raise ValueError('worker_memory_name')
                    memory=shared_memory.SharedMemory(name=name)
                    # Windows mappings may report page-rounded sizes on attach.
                    if not size<=memory.size<=size+65535:memory.close();raise ValueError('worker_memory_size')
                    buffers.append(memory);arrays.append(np.ndarray(shape,dtype=np.float32,buffer=memory.buf))
                print(json.dumps({'status':'bound','request_id':previous_request}),flush=True)
            elif request.get('mode')=='run':
                if len(arrays)!=2 or not np.isfinite(arrays[0]).all():raise ValueError('worker_invalid_input')
                result=session.run([outputs[0].name],{inputs[0].name:arrays[0]})[0]
                if list(result.shape)!=outputs[0].shape or result.dtype!=np.float32 or not np.isfinite(result).all():raise ValueError('worker_invalid_output')
                np.copyto(arrays[1],result);print(json.dumps({'status':'completed','request_id':previous_request}),flush=True)
            else:raise ValueError('worker_invalid_command')
    finally:
        arrays=[]
        for memory in buffers:memory.close()
        if profile_directory:session.end_profiling()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--weights',required=True);parser.add_argument('--sha256',required=True);args=parser.parse_args()
    try:run(args.weights,args.sha256)
    except Exception as error:
        print(json.dumps({'status':'failed','error_type':type(error).__name__}),flush=True);sys.exit(2)
