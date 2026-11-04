"""Explicit CPU execution profiles and owned per-model inference workers."""
from concurrent.futures import ThreadPoolExecutor,wait
from threading import Lock
from .models import OnnxModel

PROFILES={'default':{'cpu_threads':2,'allow_spinning':True,'model_workers':1},
          'fast':{'cpu_threads':8,'allow_spinning':False,'model_workers':3}}


def profile_options(name):
    if not isinstance(name,str) or name not in PROFILES:raise ValueError('unknown_cpu_profile')
    return dict(PROFILES[name])


def close_models(models):
    for model in models.values():
        close=getattr(model,'close',None)
        if close:close()


def create_models(manifests,name='default',model_factory=OnnxModel,*,inference_backend='cpu',gpu_profile=None):
    options=profile_options(name)
    if inference_backend not in ('cpu','cuda'):raise ValueError('unknown_inference_backend')
    if inference_backend=='cpu' and gpu_profile is not None:raise ValueError('gpu_profile_requires_cuda')
    if inference_backend=='cuda':
        if gpu_profile is None or model_factory is not OnnxModel:raise ValueError('verified_native_gpu_profile_required')
        from .gpu_worker_client import gpu_bundle_manifests
        manifests=gpu_bundle_manifests(manifests,gpu_profile)
    result={}
    try:
        for task,path in manifests.items():
            if inference_backend=='cuda':result[task]=model_factory(path,backend='onnxruntime-cuda')
            elif name=='default' or model_factory is not OnnxModel:result[task]=model_factory(path)
            else:result[task]=model_factory(path,cpu_threads=options['cpu_threads'],allow_spinning=options['allow_spinning'])
        return result
    except BaseException:close_models(result);raise


class InferenceRunner:
    def __init__(self,models,profile='default'):
        self.models=models;self.options=profile_options(profile);self.profile=profile
        if len({id(model) for model in models.values()})!=len(models):raise ValueError('independent_model_sessions_required')
        self.lock=Lock();self.closed=False
        self.pool=ThreadPoolExecutor(max_workers=self.options['model_workers'],thread_name_prefix='road-inference') if self.options['model_workers']>1 else None
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def predict(self,frame):
        with self.lock:
            if self.closed:raise RuntimeError('inference_runner_closed')
            if self.pool is None:return {task:model.predict(frame) for task,model in self.models.items()}
            futures={task:self.pool.submit(model.predict,frame) for task,model in self.models.items()}
            try:return {task:future.result() for task,future in futures.items()}
            except BaseException:
                for future in futures.values():future.cancel()
                # Running mutable sessions must finish before another batch/close.
                wait(futures.values());raise
    def close(self):
        with self.lock:
            if not self.closed:
                self.closed=True
                if self.pool:self.pool.shutdown(wait=True,cancel_futures=True)
