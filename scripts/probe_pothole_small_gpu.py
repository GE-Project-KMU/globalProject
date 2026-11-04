"""One development TRAIN batch, no update/export/TEST: capacity memory check."""
import copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    import torch,cv2
    from agent.paths import atomic_json,sha256
    from agent.dataset import validate_research_release
    from agent.pothole_small_network import make_pothole_small
    from scripts.train_pothole_small import OUT,INIT
    from scripts.improve_pothole_train import OriginalSamples
    from scripts.cycle2_device_memory import DeviceMemoryObserver
    from yolox.data.datasets.mosaicdetection import MosaicDetection
    from yolox.data.data_augment import TrainTransform
    output=ROOT/OUT;receipt=output/'GPU-memory-preflight.json'
    if receipt.exists():raise ValueError('one_memory_probe_preserve_evidence')
    plan=json.loads((output/'plan.json').read_text());rows=json.loads((output/'release.json').read_text())
    if sha256(ROOT/'scripts/train_pothole_small.py')!=plan['source_sha256']:raise ValueError('learning_source_changed')
    selected=sorted((r for r in rows if r['split']=='train'),key=lambda r:(-len(r['objects']),r['sample_id']))[:8]
    if len(selected)!=8 or not validate_research_release(selected)['valid']:raise ValueError('licensed_hashed_TRAIN_only_required')
    torch.set_num_threads(2);cv2.setNumThreads(1);torch.manual_seed(plan['seed'])
    observer=DeviceMemoryObserver(ROOT/'.cache/cycle2-small-memory-observer.log').start();report={'optimizer_updates':0,'TEST_images_used':0,'source_sha256':sha256(Path(__file__)),'plan_sha256':sha256(output/'plan.json'),'status':'started'}
    try:
        model,_=make_pothole_small(pretrained=INIT);model.cuda();ema=copy.deepcopy(model).eval();model.train();model.head.use_l1=True
        for m in model.modules():
            if isinstance(m,torch.nn.modules.batchnorm._BatchNorm):m.eval()
        ds=MosaicDetection(OriginalSamples(selected,640,0),img_size=(640,640),mosaic=False,preproc=TrainTransform(max_labels=120,hsv_prob=0.,flip_prob=0.),enable_mixup=False)
        images,labels,_,_=next(iter(torch.utils.data.DataLoader(ds,batch_size=8,num_workers=0)))
        if (labels[:,:,3]>0).sum(1).tolist()!=[len(r['objects']) for r in selected]:raise ValueError('all_GT_required')
        with torch.autocast('cuda',dtype=torch.float16):loss=model(images.cuda(),labels.cuda())['total_loss']
        loss.backward();torch.cuda.synchronize();observer.raise_if_unsafe()
        if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('finite_backward_required')
        report.update(status='passed',batch_size=8,all_GT_preserved=True,loss=float(loss.detach()),own_peak_reserved_bytes=torch.cuda.max_memory_reserved(),own_peak_allocated_bytes=torch.cuda.max_memory_allocated(),own_OOM_counter=torch.cuda.memory_stats().get('num_ooms',0),optimizer_state_not_allocated=True)
        if report['own_peak_reserved_bytes']>5*2**30 or report['own_OOM_counter']:raise ValueError('own_memory_budget_exceeded')
        observer.wait_additional_samples(2)
    except BaseException as error:
        report.update(status='failed',error=type(error).__name__+':'+str(error));raise
    finally:
        observer.close();report['device_memory']=observer.summary();atomic_json(receipt,report);atomic_json(ROOT/'docs/reports/POTHOLE-SMALL-V9-GPU-PREFLIGHT.json',report)
    print(json.dumps(report))
if __name__=='__main__':main()
