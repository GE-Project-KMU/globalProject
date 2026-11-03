"""Actual worst-target TRAIN batch hardware smoke; weights are discarded."""
import argparse,json,time,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import improve_pothole_train as m

def main():
    parser=argparse.ArgumentParser();parser.add_argument("release");args=parser.parse_args()
    m.cv2.setNumThreads(1);m.seed_all(20261104)
    rows=[r for r in json.loads(m.workspace_path(args.release,exists=True).read_text()) if r["split"]=="train"]
    worst=max(rows,key=lambda r:len(r["objects"]))
    dataset=m.MosaicDetection(m.OriginalSamples([worst]*32,640),img_size=(640,640),
        preproc=m.TrainTransform(max_labels=120),mosaic_prob=1.,enable_mixup=False)
    images,labels,_,_=next(iter(m.torch.utils.data.DataLoader(dataset,batch_size=16,num_workers=0)))
    model,_=m.make_model(["pothole"],"data/research/yolox_nano_coco.pth");model=model.cuda();start=time.perf_counter()
    with m.torch.autocast("cuda",dtype=m.torch.float16):loss=model(images.cuda(),labels.cuda())["total_loss"]
    m.torch.amp.GradScaler("cuda").scale(loss).backward()
    report={"actual_train_data":True,"test_data_accessed":False,"batch_shape":list(images.shape),
        "worst_original_targets":len(worst["objects"]),"native_mosaic_probability":1.,"finite_loss":bool(m.torch.isfinite(loss)),
        "loss":float(loss.detach()),"peak_cuda_allocated_MiB":m.torch.cuda.max_memory_allocated()/2**20,
        "seconds":time.perf_counter()-start,"timing_is_not_isolated_benchmark":True,"weights_discarded":True}
    m.atomic_json(m.ROOT/"docs/reports/POTHOLE-IMPROVEMENT-BATCH16-SMOKE.json",report);print(json.dumps(report),flush=True)
    if not report["finite_loss"]:raise ValueError("non_finite_smoke")

if __name__=="__main__":main()
