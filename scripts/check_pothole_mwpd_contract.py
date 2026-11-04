"""Actual public TRAIN samples: collate, full GT and backward preflight on CPU."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'.cache/vendor/YOLOX'))
import cv2,torch
from agent.paths import workspace_path,sha256,atomic_json
from scripts.train_yolox_research import make_model
from scripts.improve_pothole_train import OriginalSamples
from yolox.data.datasets.mosaicdetection import MosaicDetection
from yolox.data.data_augment import TrainTransform


def main():
    cv2.setNumThreads(1);torch.set_num_threads(2);torch.manual_seed(20261106)
    if torch.cuda.is_initialized():raise ValueError('CPU_contract_must_not_initialize_GPU')
    release=ROOT/'data/training/pothole-mwpd-v6/release.json';rows=json.loads(release.read_text(encoding='utf-8'))
    train=[r for r in rows if r['split']=='train']
    if max(len(r['objects']) for r in train)>30:raise ValueError('source_target_cap_exceeded')
    chosen=[next(r for r in train if r['development_domain']==domain and r['objects']) for domain in ('original','mwpd')]
    dataset=MosaicDetection(OriginalSamples(chosen,640,0),img_size=(640,640),mosaic=False,preproc=TrainTransform(max_labels=120,hsv_prob=0.,flip_prob=0.),enable_mixup=False)
    images,labels,_,_=next(iter(torch.utils.data.DataLoader(dataset,batch_size=2,shuffle=False,num_workers=0)))
    expected=[len(r['objects']) for r in chosen];actual=(labels[:,:,3]>0).sum(1).tolist()
    if actual!=expected or list(images.shape)!=[2,3,640,640] or list(labels.shape)!=[2,120,5]:raise ValueError('complete_GT_collate_contract_failed')
    model,_=make_model(['pothole']);checkpoint=ROOT/'data/training/pothole-nano-rdd2020-v4/best.pt'
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model'],strict=True);model.train()
    for module in model.modules():
        if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.eval()
    loss=model(images,labels)['total_loss']
    if not torch.isfinite(loss):raise ValueError('nonfinite_real_training_batch_loss')
    loss.backward()
    gradients=[p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(v).all() for v in gradients) or not any(torch.count_nonzero(v) for v in gradients):raise ValueError('finite_nonzero_backward_required')
    report={'format':'pothole-mwpd-training-contract-v1','source_sha256':sha256(Path(__file__)),'release_sha256':sha256(release),'checkpoint_sha256':sha256(checkpoint),'public_train_domains':2,'batch_shape':list(images.shape),'target_shape':list(labels.shape),'all_source_GT_preserved':actual==expected,'max_training_targets':max(len(r['objects']) for r in train),'finite_loss':float(loss.detach()),'finite_nonzero_gradients':True,'optimizer_updates':0,'test_images_used':0,'GPU_initialized':torch.cuda.is_initialized(),'user_recordings_used':False}
    atomic_json(ROOT/'docs/reports/POTHOLE-MWPD-TRAINING-CONTRACT.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
