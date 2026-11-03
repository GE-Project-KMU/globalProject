"""Predeclared high-resolution paint experiment; development rows only during learning."""
import argparse,copy,json,math,random,sys,time
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import workspace_path,atomic_json,sha256
from agent.dataset import validate_research_release
from agent.governance import begin_training,register_asset,link_model
from agent.train_lane import prepare_pair
from agent.lane_network import make_lane_network,freeze_batch_norm_statistics
from agent.lane_metrics import PAINT_THRESHOLDS,threshold_confusions

MEAN=np.asarray([.485,.456,.406],np.float32)
STD=np.asarray([.229,.224,.225],np.float32)


def training_patch(row,size=640,view_size=1024):
    image=cv2.imread(str(workspace_path(row["path"],exists=True)))
    labels=cv2.imread(str(workspace_path(row["mask_path"],exists=True)),cv2.IMREAD_GRAYSCALE)
    h,w=labels.shape
    scale=view_size/max(h,w)*random.uniform(.75,1.5)
    rw,rh=max(1,round(w*scale)),max(1,round(h*scale))
    image=cv2.resize(image,(rw,rh));labels=cv2.resize(labels,(rw,rh),interpolation=cv2.INTER_NEAREST)
    painted=np.argwhere(labels==1)
    if len(painted) and random.random()<.75:
        cy,cx=painted[random.randrange(len(painted))]
        left=int(np.clip(cx-random.uniform(.25,.75)*size,0,max(0,rw-size)))
        top=int(np.clip(cy-random.uniform(.25,.75)*size,0,max(0,rh-size)))
    else:
        left=random.randint(0,max(0,rw-size));top=random.randint(0,max(0,rh-size))
    canvas=np.full((size,size,3),114,np.uint8);mask=np.full((size,size),255,np.uint8)
    cut=image[top:top+size,left:left+size];target=labels[top:top+size,left:left+size]
    canvas[:cut.shape[0],:cut.shape[1]]=cut;mask[:target.shape[0],:target.shape[1]]=target
    if random.random()<.5:canvas=canvas[:,::-1].copy();mask=mask[:,::-1].copy()
    raw=cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB).astype(np.float32)/255
    raw=np.clip((raw-.5)*random.uniform(.8,1.2)+.5,0,1)
    raw=np.clip(raw*random.uniform(.85,1.15),0,1)
    return ((raw-MEAN)/STD).transpose(2,0,1).copy(),mask.astype(np.int64)


def validation_scores(model,rows,size,torch):
    totals={t:{"tp":0,"fp":0,"fn":0,"tn":0,"ignored":0} for t in PAINT_THRESHOLDS}
    model.eval()
    with torch.no_grad():
        for row in rows:
            pixels,_=prepare_pair(row["path"],row["mask_path"],size=size)
            # FP32 validation matches the exported model's numeric contract.
            logits=model(torch.from_numpy(pixels[None]).cuda())["out"][0].float().cpu().numpy()
            truth=cv2.imread(str(workspace_path(row["mask_path"])),cv2.IMREAD_GRAYSCALE)
            h,w=truth.shape;scale=min(size/w,size/h)
            delta=(logits[1]-logits[0])[:max(1,round(h*scale)),:max(1,round(w*scale))]
            for t,c in threshold_confusions(truth,delta).items():
                for key,value in c.items():totals[t][key]+=value
    scores=[{"paint_threshold":t,"paint_iou":c["tp"]/(c["tp"]+c["fp"]+c["fn"]) if c["tp"]+c["fp"]+c["fn"] else 0.,"confusion":c}
            for t,c in totals.items()]
    return max(scores,key=lambda s:(s["paint_iou"],-abs(s["paint_threshold"]-.5))),scores


def run(args):
    import torch,torchvision
    torch.set_num_threads(3)
    cv2.setNumThreads(1)
    if not torch.cuda.is_available():raise ValueError("cuda_training_required")
    torch.backends.cudnn.benchmark=True
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    dev=workspace_path(args.development,exists=True);release=workspace_path(args.release,exists=True);output=workspace_path(args.output)
    rows=json.loads(dev.read_text(encoding="utf-8"))
    if any(r["split"] not in ("train","validation") for r in rows):raise ValueError("test_in_development_release")
    state=validate_research_release(rows)
    if not state["valid"]:raise ValueError("invalid_development_release")
    train=[r for r in rows if r["split"]=="train"];valid=[r for r in rows if r["split"]=="validation"]
    if len(train)<args.batch_size or not valid:raise ValueError("nonempty_development_splits_required")
    for r in rows:
        if sha256(workspace_path(r["mask_path"],exists=True))!=r["mask_sha256"]:raise ValueError("mask_changed")
    provenance_path,provenance=begin_training(output,release,rows,"lane")
    provenance.update(research_only=True,development_release_path=dev.relative_to(ROOT).as_posix(),development_release_sha256=sha256(dev))
    provenance["assets"]=[register_asset(provenance,r,[]) for r in rows];atomic_json(provenance_path,provenance)
    rates=[6,12,18]
    model=make_lane_network(pretrained=True,aspp_dilations=rates).cuda()
    ema=copy.deepcopy(model).eval()
    for p in ema.parameters():p.requires_grad_(False)
    optimizer=torch.optim.AdamW([{"params":model.backbone.parameters(),"lr":.0001,"base_lr":.0001},
                                {"params":model.classifier.parameters(),"lr":.001,"base_lr":.001}],weight_decay=.0001)
    scaler=torch.amp.GradScaler("cuda")
    class Patches(torch.utils.data.Dataset):
        def __len__(self):return len(train)
        def __getitem__(self,index):
            pixels,labels=training_patch(train[index],args.patch_size,args.input_size)
            return torch.from_numpy(pixels),torch.from_numpy(labels)
    loader=torch.utils.data.DataLoader(Patches(),batch_size=args.batch_size,shuffle=True,drop_last=True,num_workers=0)
    criterion=torch.nn.CrossEntropyLoss(weight=torch.tensor([1.,4.],device="cuda"),ignore_index=255)
    config={"source_code_sha256":sha256(Path(__file__)),"factory_code_sha256":sha256(workspace_path("agent/lane_network.py")),"development_sha256":sha256(dev),"full_release_sha256":sha256(release),
        "seed":args.seed,"max_epochs":args.epochs,"min_epochs":args.min_epochs,"patience":args.patience,"min_delta":.001,
        "batch_size":args.batch_size,"patch_size":args.patch_size,"input_size":args.input_size,"aspp_dilations":rates,
        "freeze_bn_statistics":True,"backbone_lr":.0001,"decoder_lr":.001,"ema_decay":.99,"dice_weight":.5,
        "checkpoint_metric":"maximum validation original-resolution paint IoU over predeclared threshold grid",
        "threshold_grid":list(PAINT_THRESHOLDS),"test_images_read":0,"source_licenses":sorted({r["allowed_use"] for r in rows}),
        "torch":torch.__version__,"torchvision":torchvision.__version__,"device":torch.cuda.get_device_name(),"research_only":True}
    atomic_json(output/"pretraining-config.json",config)
    best=-1.;patience_best=-1.;stale=0;history=[];start=time.perf_counter()
    for epoch in range(args.epochs):
        factor=min(1.,(epoch+1)/3)*(.05+.95*(1+math.cos(math.pi*max(0,epoch-3)/max(1,args.epochs-4)))/2)
        for group in optimizer.param_groups:group["lr"]=group["base_lr"]*factor
        model.train();freeze_batch_norm_statistics(model);losses=[]
        for step,(pixels,labels) in enumerate(loader):
            pixels,labels=pixels.cuda(),labels.cuda();optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.float16):
                logits=model(pixels)["out"];ce=criterion(logits,labels)
                usable=labels!=255;prob=torch.softmax(logits.float(),1)[:,1]*usable;target=(labels==1).float()
                dice=1-(2*(prob*target).sum()+1)/(prob.sum()+target.sum()+1);loss=.5*ce+.5*dice
            if not torch.isfinite(loss):raise ValueError("non_finite_loss")
            scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            scaler.step(optimizer);scaler.update();losses.append(float(loss.detach()))
            with torch.no_grad():
                source=model.state_dict()
                for key,value in ema.state_dict().items():
                    if value.is_floating_point():value.mul_(.99).add_(source[key],alpha=.01)
                    else:value.copy_(source[key])
            if step%50==0:print(json.dumps({"epoch":epoch+1,"step":step,"loss":losses[-1]}),flush=True)
        selected,scores=validation_scores(ema,valid,args.input_size,torch)
        entry={"epoch":epoch+1,"loss":float(np.mean(losses)),"validation_paint_iou":selected["paint_iou"],"validation_threshold":selected["paint_threshold"]}
        history.append(entry);print(json.dumps(entry),flush=True)
        if selected["paint_iou"]>best:
            best=selected["paint_iou"];torch.save(ema.state_dict(),output/"lane_best.pt")
            atomic_json(output/"best-validation.json",{"selected_on":"validation_only","epoch":epoch+1,"selected":selected,"scores":scores,"test_images_used":0})
        if selected["paint_iou"]>patience_best+.001:patience_best=selected["paint_iou"];stale=0
        else:stale+=1
        atomic_json(output/"training.json",{"history":history,"elapsed_seconds":time.perf_counter()-start,"research_only":True,
            "production_approved":False,"test_split_used":False,"code_sha256":config["source_code_sha256"],"manifest_sha256":sha256(release)})
        if epoch+1>=args.min_epochs and stale>=args.patience:break
    model.cpu();model.load_state_dict(torch.load(output/"lane_best.pt",map_location="cpu",weights_only=True));model.eval()
    class Output(torch.nn.Module):
        def __init__(self,network):super().__init__();self.network=network
        def forward(self,pixels):return self.network(pixels)["out"]
    weights=output/"lane.onnx"
    torch.onnx.export(Output(model),torch.zeros(1,3,args.input_size,args.input_size),str(weights),opset_version=17,
                      input_names=["images"],output_names=["logits"],dynamo=False)
    artifact={"format":"onnx","task":"lane","architecture":"deeplabv3_mobilenet_v3_large","aspp_dilations":rates,
        "classes":["background","paint"],"weights":weights.relative_to(ROOT).as_posix(),"sha256":sha256(weights),"target_trained":True,
        "dataset_release":"manifest-sha256:"+sha256(release),"source_release_path":release.relative_to(ROOT).as_posix(),
        "training_development_release_path":dev.relative_to(ROOT).as_posix(),"training_development_release_sha256":sha256(dev),
        "training_provenance":provenance_path.relative_to(ROOT).as_posix(),"input_size":[args.input_size,args.input_size],
        "mean":MEAN.tolist(),"std":STD.tolist(),"production_approved":False,"research_only":True,"human_privacy_reviewed":False,"project_acceptance_eligible":False,
        "source_licenses":config["source_licenses"],"independent_route_verified":False,"data_revoked":False,"needs_revalidation":False,
        "test_split_used":False,"validation_paint_iou":best}
    atomic_json(output/"lane.json",artifact);link_model(provenance_path,output/"lane.json")
    print(json.dumps({"status":"trained_research_lane","epochs":len(history),"best_validation_iou":best}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("development");p.add_argument("release");p.add_argument("output")
    p.add_argument("--epochs",type=int,default=60);p.add_argument("--min-epochs",type=int,default=15);p.add_argument("--patience",type=int,default=12)
    p.add_argument("--batch-size",type=int,default=4);p.add_argument("--patch-size",type=int,default=640);p.add_argument("--input-size",type=int,default=1024);p.add_argument("--seed",type=int,default=20261104)
    a=p.parse_args()
    if not 1<=a.min_epochs<=a.epochs or a.batch_size<2 or a.patience<1 or a.patch_size%32 or a.input_size%32:raise ValueError("invalid_training_recipe")
    run(a)
