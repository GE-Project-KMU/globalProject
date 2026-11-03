"""Approved detection release -> COCO -> official workspace YOLOX train/export.

Does not download data, import legacy training dependencies in the app runtime,
or treat exported weights as accepted. Use a separate Python 3.10 environment.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from PIL import Image
from .paths import ROOT, workspace_path, atomic_json, sha256
from .dataset import validate_release,validate_research_release
from .governance import begin_training,register_asset,link_model


def export_coco(release_path, output, task,*,research=False):
    if task not in ("pothole","redaction"): raise ValueError("unsupported_detection_task")
    release_path=workspace_path(release_path,exists=True)
    if release_path.with_suffix(".quarantine.json").exists():raise ValueError("source_release_quarantined")
    rows=json.loads(release_path.read_text(encoding="utf-8"))
    state=(validate_research_release if research else validate_release)(rows)
    if not state["valid"]: raise ValueError("release_invalid:"+json.dumps(state["errors"]))
    output=workspace_path(output); output.mkdir(parents=True,exist_ok=True)
    provenance_path,provenance=begin_training(output,release_path,[r for r in rows if r["split"] in ("train","validation")],task)
    provenance["research_only"]=research
    atomic_json(provenance_path,provenance)
    classes=["pothole"] if task=="pothole" else ["face","license_plate"]
    for split,folder in (("train","train2017"),("validation","val2017")):
        images,annotations=[],[]
        destination=output/folder;destination.mkdir(exist_ok=True)
        for row in rows:
            if row["split"]!=split: continue
            # Explicit exhaustive review required: missing labels are not negatives.
            if row.get("detection_labels_verified") is not True: raise ValueError("verified_detection_labels_required")
            objects=row.get("objects")
            if not isinstance(objects,list): raise ValueError("explicit_objects_list_required")
            if any(o.get("certainty")!="confirmed" or o.get("ignore") is True for o in objects):
                raise ValueError("uncertain_objects_require_safe_crop_or_hold")
            path=workspace_path(row["path"],exists=True)
            with Image.open(path) as image: width,height=image.size
            image_id=len(images)+1;name=str(row["sample_id"])+path.suffix.lower()
            try:shutil.copyfile(path,destination/name)
            except OSError:
                (destination/name).unlink(missing_ok=True)
                raise
            label_path=output/"annotations"/f"instances_{folder}.json"
            provenance["assets"].append(register_asset(provenance,row,[destination/name],label_refs=[{"path":label_path.relative_to(ROOT).as_posix(),"image_id":image_id}]))
            atomic_json(provenance_path,provenance)
            images.append({"id":image_id,"file_name":name,"width":width,"height":height})
            for item in objects:
                if item["class"] not in classes: continue
                x,y,w,h=map(float,item["bbox_xywh_px"])
                if min(x,y)<0 or min(w,h)<=0 or x+w>width or y+h>height: raise ValueError("bbox_out_of_image")
                annotations.append({"id":len(annotations)+1,"image_id":image_id,"category_id":classes.index(item["class"])+1,
                                    "bbox":[x,y,w,h],"area":w*h,"iscrowd":0})
        if not images: raise ValueError("train_and_validation_images_required")
        atomic_json(output/"annotations"/f"instances_{folder}.json",{"images":images,"annotations":annotations,
                    "categories":[{"id":i+1,"name":c} for i,c in enumerate(classes)]})
        label_path=output/"annotations"/f"instances_{folder}.json"
        for asset in provenance["assets"]:
            for ref in asset["label_refs"]:
                if ref["path"]==label_path.relative_to(ROOT).as_posix():ref["sha256"]=sha256(label_path)
        atomic_json(provenance_path,provenance)
    return output


def train_export(release, output, task, repository, interpreter, *, epochs=100, batch_size=8, device="cpu",research=False):
    if epochs<1 or batch_size<2 or device not in ("cpu","gpu"): raise ValueError("invalid_training_options")
    repository,interpreter=workspace_path(repository,exists=True),workspace_path(interpreter,exists=True)
    if not (repository/"tools/train.py").is_file() or not (repository/"tools/export_onnx.py").is_file() or not (repository/"exps/default/yolox_nano.py").is_file():
        raise ValueError("official_yolox_repository_required")
    output=export_coco(release,output,task,research=research)
    count=1 if task=="pothole" else 2
    experiment=output/"experiment.py"
    # repr encodes filesystem paths safely into Python source; no shell interpolation.
    experiment.write_text("from runpy import run_path\n"+f"BaseExp=run_path({str(repository/'exps/default/yolox_nano.py')!r})['Exp']\nclass Exp(BaseExp):\n"
        " def __init__(self):\n  super().__init__()\n"
        f"  self.num_classes={count}\n  self.depth=.33\n  self.width=.25\n  self.depthwise=True\n"
        f"  self.input_size=(416,416)\n  self.test_size=(416,416)\n  self.random_size=(10,20)\n"
        f"  self.data_dir={str(output)!r}\n  self.output_dir={str(output/'checkpoints')!r}\n"
        f"  self.exp_name='target_nano'\n  self.max_epoch={epochs}\n  self.seed=2026\n  self.data_num_workers=0\n  self.enable_mixup=False\n",encoding="utf-8")
    environment=os.environ.copy()
    environment.update(PYTHONPATH=str(repository),TORCH_HOME=str(ROOT/".cache/torch"),TEMP=str(ROOT/".cache/tmp"),TMP=str(ROOT/".cache/tmp"))
    environment.update(CUDA_CACHE_PATH=str(ROOT/".cache/cuda"),XDG_CACHE_HOME=str(ROOT/".cache"),MPLCONFIGDIR=str(ROOT/".cache/matplotlib"),WANDB_MODE="disabled")
    # Official YOLOX trainer expects CUDA; CPU export is supported but CPU training
    # must not silently invoke a CUDA trainer with fabricated success.
    if device=="cpu": raise RuntimeError("coco_exported_yolox_training_requires_cuda_training_environment")
    log=output/"training.log"
    with log.open("w",encoding="utf-8") as stream:
        subprocess.run([str(interpreter),str(repository/"tools/train.py"),"-f",str(experiment),"-d","1","-b",str(batch_size),"--fp16"],
                       cwd=repository,env=environment,stdout=stream,stderr=subprocess.STDOUT,check=True)
        weights=output/"checkpoints/target_nano/best_ckpt.pth"
        onnx=output/"target_nano.onnx"
        subprocess.run([str(interpreter),str(repository/"tools/export_onnx.py"),"-f",str(experiment),"-c",str(weights),
                        "--output-name",str(onnx),"--decode_in_inference"],cwd=repository,env=environment,
                       stdout=stream,stderr=subprocess.STDOUT,check=True)
    artifact={"format":"onnx","task":task,"classes":["pothole"] if task=="pothole" else ["face","license_plate"],
              "weights":onnx.relative_to(ROOT).as_posix(),"sha256":sha256(onnx),"target_trained":True,
              "dataset_release":"manifest-sha256:"+sha256(workspace_path(release,exists=True)),
              "input_size":[416,416],"output_layout":"yolox_decoded_xywh","architecture":"YOLOX-Nano",
              "production_approved":False,"research_only":research,"test_split_used":False,"data_revoked":False,"needs_revalidation":False,
              "source_release_path":workspace_path(release,exists=True).relative_to(ROOT).as_posix(),
              "training_provenance":(output/"provenance.json").relative_to(ROOT).as_posix()}
    atomic_json(output/"model.json",artifact)
    link_model(output/"provenance.json",output/"model.json")
    return artifact


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release");parser.add_argument("output");parser.add_argument("--task",choices=["pothole","redaction"],required=True)
    parser.add_argument("--repository");parser.add_argument("--interpreter");parser.add_argument("--epochs",type=int,default=100)
    parser.add_argument("--batch-size",type=int,default=8);parser.add_argument("--device",choices=["cpu","gpu"],default="cpu")
    parser.add_argument("--export-data-only",action="store_true")
    parser.add_argument("--research",action="store_true",help="Offline publisher research labels; never deployment approval")
    args=parser.parse_args()
    try:
        if args.export_data_only: result={"coco_directory":str(export_coco(args.release,args.output,args.task,research=args.research)),"test_split_exported":False}
        else:
            if not args.repository or not args.interpreter: raise ValueError("workspace_training_repository_and_interpreter_required")
            result=train_export(args.release,args.output,args.task,args.repository,args.interpreter,epochs=args.epochs,batch_size=args.batch_size,device=args.device,research=args.research)
        print(json.dumps(result,ensure_ascii=False,indent=2));return 0
    except (ValueError,RuntimeError,OSError,KeyError,subprocess.CalledProcessError) as exc:
        print(json.dumps({"status":"training_blocked","reason":str(exc)},ensure_ascii=False));return 2

if __name__=="__main__":sys.exit(main())
