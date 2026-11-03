"""Publisher-source research data acquisition; never fabricates human approval.

All files/caches remain inside this workspace. RDD country archives are read
from the authors' Figshare ZIP via authenticated HTTPS ranges and ZIP CRC checks.
"""
import argparse
from collections import Counter,OrderedDict
from datetime import datetime,timezone,timedelta
import hashlib
import io
import json
from pathlib import Path
import random
import shutil
import sys
import time
import uuid
import xml.etree.ElementTree as ET
import zipfile
import tarfile
from concurrent.futures import ThreadPoolExecutor
import re
import html

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import httpx
import cv2
import numpy as np
from agent.paths import workspace_path,atomic_json,sha256
from agent.dataset import validate_research_release

RDD_ARTICLE="https://api.figshare.com/v2/articles/21431547"
RDD_URL="https://ndownloader.figshare.com/files/38030910"
RDD_SIZE=13_264_172_619
NAMESPACE=uuid.UUID("3c444c1d-42d3-4353-a708-71793ac7edd0")


class RemoteRange(io.RawIOBase):
    def __init__(self,url,size):
        self.client=httpx.Client(follow_redirects=True,timeout=90)
        self.url,self.size,self.position=url,size,0
        self.cache=OrderedDict();self.block=8*1024*1024;self.bytes_downloaded=0
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        self.position=offset if whence==0 else self.position+offset if whence==1 else self.size+offset
        if self.position<0:raise ValueError("negative_seek")
        return self.position
    def read(self,size=-1):
        end=self.size if size<0 else min(self.size,self.position+size)
        parts=[]
        while self.position<end:
            number=self.position//self.block;start=number*self.block
            if number not in self.cache:
                for attempt in range(4):
                    try:
                        response=self.client.get(self.url,headers={"Range":f"bytes={start}-{min(self.size-1,start+self.block-1)}"})
                        response.raise_for_status()
                        if response.status_code!=206 or len(response.content)!=min(self.block,self.size-start):raise ValueError("range_response_mismatch")
                        self.cache[number]=response.content;self.bytes_downloaded+=len(response.content)
                        break
                    except (httpx.HTTPError,ValueError):
                        if attempt==3:raise
                        time.sleep(1+attempt)
                while len(self.cache)>3:self.cache.popitem(last=False)
            data=self.cache[number];local=self.position-start;count=min(end-self.position,len(data)-local)
            parts.append(data[local:local+count]);self.position+=count
        return b"".join(parts)


def fetch_rdd(countries):
    folder=workspace_path("data/research/rdd2022");folder.mkdir(parents=True,exist_ok=True)
    response=httpx.get(RDD_ARTICLE,timeout=30);response.raise_for_status();metadata=response.json()
    if metadata["license"]["name"]!="CC BY 4.0":raise ValueError("publisher_license_changed_review_required")
    atomic_json(folder/"publisher-metadata.json",metadata)
    evidence={"research_allowed":True,"license_id":"CC-BY-4.0","source_url":metadata["url_public_html"],
        "license_url":metadata["license"]["url"],"publisher":"Arya, Maeda, Ghosh, Toshniwal, Sekimoto",
        "code_repository":"https://github.com/sekilab/RoadDamageDetector","human_privacy_reviewed":False,
        "decision":"User-authorized offline research under publisher license; no deployment or redistribution approval."}
    atomic_json(folder/"license-evidence.json",evidence)
    remote=RemoteRange(RDD_URL+"?cache="+str(time.time()),RDD_SIZE)
    try:
        with zipfile.ZipFile(remote) as archive:
            for country in countries:
                member=next(x for x in archive.infolist() if x.filename==f"RDD2022/{country}.zip")
                target=folder/(country+".zip")
                record=target.with_suffix(".source.json")
                if target.exists() and record.exists() and sha256(target)==json.loads(record.read_text())["sha256"]:
                    print("cached",country,flush=True);continue
                temporary=target.with_suffix(".download")
                print("downloading",country,member.file_size,flush=True)
                with archive.open(member) as source,temporary.open("wb") as destination:
                    shutil.copyfileobj(source,destination,length=1024*1024)
                temporary.replace(target)
                atomic_json(record,{"source_url":RDD_URL,"outer_size":RDD_SIZE,"outer_member":member.filename,
                    "zip_crc32":member.CRC,"sha256":sha256(target),"downloaded_at":datetime.now(timezone.utc).isoformat(),
                    "expires_at":(datetime.now(timezone.utc)+timedelta(days=7)).isoformat(),"full_outer_md5_verified":False,
                    "integrity":"selected complete ZIP member verified by zipfile CRC; outer publisher MD5 not claimed for ranged subset"})
                print("downloaded",country,"bytes_total",remote.bytes_downloaded,flush=True)
    finally:remote.client.close()


def d_hash(image):
    gray=cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),(9,8))
    return int.from_bytes(np.packbits(gray[:,1:]>gray[:,:-1]).tobytes(),"big")


def fetch_file(client,url,path,limit):
    path=workspace_path(path);path.parent.mkdir(parents=True,exist_ok=True)
    record=path.with_suffix(path.suffix+".source.json")
    if path.exists() and record.exists() and sha256(path)==json.loads(record.read_text())["sha256"]:return
    temporary=path.with_suffix(path.suffix+".download")
    with client.stream("GET",url) as response:
        response.raise_for_status();count=0
        with temporary.open("wb") as output:
            for block in response.iter_bytes(1024*1024):
                count+=len(block)
                if count>limit:raise ValueError("download_exceeds_limit")
                output.write(block)
    temporary.replace(path)
    atomic_json(record,{"source_url":url,"sha256":sha256(path),"bytes":count,
                       "downloaded_at":datetime.now(timezone.utc).isoformat(),"expires_at":(datetime.now(timezone.utc)+timedelta(days=7)).isoformat()})
    print("downloaded",path.name,count,flush=True)


def fetch_redaction():
    root=workspace_path("data/research/redaction");root.mkdir(parents=True,exist_ok=True)
    with httpx.Client(follow_redirects=True,timeout=90) as client:
        metadata=client.get("https://huggingface.co/api/datasets/CUHK-CSE/wider_face").json();revision=metadata["sha"]
        card=client.get(f"https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/{revision}/README.md");card.raise_for_status()
        (root/"WIDER-MODEL-CARD.md").write_text(card.text,encoding="utf-8")
        atomic_json(root/"wider-license.json",{"research_allowed":True,"license_id":"CC-BY-NC-ND-4.0",
            "source_url":"https://huggingface.co/datasets/CUHK-CSE/wider_face","source_revision":revision,
            "license_url":"https://creativecommons.org/licenses/by-nc-nd/4.0/","primary_homepage":"https://shuoyang1213.me/WIDERFACE/index.html",
            "card_sha256":sha256(root/"WIDER-MODEL-CARD.md"),"restriction":"Private non-commercial research only; no dataset adaptations or model release claimed permitted."})
        for name in ("wider_face_split.zip","WIDER_train.zip","WIDER_val.zip"):
            print("fetching",name,flush=True)
            fetch_file(client,f"https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/{revision}/data/{name}",root/name,2_000_000_000)
        readme=client.get("https://raw.githubusercontent.com/detectRecog/CCPD/master/README.md");readme.raise_for_status()
        license_text=client.get("https://raw.githubusercontent.com/detectRecog/CCPD/master/LICENSE");license_text.raise_for_status()
        if "This dataset is open-source under MIT license" not in readme.text:raise ValueError("ccpd_dataset_license_changed")
        (root/"CCPD-README.md").write_text(readme.text,encoding="utf-8");(root/"CCPD-LICENSE.txt").write_text(license_text.text,encoding="utf-8")
        atomic_json(root/"ccpd-license.json",{"research_allowed":True,"license_id":"MIT","source_url":"https://github.com/detectRecog/CCPD",
            "license_url":"https://github.com/detectRecog/CCPD/blob/master/LICENSE","readme_sha256":sha256(root/"CCPD-README.md"),
            "decision":"Publisher explicitly applies MIT to dataset; local research only, no human privacy/vehicle-owner consent claim."})
        warning=client.get("https://drive.google.com/uc?export=download&id=1m8w1kFxnCEiqz_-t2vTcgrgqNIv986PR");warning.raise_for_status()
        action=re.search(r'<form[^>]*action="([^"]+)"',warning.text)
        if not action:raise ValueError("public_drive_download_form_not_found_no_auth_bypass")
        values=dict(re.findall(r'<input[^>]*type="hidden"[^>]*name="([^"]+)"[^>]*value="([^"]+)"',warning.text))
        url=str(httpx.URL(html.unescape(action.group(1)),params=values))
        fetch_file(client,url,root/"CCPD2020.zip",2_000_000_000)


def fetch_pretrained():
    path=workspace_path("data/research/yolox_nano_coco.pth")
    url="https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth"
    expected="cd28f55fbbc1829f99d9ac9b38a16d259a22889739c8728ea877610201feff7b"
    with httpx.Client(follow_redirects=True,timeout=60) as client:
        if not path.exists():fetch_file(client,url,path,20_000_000)
        if sha256(path)!=expected:raise ValueError("official_transfer_checkpoint_digest_mismatch")
        license_response=client.get("https://raw.githubusercontent.com/Megvii-BaseDetection/YOLOX/6ddff4824372906469a7fae2dc3206c7aa4bbaee/LICENSE");license_response.raise_for_status()
        license_path=path.parent/"YOLOX-LICENSE.txt";license_path.write_text(license_response.text,encoding="utf-8")
    atomic_json(path.with_suffix(".source.json"),{"source_url":url,"sha256":expected,"publisher":"Megvii-BaseDetection/YOLOX",
        "architecture":"YOLOX-Nano COCO 80class","purpose":"transfer initialization only; not pothole or face/plate target checkpoint",
        "repository_license":"Apache-2.0","license_sha256":sha256(license_path),"production_approved":False})


def fetch_yolox():
    revision="6ddff4824372906469a7fae2dc3206c7aa4bbaee"
    url=f"https://codeload.github.com/Megvii-BaseDetection/YOLOX/zip/{revision}"
    archive=workspace_path(".cache/vendor/yolox-source.zip");target=workspace_path(".cache/vendor/YOLOX")
    expected="647bba7d16cc7e13feb5f2beca0296f86186c944f5a8fa9ccefc2629758bccba"
    with httpx.Client(follow_redirects=True,timeout=60) as client:
        if not archive.exists():fetch_file(client,url,archive,20_000_000)
    if sha256(archive)!=expected:raise ValueError("pinned_yolox_source_archive_hash_mismatch")
    with zipfile.ZipFile(archive) as source:
        prefix=f"YOLOX-{revision}/"
        for member in source.infolist():
            if not member.filename.startswith(prefix):raise ValueError("unexpected_source_archive_prefix")
            relative=member.filename[len(prefix):]
            if not relative:continue
            destination=workspace_path(target/relative)
            if not destination.is_relative_to(target):raise ValueError("source_archive_path_traversal")
            if member.is_dir():destination.mkdir(parents=True,exist_ok=True);continue
            raw=source.read(member)
            if destination.exists():
                if sha256(destination)!=hashlib.sha256(raw).hexdigest():raise ValueError("existing_yolox_source_modified_no_overwrite")
            else:
                destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(raw)
    atomic_json(target/".source.json",{"source_url":url,"commit":revision,"archive_sha256":expected,"license":"Apache-2.0",
        "nano_exp_sha256":sha256(target/"exps/default/yolox_nano.py"),"purpose":"Official native Nano architecture and loss; no source code patch."})
    print("pinned_official_yolox_source_ready",revision,flush=True)


def prepare_rdd_mirror(limit=1600):
    """Transparent CC-BY-SA redistribution fallback; original publisher evidence retained."""
    raise RuntimeError("quarantined_semantic_mapping: mirror class3 contains markings/cracks; use literal publisher VOC D40 instead")
    repository="dronefreak/RDD2022";revision="72b01f94c2e5bfe430861afd73995f2d412b8855"
    root=workspace_path("data/research/rdd2022");root.mkdir(parents=True,exist_ok=True)
    output=workspace_path("data/training/research-pothole/source");output.mkdir(parents=True,exist_ok=True)
    client=httpx.Client(follow_redirects=True,timeout=45,limits=httpx.Limits(max_connections=16))
    metadata=client.get(RDD_ARTICLE);metadata.raise_for_status();atomic_json(root/"publisher-metadata.json",metadata.json())
    card=client.get(f"https://huggingface.co/datasets/{repository}/resolve/{revision}/README.md");card.raise_for_status()
    (root/"mirror-card.md").write_text(card.text,encoding="utf-8")
    evidence={"research_allowed":True,"license_id":"CC-BY-4.0 original + CC-BY-SA-4.0 mirror export",
        "source_url":"https://github.com/sekilab/RoadDamageDetector","license_url":"https://creativecommons.org/licenses/by/4.0/",
        "primary_figshare":RDD_ARTICLE,"mirror":"https://huggingface.co/datasets/"+repository,"mirror_revision":revision,
        "mirror_license_url":"https://creativecommons.org/licenses/by-sa/4.0/","mirror_card_sha256":sha256(root/"mirror-card.md"),
        "integrity_limit":"Original country download returns403/timeouts; mirror pinned source and individual download hashes, not publisher archive MD5 verification."}
    atomic_json(root/"license-evidence.json",evidence);evidence_path=root/"license-evidence.json"
    meta=client.get(f"https://huggingface.co/api/datasets/{repository}/revision/{revision}").json()
    paths=[r["rfilename"] for r in meta["siblings"] if r["rfilename"].endswith("metadata.jsonl")]
    def get_metadata(path):
        response=client.get(f"https://huggingface.co/datasets/{repository}/resolve/{revision}/{path}");response.raise_for_status()
        return [(path.rsplit("/",1)[0]+"/"+row["file_name"],row) for line in response.text.splitlines() if (row:=json.loads(line))]
    records=[r for group in ThreadPoolExecutor(12).map(get_metadata,paths) for r in group]
    by_country={}
    for path,record in records:
        country=re.sub(r"_\d+\.jpg$","",record["file_name"])
        by_country.setdefault(country,[]).append((path,record))
    stats={country:{"positive":sum(3 in row["objects"]["categories"] for _,row in group),"images":len(group)} for country,group in by_country.items()}
    print("mirror_country_counts",json.dumps(stats),flush=True)
    # Hold out entire countries; selection does not use model outputs.
    splits={"India":"train","Japan":"train","United_States":"validation","China_MotorBike":"test"}
    if any(stats.get(country,{}).get("positive",0)<30 for country in splits):
        positive_countries=sorted([c for c,s in stats.items() if s["positive"]>=30],key=lambda c:stats[c]["positive"],reverse=True)
        if len(positive_countries)<3:raise ValueError("insufficient_country_independent_pothole_groups")
        splits={c:"train" for c in positive_countries[:-2]};splits[positive_countries[-2]]="validation";splits[positive_countries[-1]]="test"
    chosen=[]
    for country,split in splits.items():
        group=by_country[country];positive=[r for r in group if 3 in r[1]["objects"]["categories"]];negative=[r for r in group if 3 not in r[1]["objects"]["categories"]]
        random.Random(2026).shuffle(positive);random.Random(2027).shuffle(negative)
        count=limit if split=="train" else min(400,limit//3)
        chosen.extend((country,split,path,row) for path,row in positive[:count]+negative[:min(count//2,len(positive)//2)])
    now=datetime.now(timezone.utc);failures=[]
    def download(item):
        country,split,path,record=item
        sample_id=str(uuid.uuid5(NAMESPACE,repository+":"+path));target=output/(sample_id+".jpg")
        for attempt in range(3):
            try:
                if target.exists():raw=target.read_bytes()
                else:
                    response=client.get(f"https://huggingface.co/datasets/{repository}/resolve/{revision}/{path}");response.raise_for_status();raw=response.content
                image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
                if image is None:raise ValueError("image_decode_failed")
                target.write_bytes(raw);digest=hashlib.sha256(raw).hexdigest();h,w=image.shape[:2]
                objects=[]
                for box,label in zip(record["objects"]["bbox"],record["objects"]["categories"]):
                    if label!=3:continue
                    x,y,bw,bh=map(float,box)
                    if min(x,y)<0 or min(bw,bh)<=0 or x+bw>w+1 or y+bh>h+1:raise ValueError("publisher_bbox_invalid")
                    objects.append({"class":"pothole","bbox_xywh_px":[x,y,min(bw,w-x),min(bh,h-y)],"certainty":"confirmed"})
                return {"sample_id":sample_id,"source_id":str(uuid.uuid5(NAMESPACE,"source:"+digest)),"source_hash":digest,"leakage_group_id":"country:"+country,
                    "annotation_version":"RDD2022-D40-pinned-mirror-bbox-v1","path":target.relative_to(ROOT).as_posix(),"sha256":digest,"split":split,"country":country,
                    "privacy_status":"restricted_research","qc_status":"publisher_annotations_machine_checked","release_scope":"offline_research","human_privacy_reviewed":False,
                    "not_for_deployment":True,"annotation_review":"publisher_original_machine_dimension_and_bbox_checks; mirror conversion provenance",
                    "allowed_use":"Private offline research; original CC-BY4.0 and mirrorCC-BY-SA4.0 attribution retained; no human privacy approval.",
                    "expires_at":(now+timedelta(days=7)).isoformat(),"source_url":f"https://huggingface.co/datasets/{repository}/blob/{revision}/{path}",
                    "license_evidence_path":evidence_path.relative_to(ROOT).as_posix(),"license_evidence_sha256":sha256(evidence_path),"detection_labels_verified":True,
                    "objects":objects,"perceptual_dhash":f"{d_hash(image):016x}"}
            except (httpx.HTTPError,ValueError):
                if attempt==2:failures.append(path);return None
                time.sleep(attempt+1)
    rows=[];seen=set();visual=[];duplicates=0
    for i,row in enumerate(ThreadPoolExecutor(12).map(download,chosen)):
        if row:
            phash=int(row["perceptual_dhash"],16)
            if row["sha256"] in seen or any(split!=row["split"] and (phash^old).bit_count()<=3 for old,split in visual):duplicates+=1
            else:rows.append(row);seen.add(row["sha256"]);visual.append((phash,row["split"]))
        if i%100==0:print("prepared",i,"of",len(chosen),flush=True)
    state=validate_research_release(rows)
    if not state["valid"]:raise ValueError(json.dumps(state["errors"][:10]))
    release=output.parent/"release.json";atomic_json(release,rows)
    atomic_json(output.parent/"split-evidence.json",{"country_splits":splits,"available_counts":stats,"counts":dict(Counter(r["split"] for r in rows)),
        "duplicate_or_cross_split_dhash_removed":duplicates,"download_failures":failures,"release_sha256":sha256(release),"human_reviewed":False,
        "physical_identity_available":False,"independence":"country-disjoint; exactSHA dedup and cross-split64bit dHash Hamming<=3 excluded; no exhaustive identity audit claim"})
    print("research_release",len(rows),state["valid"],flush=True)
    client.close()


def finalize_rdd_sources():
    """Freeze first-acquisition TTL and remove only this tool's unused UUID JPEGs."""
    folder=workspace_path("data/training/research-pothole")
    release=folder/"release.json";rows=json.loads(release.read_text(encoding="utf-8"))
    retained=set()
    for row in rows:
        path=workspace_path(row["path"],exists=True);stat=path.stat()
        acquisition=datetime.fromtimestamp(getattr(stat,"st_birthtime",stat.st_mtime),timezone.utc)
        expiry=min(datetime.fromisoformat(row["expires_at"]),acquisition+timedelta(days=7))
        row["source_expires_at"]=expiry.isoformat();row["expires_at"]=expiry.isoformat();retained.add(path)
    removed=0
    for path in (folder/"source").glob("*.jpg"):
        try:uuid.UUID(path.stem)
        except ValueError:continue
        if path not in retained:path.unlink();removed+=1
    state=validate_research_release(rows)
    if not state["valid"]:raise ValueError(json.dumps(state["errors"][:10]))
    atomic_json(release,rows)
    report=json.loads((folder/"split-evidence.json").read_text(encoding="utf-8"))
    report.update(release_sha256=sha256(release),unused_tool_generated_jpegs_removed=removed,ttl="earliest retained first file creation/acquisition +7days; never renewed by repeat prep")
    atomic_json(folder/"split-evidence.json",report)
    print(json.dumps({"rows":len(rows),"removed_unused":removed,"release_sha256":sha256(release)}),flush=True)


def fetch_rdd2020():
    root=workspace_path("data/research/rdd2020");root.mkdir(parents=True,exist_ok=True)
    with httpx.Client(follow_redirects=True,timeout=60,limits=httpx.Limits(max_connections=12)) as client:
        endpoint="https://data.mendeley.com/public-api/datasets/5ty2wb6gvg/files?folder_id=root&version=1"
        response=client.get(endpoint);response.raise_for_status();metadata=response.json();atomic_json(root/"publisher-files.json",metadata)
        source=next(row for row in metadata if row["filename"]=="train.tar.gz");details=source["content_details"]
        expected=details["sha256_hash"];size=source["size"];target=root/"train.tar.gz"
        if target.exists() and sha256(target)==expected:print("cached_verified_rdd2020",flush=True);return
        with client.stream("GET",details["download_url"],headers={"Range":"bytes=0-0"}) as probe:
            probe.raise_for_status();direct=str(probe.url)
        parts=root/"download-parts";parts.mkdir(exist_ok=True);block=8*1024*1024
        ranges=[(i,start,min(size-1,start+block-1)) for i,start in enumerate(range(0,size,block))]
        def get_part(item):
            i,start,end=item;path=parts/f"{i:04d}.part"
            if path.exists() and path.stat().st_size==end-start+1:return path
            for attempt in range(4):
                try:
                    response=client.get(direct,headers={"Range":f"bytes={start}-{end}"})
                    response.raise_for_status()
                    if response.status_code!=206 or response.headers.get("content-range")!=f"bytes {start}-{end}/{size}" or len(response.content)!=end-start+1:
                        raise ValueError("publisher_range_mismatch")
                    path.write_bytes(response.content);return path
                except (httpx.HTTPError,ValueError):
                    if attempt==3:raise
                    time.sleep(attempt+1)
        for i,_ in enumerate(ThreadPoolExecutor(8).map(get_part,ranges)):
            if i%16==0:print("publisher_range_parts",i+1,"/",len(ranges),flush=True)
        assembled=root/"train.tar.gz.assembled"
        with assembled.open("wb") as stream:
            for i,_,_ in ranges:
                with (parts/f"{i:04d}.part").open("rb") as source:shutil.copyfileobj(source,stream)
        actual=sha256(assembled)
        if actual!=expected:raise ValueError("original_publisher_sha256_mismatch")
        assembled.replace(target)
        atomic_json(root/"source.json",{"source_url":details["download_url"],"sha256":actual,"size":size,"publisher_hash_verified":True,
            "downloaded_at":datetime.now(timezone.utc).isoformat(),"expires_at":(datetime.now(timezone.utc)+timedelta(days=7)).isoformat(),
            "doi":"10.17632/5ty2wb6gvg.1","license_id":"CC-BY-NC-3.0"})
        atomic_json(root/"license-evidence.json",{"research_allowed":True,"license_id":"CC-BY-NC-3.0",
            "source_url":"https://data.mendeley.com/datasets/5ty2wb6gvg/1","license_url":"https://creativecommons.org/licenses/by-nc/3.0/",
            "publisher":"Deeksha Arya, Hiroya Maeda et al.","semantic_mapping":"Publisher explicitly D40=Potholes; literal original XML selected, no YOLO numeric remap."})
        for i,_,_ in ranges:(parts/f"{i:04d}.part").unlink()
        parts.rmdir();print("original_publisher_sha256_verified",actual,flush=True)


def prepare_rdd2020():
    root=workspace_path("data/research/rdd2020");archive=root/"train.tar.gz"
    expected="e06f74463b2d7410fa7e02f478818025be9b051087f80e0c51631abb32c0af55"
    if sha256(archive)!=expected:raise ValueError("publisher_archive_hash_not_verified")
    candidates={};counts={};seen_xml=set()
    # Streaming passes avoid repeated gzip seeks and never unpack arbitrary paths.
    with tarfile.open(archive,mode="r|gz") as source:
        for member in source:
            if not member.isfile() or not member.name.endswith(".xml"):continue
            xml=ET.fromstring(source.extractfile(member).read());country=member.name.split("/")[1]
            counts.setdefault(country,Counter()).update(obj.findtext("name") for obj in xml.findall("object"))
            filename=xml.findtext("filename") or Path(member.name).with_suffix(".jpg").name
            if not filename.lower().endswith(".jpg"):filename+=".jpg"
            image_name=f"train/{country}/images/{filename}"
            width,height=int(xml.findtext("size/width")),int(xml.findtext("size/height"));objects=[]
            for obj in xml.findall("object"):
                if obj.findtext("name")!="D40":continue
                box=obj.find("bndbox");xmin,ymin,xmax,ymax=[float(box.findtext(k)) for k in ("xmin","ymin","xmax","ymax")]
                x,y=max(0.,xmin-1),max(0.,ymin-1);right,bottom=min(float(width),xmax),min(float(height),ymax)
                if right<=x or bottom<=y:raise ValueError("original_publisher_D40_bbox_invalid")
                objects.append({"class":"pothole","bbox_xywh_px":[x,y,right-x,bottom-y],"certainty":"confirmed"})
            if image_name in seen_xml:raise ValueError("duplicate_original_annotation")
            seen_xml.add(image_name);candidates[image_name]={"country":country,"objects":objects,"width":width,"height":height,
                "xml_sha256":hashlib.sha256(ET.tostring(xml)).hexdigest()}
    print("original_literal_VOC_counts",json.dumps(counts),flush=True)
    split_countries={"India":"train","Japan":"validation","Czech":"test"};selected={};selection={}
    for country,split in split_countries.items():
        entries=[(name,row) for name,row in candidates.items() if row["country"]==country]
        positive=[r for r in entries if r[1]["objects"]];negative=[r for r in entries if not r[1]["objects"]]
        random.Random(2026).shuffle(positive);random.Random(2027).shuffle(negative)
        maximum=1200 if split=="train" else 250
        chosen=positive[:maximum]+negative[:min(len(positive)//2,maximum//2)]
        for name,row in chosen:selected[name]=row|{"split":split}
        selection[country]={"positive_available":len(positive),"negative_available":len(negative),"selected":len(chosen),"split":split}
    output=workspace_path("data/training/research-pothole-rdd2020/source");output.mkdir(parents=True,exist_ok=True)
    expiry=json.loads((root/"source.json").read_text())["expires_at"];evidence=root/"license-evidence.json";evidence_sha=sha256(evidence)
    rows=[];seen=set();visual=[];duplicates=0
    with tarfile.open(archive,mode="r|gz") as source:
        for member in source:
            if member.name not in selected:continue
            raw=source.extractfile(member).read();info=selected[member.name];digest=hashlib.sha256(raw).hexdigest()
            image=cv2.imdecode(np.frombuffer(raw,np.uint8),1)
            if image is None or image.shape[:2]!=(info["height"],info["width"]):raise ValueError("original_image_XML_dimension_mismatch")
            phash=d_hash(image)
            if digest in seen or any(split!=info["split"] and (phash^old).bit_count()<=3 for old,split in visual):duplicates+=1;continue
            seen.add(digest);visual.append((phash,info["split"]))
            identifier=str(uuid.uuid5(NAMESPACE,"RDD2020:"+member.name));path=output/(identifier+".jpg");path.write_bytes(raw)
            rows.append({"sample_id":identifier,"source_id":str(uuid.uuid5(NAMESPACE,"source:"+digest)),"source_hash":digest,
                "leakage_group_id":"RDD2020:country:"+info["country"],"country":info["country"],"split":info["split"],
                "annotation_version":"RDD2020-publisher-literal-VOC-D40-v1","xml_sha256":info["xml_sha256"],"path":path.relative_to(ROOT).as_posix(),"sha256":digest,
                "privacy_status":"restricted_research","qc_status":"publisher_annotations_machine_checked","release_scope":"offline_research",
                "human_privacy_reviewed":False,"not_for_deployment":True,"project_acceptance_eligible":False,"independent_route_verified":False,
                "annotation_review":"original publisher XML literal D40 only; dimensions and bounds machine checked; other D-codes excluded without remapping",
                "allowed_use":"Private noncommercial research CC-BY-NC3.0, attribution required; no human privacy or deployment approval.",
                "expires_at":expiry,"source_expires_at":expiry,"source_url":"https://data.mendeley.com/datasets/5ty2wb6gvg/1",
                "license_evidence_path":evidence.relative_to(ROOT).as_posix(),"license_evidence_sha256":evidence_sha,"detection_labels_verified":True,
                "objects":info["objects"],"original_member":member.name,"perceptual_dhash":f"{phash:016x}"})
    validation=validate_research_release(rows)
    if not validation["valid"]:raise ValueError(json.dumps(validation["errors"][:10]))
    release=output.parent/"release.json";atomic_json(release,rows)
    atomic_json(output.parent/"source-qc.json",{"source":"author RDD2020 release DOI10.17632/5ty2wb6gvg.1","publisher_archive_sha256":expected,
        "publisher_hash_verified":True,"license":"CC-BY-NC-3.0","literal_XML_counts":counts,"selection":selection,
        "final_split_counts":dict(Counter(row["split"] for row in rows)),"exact_or_cross_split_dhash_duplicates_removed":duplicates,
        "release_sha256":sha256(release),"independence":"India/Japan/Czech disjoint countries; exactSHA and cross-split64bit dHashHamming<=3 excluded; no exhaustive physical identity/route audit",
        "human_privacy_reviewed":False,"not_for_deployment":True})
    print("publisher_XML_research_release",len(rows),validation["valid"],flush=True)


def prepare_rdd(train_countries,val_country,test_country,limit=2500):
    root=workspace_path("data/research/rdd2022");output=workspace_path("data/training/research-pothole/source")
    output.mkdir(parents=True,exist_ok=True)
    evidence=root/"license-evidence.json";evidence_hash=sha256(evidence)
    now=datetime.now(timezone.utc);rows=[];seen_hashes=set();seen_perceptual=[];report={}
    country_split={country:"train" for country in train_countries};country_split[val_country]="validation";country_split[test_country]="test"
    if len(country_split)!=len(train_countries)+2:raise ValueError("country_splits_must_be_disjoint")
    for country,split in country_split.items():
        candidates=[]
        with zipfile.ZipFile(root/(country+".zip")) as archive:
            names=set(archive.namelist())
            for name in names:
                if "/train/" not in name or not name.endswith(".xml"):continue
                xml=ET.fromstring(archive.read(name))
                image_name=xml.findtext("filename") or Path(name).with_suffix(".jpg").name
                image_candidates=[p for p in (name.replace("annotations/xmls/","images/").replace(".xml",".jpg"),
                    name.split("/train/")[0]+"/train/images/"+image_name) if p in names]
                if not image_candidates:continue
                width=int(xml.findtext("size/width"));height=int(xml.findtext("size/height"));objects=[]
                for obj in xml.findall("object"):
                    if obj.findtext("name")!="D40":continue
                    box=obj.find("bndbox");xmin,ymin,xmax,ymax=[float(box.findtext(k)) for k in ("xmin","ymin","xmax","ymax")]
                    x,y=max(0.,xmin-1),max(0.,ymin-1);right,bottom=min(float(width),xmax),min(float(height),ymax)
                    if right>x and bottom>y:objects.append({"class":"pothole","bbox_xywh_px":[x,y,right-x,bottom-y],"certainty":"confirmed"})
                candidates.append((image_candidates[0],width,height,objects))
            positives=[r for r in candidates if r[3]];negatives=[r for r in candidates if not r[3]]
            random.Random(2026).shuffle(positives);random.Random(2027).shuffle(negatives)
            maximum=limit if split=="train" else max(200,limit//3)
            chosen=positives[:maximum]+negatives[:min(len(positives)//2,maximum//2)]
            stats={"publisher_positive_images":len(positives),"publisher_negative_images":len(negatives),"selected":0,"near_duplicates_removed":0}
            for name,width,height,objects in chosen:
                raw=archive.read(name);digest=hashlib.sha256(raw).hexdigest()
                if digest in seen_hashes:continue
                image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
                if image is None or image.shape[:2]!=(height,width):continue
                phash=d_hash(image)
                # Conservatively drop cross-split visual near duplicates as well as exact hashes.
                if any(old_split!=split and (phash^old).bit_count()<=3 for old,old_split in seen_perceptual):
                    stats["near_duplicates_removed"]+=1;continue
                seen_hashes.add(digest);seen_perceptual.append((phash,split))
                sample_id=str(uuid.uuid5(NAMESPACE,country+":"+name));path=output/(sample_id+".jpg");path.write_bytes(raw)
                rows.append({"sample_id":sample_id,"source_id":str(uuid.uuid5(NAMESPACE,"source:"+digest)),"source_hash":digest,
                    "leakage_group_id":"country:"+country,"annotation_version":"RDD2022-D40-publisher-VOC-to-COCO-v1",
                    "path":path.relative_to(ROOT).as_posix(),"sha256":digest,"split":split,"country":country,"original_member":name,
                    "privacy_status":"restricted_research","qc_status":"publisher_annotations_machine_checked","release_scope":"offline_research",
                    "human_privacy_reviewed":False,"not_for_deployment":True,"annotation_review":"publisher_original_bbox_syntax_image_dimension_machine_checks",
                    "allowed_use":"Offline research under CC-BY-4.0; attribution required; no human privacy approval; no image upload/redistribution.",
                    "expires_at":(now+timedelta(days=7)).isoformat(),"source_expires_at":json.loads((root/(country+".source.json")).read_text())["expires_at"],
                    "source_url":RDD_URL,"license_evidence_path":evidence.relative_to(ROOT).as_posix(),"license_evidence_sha256":evidence_hash,
                    "detection_labels_verified":True,"objects":objects,"perceptual_dhash":f"{phash:016x}"})
                stats["selected"]+=1
            report[country]=stats
    state=validate_research_release(rows)
    if not state["valid"]:raise ValueError(json.dumps(state["errors"][:10]))
    release=output.parent/"release.json";atomic_json(release,rows)
    atomic_json(output.parent/"split-evidence.json",{"countries":country_split,"stats":report,"split_counts":dict(Counter(r["split"] for r in rows)),
        "exact_hash_unique":len(seen_hashes)==len(rows),"cross_split_dhash_max_removed":3,"physical_object_identity_available":False,
        "independence":"disjoint country sources, identical image hash removal and conservative cross-split dHash filtering; no claim of exhaustive manual real-object audit",
        "human_reviewed":False,"release_sha256":sha256(release)})
    print(json.dumps({"release":str(release),"stats":report,"validation":state},indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest="command",required=True)
    fetch=sub.add_parser("fetch-rdd");fetch.add_argument("countries",nargs="+")
    prep=sub.add_parser("prepare-rdd");prep.add_argument("--train",nargs="+",default=["India","Japan"])
    prep.add_argument("--validation",default="China_MotorBike");prep.add_argument("--test",default="United_States");prep.add_argument("--limit",type=int,default=2500)
    mirror=sub.add_parser("prepare-rdd-mirror");mirror.add_argument("--limit",type=int,default=1600)
    sub.add_parser("fetch-redaction")
    sub.add_parser("finalize-rdd")
    sub.add_parser("fetch-pretrained")
    sub.add_parser("fetch-yolox")
    sub.add_parser("fetch-rdd2020")
    sub.add_parser("prepare-rdd2020")
    args=parser.parse_args()
    if args.command=="fetch-rdd":fetch_rdd(args.countries)
    elif args.command=="prepare-rdd":prepare_rdd(args.train,args.validation,args.test,args.limit)
    elif args.command=="prepare-rdd-mirror":prepare_rdd_mirror(args.limit)
    elif args.command=="fetch-redaction":fetch_redaction()
    elif args.command=="finalize-rdd":finalize_rdd_sources()
    elif args.command=="fetch-pretrained":fetch_pretrained()
    elif args.command=="fetch-yolox":fetch_yolox()
    elif args.command=="fetch-rdd2020":fetch_rdd2020()
    else:prepare_rdd2020()

if __name__=="__main__":main()
