"""Pinned official COCO initialization, downloaded only into project cache."""
from pathlib import Path
import hashlib,json,shutil,sys,urllib.request
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
URL='https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.pth'
SHA='9de513de589ac98bb92d3bca53b5af7b9acfa9b0bacb831f7999d0f7afaee8f0'

def main():
    sys.path.insert(0,str(ROOT))
    from agent.paths import workspace_path,atomic_json,sha256
    folder=workspace_path('.cache/training/pretrained');folder.mkdir(parents=True,exist_ok=True)
    path=folder/'yolox_tiny_0.1.1rc0.pth'
    if not path.exists():
        part=folder/'yolox_tiny_0.1.1rc0.download'
        if part.exists():raise ValueError('preserve_incomplete_official_download')
        with urllib.request.urlopen(URL,timeout=60) as response,part.open('xb') as stream:
            total=0
            while block:=response.read(1024*1024):
                total+=len(block)
                if total>80*1024*1024:raise ValueError('unexpected_initialization_size')
                stream.write(block)
        if sha256(part)!=SHA:raise ValueError('official_initialization_hash_mismatch')
        part.replace(path)
    if sha256(path)!=SHA:raise ValueError('official_initialization_hash_mismatch')
    evidence=folder/'yolox_tiny_0.1.1rc0-source.json'
    if not evidence.exists():atomic_json(evidence,{'source_url':URL,'primary_reference':'https://github.com/Megvii-BaseDetection/YOLOX','acquired_at':datetime.now(timezone.utc).isoformat(),'sha256':SHA,'bytes':path.stat().st_size,'upstream_software_license':'Apache-2.0','role':'official COCO initialization, not a trained pothole detector','publisher_digest_claimed':False,'research_only':True})
    print(json.dumps({'sha256':SHA,'bytes':path.stat().st_size,'weights_uploaded':False}))

if __name__=='__main__':main()
