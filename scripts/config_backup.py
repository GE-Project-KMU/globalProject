"""Configuration-only backup: detailed records, .env and keys are never copied."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
ALLOWED={"compose.yaml","compose.tls.yaml","Dockerfile",".dockerignore","requirements.txt","requirements-vision.txt","frontend/package.json","frontend/package-lock.json","frontend/vite.config.ts","scripts/nginx.conf","scripts/nginx-tls.conf"}
LIMIT=2_000_000


def scoped(path):
    path=Path(path)
    path=(path if path.is_absolute() else ROOT/path).resolve()
    if not path.is_relative_to(ROOT):raise ValueError("Path outside workspace")
    return path


def create(output):
    output=scoped(output)
    if output.exists():raise ValueError("Refusing to overwrite existing backup")
    entries={name:(ROOT/name).read_bytes() for name in sorted(ALLOWED) if (ROOT/name).is_file()}
    manifest={"schema_version":1,"policy":"configuration_only","detail_data_included":False,"files":{name:hashlib.sha256(body).hexdigest() for name,body in entries.items()}}
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,"x",zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json",json.dumps(manifest,sort_keys=True))
        for name,body in entries.items():archive.writestr(name,body)
    return manifest


def inspect(path):
    path=scoped(path)
    with zipfile.ZipFile(path) as archive:
        infos=archive.infolist();names=[item.filename for item in infos]
        if len(names)!=len(set(names)) or set(names)-ALLOWED-{"manifest.json"}:raise ValueError("Forbidden backup entries")
        if sum(item.file_size for item in infos)>LIMIT:raise ValueError("Backup too large")
        manifest=json.loads(archive.read("manifest.json"))
        if manifest.get("schema_version")!=1 or manifest.get("policy")!="configuration_only" or manifest.get("detail_data_included") is not False:raise ValueError("Unsupported backup policy")
        if set(manifest["files"])!=set(names)-{"manifest.json"}:raise ValueError("Manifest mismatch")
        entries={name:archive.read(name) for name in manifest["files"]}
        for name,body in entries.items():
            if hashlib.sha256(body).hexdigest()!=manifest["files"][name]:raise ValueError("Backup hash mismatch")
    return manifest,entries


def restore(path,apply=False):
    manifest,entries=inspect(path)
    # No data/DB/session/secret restore route exists: deleted observations cannot reappear.
    if apply:
        for name,body in entries.items():
            target=scoped(name);target.parent.mkdir(parents=True,exist_ok=True)
            temporary=target.with_name(target.name+".restore-tmp")
            temporary.write_bytes(body);temporary.replace(target)
    return {"verified":True,"applied":apply,"detail_records_restored":0,"files":len(entries)}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["create","verify","restore"])
    parser.add_argument("path")
    parser.add_argument("--apply",action="store_true")
    args=parser.parse_args()
    result=create(args.path) if args.action=="create" else restore(args.path,args.action=="restore" and args.apply)
    print(json.dumps(result,ensure_ascii=False,indent=2))
