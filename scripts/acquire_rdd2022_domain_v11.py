"""Separately authorized bounded full official archive transfer and ZIP CRC audit.

No image decoder, model, split creation, or training. Partial bytes are preserved.
Signed URLs/error bodies never enter receipts or console output.
"""
from __future__ import annotations
import argparse,hashlib,json,os,re,shutil,stat,sys,time,zipfile,zlib
from datetime import datetime,timezone
from pathlib import Path,PurePosixPath
import httpx
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import workspace_path,sha256
from scripts import probe_rdd2022_domain_v11 as prior

DEST=prior.DEST;NAME='RDD2022_released_through_CRDDC2022.zip'
PLAN=DEST/'transfer-plan.json';PARTIAL=DEST/(NAME+'.download');ARCHIVE=DEST/NAME
MAX_ATTEMPTS=2;MAX_WALL=45*60;READ_TIMEOUT=60;RESERVE=5*1024**3
MAX_ENTRIES=200_000;MAX_EXPANDED=50*1024**3
PARENT_PLAN='2ddc84941daa5bfed1e25b272ba1d7cb8788c966857fc27a91d9ef71cf93c93a'


def now():return datetime.now(timezone.utc).isoformat()


def alive(deadline):
    if time.monotonic()>=deadline:raise ValueError('transfer_wall_budget_exhausted')
    if datetime.now(timezone.utc)>=datetime.fromisoformat(prior.EXPIRY):raise ValueError('original_source_expired')


def scoped_files():
    root=workspace_path(DEST,exists=True)
    for path in (PARTIAL,ARCHIVE):
        resolved=workspace_path(path)
        if resolved.parent!=root or path.is_symlink() or getattr(path,'is_junction',lambda:False)():raise ValueError('exact_same_root_archive_paths_required')


def prepare():
    scoped_files()
    if PLAN.exists() or PARTIAL.exists() or ARCHIVE.exists():raise ValueError('new_full_transfer_phase_required')
    if sha256(DEST/'acquisition-plan.json')!=PARENT_PLAN:raise ValueError('original_probe_plan_changed')
    parent=json.loads((DEST/'availability-results.json').read_text())
    metadata=json.loads((DEST/'publisher-metadata.json').read_text())
    item=next(item for item in metadata['files'] if item['id']==38030910)
    if parent['file_transport_attempts']!=3 or item['size']!=prior.SIZE or item['computed_md5']!=prior.MD5:raise ValueError('verified_publisher_metadata_required')
    if shutil.disk_usage(DEST).free<prior.SIZE+RESERVE:raise ValueError('archive_plus_5GiB_reserve_required')
    plan={'format':'rdd2022-domain-v11-full-transfer-plan-v1','created_at':now(),
        'parent_probe_plan_sha256':PARENT_PLAN,'parent_availability_sha256':sha256(DEST/'availability-results.json'),
        'publisher_metadata_sha256':sha256(DEST/'publisher-metadata.json'),'transport_source_sha256':sha256(Path(__file__)),
        'official_url':prior.OUTER,'file_id':38030910,'archive_bytes':prior.SIZE,'publisher_md5':prior.MD5,
        'source_expires_at':prior.EXPIRY,'maximum_transfer_attempts':MAX_ATTEMPTS,'maximum_total_wall_seconds':MAX_WALL,
        'read_timeout_seconds':READ_TIMEOUT,'free_disk_reserve_bytes':RESERVE,'resume_requires_exact206_offset_total':True,
        'signed_URL_modified':False,'signed_URL_logged':False,'error_body_logged':False,
        'ZIP_limits':{'entries':MAX_ENTRIES,'total_uncompressed_bytes':MAX_EXPANDED,'JPG_bytes':32*1024**2,'XML_bytes':2*1024**2},
        'whole_publisher_MD5_required_before_archive_publish':True,'source_expiry_extended':False,
        'image_decode':False,'model_inference':False,'new_training':False,'new_split':False,'GPU_used':False}
    prior.new_json(PLAN,plan)
    return {'status':'transfer_plan_fixed','plan_sha256':sha256(PLAN)}


def response_contract(status,headers,offset):
    expected=prior.SIZE-offset
    if status==206:
        match=re.fullmatch(r'bytes ([0-9]+)-([0-9]+)/([0-9]+)',headers.get('content-range',''))
        if not match or tuple(map(int,match.groups()))!=(offset,prior.SIZE-1,prior.SIZE):raise ValueError('exact_resume_ContentRange_required')
    elif status!=200 or offset!=0:raise ValueError('resume_cannot_append_non206_response')
    raw=headers.get('content-length')
    if raw is None or not raw.isdigit() or int(raw)!=expected:raise ValueError('exact_remaining_ContentLength_required')
    if headers.get('content-encoding','identity')!='identity':raise ValueError('unmodified_archive_body_required')


def hash_partial(deadline):
    md5=hashlib.md5();sha=hashlib.sha256();count=0
    if PARTIAL.exists():
        with PARTIAL.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024**2),b''):
                alive(deadline);count+=len(chunk)
                if count>prior.SIZE:raise ValueError('partial_exceeds_archive_bound')
                md5.update(chunk);sha.update(chunk)
    return count,md5,sha


def transfer_attempt(client,deadline,index):
    scoped_files();offset,md5,sha=hash_partial(deadline)
    if offset==prior.SIZE:return md5.hexdigest(),sha.hexdigest(),0
    if shutil.disk_usage(DEST).free<prior.SIZE-offset+RESERVE:raise ValueError('remaining_archive_plus_reserve_required')
    current=httpx.URL(prior.OUTER+'?download='+str(time.time_ns()))
    for hop in range(5):
        alive(deadline);prior.allowed(current)
        headers={'Accept-Encoding':'identity','Cache-Control':'no-cache'}
        if offset:headers['Range']=f'bytes={offset}-'
        timeout=min(READ_TIMEOUT,max(.1,deadline-time.monotonic()))
        with client.stream('GET',current,headers=headers,timeout=timeout) as response:
            if response.status_code in (301,302,303,307,308):
                location=response.headers.get('location')
                if not location:raise ValueError('official_redirect_location_missing')
                current=response.url.join(location);continue
            if response.status_code in (401,403):raise ValueError('official_access_denied')
            response_contract(response.status_code,response.headers,offset)
            written=offset;next_progress=time.monotonic()+15
            with PARTIAL.open('ab' if offset else 'xb') as output:
                for chunk in response.iter_raw(1024**2):
                    alive(deadline)
                    if written+len(chunk)>prior.SIZE:raise ValueError('archive_body_exceeds_publisher_size')
                    if written==0 and chunk[:4]!=b'PK\x03\x04':raise ValueError('archive_ZIP_magic_required')
                    output.write(chunk);md5.update(chunk);sha.update(chunk);written+=len(chunk)
                    if time.monotonic()>=next_progress:
                        output.flush();print(json.dumps({'attempt':index,'downloaded_bytes':written,'expected_bytes':prior.SIZE}),flush=True);next_progress=time.monotonic()+15
                output.flush();os.fsync(output.fileno())
            if written!=prior.SIZE:raise ValueError('interrupted_archive_body')
            return md5.hexdigest(),sha.hexdigest(),written-offset
    raise ValueError('bounded_official_redirect_hops_exceeded')


def safe_members(archive,budget):
    infos=archive.infolist();budget['entries']+=len(infos)
    if budget['entries']>MAX_ENTRIES:raise ValueError('archive_entry_count_exceeded')
    for item in infos:
        path=PurePosixPath(item.filename)
        if '\\' in item.filename or '\\' in item.orig_filename or path.is_absolute() or '..' in path.parts or ':' in item.filename or not path.parts or stat.S_ISLNK(item.external_attr>>16) or item.flag_bits&1:raise ValueError('ZIP_path_link_or_encryption_refused')
        budget['expanded']+=item.file_size
        if budget['expanded']>MAX_EXPANDED:raise ValueError('archive_total_uncompressed_bound_exceeded')
        limit=32*1024**2 if path.suffix.lower() in ('.jpg','.jpeg') else 2*1024**2 if path.suffix.lower()=='.xml' else MAX_EXPANDED
        if item.file_size>limit:raise ValueError('individual_source_file_size_exceeded')
    return infos


def crc_member(archive,item,deadline):
    total=0;crc=0
    with archive.open(item) as stream:
        for chunk in iter(lambda:stream.read(1024**2),b''):
            alive(deadline);total+=len(chunk)
            if total>item.file_size:raise ValueError('uncompressed_member_exceeds_metadata')
            crc=zlib.crc32(chunk,crc)
    if total!=item.file_size or crc!=item.CRC:raise ValueError('member_size_or_CRC_mismatch')


def inspect_zip(path,deadline):
    budget={'entries':0,'expanded':0};countries={}
    with zipfile.ZipFile(path) as outer:
        for item in safe_members(outer,budget):
            alive(deadline)
            if item.is_dir():continue
            crc_member(outer,item,deadline)
            if PurePosixPath(item.filename).suffix.lower()!='.zip':continue
            country=PurePosixPath(item.filename).stem
            if country in countries:raise ValueError('duplicate_country_archive')
            counts={'JPG':0,'XML':0,'train_JPG':0,'train_XML':0,'test_JPG':0,'test_XML':0,'entries':0}
            with outer.open(item) as source,zipfile.ZipFile(source) as nested:
                for member in safe_members(nested,budget):
                    alive(deadline)
                    if member.is_dir():continue
                    crc_member(nested,member,deadline);counts['entries']+=1
                    parts=PurePosixPath(member.filename).parts;suffix=PurePosixPath(member.filename).suffix.lower()
                    kind='JPG' if suffix in ('.jpg','.jpeg') else 'XML' if suffix=='.xml' else None
                    if kind:
                        counts[kind]+=1
                        for split in ('train','test'):
                            if split in parts:counts[split+'_'+kind]+=1
            countries[country]=counts
    return {'entries':budget['entries'],'uncompressed_bytes_with_nested_containers':budget['expanded'],'countries':countries,'all_members_CRC_verified':True}


def run():
    plan=json.loads(PLAN.read_text());scoped_files()
    if plan['transport_source_sha256']!=sha256(Path(__file__)) or plan['parent_probe_plan_sha256']!=sha256(DEST/'acquisition-plan.json') or plan['parent_availability_sha256']!=sha256(DEST/'availability-results.json') or plan['publisher_metadata_sha256']!=sha256(DEST/'publisher-metadata.json') or plan['source_expires_at']!=prior.EXPIRY:raise ValueError('frozen_bulk_phase_binding_changed')
    start=DEST/'full-transfer-started.json'
    if start.exists() or ARCHIVE.exists():raise ValueError('bulk_phase_already_started')
    deadline=time.monotonic()+MAX_WALL;alive(deadline)
    prior.new_json(start,{'started_at':now(),'plan_sha256':sha256(PLAN),'maximum_wall_seconds':MAX_WALL})
    started=time.monotonic();attempts=[];result={'status':'not_verified','source_expires_at':prior.EXPIRY,'source_expiry_extended':False,'usable_source':False}
    try:
        with httpx.Client(timeout=READ_TIMEOUT,follow_redirects=False) as client:
            for index in range(1,MAX_ATTEMPTS+1):
                try:
                    md5,digest,received=transfer_attempt(client,deadline,index)
                    attempts.append({'attempt':index,'status':'body_complete','received_bytes':received})
                    if md5!=prior.MD5:raise ValueError('whole_publisher_MD5_mismatch')
                    scoped_files()
                    # Atomic no-replace publication; preserve partial if final
                    # destination unexpectedly appears on either OS.
                    os.link(PARTIAL,ARCHIVE);PARTIAL.unlink()
                    prior.new_json(DEST/'whole-MD5-verified.json',{'bytes':prior.SIZE,'publisher_md5':md5,'sha256':digest,'source_expires_at':prior.EXPIRY,'ZIP_CRC_verified':False,'usable_source':False})
                    structure=inspect_zip(ARCHIVE,deadline)
                    result.update(status='verified_archive_structure_CRC',usable_source=True,bytes=prior.SIZE,md5=md5,sha256=digest,**structure)
                    break
                except (httpx.HTTPError,ValueError,OSError,zipfile.BadZipFile) as error:
                    reason=str(error) if isinstance(error,ValueError) and re.fullmatch('[A-Za-z_]+',str(error)) else type(error).__name__
                    attempts.append({'attempt':index,'status':'stopped','reason':reason})
                    if reason in ('official_access_denied','whole_publisher_MD5_mismatch','transfer_wall_budget_exhausted','original_source_expired') or ARCHIVE.exists():break
    finally:
        result.update(finished_at=now(),wall_seconds=time.monotonic()-started,attempts=attempts,
            partial_bytes=PARTIAL.stat().st_size if PARTIAL.exists() else 0,
            image_decodes=0,model_inferences=0,training_updates=0,new_splits=0,GPU_used=False,
            signed_URL_logged=False,error_body_logged=False,existing_source_cache_modified=False)
        prior.new_json(DEST/'full-transfer-results.json',result)
    return {key:result[key] for key in ('status','usable_source','wall_seconds')}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','transfer']);args=parser.parse_args()
    try:print(json.dumps(prepare() if args.action=='prepare' else run()))
    except Exception as error:print(json.dumps({'status':'bounded_transfer_stopped','error_type':type(error).__name__}));return 1
    return 0


if __name__=='__main__':raise SystemExit(main())
