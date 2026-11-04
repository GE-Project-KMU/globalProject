"""Three bounded official availability probes; no extraction, model, or training.

Signed redirect URLs remain in memory only. Existing failed/cache data is never
opened or modified except the pinned source.json expiry evidence.
"""
from __future__ import annotations
import argparse,hashlib,json,re,shutil,sys,time
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import urlparse,parse_qs
import httpx
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import workspace_path,sha256

DEST=ROOT/'data/research/rdd2022-domain-v11'
API='https://api.figshare.com/v2/articles/21431547'
README='https://raw.githubusercontent.com/sekilab/RoadDamageDetector/master/README.md'
OUTER='https://ndownloader.figshare.com/files/38030910'
COUNTRY_BASE='https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022/'
SIZE=13_264_172_619;MD5='b62bd51d2ffcfaa76c60f234f0cc2bb3'
EVIDENCE=ROOT/'data/research/rdd2022-blind-v7/source.json'
EXPIRY='2026-11-10T17:07:24.286159+00:00'
MAX_ATTEMPTS=3;REQUEST_SECONDS=60;PROBE_BYTES=64


def new_json(path,value):
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False)


def utc():return datetime.now(timezone.utc).isoformat()


def allowed(url):
    parsed=urlparse(str(url));host=parsed.hostname or ''
    if parsed.scheme!='https' or parsed.username or parsed.password or not (host=='ndownloader.figshare.com' or host.endswith('.amazonaws.com')):raise ValueError('unexpected_official_redirect_host')


def read_metadata(url,maximum):
    started=time.monotonic();raw=bytearray()
    with httpx.Client(timeout=REQUEST_SECONDS,follow_redirects=False) as client:
        with client.stream('GET',url) as response:
            if response.status_code!=200:raise ValueError('official_metadata_not_available')
            for chunk in response.iter_bytes(65536):
                raw.extend(chunk)
                if len(raw)>maximum or time.monotonic()-started>REQUEST_SECONDS:raise ValueError('bounded_metadata_request_exceeded')
    return bytes(raw)


def prepare():
    if DEST.exists():raise ValueError('new_acquisition_namespace_required')
    previous=json.loads(EVIDENCE.read_text(encoding='utf-8'))
    if previous.get('expires_at')!=EXPIRY or datetime.fromisoformat(EXPIRY)<=datetime.now(timezone.utc):raise ValueError('fixed_original_expiry_required')
    evidence_sha=sha256(EVIDENCE)
    api_bytes=read_metadata(API,1_000_000);article=json.loads(api_bytes)
    archive=next(item for item in article['files'] if item['id']==38030910)
    if archive['size']!=SIZE or archive['computed_md5']!=MD5 or archive['download_url']!=OUTER or article['license']['name']!='CC BY 4.0':raise ValueError('official_archive_contract_changed')
    readme=read_metadata(README,1_000_000)
    countries=sorted(set(re.findall(r'https://bigdatacup[^\s)]+Country_Specific[^\s)]+',readme.decode('utf-8'))))
    alternatives=[COUNTRY_BASE+'RDD2022_'+name+'.zip' for name in ('United_States','China_MotorBike')]
    if any(url not in countries for url in alternatives):raise ValueError('author_country_alternative_not_declared')
    if shutil.disk_usage(ROOT).free<SIZE+2*1024**3:raise ValueError('bounded_archive_plus_reserve_disk_required')
    DEST.mkdir(parents=True)
    new_json(DEST/'publisher-metadata.json',{'article_api':API,'metadata_sha256':hashlib.sha256(api_bytes).hexdigest(),
        'article_id':article['id'],'license':article['license'],
        'files':[{key:item[key] for key in ('id','name','size','computed_md5','download_url')} for item in article['files']],
        'author_readme':README,'author_readme_sha256':hashlib.sha256(readme).hexdigest(),'country_archive_urls':countries})
    plan={'format':'rdd2022-domain-v11-bounded-acquisition-plan-v1','created_at':utc(),
        'publisher_metadata_sha256':sha256(DEST/'publisher-metadata.json'),'transport_source_sha256':sha256(Path(__file__)),
        'parent_expiry_evidence':{'path':EVIDENCE.relative_to(ROOT).as_posix(),'sha256':evidence_sha},
        'source_expires_at':EXPIRY,'source_expiry_extended':False,'official_archive_size':SIZE,'official_archive_md5':MD5,
        'candidate_urls':[OUTER,*alternatives],'maximum_file_transport_attempts':MAX_ATTEMPTS,
        'maximum_seconds_per_attempt':REQUEST_SECONDS,'range_header':f'bytes=0-{PROBE_BYTES-1}',
        'maximum_probe_payload_bytes':MAX_ATTEMPTS*PROBE_BYTES,'maximum_archive_bytes':SIZE,
        'disk_reserve_bytes':2*1024**3,'phase':'availability_and_checksum_evidence_only',
        'archive_acceptance':'Only complete publisher-checksum-verified archives may become usable source; 206 or ZIP prefix alone is not sufficient',
        'no_bulk_transfer_without_verified_checksum_and_separate_bounded_phase':True,
        'signed_URL_modified':False,'signed_URL_logged':False,'authentication_acquired':False,'old_cache_modified':False,
        'no_training_blacklist':'All historical TEST including US100, old datasets and P2 TEST remain permanently excluded from future TRAIN',
        'model_inferences':0,'training_updates':0,'split_creation':0,'GPU_used':False}
    new_json(DEST/'acquisition-plan.json',plan)
    return {'status':'plan_fixed','plan_sha256':sha256(DEST/'acquisition-plan.json')}


def probe(url,index,client,*,clock=time.monotonic):
    started=clock();current=httpx.URL(url);chain=[]
    for hop in range(5):
        remaining=REQUEST_SECONDS-(clock()-started)
        if remaining<=0:raise ValueError('absolute_attempt_deadline_exceeded')
        allowed(current)
        with client.stream('GET',current,headers={'Range':f'bytes=0-{PROBE_BYTES-1}','Cache-Control':'no-cache'},timeout=remaining) as response:
            chain.append(response.status_code)
            if response.status_code in (301,302,303,307,308):
                location=response.headers.get('location')
                if not location:raise ValueError('official_redirect_location_missing')
                current=response.url.join(location);continue
            info={'attempt':index,'status_code':response.status_code,'redirect_statuses':chain[:-1],
                'final_host':current.host,'elapsed_seconds':clock()-started,'bytes_received':0,
                'source_expires_at':EXPIRY,'date':response.headers.get('date'),
                'content_range':response.headers.get('content-range'),'content_type':response.headers.get('content-type')}
            query=parse_qs(urlparse(str(current)).query)
            if query.get('X-Amz-Date') and query.get('X-Amz-Expires'):
                try:
                    signed=datetime.strptime(query['X-Amz-Date'][0],'%Y%m%dT%H%M%SZ').replace(tzinfo=timezone.utc)
                    info['signature_age_seconds_at_observation']=(datetime.now(timezone.utc)-signed).total_seconds()
                    info['signature_lifetime_seconds']=int(query['X-Amz-Expires'][0])
                except (ValueError,TypeError):info['signature_metadata_parse_failed']=True
            if response.status_code!=206:return info|{'status':'source_unavailable','whole_archive_checksum_verified':False}
            match=re.fullmatch(r'bytes 0-63/([0-9]+)',response.headers.get('content-range',''))
            if not match or not PROBE_BYTES<=int(match[1])<=SIZE:raise ValueError('exact_bounded_content_range_required')
            if index==1 and int(match[1])!=SIZE:raise ValueError('outer_archive_size_changed')
            raw=bytearray()
            for chunk in response.iter_bytes(PROBE_BYTES+1):
                raw.extend(chunk)
                if len(raw)>PROBE_BYTES or clock()-started>REQUEST_SECONDS:raise ValueError('probe_body_or_absolute_time_exceeded')
            if len(raw)!=PROBE_BYTES or bytes(raw[:4])!=b'PK\x03\x04':raise ValueError('exact_ZIP_prefix_required')
            # Do not save original payload or mistake an opaque/multipart ETag
            # for an independently published complete-archive checksum.
            return info|{'status':'bounded_ZIP_range_available','bytes_received':len(raw),'archive_bytes':int(match[1]),
                'probe_sha256':hashlib.sha256(raw).hexdigest(),'etag':response.headers.get('etag'),
                'whole_archive_checksum_verified':False,'ZIP_CRC_verified':False}
    raise ValueError('bounded_redirect_hops_exceeded')


def run():
    plan=json.loads((DEST/'acquisition-plan.json').read_text(encoding='utf-8'))
    if plan['transport_source_sha256']!=sha256(Path(__file__)) or plan['source_expires_at']!=EXPIRY or sha256(EVIDENCE)!=plan['parent_expiry_evidence']['sha256'] or datetime.fromisoformat(EXPIRY)<=datetime.now(timezone.utc):raise ValueError('frozen_source_or_expiry_changed')
    if (DEST/'transport-attempt-started.json').exists():raise ValueError('bounded_attempts_already_started')
    new_json(DEST/'transport-attempt-started.json',{'started_at':utc(),'plan_sha256':sha256(DEST/'acquisition-plan.json'),'maximum_attempts':3})
    results=[]
    with httpx.Client(timeout=REQUEST_SECONDS,follow_redirects=False) as client:
        for index,source in enumerate(plan['candidate_urls'],1):
            if index>MAX_ATTEMPTS:raise ValueError('transport_attempt_budget_exceeded')
            # Only the initial official public downloader receives a new nonce;
            # signed Location is followed unchanged and never persisted.
            url=source+'?download='+str(time.time_ns()) if index==1 else source
            try:result=probe(url,index,client)
            except (httpx.HTTPError,ValueError) as error:
                result={'attempt':index,'status':'bounded_probe_stopped','error_type':type(error).__name__,'whole_archive_checksum_verified':False}
            results.append(result)
            new_json(DEST/f'attempt-{index}.json',result)
            print(json.dumps({'attempt':index,'status':result['status'],'status_code':result.get('status_code')}),flush=True)
    summary={'format':'rdd2022-domain-v11-availability-results-v1','finished_at':utc(),'plan_sha256':sha256(DEST/'acquisition-plan.json'),
        'attempts':results,'file_transport_attempts':len(results),'source_expires_at':EXPIRY,'source_expiry_extended':False,
        'official_complete_archives_verified':0,'new_usable_source_images':0,'image_decodes':0,'model_inferences':0,'training_updates':0,
        'split_creation':0,'old_cache_modified':False,'existing_inventories_modified':False,'signed_URL_logged':False,
        'status':'availability_only_no_verified_archive' if any(r['status']=='bounded_ZIP_range_available' for r in results) else 'official_transport_unavailable_within_budget'}
    new_json(DEST/'availability-results.json',summary)
    return {key:summary[key] for key in ('status','file_transport_attempts','official_complete_archives_verified','new_usable_source_images')}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','probe']);args=parser.parse_args()
    try:print(json.dumps(prepare() if args.action=='prepare' else run()))
    except Exception as error:
        # HTTP errors may contain signed secrets. Never emit error text/trace.
        print(json.dumps({'status':'stopped','error_type':type(error).__name__}));return 1
    return 0


if __name__=='__main__':raise SystemExit(main())
