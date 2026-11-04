"""Acquire author-published country ZIPs without touching prior experiments."""
import argparse,json,shutil,sys,time,zipfile
from datetime import datetime,timedelta,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import httpx
from agent.paths import workspace_path,atomic_json,sha256
from scripts.research_detection_data import RemoteRange,RDD_URL,RDD_SIZE,RDD_ARTICLE

COUNTRIES={'Japan','Czech','United_States','China_MotorBike'}
DEST='data/research/rdd2022-official-v5'

class FreshRange(RemoteRange):
    def read(self,size=-1):
        # Avoid reusing an expired redirect cached by a large-download proxy.
        self.url=RDD_URL+'?cache='+str(time.time_ns())
        return super().read(size)

def fetch(countries,output=DEST):
    if not countries or len(set(countries))!=len(countries) or set(countries)-COUNTRIES:
        raise ValueError('explicit_supported_country_subset_required')
    folder=workspace_path(output)
    if not folder.is_relative_to(ROOT/'data/research') or folder==ROOT/'data/research':raise ValueError('new_research_subdirectory_required')
    folder.mkdir(parents=True,exist_ok=True)
    metadata=httpx.get(RDD_ARTICLE,timeout=30);metadata.raise_for_status();metadata=metadata.json()
    archive=next(f for f in metadata['files'] if f['id']==38030910)
    if metadata['license']['name']!='CC BY 4.0' or archive['size']!=RDD_SIZE or archive['computed_md5']!='b62bd51d2ffcfaa76c60f234f0cc2bb3':raise ValueError('publisher_contract_changed')
    publisher=folder/'publisher-contract.json'
    contract={'article_url':RDD_ARTICLE,'source_url':metadata['url_public_html'],'license_id':'CC-BY-4.0','license_url':metadata['license']['url'],'archive_size':RDD_SIZE,'publisher_outer_md5':archive['computed_md5'],'selected_member_crc_only':True,'full_outer_md5_verified':False,'human_privacy_reviewed':False,'research_only':True}
    if publisher.exists():
        if json.loads(publisher.read_text())!=contract:raise ValueError('publisher_contract_changed')
    else:atomic_json(publisher,contract)
    remote=FreshRange(RDD_URL,RDD_SIZE)
    try:
        with zipfile.ZipFile(remote) as outer:
            for country in countries:
                member=outer.getinfo('RDD2022/'+country+'.zip')
                if member.file_size>1_500_000_000 or member.is_dir():raise ValueError('country_archive_size_exceeds_budget')
                target=folder/(country+'.zip');proof=folder/(country+'.zip.source.json');start=folder/(country+'.acquisition-start.json')
                if proof.exists():
                    record=json.loads(proof.read_text())
                    if not target.is_file() or sha256(target)!=record['sha256']:raise ValueError('recorded_source_missing_or_changed')
                    if datetime.fromisoformat(record['expires_at'])<=datetime.now(timezone.utc):raise ValueError('source_expired')
                    print(json.dumps({'country':country,'status':'cached_original_deadline'}),flush=True);continue
                if target.exists():raise ValueError('unreceipted_archive_requires_review')
                if start.exists():dates=json.loads(start.read_text())
                else:
                    acquired=datetime.now(timezone.utc)
                    dates={'download_started_at':acquired.isoformat(),'expires_at':(acquired+timedelta(days=7)).isoformat()}
                    atomic_json(start,dates)
                if datetime.fromisoformat(dates['expires_at'])<=datetime.now(timezone.utc):raise ValueError('acquisition_deadline_expired')
                if shutil.disk_usage(folder).free<member.file_size+2*1024**3:raise ValueError('insufficient_workspace_disk')
                temporary=target.with_suffix('.zip.download')
                print(json.dumps({'country':country,'status':'downloading','expected_bytes':member.file_size}),flush=True)
                with outer.open(member) as source,temporary.open('wb') as destination:
                    shutil.copyfileobj(source,destination,1024*1024)
                if temporary.stat().st_size!=member.file_size:raise ValueError('country_size_mismatch')
                with zipfile.ZipFile(temporary) as nested:
                    if nested.testzip() is not None:raise ValueError('country_member_crc_failed')
                temporary.replace(target)
                atomic_json(proof,{**dates,'source_url':RDD_URL,'outer_member':member.filename,'zip_crc32':member.CRC,'bytes':member.file_size,'sha256':sha256(target),'license_id':'CC-BY-4.0','license_evidence_path':publisher.relative_to(ROOT).as_posix(),'license_evidence_sha256':sha256(publisher),'full_outer_md5_verified':False,'integrity':'Complete country member CRC and nested ZIP CRC checked. Full outer MD5 not claimed.','human_privacy_reviewed':False,'not_for_deployment':True})
                print(json.dumps({'country':country,'status':'verified','downloaded_bytes_total':remote.bytes_downloaded}),flush=True)
    finally:remote.client.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--countries',nargs='+',default=['China_MotorBike','United_States','Czech','Japan']);parser.add_argument('--output',default=DEST);args=parser.parse_args()
    fetch(args.countries,args.output)
