"""Bounded official signed-range transport for frozen RDD2022 blind selection.

The failed original/API attempts remain unchanged. No model is invoked. Private
range cache inherits the existing source deadline; signatures are never logged.
"""
from __future__ import annotations
import hashlib,json,os,sys,time
from datetime import datetime,timedelta,timezone
from pathlib import Path
import httpx

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import atomic_json,sha256
from scripts import prepare_rdd2022_blind as selection
from scripts.acquire_rdd2022_blind_api import SELECTION_SHA,PROTOCOL_SHA

def checked_host(url):
    host=url.host or ''
    if url.scheme!='https' or not (host.endswith('.figshare.com') or host.endswith('.amazonaws.com')):
        raise ValueError('unexpected_publisher_redirect_host')

class SignedRange(selection.OfficialRange):
    def __init__(self,*,client=None,block=4*1024*1024,folder=None):
        super().__init__(client=client,block=block)
        self.folder=folder or selection.scoped_output(selection.DEST)
        self.expiry=selection.aware(json.loads((self.folder/'source.json').read_text(encoding='utf-8'))['expires_at'])
        self.disk=self.folder/'range-cache'
        if self.disk.is_symlink() or getattr(self.disk,'is_junction',lambda:False)():raise ValueError('source_links_forbidden')
        self.disk.mkdir(exist_ok=True)
        self.signed_url=None;self.signed_until=datetime.min.replace(tzinfo=timezone.utc)
        self.resolutions=0;self.disk_hits=0
    def alive(self):
        if datetime.now(timezone.utc)>=self.expiry:raise ValueError('original_source_expired')
    def resolve(self):
        self.alive();self.resolutions+=1
        # Short increasing public query avoids reusing a stale redirect. The
        # resulting signed URL itself is preserved byte-for-byte thereafter.
        nonce=str(int(time.time()))+str(self.resolutions%100)
        url='https://ndownloader.figshare.com/files/38030910?download='+nonce
        with self.client.stream('GET',url,headers={'Range':'bytes=0-0','Cache-Control':'no-cache','Pragma':'no-cache'}) as response:
            response.raise_for_status()
            for item in [*response.history,response]:checked_host(item.url)
            if response.status_code!=206 or response.headers.get('content-range')!=f'bytes 0-0/{self.size}':raise ValueError('exact_resolution_range_required')
            received=0
            for chunk in response.iter_bytes(2):
                received+=len(chunk)
                if received>1:raise ValueError('resolution_size_mismatch')
            if received!=1:raise ValueError('resolution_size_mismatch')
            final=response.url
            date=final.params.get('X-Amz-Date');seconds=final.params.get('X-Amz-Expires')
            if not date or not seconds or not seconds.isdigit() or not 1<=int(seconds)<=3600:raise ValueError('bounded_official_signature_required')
            until=datetime.strptime(date,'%Y%m%dT%H%M%SZ').replace(tzinfo=timezone.utc)+timedelta(seconds=int(seconds))
            if until<=datetime.now(timezone.utc):raise ValueError('stale_official_signature')
            self.signed_url=final;self.signed_until=until
    def fetch(self,start,end):
        self.alive();expected=end-start+1
        if start<0 or end<start or end>=self.size or expected>self.block:raise ValueError('bounded_archive_range_required')
        cache=self.disk/f'{start}-{end}.bin';proof=cache.with_suffix('.sha256')
        if cache.is_symlink() or proof.is_symlink():raise ValueError('source_links_forbidden')
        if cache.exists() and proof.exists():
            data=cache.read_bytes()
            if len(data)!=expected or hashlib.sha256(data).hexdigest()!=proof.read_text(encoding='utf-8'):raise ValueError('range_cache_integrity_failure')
            self.disk_hits+=1;return data
        for attempt in range(5):
            try:
                self.alive()
                if self.signed_url is None or datetime.now(timezone.utc)+timedelta(seconds=1)>=self.signed_until:self.resolve()
                with self.client.stream('GET',self.signed_url,headers={'Range':f'bytes={start}-{end}'}) as response:
                    response.raise_for_status()
                    for item in [*response.history,response]:checked_host(item.url)
                    if response.status_code!=206 or response.headers.get('content-range')!=f'bytes {start}-{end}/{self.size}':raise ValueError('exact_content_range_required')
                    result=bytearray()
                    for chunk in response.iter_bytes(1024*1024):
                        result.extend(chunk)
                        if len(result)>expected:raise ValueError('oversized_range_response')
                    if len(result)!=expected:raise ValueError('truncated_range_response')
                self.alive()
                partial=cache.with_suffix('.partial')
                with partial.open('xb') as stream:stream.write(result)
                os.replace(partial,cache)
                proof.write_text(hashlib.sha256(result).hexdigest(),encoding='utf-8')
                self.network_bytes+=len(result);self.range_responses+=1
                return bytes(result)
            except (httpx.HTTPError,ValueError):
                self.signed_url=None
                if attempt==4:raise
                time.sleep(min(4,attempt+1))

def run():
    folder=selection.scoped_output(selection.DEST)
    if sha256(Path(selection.__file__))!=SELECTION_SHA:raise ValueError('frozen_selection_source_changed')
    if sha256(folder/'selection-protocol.json')!=PROTOCOL_SHA:raise ValueError('frozen_selection_protocol_changed')
    source=json.loads((folder/'source.json').read_text(encoding='utf-8'));expiry=selection.aware(source['expires_at'])
    if expiry<=datetime.now(timezone.utc):raise ValueError('original_source_expired')
    receipt=folder/'signed-transport-attempt.json'
    if receipt.exists():raise ValueError('attempt_receipt_already_exists')
    record={'status':'running','started_at':datetime.now(timezone.utc).isoformat(),
            'selection_source_sha256':SELECTION_SHA,'selection_protocol_sha256':PROTOCOL_SHA,
            'transport_source_sha256':sha256(Path(__file__)),'source_expires_at':expiry.isoformat(),
            'official_public_transport_only':True,'selection_unchanged':True,'model_inferences':0,
            'whole_outer_MD5_verified':False,'range_cache_expires_at':expiry.isoformat(),'signed_urls_logged':False}
    atomic_json(receipt,record)
    selection.OfficialRange=SignedRange
    try:selection.prepare()
    except BaseException as error:
        record.update(status='failed',error_type=type(error).__name__,finished_at=datetime.now(timezone.utc).isoformat())
        atomic_json(receipt,record);raise
    record.update(status='completed',finished_at=datetime.now(timezone.utc).isoformat());atomic_json(receipt,record)

def main():
    try:run()
    except BaseException as error:
        # HTTP exceptions may embed temporary signed credentials in their text.
        # Keep the private receipt's type-only evidence and never print a stack.
        print(json.dumps({'status':'acquisition_stopped','error_type':type(error).__name__,
                          'reason':'Official acquisition did not complete; no success is claimed.'}),file=sys.stderr)
        return 1
    return 0

if __name__=='__main__':sys.exit(main())
