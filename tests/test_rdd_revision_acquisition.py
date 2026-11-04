"""Synthetic ZIP checks: no network, original experiment or personal images."""
import io,json,tempfile,zipfile
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts import fetch_rdd2022_revision as acquisition

@pytest.fixture
def case(monkeypatch):
    base=acquisition.ROOT/'.cache/rdd-revision-tests';base.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=base) as directory:
        root=Path(directory);folder=root/'data/research/new';folder.mkdir(parents=True)
        monkeypatch.setattr(acquisition,'ROOT',root)
        def scoped(value,exists=False):
            p=Path(value);p=(root/p).resolve() if not p.is_absolute() else p.resolve()
            if not p.is_relative_to(root):raise ValueError('outside_synthetic_workspace')
            if exists and not p.exists():raise FileNotFoundError()
            return p
        monkeypatch.setattr(acquisition,'workspace_path',scoped)
        monkeypatch.setattr(acquisition,'sha256',lambda p:__import__('hashlib').sha256(Path(p).read_bytes()).hexdigest())
        def write(path,value):Path(path).write_text(json.dumps(value),encoding='utf-8')
        monkeypatch.setattr(acquisition,'atomic_json',write)
        nested=io.BytesIO()
        with zipfile.ZipFile(nested,'w') as z:z.writestr('SYNTHETIC.txt','Synthetic publisher member')
        outer=io.BytesIO()
        with zipfile.ZipFile(outer,'w') as z:z.writestr('RDD2022/Japan.zip',nested.getvalue())
        wire=outer.getvalue();monkeypatch.setattr(acquisition,'RDD_SIZE',len(wire))
        class Remote(io.BytesIO):
            def __init__(self,*args):super().__init__(wire);self.client=SimpleNamespace(close=lambda:None);self.bytes_downloaded=len(wire)
        monkeypatch.setattr(acquisition,'FreshRange',Remote)
        metadata={'license':{'name':'CC BY 4.0','url':'https://creativecommons.org/licenses/by/4.0/'},'url_public_html':'https://figshare.com/articles/dataset/example/21431547','files':[{'id':38030910,'size':len(wire),'computed_md5':'b62bd51d2ffcfaa76c60f234f0cc2bb3'}]}
        monkeypatch.setattr(acquisition.httpx,'get',lambda *a,**kw:SimpleNamespace(raise_for_status=lambda:None,json=lambda:metadata))
        yield folder,metadata

def test_complete_nested_zip_receipt_and_cached_deadline_are_preserved(case):
    folder,_=case;acquisition.fetch(['Japan'],folder)
    proof=folder/'Japan.zip.source.json';receipt=proof.read_bytes();start=(folder/'Japan.acquisition-start.json').read_bytes()
    acquisition.fetch(['Japan'],folder)
    assert proof.read_bytes()==receipt and (folder/'Japan.acquisition-start.json').read_bytes()==start
    data=json.loads(receipt);assert data['bytes']==(folder/'Japan.zip').stat().st_size and data['full_outer_md5_verified'] is False
    assert data['expires_at']==json.loads(start)['expires_at']

def test_expired_acquisition_cannot_restart_with_a_new_deadline(case):
    folder,_=case;start=folder/'Japan.acquisition-start.json';start.write_text(json.dumps({'download_started_at':'2020-01-01T00:00:00+00:00','expires_at':'2020-01-08T00:00:00+00:00'}));original=start.read_bytes()
    with pytest.raises(ValueError,match='deadline_expired'):acquisition.fetch(['Japan'],folder)
    assert start.read_bytes()==original and not (folder/'Japan.zip.source.json').exists()

def test_modified_cached_archive_is_not_replaced_or_relicensed(case):
    folder,_=case;acquisition.fetch(['Japan'],folder);proof=(folder/'Japan.zip.source.json').read_bytes()
    (folder/'Japan.zip').write_bytes(b'CHANGED')
    with pytest.raises(ValueError,match='missing_or_changed'):acquisition.fetch(['Japan'],folder)
    assert (folder/'Japan.zip').read_bytes()==b'CHANGED' and (folder/'Japan.zip.source.json').read_bytes()==proof

def test_publisher_license_change_blocks_acquisition(case):
    folder,metadata=case;metadata['license']['name']='ALL RIGHTS RESERVED'
    with pytest.raises(ValueError,match='publisher_contract_changed'):acquisition.fetch(['Japan'],folder)
    assert not (folder/'Japan.zip').exists()

@pytest.mark.parametrize('countries',[[],['Japan','Japan'],['Norway'],['../outside']])
def test_country_budget_and_traversal_inputs_refused(case,countries):
    folder,_=case
    with pytest.raises(ValueError,match='explicit_supported_country'):acquisition.fetch(countries,folder)
