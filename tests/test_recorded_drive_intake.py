import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import recorded_drive_intake as tool


@pytest.fixture
def workspace(tmp_path):
    (tmp_path/'data').mkdir()
    (tmp_path/'docs/reports').mkdir(parents=True)
    (tmp_path/'recordings').mkdir()
    return tmp_path


def sample(workspace,name='a.svo2',content=b'not a decoded video'):
    path=workspace/'recordings'/name
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(content)
    return path


def test_sorted_recursive_inventory_duplicates_and_private_public_boundary(workspace):
    sample(workspace,'Z.svo2',b'same')
    sample(workspace,'nested/B.SVO2',b'same')
    sample(workspace,'a-camera-private.svo2',b'other')
    sample(workspace,'ignored.txt',b'ignored')
    destination,public=tool.intake(workspace/'recordings',workspace/'data/recorded-drive-intake',public_report=workspace/'docs/reports/intake.json',workspace=workspace)
    private=json.loads((destination/'private-manifest.json').read_text(encoding='utf-8'))
    assert [r['source_relative_name'] for r in private['recordings']]==['a-camera-private.svo2','nested/B.SVO2','Z.svo2']
    assert public['recordings']==3 and public['total_bytes']==13
    assert public['duplicate_sha256_groups']==1 and public['duplicate_extra_files']==1
    assert public['distinct_byte_contents']==2
    assert json.loads((workspace/'docs/reports/intake.json').read_text())==public
    serialized=json.dumps(public)
    assert 'camera-private' not in serialized and 'source_path' not in serialized
    assert not any(r['sha256'] in serialized for r in private['recordings'])
    assert public['playback_status']=='not_verified' and public['sdk_reader_executed'] is False
    assert public['svo_frames_decoded'] is None and public['content_format_verified'] is False
    assert private['retention_registration_performed'] is False


def test_mtime_never_becomes_capture_time_or_expiry_and_source_unchanged(workspace):
    path=sample(workspace)
    os.utime(path,(946684800,946684800))
    before=path.read_bytes(),path.stat().st_mtime_ns
    rows,_,_=tool.collect(workspace/'recordings',workspace)
    assert rows[0]['capture_time'] is None and rows[0]['original_expiry'] is None
    assert rows[0]['camera_serial'] is None
    assert rows[0]['filesystem_mtime_utc_auxiliary_only'].startswith('2000-01-01')
    assert (path.read_bytes(),path.stat().st_mtime_ns)==before


@pytest.mark.parametrize('value',['../outside','recordings/../data'])
def test_parent_traversal_refused(workspace,value):
    with pytest.raises(tool.IntakeError,match='PARENT_TRAVERSAL'):
        tool.scoped(value,workspace)


def test_absolute_outside_refused_without_reading(workspace):
    with pytest.raises(tool.IntakeError,match='PATH_OUTSIDE'):
        tool.scoped(workspace.parent/'does-not-exist',workspace)


def test_junction_guard_without_following_target(workspace,monkeypatch):
    path=sample(workspace)
    original=Path.is_junction
    monkeypatch.setattr(Path,'is_junction',lambda self:self==path.parent or original(self))
    with pytest.raises(tool.IntakeError,match='LINK_OR_REPARSE'):
        tool.collect(path.parent,workspace)


def test_symbolic_link_refused(workspace):
    path=sample(workspace)
    link=workspace/'recordings/link.svo2'
    try:link.symlink_to(path)
    except OSError:pytest.skip('OS account cannot create symlinks; junction detection tested separately')
    with pytest.raises(tool.IntakeError,match='LINK_OR_REPARSE'):
        tool.collect(path.parent,workspace)


def test_changing_file_during_hash_refused(workspace,monkeypatch):
    path=sample(workspace,content=b'abcd')
    original=tool.read_chunk
    changed=False
    def mutate(stream,size):
        nonlocal changed
        result=original(stream,size)
        if result and not changed:
            changed=True
            with path.open('ab') as output:output.write(b'x')
        return result
    monkeypatch.setattr(tool,'read_chunk',mutate)
    with pytest.raises(tool.IntakeError,match='SOURCE_CHANGED_WHILE_HASHING'):
        tool.hash_file(path,workspace,chunk_size=2)


def test_windows_cross_api_ctime_difference_is_not_a_false_change(workspace,monkeypatch):
    path=sample(workspace)
    original=tool.os.fstat
    def alternate_ctime(fd):
        info=original(fd)
        return SimpleNamespace(**{key:getattr(info,key) for key in ('st_dev','st_ino','st_size','st_mtime_ns')},st_ctime_ns=1)
    monkeypatch.setattr(tool.os,'fstat',alternate_ctime)
    assert tool.hash_file(path,workspace)['size_bytes']==len(path.read_bytes())


def test_same_api_ctime_change_is_rejected(workspace,monkeypatch):
    path=sample(workspace)
    original=tool.os.fstat
    calls=0
    def changing_ctime(fd):
        nonlocal calls
        calls+=1
        info=original(fd)
        return SimpleNamespace(**{key:getattr(info,key) for key in ('st_dev','st_ino','st_size','st_mtime_ns')},st_ctime_ns=calls)
    monkeypatch.setattr(tool.os,'fstat',changing_ctime)
    with pytest.raises(tool.IntakeError,match='SOURCE_CHANGED_WHILE_HASHING'):
        tool.hash_file(path,workspace)


def test_change_after_hash_refused_and_no_partial_inventory(workspace,monkeypatch):
    path=sample(workspace)
    original=tool.hash_file
    def mutate(*args,**kwargs):
        result=original(*args,**kwargs)
        path.write_bytes(b'changed after hash')
        return result
    monkeypatch.setattr(tool,'hash_file',mutate)
    with pytest.raises(tool.IntakeError,match='SOURCE_CHANGED_DURING_INTAKE'):
        tool.intake(path.parent,workspace/'data/recorded-drive-intake',workspace=workspace)
    assert not (workspace/'data/recorded-drive-intake').exists()


def test_added_file_refused(workspace,monkeypatch):
    path=sample(workspace)
    original=tool.hash_file
    def add(*args,**kwargs):
        result=original(*args,**kwargs)
        sample(workspace,'new.svo2')
        return result
    monkeypatch.setattr(tool,'hash_file',add)
    with pytest.raises(tool.IntakeError,match='SOURCE_SET_CHANGED'):
        tool.collect(path.parent,workspace)


def test_empty_input_and_wrong_output_root_refused(workspace):
    with pytest.raises(tool.IntakeError,match='NO_SVO2'):
        tool.collect(workspace/'recordings',workspace)
    sample(workspace)
    with pytest.raises(tool.IntakeError,match='PRIVATE_INTAKE_ROOT'):
        tool.intake(workspace/'recordings',workspace/'recordings/out',workspace=workspace)


def test_public_report_cannot_overwrite_or_escape_aggregate_directory(workspace):
    sample(workspace)
    report=workspace/'docs/reports/existing.json'
    report.write_text('keep')
    with pytest.raises(tool.IntakeError,match='PUBLIC_REPORT_ALREADY_EXISTS'):
        tool.intake(workspace/'recordings',workspace/'data/recorded-drive-intake',public_report=report,workspace=workspace)
    assert report.read_text()=='keep'
    with pytest.raises(tool.IntakeError,match='PUBLIC_AGGREGATE_LOCATION'):
        tool.intake(workspace/'recordings',workspace/'data/recorded-drive-intake',public_report=workspace/'private.json',workspace=workspace)


def test_repeated_receipts_have_unique_destinations(workspace):
    path=sample(workspace)
    first,_=tool.intake(path.parent,workspace/'data/recorded-drive-intake',workspace=workspace)
    second,_=tool.intake(path.parent,workspace/'data/recorded-drive-intake',workspace=workspace)
    assert first!=second and first.is_dir() and second.is_dir()
    assert path.read_bytes()==b'not a decoded video'


def test_recording_and_private_inventory_ignored():
    patterns=(tool.ROOT/'.gitignore').read_text().splitlines()
    assert '*.svo2' in patterns and 'data/' in patterns
