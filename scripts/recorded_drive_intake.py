"""Read-only byte inventory for user recordings; never SDK playback or retention.

Paths/names/individual hashes/mtime are private data. Public output contains only
aggregate byte/format/integrity counts and the actual unverified playback state.
No capture time, camera identity, calibration or expiry is inferred from names or
filesystem times. Source files are never moved, modified, deleted or registered
for automatic retention by this tool.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib,json,os,stat,uuid
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

class IntakeError(ValueError):pass

def no_links(path:Path,workspace:Path):
    if not path.is_relative_to(workspace):raise IntakeError('PATH_OUTSIDE_WORKSPACE')
    try:info=workspace.lstat()
    except OSError:raise IntakeError('SOURCE_UNREADABLE') from None
    if stat.S_ISLNK(info.st_mode) or workspace.is_junction() or getattr(info,'st_file_attributes',0)&0x400:raise IntakeError('LINK_OR_REPARSE_POINT_FORBIDDEN')
    current=workspace
    for part in path.relative_to(workspace).parts:
        current=current/part
        try:info=current.lstat()
        except OSError:raise IntakeError('SOURCE_UNREADABLE') from None
        if stat.S_ISLNK(info.st_mode) or current.is_junction() or getattr(info,'st_file_attributes',0)&0x400:raise IntakeError('LINK_OR_REPARSE_POINT_FORBIDDEN')

def scoped(value:str|Path,workspace:Path=ROOT,*,directory=False)->Path:
    workspace=workspace.absolute();candidate=Path(value)
    if '..' in candidate.parts:raise IntakeError('PARENT_TRAVERSAL_FORBIDDEN')
    candidate=candidate if candidate.is_absolute() else workspace/candidate
    candidate=candidate.absolute();no_links(candidate,workspace)
    if candidate.resolve()!=candidate:raise IntakeError('CANONICAL_PATH_MISMATCH')
    if directory and not candidate.is_dir():raise IntakeError('SOURCE_DIRECTORY_REQUIRED')
    return candidate

def enumerate_sources(source:Path,workspace:Path=ROOT)->list[Path]:
    source=scoped(source,workspace,directory=True);files=[];pending=[source]
    while pending:
        directory=pending.pop();no_links(directory,workspace)
        try:children=list(directory.iterdir())
        except OSError:raise IntakeError('SOURCE_UNREADABLE') from None
        for child in children:
            no_links(child,workspace)
            if child.is_dir():pending.append(child)
            elif child.suffix.lower()=='.svo2':
                if not stat.S_ISREG(child.lstat().st_mode):raise IntakeError('REGULAR_FILE_REQUIRED')
                files.append(child)
    return sorted(files,key=lambda p:(p.relative_to(source).as_posix().casefold(),p.relative_to(source).as_posix()))

def identity(info):return (info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns)

def change_token(info):return (*identity(info),getattr(info,'st_ctime_ns',None))
def read_chunk(stream,size):return stream.read(size)

def hash_file(path:Path,workspace:Path=ROOT,*,chunk_size=4*1024*1024):
    path=scoped(path,workspace);before=path.lstat()
    if not stat.S_ISREG(before.st_mode):raise IntakeError('REGULAR_FILE_REQUIRED')
    flags=os.O_RDONLY|getattr(os,'O_BINARY',0)|getattr(os,'O_NOFOLLOW',0)
    try:
        descriptor=os.open(path,flags)
        with os.fdopen(descriptor,'rb') as stream:
            opened=os.fstat(stream.fileno())
            if identity(opened)!=identity(before):raise IntakeError('SOURCE_CHANGED_WHILE_HASHING')
            h=hashlib.sha256();count=0
            while chunk:=read_chunk(stream,chunk_size):h.update(chunk);count+=len(chunk)
            after=os.fstat(stream.fileno())
        no_links(path,workspace);current=path.lstat()
        # Windows lstat/fstat can expose different ctime semantics. Compare
        # identity across APIs, and ctime only between calls of the same API.
        if identity(before)!=identity(after) or identity(after)!=identity(current) or change_token(opened)!=change_token(after) or change_token(before)!=change_token(current) or count!=before.st_size:raise IntakeError('SOURCE_CHANGED_WHILE_HASHING')
    except OSError:raise IntakeError('SOURCE_UNREADABLE_OR_REPLACED') from None
    return {'sha256':h.hexdigest(),'size_bytes':count,'filesystem_mtime_ns_auxiliary_only':before.st_mtime_ns,'filesystem_mtime_utc_auxiliary_only':datetime.fromtimestamp(before.st_mtime,timezone.utc).isoformat(),'_identity':change_token(before)}

def collect(source:Path,workspace:Path=ROOT):
    source=scoped(source,workspace,directory=True);files=enumerate_sources(source,workspace)
    if not files:raise IntakeError('NO_SVO2_RECORDINGS')
    rows=[]
    for path in files:
        item=hash_file(path,workspace)
        rows.append({'source_path':path.relative_to(workspace).as_posix(),'source_relative_name':path.relative_to(source).as_posix(),'declared_extension':'.svo2',**item,'capture_time':None,'capture_time_status':'unknown','original_expiry':None,'retention_status':'unknown_do_not_auto_register','camera_serial':None,'camera_serial_status':'not_verified_from_filename','content_format_verified':False,'frame_count':None,'playback_status':'not_verified'})
    if files!=enumerate_sources(source,workspace):raise IntakeError('SOURCE_SET_CHANGED_DURING_INTAKE')
    for path,row in zip(files,rows):
        no_links(path,workspace)
        if change_token(path.lstat())!=row.pop('_identity'):raise IntakeError('SOURCE_CHANGED_DURING_INTAKE')
    groups={}
    for row in rows:groups.setdefault(row['sha256'],[]).append(row['source_path'])
    duplicates=[{'sha256':key,'source_paths':paths} for key,paths in sorted(groups.items()) if len(paths)>1]
    public={'schema_version':1,'recordings':len(rows),'total_bytes':sum(r['size_bytes'] for r in rows),'declared_formats':{'.svo2':len(rows)},'content_format_verified':False,'sha256_inventory_complete':True,'distinct_byte_contents':len(groups),'duplicate_sha256_groups':len(duplicates),'duplicate_extra_files':sum(len(x['source_paths'])-1 for x in duplicates),'playback_status':'not_verified','sdk_reader_executed':False,'svo_frames_decoded':None,'capture_time_known':False,'original_expiry_known':False}
    return rows,duplicates,public

def write_json(path,value):
    temporary=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('x',encoding='utf-8') as stream:json.dump(value,stream,ensure_ascii=False,allow_nan=False,indent=2);stream.write('\n')
        os.replace(temporary,path)
    finally:temporary.unlink(missing_ok=True)

def intake(source:Path,output_root:Path,*,public_report:Path|None=None,workspace:Path=ROOT):
    workspace=workspace.absolute();source=scoped(source,workspace,directory=True)
    # Validate the existing output parent before creating any private metadata.
    allowed=workspace/'data/recorded-drive-intake'
    output_root=output_root if output_root.is_absolute() else workspace/output_root
    output_root=output_root.absolute()
    if output_root!=allowed:raise IntakeError('PRIVATE_INTAKE_ROOT_REQUIRED')
    scoped(workspace/'data',workspace,directory=True)
    if os.path.lexists(output_root):scoped(output_root,workspace,directory=True)
    if source==output_root or output_root.is_relative_to(source):raise IntakeError('OUTPUT_INSIDE_SOURCE_FORBIDDEN')
    if public_report is not None:
        public_report=public_report if public_report.is_absolute() else workspace/public_report
        if '..' in public_report.parts or not public_report.absolute().is_relative_to(workspace/'docs/reports'):raise IntakeError('PUBLIC_AGGREGATE_LOCATION_REQUIRED')
        scoped(public_report.parent,workspace,directory=True)
        if os.path.lexists(public_report):raise IntakeError('PUBLIC_REPORT_ALREADY_EXISTS')
    rows,duplicates,public=collect(source,workspace)
    identifier=str(uuid.uuid4());destination=output_root/identifier
    output_root.mkdir(exist_ok=True);scoped(output_root,workspace,directory=True);destination.mkdir()
    manifest={'schema_version':1,'intake_id':identifier,'inventory_created_at':datetime.now(timezone.utc).isoformat(),'source_directory':source.relative_to(workspace).as_posix(),'private':True,'source_bytes_written':False,'sources_moved_or_deleted':False,'filesystem_times_are_not_capture_times':True,'capture_time':None,'original_expiry':None,'retention_registration_performed':False,'retention_warning':'Original capture time/deadline is unknown. Do not derive either from filename, mtime or intake date; no automatic retention/deletion registration.','sdk_reader_executed':False,'playback_status':'not_verified','recordings':rows,'duplicate_contents':duplicates}
    write_json(destination/'private-manifest.json',manifest)
    write_json(destination/'aggregate.json',public)
    if public_report is not None:write_json(public_report,public)
    return destination,public

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',default='데이터');parser.add_argument('--public-report');args=parser.parse_args()
    try:
        _,summary=intake(Path(args.source),ROOT/'data/recorded-drive-intake',public_report=Path(args.public_report) if args.public_report else None)
        print(json.dumps(summary,ensure_ascii=True));return 0
    except (IntakeError,OSError) as error:
        reason=str(error) if isinstance(error,IntakeError) else 'FILESYSTEM_OPERATION_FAILED'
        print(json.dumps({'status':'intake_refused','reason':reason,'source_bytes_written':False}));return 2

if __name__=='__main__':raise SystemExit(main())
