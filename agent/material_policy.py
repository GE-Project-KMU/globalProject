"""Local review material TTLs. No capture time or source deadline is guessed."""
from datetime import datetime,timezone,timedelta
from pathlib import Path
from .paths import ROOT,workspace_path

def instant(value):
    if not isinstance(value,str):raise ValueError('explicit_aware_time_required')
    try:date=datetime.fromisoformat(value.replace('Z','+00:00'))
    except ValueError:raise ValueError('explicit_aware_time_required') from None
    if date.tzinfo is None:raise ValueError('explicit_aware_time_required')
    return date.astimezone(timezone.utc)

def current():return datetime.now(timezone.utc)

def source_deadline(value,*,required=False,now=None):
    if value is None:
        if required:raise ValueError('explicit_source_expiry_required_before_reading_pixels')
        return None
    deadline=instant(value)
    if deadline<=(now or current()):raise ValueError('source_material_expired')
    return deadline

def capture_time(value,*,now=None):
    date=instant(value);now=now or current()
    if date>now+timedelta(seconds=2):raise ValueError('captured_at_in_future')
    return date

def expiry(captured,*,approved=False,source_expires_at=None,now=None):
    now=now or current();date=capture_time(captured,now=now)
    deadline=date+timedelta(days=90 if approved else 7)
    if source_expires_at is not None:deadline=min(deadline,instant(source_expires_at))
    if deadline<=now:raise ValueError('local_material_expired')
    return deadline.isoformat()

def guarded_samples(samples,expires_at):
    iterator=iter(samples)
    try:
        while True:
            source_deadline(expires_at,required=True)
            try:sample=next(iterator)
            except StopIteration:return
            source_deadline(expires_at,required=True)
            yield sample
    finally:
        close=getattr(iterator,'close',None)
        if close:close()

def safe_file(value,folder,*,exists=True,suffixes=('.png','.jpg','.jpeg')):
    lexical=Path(value)
    if '..' in lexical.parts:raise ValueError('unsafe_material_path')
    lexical=lexical if lexical.is_absolute() else ROOT/lexical
    # Inspect links before resolve, including Windows junctions.
    cursor=lexical
    while cursor!=ROOT:
        if not cursor.is_relative_to(ROOT):raise ValueError('unsafe_material_path')
        if cursor.is_symlink() or getattr(cursor,'is_junction',lambda:False)():raise ValueError('material_links_forbidden')
        cursor=cursor.parent
    path=workspace_path(lexical,exists=exists);folder=workspace_path(folder)
    if path.parent!=folder or path.suffix.lower() not in suffixes:raise ValueError('material_file_outside_owned_folder')
    return path
