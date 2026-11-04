"""Read-only provenance views. Original observations and reviews are immutable.

Legacy HTTP fixtures require exact organization/run/device/profile/model IDs from
an operator-generated, cross-checked presentation policy; no fuzzy user-data rule.
"""
import json
import os
from pathlib import Path
import re
from typing import Literal
from common.runtime import DATA,ROOT,digest,fail

SCOPES={'all','presentation','test','field','synthetic'}
DataScope=Literal['all','presentation','test','field','synthetic']
KEYS=('org_id','run_id','device_id','profile_id','model_version')

def policy():
    path=Path(os.getenv('ROAD_PRESENTATION_POLICY',str(DATA/'presentation/policy.json'))).resolve()
    if not path.is_relative_to(ROOT):fail(503,'PRESENTATION_POLICY_UNAVAILABLE')
    if not path.exists():return [],digest([])
    try:
        if path.stat().st_size>2_000_000:raise ValueError()
        raw=path.read_bytes();value=json.loads(raw)
        if value.get('schema_version')!=1 or not isinstance(value.get('legacy_http_fixtures'),list):raise ValueError()
        rows=value['legacy_http_fixtures']
        if len(rows)>10000 or any(not isinstance(r,dict) or any(not isinstance(r.get(k),str) or not r[k] for k in KEYS) or r.get('reason_code')!='legacy_http_fixture_crosschecked' for r in rows):raise ValueError()
        return rows,digest(raw)
    except (ValueError,OSError,TypeError):fail(503,'PRESENTATION_POLICY_UNAVAILABLE')

def data_origin(body,org_id,rows=None):
    if rows is None:rows,_=policy()
    if any(all((org_id if k=='org_id' else body.get(k))==row[k] for k in KEYS) for row in rows):return 'http_test'
    version=body.get('model_version') if isinstance(body.get('model_version'),str) else ''
    if body.get('demo') and re.fullmatch(r'http-[0-9a-f]{10}',version):return 'http_test'
    if body.get('demo') or body.get('synthetic'):return 'synthetic_demo'
    return 'field'

def validate_scope(scope):
    if scope not in SCOPES:fail(422,'INVALID_DATA_SCOPE')

def included(origin,scope):
    validate_scope(scope)
    return scope=='all' or (scope=='presentation' and origin!='http_test') or (scope=='test' and origin=='http_test') or (scope=='field' and origin=='field') or (scope=='synthetic' and origin=='synthetic_demo')
