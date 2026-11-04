"""Provision an isolated HTTP-test tenant through the supported identity CLI.

Never uses demo credentials and never deletes existing organizations or records.
Returned secrets belong in memory only and must not be printed.
"""
import os
from pathlib import Path
import secrets
import subprocess
import sys
from uuid import uuid4

ROOT=Path(__file__).resolve().parents[1]

def isolated_identity(settings,*,prefix='http-test'):
    org=prefix+'-'+uuid4().hex
    login='admin@http-test.local';password=secrets.token_urlsafe(32)
    env={**os.environ,**settings,'ROAD_BOOTSTRAP_PASSWORD':'','ROAD_HTTP_TEST_PASSWORD':password,'PYTHONPATH':str(ROOT),'PYTHONPYCACHEPREFIX':str(ROOT/'.cache/pycache')}
    result=subprocess.run([sys.executable,'-m','services.identity.bootstrap','--org',org,'--login',login,'--password-env','ROAD_HTTP_TEST_PASSWORD'],cwd=ROOT,env=env,capture_output=True,text=True,timeout=30)
    if result.returncode:raise RuntimeError('Isolated HTTP-test identity bootstrap failed')
    return {'org_code':org,'login':login,'password':password}
