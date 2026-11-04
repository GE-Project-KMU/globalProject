"""Cross-check legacy HTTP fixtures via APIs; add exact read-only presentation rules.

No observation, image, case, user or historical payload is changed or deleted.
"""
import argparse,json,sys
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import httpx
from agent.paths import atomic_json,sha256
from scripts.manage import load_env

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    env=load_env();rows=[]
    with httpx.Client(base_url=env.get('ROAD_TEST_BASE_URL','http://127.0.0.1:8000'),timeout=30) as client:
        def req(path):
            r=client.get('/api/v1'+path,headers=headers);r.raise_for_status();return r.json()['data']
        login=client.post('/api/v1/auth/login',json={'org_code':env.get('ROAD_BOOTSTRAP_ORG','demo'),'login':env.get('ROAD_BOOTSTRAP_LOGIN','admin@road.local'),'password':env['ROAD_BOOTSTRAP_PASSWORD']});login.raise_for_status()
        headers={'Authorization':'Bearer '+login.json()['data']['access_token']};me=req('/me')
        devices={r['id']:r for r in req('/devices')['items']};profiles={r['id']:r for r in req('/profiles')['items']};models={r['version']:r for r in req('/model-bundles')['items']}
        observations=[];cursor=None
        while True:
            result=req('/observations?limit=200'+('&cursor='+cursor if cursor else ''));observations+=result['items'];cursor=result.get('next_cursor')
            if not cursor:break
        seen=set()
        for obs in observations:
            device=devices.get(obs['device_id'],{});profile=profiles.get(obs['profile_id'],{});model=models.get(obs['model_version'],{})
            if not(device.get('name','').startswith('HTTP synthetic ') and profile.get('name','').startswith('HTTP synthetic ') and profile.get('camera_serial')=='SYNTHETIC-NOT-A-CAMERA' and model.get('code_commit')=='synthetic-contract-test' and obs['model_version'].startswith('http-')):continue
            key=(obs['run_id'],obs['device_id'],obs['profile_id'],obs['model_version'])
            if key in seen:continue
            seen.add(key);rows.append(dict(zip(('run_id','device_id','profile_id','model_version'),key),org_id=me['org_id'],reason_code='legacy_http_fixture_crosschecked'))
        client.post('/api/v1/auth/logout',headers=headers)
    path=Path(env.get('ROAD_DATA_DIR',str(ROOT/'data')))/'presentation/policy.json';path=path.resolve()
    if not path.is_relative_to(ROOT):raise ValueError('policy_outside_workspace')
    if path.exists():
        old=json.loads(path.read_text());rows=list({tuple(r[k] for k in ('org_id','run_id','device_id','profile_id','model_version')):r for r in old['legacy_http_fixtures']+rows}.values())
    if args.apply:
        atomic_json(path,{'schema_version':1,'created_at':datetime.now(timezone.utc).isoformat(),'legacy_http_fixtures':rows,'original_observations_modified':False,'classification_basis':'exact API metadata: device+profile HTTP synthetic, SYNTHETIC camera serial, synthetic-contract-test model commit; all original IDs retained'})
    print(json.dumps({'dry_run':not args.apply,'exact_legacy_test_runs':len(rows),'observations_examined':len(observations),'original_observations_modified':False,'policy_sha256':sha256(path) if args.apply else None}))

if __name__=='__main__':main()
