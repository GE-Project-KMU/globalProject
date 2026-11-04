"""Read-only HTTP smoke checks after an isolated Compose build/start and seed."""
import argparse,json,re,sys
from pathlib import Path
from urllib.parse import urlsplit
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import httpx
from scripts.manage import load_env
from agent.paths import atomic_json,workspace_path

def run(base_url='http://127.0.0.1:8080',output='data/compose-probe-results.json'):
    parts=urlsplit(base_url)
    if parts.scheme!='http' or parts.hostname not in ('127.0.0.1','localhost') or parts.username or parts.password or parts.path not in ('','/') or parts.query or parts.fragment:raise ValueError('isolated_loopback_http_origin_required')
    env=load_env();checks=[]
    with httpx.Client(base_url=base_url,timeout=20) as client:
        def get(path,headers=None):
            result=client.get(path,headers=headers)
            if result.status_code!=200:raise RuntimeError('compose_http_smoke_failed')
            return result
        page=get('/');assert 'text/html' in page.headers.get('content-type','')
        assets=re.findall(r'(?:src|href)="(/assets/[^"<>]+)"',page.text)
        assert assets,'production_bundle_assets_required'
        for path in assets:
            asset=get(path);assert len(asset.content)>0
        checks.extend(['nginx_frontend_html','production_bundle_assets'])
        health=get('/health').json();assert health.get('status')=='ok';checks.append('nginx_gateway_health_proxy')
        unauth=client.get('/api/v1/me');assert unauth.status_code==401 and 'error' in unauth.json();checks.append('anonymous_request_denied')
        login=client.post('/api/v1/auth/login',json={'org_code':'demo','login':'admin@road.local','password':env['ROAD_BOOTSTRAP_PASSWORD']})
        if login.status_code!=200:raise RuntimeError('compose_login_failed')
        auth={'Authorization':'Bearer '+login.json()['data']['access_token']}
        assert get('/api/v1/me',auth).json()['data']['role']=='admin';checks.append('authenticated_nginx_gateway_identity_route')
        observations=get('/api/v1/observations',auth).json()['data']['items'];assert len(observations)>=9
        assert get('/api/v1/summary',auth).json()['data']['total']>=9
        checks.append('postgres_observation_and_summary_routes')
        path=observations[0]['image_content_path'];assert path.startswith('/api/v1/images/') and path.endswith('/content')
        image=get(path,auth);assert image.headers.get('content-type','').startswith('image/jpeg') and image.content[:2]==b'\xff\xd8' and image.content[-2:]==b'\xff\xd9'
        denied=client.get(path);assert denied.status_code==401;checks.append('private_media_bytes_and_anonymous_denial')
        bad=client.get('/api/v1/observations/not-a-uuid',headers=auth);assert bad.status_code==422 and 'error' in bad.json();checks.append('invalid_input_error_envelope')
    result={'format':'compose-runtime-smoke-v1','checks_passed':checks,'production_assets_checked':len(assets),'http_smoke_passed':True,'synthetic_seed_only':True,'field_acceptance':False,'scope':'Remote isolated CI Compose frontend/gateway/six MSA services/five PostgreSQL stores; Windows Docker availability is separate.'}
    atomic_json(workspace_path(output),result);print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--base-url',default='http://127.0.0.1:8080');parser.add_argument('--output',default='data/compose-probe-results.json');args=parser.parse_args();run(args.base_url,args.output)
