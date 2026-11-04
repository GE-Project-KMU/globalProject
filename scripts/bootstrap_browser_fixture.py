"""Return ephemeral browser-test credentials on stdout for the parent test only.

Every entity is created in its own UUID tenant. Do not print this command's output
in a public report; the Playwright caller consumes it in memory.
"""
import json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import httpx
from scripts.http_fixture_identity import isolated_identity
from tests.test_http import env_values,setup

def main():
    settings=env_values();identity=isolated_identity(settings,prefix='browser-test')
    with httpx.Client(base_url=os.getenv('ROAD_TEST_BASE_URL','http://127.0.0.1:8000'),timeout=30) as client:
        r=client.post('/api/v1/auth/login',json=identity);r.raise_for_status();headers={'Authorization':'Bearer '+r.json()['data']['access_token']}
        fixture=setup.__wrapped__(client,headers)
        r=client.post('/api/v1/observations',headers=fixture['dh'],json=fixture['observation']);r.raise_for_status()
        client.post('/api/v1/auth/logout',headers=headers)
    print(json.dumps(identity))
if __name__=='__main__':main()
