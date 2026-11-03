
import argparse, os
from services.identity.app import bootstrap
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--org",required=True);p.add_argument("--login",required=True);p.add_argument("--password-env",default="ROAD_BOOTSTRAP_PASSWORD");a=p.parse_args()
    print(bootstrap(a.org,a.login,os.environ[a.password_env]))
