
import hashlib, os, secrets
from argon2 import PasswordHasher
from argon2.exceptions import VerificationError
from fastapi import Request, Response
from pydantic import BaseModel, ConfigDict, Field
from common.runtime import *
store=Store("identity");app=app_for("identity");ph=PasswordHasher()
class Login(BaseModel):
    model_config=ConfigDict(extra="forbid")
    org_code:str=Field(min_length=1,max_length=64)
    login:str=Field(min_length=1,max_length=254)
    password:str=Field(min_length=1,max_length=128)
def bootstrap(org_code,login,password):
    if not password or len(password)<12:raise RuntimeError("Bootstrap password must be >=12 chars")
    org_id=str(uuid.uuid5(uuid.NAMESPACE_DNS,"roadwatch."+org_code))
    if any(x["code"]==org_code for x in store.list(None,"organization")):return org_id
    store.create(org_id,org_id,"organization",{"code":org_code,"name":org_code,"active":True})
    store.create(uid(),org_id,"user",{"login":login,"password_hash":ph.hash(password),"role":"admin","active":True})
    return org_id
if os.getenv("ROAD_BOOTSTRAP_PASSWORD"):bootstrap(os.getenv("ROAD_BOOTSTRAP_ORG","demo"),os.getenv("ROAD_BOOTSTRAP_LOGIN","admin@road.local"),os.environ["ROAD_BOOTSTRAP_PASSWORD"])
@app.post("/api/v1/auth/login")
def login(b:Login):
    orgs=[x for x in store.list(None,"organization") if x["code"]==b.org_code and x["active"]]
    user=next((x for x in store.list(orgs[0]["id"],"user") if x["login"]==b.login and x["active"]),None) if orgs else None
    try:
        if not user:ph.verify(ph.hash("invalid-login-timing-padding"),b.password);raise VerificationError()
        ph.verify(user["password_hash"],b.password)
    except VerificationError:fail(401,"INVALID_CREDENTIALS")
    token=secrets.token_urlsafe(32)
    store.create(digest(token),user["org_id"],"session",{"user_id":user["id"]},now()+1800)
    return ok({"access_token":token,"token_type":"Bearer","expires_in":1800,"user":{"id":user["id"],"role":user["role"],"login":user["login"]}})
@app.post("/internal/introspect")
async def introspect(req:Request):
    b=await req.json();token=b.get("token","")
    try:r=store.get(digest(token),expired=False)
    except HTTPException:fail(401,"UNAUTHENTICATED")
    if r["kind"]=="session":
        user=store.get(r["user_id"],r["org_id"],"user")
        if not user["active"]:fail(401,"UNAUTHENTICATED")
        return ok({"sub":user["id"],"org_id":r["org_id"],"role":user["role"],"session_id":r["id"],"login":user["login"]})
    if r["kind"]=="device_credential":
        device=store.get(r["device_id"],r["org_id"],"device")
        if not device["active"] or device["credential_id"]!=r["id"]:fail(401,"UNAUTHENTICATED")
        return ok({"sub":device["id"],"device_id":device["id"],"org_id":r["org_id"],"role":"device"})
    fail(401,"UNAUTHENTICATED")
@app.get("/api/v1/me")
def me(req:Request):
    p=principal(req);return ok({k:v for k,v in p.items() if k not in ("exp","aud","session_id")})
@app.post("/api/v1/auth/logout",status_code=204)
def logout(req:Request):
    p=principal(req)
    if p.get("session_id"):store.delete(p["session_id"],p["org_id"])
    return Response(status_code=204)
