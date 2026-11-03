
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
def public_user(x):return {k:x[k] for k in ("id","login","role","active")}
@app.get("/api/v1/users")
def users(req:Request):
    p=principal(req);role(p,"admin");return ok({"items":[public_user(x) for x in store.list(p["org_id"],"user")]})
@app.post("/api/v1/users",status_code=201)
async def create_user(req:Request):
    p=principal(req);role(p,"admin");b=await req.json()
    if set(b)-{"login","password","role"} or not 1<=len(b.get("login",""))<=254 or not 12<=len(b.get("password",""))<=128 or b.get("role") not in ("admin","reviewer","viewer"):fail(422,"VALIDATION_ERROR")
    if any(x["login"]==b["login"] for x in store.list(p["org_id"],"user")):fail(409,"LOGIN_EXISTS")
    x=store.create(uid(),p["org_id"],"user",{"login":b["login"],"password_hash":ph.hash(b["password"]),"role":b["role"],"active":True})
    store.audit(p,"user.create",x["id"]);return ok(public_user(x))
@app.patch("/api/v1/users/{id}")
async def change_user(id:str,req:Request):
    p=principal(req);role(p,"admin");b=await req.json()
    if not b or set(b)-{"role","active","password"} or ("role" in b and b["role"] not in ("admin","reviewer","viewer")) or ("active" in b and not isinstance(b["active"],bool)) or ("password" in b and not 12<=len(b["password"])<=128):fail(422,"VALIDATION_ERROR")
    with store.tx() as s:
        x=store.get(id,p["org_id"],"user",s)
        if x["role"]=="admin" and x["active"] and (b.get("active") is False or b.get("role",x["role"])!="admin"):
            if sum(u["active"] and u["role"]=="admin" for u in store.list(p["org_id"],"user",s))<=1:fail(409,"LAST_ADMIN")
        payload={k:x[k] for k in ("login","password_hash","role","active")}
        payload.update({k:v for k,v in b.items() if k!="password"})
        if "password" in b:payload["password_hash"]=ph.hash(b["password"])
        x=store.change(id,p["org_id"],payload,x["version"],s)
        for session in store.list(p["org_id"],"session",s):
            if session["user_id"]==id:store.delete(session["id"],p["org_id"],s)
        store.audit(p,"user.update",id,s)
    return ok(public_user(x))
@app.get("/api/v1/devices")
def devices(req:Request):
    p=principal(req);role(p,"admin","reviewer");return ok({"items":[{k:v for k,v in x.items() if k!="credential_id"} for x in store.list(p["org_id"],"device")]})
@app.get("/internal/devices/{id}")
def device_reference(id:str,req:Request):
    p=principal(req)
    if p["role"]=="device" and p["sub"]!=id:fail(403,"FORBIDDEN")
    x=store.get(id,p["org_id"],"device")
    if not x["active"]:fail(403,"DEVICE_INACTIVE")
    return ok({"id":id,"name":x["name"],"active":x["active"]})
@app.post("/api/v1/devices",status_code=201)
async def new_device(req:Request):
    p=principal(req);role(p,"admin");b=await req.json()
    if set(b)!={"name"} or not isinstance(b["name"],str) or not 1<=len(b["name"])<=100:fail(422,"VALIDATION_ERROR")
    id=uid();token=secrets.token_urlsafe(32);tid=digest(token)
    with store.tx() as s:
        store.create(tid,p["org_id"],"device_credential",{"device_id":id},s=s)
        store.create(id,p["org_id"],"device",{"name":b["name"],"active":True,"credential_id":tid},s=s)
        store.audit(p,"device.create",id,s)
    return ok({"device_id":id,"device_token":token,"name":b["name"],"active":True})
@app.patch("/api/v1/devices/{id}")
async def update_device(id:str,req:Request):
    p=principal(req);role(p,"admin");b=await req.json()
    if not b or set(b)-{"name","active"} or ("active" in b and not isinstance(b["active"],bool)) or ("name" in b and not 1<=len(b["name"])<=100):fail(422,"VALIDATION_ERROR")
    x=store.get(id,p["org_id"],"device");payload={k:x[k] for k in ("name","active","credential_id")};payload.update(b)
    x=store.change(id,p["org_id"],payload,x["version"]);return ok({k:v for k,v in x.items() if k!="credential_id"})
@app.post("/api/v1/devices/{id}/credentials")
async def rotate(id:str,req:Request):
    p=principal(req);role(p,"admin");b=await req.json()
    if not 1<=len(b.get("reason",""))<=500:fail(422,"REASON_REQUIRED")
    x=store.get(id,p["org_id"],"device");token=secrets.token_urlsafe(32);tid=digest(token)
    with store.tx() as s:
        store.delete(x["credential_id"],p["org_id"],s)
        store.create(tid,p["org_id"],"device_credential",{"device_id":id},s=s)
        store.change(id,p["org_id"],{"name":x["name"],"active":x["active"],"credential_id":tid},x["version"],s)
        store.audit(p,"device.rotate",id,s)
    return ok({"device_id":id,"device_token":token})
