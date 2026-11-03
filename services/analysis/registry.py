"""Quarantine invalid legacy manifests without inventing replacement evidence."""
from pydantic import ValidationError
from common.runtime import Aggregate,digest,iso,now

def public_profile(row):
    keys=("name","version","camera_serial","lens_label","calibration","config_hash","calibrated_by")
    return {**{k:row.get(k) for k in keys},"id":row["id"],"profile_id":row["id"],"created_at":iso(row["created"])}

def public_bundle(row):
    keys=("version","weights_hashes","preprocess","thresholds","config_hash","code_commit","environment_hash")
    return {**{k:row.get(k) for k in keys},"id":row["id"],"model_bundle_id":row["id"],"created_at":iso(row["created"]),"deployment_status":"manifest_registered_weights_not_verified"}

def quarantine_legacy(store):
    from services.response_contracts import Profile,Bundle
    count=0
    for kind,model,formatter in (("profile",Profile,public_profile),("bundle",Bundle,public_bundle)):
        for row in store.list(None,kind):
            try:model.model_validate(formatter(row))
            except (ValidationError,ValueError,TypeError,KeyError):
                with store.tx() as session:
                    record=session.get(Aggregate,row["id"])
                    if record.kind!=kind:continue
                    record.kind="quarantined_"+kind
                    record.expires=row["created"]+90*86400
                    record.version+=1
                    marker=digest("registry-quarantine:"+row["id"])
                    if session.get(Aggregate,marker) is None:
                        store.create(marker,row["org_id"],"registry_tombstone",{"target_id":row["id"],"resource":kind,"reason_code":"INVALID_LEGACY_"+kind.upper(),"quarantined_at":iso()},now()+180*86400,session)
                    store.audit({"sub":"registry-migration","org_id":row["org_id"]},kind+".quarantined",row["id"],session)
                count+=1
    return count

def counts(store,org,kind):
    return sum(x["resource"]==kind for x in store.list(org,"registry_tombstone"))

def assert_usable(store,id,org,kind):
    from common.runtime import HTTPException,fail
    try:
        marker=store.get(digest("registry-quarantine:"+id),org,"registry_tombstone")
        if marker["resource"]==kind:fail(410,marker["reason_code"])
    except HTTPException as error:
        if error.status_code!=404:raise
