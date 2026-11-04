import argparse
import json
import os
import sys
import httpx
import cv2
from cryptography.exceptions import InvalidTag
from .capture import record, replay, EncryptedStore
from .dataset import validate_annotation, validate_release, split_connected
from .models import readiness, ModelNotReady
from .outbox import Outbox
from .pipeline import analyze_video, infer_potholes
from .retention import purge_expired_runs, purge_local_material
from .paths import workspace_path, atomic_json
from .baseline import download_external_baseline
from .workflow import execute_config, approve_review, sync_run,register_capture
from .profile import calibrate_homography, load_profile
from .governance import revoke_known,blocked_identities
from .equipment import preflight, probe_gnss, stereo_depth, ZedSDK, collection_plan


def main(argv=None):
    parser = argparse.ArgumentParser(description="Local ZED 2i capture, validated analysis, and reviewed uploads")
    commands = parser.add_subparsers(dest="command", required=True)
    workflow = commands.add_parser("workflow")
    workflow.add_argument("config")
    workflow.add_argument("--inference-backend",choices=["cpu","cuda"],help="CUDA requires a verified local GPU child profile; default stays CPU")
    workflow.add_argument("--gpu-profile",help="Registered project-local GPU bundle profile")
    workflow.add_argument('--io-profile',choices=['default','overlap'],help='Optional bounded lossless PNG writer overlap; default remains synchronous')
    workflow.add_argument("--cpu-profile", choices=["default","fast"], help="CPU execution only: default2/spinning/serial; fast8/no-spin/3independent workers")
    profile = commands.add_parser("validate-profile"); profile.add_argument("profile")
    calibrate = commands.add_parser("calibrate"); calibrate.add_argument("request"); calibrate.add_argument("output")
    review = commands.add_parser("approve-review"); review.add_argument("package"); review.add_argument("decisions")
    review.add_argument("--reviewed-by", required=True)
    sync = commands.add_parser("sync-run"); sync.add_argument("package")
    sync.add_argument("--gateway", default="http://127.0.0.1:8000/api/v1")
    sync.add_argument("--ca-bundle")
    register=commands.add_parser("register-capture");register.add_argument("manifest")
    register.add_argument("--gateway",default="http://127.0.0.1:8000/api/v1");register.add_argument("--ca-bundle")
    collect = commands.add_parser("collect")
    collect.add_argument("--source", default="0"); collect.add_argument("--profile", required=True)
    collect.add_argument("--device-id", required=True); collect.add_argument("--checklist", required=True)
    collect.add_argument("--serial-port", required=True)
    collect.add_argument("--baudrate",type=int,default=115200)
    collect.add_argument("--seconds", type=float, default=60); collect.add_argument("--chunk-seconds", type=float, default=30)
    collect.add_argument("--collection-plan",required=True);collect.add_argument("--confirm-stationary",action="store_true",required=True)
    depth = commands.add_parser("stereo-depth")
    for key in ("left","right","calibration","output"):depth.add_argument(key)
    commands.add_parser("sdk-probe")
    postcheck=commands.add_parser("postcheck");postcheck.add_argument("manifest");postcheck.add_argument("checks")
    capture = commands.add_parser("capture")
    capture.add_argument("--source", default="0", help="integer camera index, or workspace video path")
    capture.add_argument("--output", default="data/agent/runs")
    capture.add_argument("--seconds", type=float, default=60)
    capture.add_argument("--chunk-seconds", type=float, default=30)
    capture.add_argument("--serial-port")
    capture.add_argument("--baudrate",type=int,default=115200)
    capture.add_argument("--collection-plan");capture.add_argument("--confirm-stationary",action="store_true")
    ready = commands.add_parser("model-status")
    ready.add_argument("manifest")
    baseline = commands.add_parser("download-baseline")
    baseline.add_argument("--output", default="data/models/external")
    infer = commands.add_parser("infer-potholes")
    infer.add_argument("source")
    infer.add_argument("--manifest", required=True)
    infer.add_argument("--output", required=True)
    analyze = commands.add_parser("analyze")
    analyze.add_argument("source")
    analyze.add_argument("--output", required=True)
    analyze.add_argument("--pothole-manifest", required=True)
    analyze.add_argument("--lane-manifest", required=True)
    analyze.add_argument("--redaction-manifest", required=True)
    analyze.add_argument("--profile")
    analyze.add_argument("--stereo", action="store_true")
    analyze.add_argument("--sample-fps", type=float, default=10)
    analyze.add_argument("--source-expires-at",help="Original explicit timezone-aware source deadline; never inferred from file timestamps")
    analyze.add_argument("--source-captured-at",help="Verified source capture UTC; allows capture+7d when no explicit deadline exists")
    analyze.add_argument("--inference-backend",choices=["cpu","cuda"],default="cpu")
    analyze.add_argument("--gpu-profile",help="Registered project-local GPU bundle profile")
    analyze.add_argument('--io-profile',choices=['default','overlap'],default='default',help='Optional bounded lossless PNG writer overlap')
    analyze.add_argument("--cpu-profile", choices=["default","fast"], default="default", help="Explicit CPU inference settings; weights, thresholds and runtime guards unchanged")
    validate = commands.add_parser("validate-data")
    validate.add_argument("manifest")
    validate.add_argument("--annotation", action="store_true")
    split = commands.add_parser("split-data")
    split.add_argument("manifest")
    split.add_argument("output")
    split.add_argument("--seed", type=int, default=2026)
    enqueue = commands.add_parser("enqueue")
    enqueue.add_argument("observation")
    enqueue.add_argument("image")
    enqueue.add_argument("--reviewed-by", required=True)
    enqueue.add_argument("--confirm-human-review", action="store_true")
    enqueue.add_argument("--source-expires-at",help="Original source-contract deadline, inherited by queue and server approval")
    commands.add_parser("queue-status")
    drain = commands.add_parser("upload")
    drain.add_argument("--gateway", default="http://127.0.0.1:8000/api/v1")
    drain.add_argument("--ca-bundle")
    retry = commands.add_parser("retry-blocked")
    retry.add_argument("observation_id")
    purge = commands.add_parser("purge-raw")
    purge.add_argument("--directory", default="data/agent/runs")
    purge.add_argument("--apply", action="store_true")
    purge_local=commands.add_parser("purge-local");purge_local.add_argument("--apply",action="store_true")
    purge_local.add_argument('--results',default='data/agent-jobs/results',help='Explicit generated review/diagnostic result root; unrelated files remain untouched')
    revoke=commands.add_parser("revoke-data");revoke.add_argument("request");revoke.add_argument("--apply",action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "workflow":
            result = execute_config(args.config,cpu_profile=args.cpu_profile,inference_backend=args.inference_backend,gpu_profile=args.gpu_profile,**({'io_profile':args.io_profile} if args.io_profile else {}))
        elif args.command == "sdk-probe":
            camera=ZedSDK()
            try:
                sample=camera.read()
                if sample is None: raise RuntimeError("sdk_frame_unavailable")
                result={"status":"sdk_frame_received","left_shape":list(sample["left"].shape),"timestamp_ns":sample["timestamp_ns"],"physical_validation":False}
            finally:camera.close()
        elif args.command == "stereo-depth":
            import numpy as np
            left=cv2.imread(str(workspace_path(args.left,exists=True)));right=cv2.imread(str(workspace_path(args.right,exists=True)))
            if left is None or right is None:raise ValueError("stereo_image_decode_failed")
            calibration=json.loads(workspace_path(args.calibration,exists=True).read_text(encoding="utf-8"))
            depth=stereo_depth(left,right,calibration)
            target=workspace_path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
            if target.suffix!=".npz":raise ValueError("npz_output_required")
            import io
            buffer=io.BytesIO();np.savez_compressed(buffer,**depth)
            result={"status":"depth_computed_unvalidated","artifact":EncryptedStore().encrypt_bytes(buffer.getvalue(),target),"valid_pixels":int(depth["valid"].sum())}
        elif args.command == "postcheck":
            from datetime import datetime
            manifest=json.loads(workspace_path(args.manifest,exists=True).read_text(encoding="utf-8"))
            checks=json.loads(workspace_path(args.checks,exists=True).read_text(encoding="utf-8"))
            if set(checks)!={"mount_secure","lens_clear","defect_from"} or any(type(checks[k]) is not bool for k in ("mount_secure","lens_clear")):
                raise ValueError("postcheck_fields_required")
            if checks["mount_secure"] is not True or checks["lens_clear"] is not True:
                start=datetime.fromisoformat(manifest["started_at"]);end=datetime.fromisoformat(manifest["ended_at"])
                defect=datetime.fromisoformat(checks["defect_from"])
                if not start<=defect<=end:raise ValueError("last_known_good_time_outside_run")
            elif checks["defect_from"] is not None:raise ValueError("normal_postcheck_requires_null_defect")
            manifest["postcheck"]=checks;atomic_json(args.manifest,manifest);result={"status":"postcheck_recorded","run_id":manifest["run_id"]}
        elif args.command == "collect":
            import uuid
            source = int(args.source) if args.source.isdecimal() else args.source
            device_id = str(uuid.UUID(args.device_id))
            plan=collection_plan(args.collection_plan)
            print("[수집 고지] "+plan["notice_message"]+" | 담당/연락: "+plan["operator"]+" / "+plan["controller_contact"],file=sys.stderr)
            checklist = json.loads(workspace_path(args.checklist,exists=True).read_text(encoding="utf-8"))
            nmea = probe_gnss(args.serial_port,baudrate=args.baudrate)
            checked = preflight(source,args.profile,checklist,"data/agent/preflight.json",nmea_records=nmea)
            result = record(source,seconds=args.seconds,chunk_seconds=args.chunk_seconds,serial_port=args.serial_port,
                            collection_plan_path=args.collection_plan,confirmed_stationary=args.confirm_stationary,
                            capture_metadata={"device_id":device_id,"profile_id":checked["profile_id"],"precheck":checked["precheck"],"conditions":checked["conditions"]},baudrate=args.baudrate)
            result.update(device_id=device_id,profile_id=checked["profile_id"],precheck=checked["precheck"],conditions=checked["conditions"])
            atomic_json(workspace_path("data/agent/runs") / result["run_id"] / "manifest.json",result)
        elif args.command == "validate-profile":
            value = load_profile(args.profile); result = {"valid": True, "profile_id": value["profile_id"]}
        elif args.command == "calibrate": result = calibrate_homography(args.request, args.output)
        elif args.command == "approve-review":
            result = approve_review(args.package, args.decisions, reviewed_by=args.reviewed_by)
        elif args.command == "sync-run":
            token = os.environ.get("ROAD_DEVICE_TOKEN")
            if not token: raise ValueError("ROAD_DEVICE_TOKEN_required")
            result = sync_run(args.package, args.gateway, token,ca_bundle=args.ca_bundle)
        elif args.command == "register-capture":
            token=os.environ.get("ROAD_DEVICE_TOKEN")
            if not token:raise ValueError("ROAD_DEVICE_TOKEN_required")
            result=register_capture(args.manifest,args.gateway,token,ca_bundle=args.ca_bundle)
        elif args.command == "capture":
            source = int(args.source) if args.source.isdecimal() else args.source
            if isinstance(source,int) and args.collection_plan:
                plan=collection_plan(args.collection_plan)
                print("[수집 고지] "+plan["notice_message"]+" | 담당/연락: "+plan["operator"]+" / "+plan["controller_contact"],file=sys.stderr)
            result = record(source, args.output, seconds=args.seconds, chunk_seconds=args.chunk_seconds, serial_port=args.serial_port,
                            collection_plan_path=args.collection_plan,confirmed_stationary=args.confirm_stationary,baudrate=args.baudrate)
        elif args.command == "model-status":
            result = readiness(args.manifest)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["ready"] else 2
        elif args.command == "download-baseline":
            result = download_external_baseline(args.output)
        elif args.command == "infer-potholes":
            result = infer_potholes(args.source, args.manifest, args.output)
        elif args.command == "analyze":
            result = analyze_video(args.source, args.output, pothole_manifest=args.pothole_manifest,
                                   lane_manifest=args.lane_manifest, redaction_manifest=args.redaction_manifest,
                                   stereo=args.stereo, profile_path=args.profile, sample_fps=args.sample_fps,cpu_profile=args.cpu_profile,inference_backend=args.inference_backend,gpu_profile=args.gpu_profile,
                                   source_expires_at=args.source_expires_at,source_captured_at=args.source_captured_at,io_profile=args.io_profile)
        elif args.command == "validate-data":
            value = json.loads(workspace_path(args.manifest, exists=True).read_text(encoding="utf-8"))
            if args.annotation:
                errors = validate_annotation(value)
                result = {"valid": not errors, "errors": errors}
            else:
                result = validate_release(value)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["valid"] else 2
        elif args.command == "split-data":
            values = json.loads(workspace_path(args.manifest, exists=True).read_text(encoding="utf-8"))
            result = split_connected(values, args.seed)
            atomic_json(args.output, result)
            result = {"samples": len(result), "output": args.output, "seed": args.seed}
        elif args.command == "enqueue":
            observation = json.loads(workspace_path(args.observation, exists=True).read_text(encoding="utf-8"))
            result = {"observation_id": Outbox().enqueue(observation, args.image, reviewed_by=args.reviewed_by,
                                                         confirmed_human_review=args.confirm_human_review,source_expires_at=args.source_expires_at)}
        elif args.command == "queue-status":
            result = Outbox().list()
        elif args.command == "upload":
            token = os.environ.get("ROAD_DEVICE_TOKEN")
            if not token:
                raise ValueError("ROAD_DEVICE_TOKEN_required")
            result = Outbox().drain(args.gateway, device_token=token, reviewer_token=os.environ.get("ROAD_REVIEWER_TOKEN"),ca_bundle=args.ca_bundle)
        elif args.command == "retry-blocked":
            Outbox().retry_blocked(args.observation_id)
            result = {"observation_id": args.observation_id, "status": "retry_requested"}
        elif args.command == "purge-raw":
            result = purge_expired_runs(args.directory, dry_run=not args.apply)
        elif args.command == "purge-local":
            result = {"raw":purge_expired_runs(dry_run=not args.apply),"material":purge_local_material(results=args.results,dry_run=not args.apply),
                      "queue":Outbox().purge_expired(dry_run=not args.apply)}
        elif args.command == "revoke-data":
            request=json.loads(workspace_path(args.request,exists=True).read_text(encoding="utf-8"))
            result=revoke_known(request=request,dry_run=not args.apply)
            from .governance import identity_hash
            targets=blocked_identities() if args.apply else {identity_hash(kind,value) for plural,kind in
                (("sample_ids","sample_id"),("source_ids","source_id"),("observation_ids","observation_id")) for value in request.get(plural,[])}
            queue=workspace_path("data/agent/outbox.sqlite")
            if queue.exists():result["queue"]=Outbox().revoke_observations(targets,dry_run=not args.apply)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except ModelNotReady as exc:
        print(json.dumps({"status": "model_not_ready", "reason": str(exc)}, ensure_ascii=False))
        return 2
    except (ValueError, RuntimeError, OSError, KeyError, TypeError, httpx.HTTPError, cv2.error, InvalidTag) as exc:
        print(json.dumps({"status": "error", "reason": str(exc)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
