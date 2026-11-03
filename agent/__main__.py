import argparse
import json
import os
import sys
from .capture import record, replay, EncryptedStore
from .dataset import validate_annotation, validate_release, split_connected
from .models import readiness, ModelNotReady
from .outbox import Outbox
from .pipeline import analyze_video
from .retention import purge_expired_runs
from .paths import workspace_path, atomic_json


def main(argv=None):
    parser = argparse.ArgumentParser(description="Local ZED 2i capture, validated analysis, and reviewed uploads")
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture")
    capture.add_argument("--source", default="0", help="integer camera index, or workspace video path")
    capture.add_argument("--output", default="data/agent/runs")
    capture.add_argument("--seconds", type=float, default=60)
    capture.add_argument("--chunk-seconds", type=float, default=30)
    capture.add_argument("--serial-port")
    decrypt = commands.add_parser("decrypt")
    decrypt.add_argument("source")
    decrypt.add_argument("destination")
    ready = commands.add_parser("model-status")
    ready.add_argument("manifest")
    analyze = commands.add_parser("analyze")
    analyze.add_argument("source")
    analyze.add_argument("--output", required=True)
    analyze.add_argument("--pothole-manifest", required=True)
    analyze.add_argument("--lane-manifest", required=True)
    analyze.add_argument("--redaction-manifest", required=True)
    analyze.add_argument("--profile")
    analyze.add_argument("--stereo", action="store_true")
    analyze.add_argument("--sample-fps", type=float, default=10)
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
    commands.add_parser("queue-status")
    drain = commands.add_parser("upload")
    drain.add_argument("--gateway", default="http://127.0.0.1:8000/api/v1")
    retry = commands.add_parser("retry-blocked")
    retry.add_argument("observation_id")
    purge = commands.add_parser("purge-raw")
    purge.add_argument("--directory", default="data/agent/runs")
    purge.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "capture":
            source = int(args.source) if args.source.isdecimal() else args.source
            result = record(source, args.output, seconds=args.seconds, chunk_seconds=args.chunk_seconds, serial_port=args.serial_port)
        elif args.command == "decrypt":
            result = {"path": str(EncryptedStore().decrypt(args.source, args.destination)), "warning": "Decrypted raw material remains subject to 7-day deletion and encrypted-volume policy."}
        elif args.command == "model-status":
            result = readiness(args.manifest)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["ready"] else 2
        elif args.command == "analyze":
            result = analyze_video(args.source, args.output, pothole_manifest=args.pothole_manifest,
                                   lane_manifest=args.lane_manifest, redaction_manifest=args.redaction_manifest,
                                   stereo=args.stereo, profile_path=args.profile, sample_fps=args.sample_fps)
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
                                                         confirmed_human_review=args.confirm_human_review)}
        elif args.command == "queue-status":
            result = Outbox().list()
        elif args.command == "upload":
            token = os.environ.get("ROAD_DEVICE_TOKEN")
            if not token:
                raise ValueError("ROAD_DEVICE_TOKEN_required")
            result = Outbox().drain(args.gateway, device_token=token, reviewer_token=os.environ.get("ROAD_REVIEWER_TOKEN"))
        elif args.command == "retry-blocked":
            Outbox().retry_blocked(args.observation_id)
            result = {"observation_id": args.observation_id, "status": "retry_requested"}
        elif args.command == "purge-raw":
            result = purge_expired_runs(args.directory, dry_run=not args.apply)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except ModelNotReady as exc:
        print(json.dumps({"status": "model_not_ready", "reason": str(exc)}, ensure_ascii=False))
        return 2
    except (ValueError, RuntimeError, OSError) as exc:
        print(json.dumps({"status": "error", "reason": str(exc)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
