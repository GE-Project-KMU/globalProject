"""UVC side-by-side recording with encrypted, bounded MP4 chunks at rest."""
from datetime import datetime, timezone, timedelta
from pathlib import Path
import base64
import json
import os
import shutil
import time
import uuid
import cv2
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from .paths import workspace_path, atomic_json, sha256


def utcnow():
    return datetime.now(timezone.utc)


class EncryptedStore:
    def __init__(self, key_path="data/agent/local.key"):
        self.key_path = workspace_path(key_path)
        self.key_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.key_path.exists():
            with self.key_path.open("xb") as stream:
                stream.write(AESGCM.generate_key(bit_length=256))
            try:
                self.key_path.chmod(0o600)
            except OSError:
                pass
        self.cipher = AESGCM(self.key_path.read_bytes())

    def encrypt(self, path):
        path = workspace_path(path, exists=True)
        target = path.with_suffix(path.suffix + ".aesgcm")
        nonce = os.urandom(12)
        ciphertext = self.cipher.encrypt(nonce, path.read_bytes(), path.name.encode())
        temporary = target.with_suffix(target.suffix + ".tmp")
        with temporary.open("wb") as stream:
            stream.write(b"RW01" + nonce + ciphertext)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(target)
        path.unlink()
        return {"path": str(target.relative_to(workspace_path("."))), "sha256": sha256(target),
                "encrypted": True, "aad": path.name}

    def decrypt(self, source, destination):
        source = workspace_path(source, exists=True)
        destination = workspace_path(destination)
        if not source.name.endswith(".aesgcm"):
            raise ValueError("encrypted_source_required")
        raw = source.read_bytes()
        if raw[:4] != b"RW01":
            raise ValueError("unknown_encryption_format")
        original_name = source.name[:-len(".aesgcm")]
        plaintext = self.cipher.decrypt(raw[4:16], raw[16:], original_name.encode())
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(plaintext)
        return destination


def record(source, output="data/agent/runs", *, seconds=60, chunk_seconds=30,
           minimum_free_bytes=10_000_000_000, key_path="data/agent/local.key", serial_port=None):
    if seconds <= 0 or not 1 <= chunk_seconds <= 300:
        raise ValueError("invalid_recording_duration")
    output = workspace_path(output)
    output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output).free < minimum_free_bytes:
        raise RuntimeError("insufficient_disk_space")
    camera = isinstance(source, int)
    if not camera:
        source = str(workspace_path(source, exists=True))
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError("camera_or_video_unavailable")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    capture.set(cv2.CAP_PROP_FPS, 30)
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    fps = fps if 0 < fps <= 240 else 30.0
    run_id, start = str(uuid.uuid4()), utcnow()
    folder = output / run_id
    folder.mkdir()
    store = EncryptedStore(key_path)
    manifest = {"run_id": run_id, "source_kind": "camera" if camera else "video_replay",
                "started_at": start.isoformat(), "expires_at": (start + timedelta(days=7)).isoformat(),
                "capture_fps": fps, "chunks": [], "frames": 0, "status": "recording",
                "warnings": ["Active encoder files require an encrypted OS volume; completed chunks use AES-GCM."]}
    writers, pts, nmea, chunk_start = [], [], [], 0
    serial = None
    if serial_port:
        import serial as serial_module
        serial = serial_module.Serial(serial_port, baudrate=115200, timeout=0)

    def finish_chunk():
        nonlocal writers, pts, nmea
        if not writers:
            return
        paths = []
        for writer, path in writers:
            writer.release()
            paths.append(store.encrypt(path))
        index = folder / f"{chunk_start:09d}.pts.json"
        atomic_json(index, {"frames": pts, "nmea": nmea})
        manifest["chunks"].append({"first_frame": chunk_start, "frames": len(pts),
                                   "eyes": paths, "index": store.encrypt(index)})
        writers, pts, nmea = [], [], []
        atomic_json(folder / "manifest.json", manifest)

    elapsed = time.monotonic()
    try:
        while True:
            if camera and time.monotonic() - elapsed >= seconds:
                break
            if not camera and manifest["frames"] / fps >= seconds:
                break
            if shutil.disk_usage(folder).free < minimum_free_bytes:
                raise RuntimeError("insufficient_disk_space")
            ok, frame = capture.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            if width % 2 or width < 2:
                raise ValueError("side_by_side_even_width_required")
            if not writers:
                chunk_start = manifest["frames"]
                for eye in ("left", "right"):
                    path = folder / f"{chunk_start:09d}.{eye}.mp4"
                    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width // 2, height))
                    if not writer.isOpened():
                        raise RuntimeError("mp4_encoder_unavailable")
                    writers.append((writer, path))
            writers[0][0].write(frame[:, :width // 2])
            writers[1][0].write(frame[:, width // 2:])
            position = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000 if not camera else time.monotonic() - elapsed
            pts.append({"frame_id": manifest["frames"], "pts_seconds": position,
                        "captured_at": (start + timedelta(seconds=position)).isoformat(),
                        "monotonic_seconds": time.monotonic()})
            if serial and serial.in_waiting:
                for line in serial.read(serial.in_waiting).decode("ascii", errors="replace").splitlines():
                    nmea.append({"sentence": line, "received_at": utcnow().isoformat()})
            manifest["frames"] += 1
            if len(pts) >= max(1, round(fps * chunk_seconds)):
                finish_chunk()
        if manifest["frames"] == 0:
            raise RuntimeError("no_frames_received")
        manifest["status"] = "completed"
    except Exception as exc:
        manifest.update(status="interrupted", error=str(exc))
        raise
    finally:
        capture.release()
        if serial:
            serial.close()
        finish_chunk()
        manifest["ended_at"] = utcnow().isoformat()
        atomic_json(folder / "manifest.json", manifest)
    return manifest


def replay(source, *, sample_fps=10):
    source = workspace_path(source, exists=True)
    if sample_fps <= 0:
        raise ValueError("invalid_sample_fps")
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("video_unavailable")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not 0 < fps <= 240:
        cap.release()
        raise ValueError("invalid_source_fps")
    index, next_at = 0, 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            at = index / fps
            if at + 1e-9 >= next_at:
                yield {"frame_id": index, "pts_seconds": at, "frame": frame}
                next_at += 1 / sample_fps
            index += 1
    finally:
        cap.release()
