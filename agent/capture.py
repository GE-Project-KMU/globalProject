"""UVC side-by-side recording with encrypted, bounded MP4 chunks at rest."""
from datetime import datetime, timezone, timedelta
from pathlib import Path
import base64
import json
import os
import shutil
import time
import uuid
import math
import io
from fractions import Fraction
import av
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
        result=self.encrypt_bytes(path.read_bytes(),path)
        path.unlink()
        return result

    def encrypt_bytes(self, plaintext, logical_path):
        path=workspace_path(logical_path)
        target = path.with_suffix(path.suffix + ".aesgcm")
        target.parent.mkdir(parents=True,exist_ok=True)
        nonce = os.urandom(12)
        ciphertext = self.cipher.encrypt(nonce, plaintext, path.name.encode())
        temporary = target.with_suffix(target.suffix + ".tmp")
        with temporary.open("wb") as stream:
            stream.write(b"RW01" + nonce + ciphertext)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(target)
        return {"path": target.relative_to(workspace_path(".")).as_posix(), "sha256": sha256(target),
                "encrypted": True, "aad": path.name}

    def decrypt(self, source, destination):
        destination = workspace_path(destination)
        plaintext=self.decrypt_bytes(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(plaintext)
        return destination

    def decrypt_bytes(self,source):
        source = workspace_path(source, exists=True)
        if not source.name.endswith(".aesgcm"):
            raise ValueError("encrypted_source_required")
        raw = source.read_bytes()
        if raw[:4] != b"RW01":
            raise ValueError("unknown_encryption_format")
        original_name = source.name[:-len(".aesgcm")]
        return self.cipher.decrypt(raw[4:16], raw[16:], original_name.encode())


class BoundedBuffer(io.BytesIO):
    def write(self,value):
        if self.tell()+len(value)>128*1024*1024:raise RuntimeError("encoded_chunk_memory_limit")
        return super().write(value)


class MemoryEncoder:
    def __init__(self,width,height,fps):
        if width%2 or height%2:raise ValueError("mp4_even_eye_dimensions_required")
        self.buffer=BoundedBuffer()
        self.container=av.open(self.buffer,mode="w",format="mp4")
        self.stream=self.container.add_stream("mpeg4",rate=Fraction(fps).limit_denominator(1000))
        self.stream.width,self.stream.height=width,height
        self.stream.pix_fmt="yuv420p";self.stream.bit_rate=4_000_000
    def write(self,image):
        for packet in self.stream.encode(av.VideoFrame.from_ndarray(image,format="bgr24")):
            self.container.mux(packet)
    def finish(self):
        for packet in self.stream.encode():self.container.mux(packet)
        self.container.close()
        return self.buffer.getvalue()


def replay_bytes(encoded, *, sample_fps=10):
    if not math.isfinite(sample_fps) or sample_fps<=0:raise ValueError("invalid_sample_fps")
    with av.open(io.BytesIO(encoded),mode="r",format="mp4") as container:
        stream=container.streams.video[0]
        next_at=0.
        for index,frame in enumerate(container.decode(stream)):
            at=float(frame.time) if frame.time is not None else index/float(stream.average_rate)
            if at+1e-9>=next_at:
                yield {"frame_id":index,"pts_seconds":at,"frame":frame.to_ndarray(format="bgr24")}
                next_at+=1/sample_fps


def record(source, output="data/agent/runs", *, seconds=60, chunk_seconds=30,
           minimum_free_bytes=10_000_000_000, key_path="data/agent/local.key", serial_port=None,
           collection_plan_path=None, confirmed_stationary=False,capture_metadata=None,baudrate=115200):
    if not math.isfinite(seconds) or seconds <= 0 or not 1 <= chunk_seconds <= 30:
        raise ValueError("invalid_recording_duration")
    if capture_metadata and set(capture_metadata)-{"device_id","profile_id","precheck","conditions"}:raise ValueError("invalid_capture_metadata")
    if type(baudrate) is not int or baudrate<=0:raise ValueError("invalid_gnss_baudrate")
    output = workspace_path(output)
    output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output).free < minimum_free_bytes:
        raise RuntimeError("insufficient_disk_space")
    camera = isinstance(source, int)
    plan=None
    if camera:
        if not confirmed_stationary or not collection_plan_path:raise ValueError("collection_plan_and_stationary_confirmation_required")
        from .equipment import collection_plan
        plan=collection_plan(collection_plan_path)
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
                "raw_storage": "memory_mux_then_AES256GCM", "max_chunk_seconds":30,
                "warnings": []}
    if plan:manifest.update(collection_plan=plan,stationary_start_confirmed=True)
    if serial_port:manifest["gnss_baudrate"]=baudrate
    if capture_metadata:
        manifest.update(capture_metadata)
    writers, pts, nmea, chunk_start = [], [], [], 0
    serial = None
    serial_buffer = ""

    def finish_chunk():
        nonlocal writers, pts, nmea
        if not writers:
            return
        pending,writers=writers,[]
        paths = []
        for writer, path in pending:
            paths.append(store.encrypt_bytes(writer.finish(),path))
        index = folder / f"{chunk_start:09d}.pts.json"
        encoded_index=json.dumps({"frames":pts,"nmea":nmea},allow_nan=False).encode()
        manifest["chunks"].append({"first_frame": chunk_start, "frames": len(pts),
                                   "eyes": paths, "index": store.encrypt_bytes(encoded_index,index)})
        writers, pts, nmea = [], [], []
        atomic_json(folder / "manifest.json", manifest)

    elapsed = time.monotonic()
    try:
        if serial_port:
            import serial as serial_module
            serial = serial_module.Serial(serial_port, baudrate=baudrate, timeout=0)
        while True:
            if camera and time.monotonic() - elapsed >= seconds:
                break
            if not camera and manifest["frames"] / fps >= seconds:
                break
            if shutil.disk_usage(folder).free < minimum_free_bytes:
                raise RuntimeError("insufficient_disk_space")
            ok, frame = capture.read()
            if not ok:
                if camera: raise RuntimeError("camera_disconnected_or_frame_failed")
                break
            height, width = frame.shape[:2]
            if width % 2 or width < 2:
                raise ValueError("side_by_side_even_width_required")
            if "eye_size" in manifest and manifest["eye_size"]!=[width//2,height]:raise RuntimeError("capture_resolution_changed")
            manifest["eye_size"]=[width//2,height]
            if not writers:
                chunk_start = manifest["frames"]
                for eye in ("left", "right"):
                    path = folder / f"{chunk_start:09d}.{eye}.mp4"
                    writer = MemoryEncoder(width//2,height,fps)
                    writers.append((writer, path))
            writers[0][0].write(frame[:, :width // 2])
            writers[1][0].write(frame[:, width // 2:])
            position = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000 if not camera else time.monotonic() - elapsed
            pts.append({"frame_id": manifest["frames"], "pts_seconds": position,
                        "captured_at": (start + timedelta(seconds=position)).isoformat(),
                        "monotonic_seconds": time.monotonic()})
            if serial and serial.in_waiting:
                serial_buffer += serial.read(min(serial.in_waiting,65536)).decode("ascii", errors="replace")
                if len(serial_buffer)>16384 and "\n" not in serial_buffer:
                    raise RuntimeError("nmea_line_exceeds_limit")
                lines = serial_buffer.split("\n")
                serial_buffer = lines.pop()
                for line in lines:
                    nmea.append({"sentence": line, "received_at": utcnow().isoformat()})
            manifest["frames"] += 1
            if len(pts) >= max(1, round(fps * chunk_seconds)):
                finish_chunk()
        if manifest["frames"] == 0:
            raise RuntimeError("no_frames_received")
        manifest["status"] = "completed"
    except KeyboardInterrupt:
        manifest.update(status="completed" if manifest["frames"] else "interrupted",stop_reason="operator")
    except Exception as exc:
        manifest.update(status="interrupted", error=str(exc))
        raise
    finally:
        capture.release()
        if serial:
            serial.close()
        try:
            finish_chunk()
        except Exception as exc:
            manifest.update(status="interrupted",error="chunk_finalization_failed:"+str(exc))
            raise
        finally:
            manifest["persisted_frames"]=sum(chunk["frames"] for chunk in manifest["chunks"])
            if manifest["persisted_frames"]!=manifest["frames"]:
                manifest.update(status="interrupted",error=manifest.get("error","partial_chunk_not_persisted"))
            manifest["ended_at"] = utcnow().isoformat()
            atomic_json(folder / "manifest.json", manifest)
    return manifest


def replay(source, *, sample_fps=10):
    source = workspace_path(source, exists=True)
    if not math.isfinite(sample_fps) or sample_fps <= 0:
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
