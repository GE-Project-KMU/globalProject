"""Durable encrypted SQLite outbox; user approval precedes device upload."""
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
import sqlite3
import time
import uuid
from urllib.parse import urlparse
import httpx
from PIL import Image
from cryptography.exceptions import InvalidTag
from .capture import EncryptedStore
from .paths import workspace_path, sha256


class Outbox:
    def __init__(self, path="data/agent/outbox.sqlite", key_path="data/agent/local.key"):
        self.path = workspace_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cipher = EncryptedStore(key_path).cipher
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS queue (id TEXT PRIMARY KEY, payload BLOB NOT NULL, state TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, next_at REAL NOT NULL DEFAULT 0, lease_until REAL, error TEXT, approval_id TEXT, image_ready INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL)")

    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")
        return db

    def encode(self, identifier, value):
        nonce = os.urandom(12)
        raw = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
        return nonce + self.cipher.encrypt(nonce, raw, identifier.encode())

    def decode(self, row):
        raw = bytes(row["payload"])
        return json.loads(self.cipher.decrypt(raw[:12], raw[12:], row["id"].encode()))

    def enqueue(self, observation, image, *, reviewed_by, confirmed_human_review=False):
        if not confirmed_human_review or not reviewed_by.strip():
            raise ValueError("explicit_human_review_required")
        identifier = str(uuid.UUID(observation["observation_id"]))
        path = workspace_path(image, exists=True)
        if path.suffix.lower() not in (".png", ".jpg", ".jpeg") or "raw" in [part.casefold() for part in path.parts]:
            raise ValueError("only_redacted_still_images_may_be_queued")
        if not 0 < path.stat().st_size <= 5_000_000:
            raise ValueError("image_size_limit")
        with Image.open(path) as decoded:
            decoded.verify()
        with Image.open(path) as decoded:
            if max(decoded.size) > 4096 or decoded.format not in ("PNG", "JPEG"):
                raise ValueError("unsupported_image")
            mime = "image/png" if decoded.format == "PNG" else "image/jpeg"
        digest = sha256(path)
        if observation.get("image", {}).get("sha256") != digest:
            raise ValueError("reviewed_image_hash_mismatch")
        captured = datetime.fromisoformat(observation["captured_at"].replace("Z", "+00:00"))
        if captured.tzinfo is None or captured + timedelta(days=90) <= datetime.now(timezone.utc):
            raise ValueError("observation_expired")
        item = {"observation": observation, "image_path": str(path), "mime": mime,
                "reviewed_by": reviewed_by, "reviewed_at": datetime.now(timezone.utc).isoformat()}
        with self.connect() as db:
            existing = db.execute("SELECT * FROM queue WHERE id=?", (identifier,)).fetchone()
            if existing:
                old = self.decode(existing)
                if old["observation"] != observation or old["image_path"] != str(path):
                    raise ValueError("immutable_outbox_id_conflict")
                return identifier
            db.execute("INSERT INTO queue(id,payload,state,created_at) VALUES(?,?,?,?)",
                       (identifier, self.encode(identifier, item), "pending", time.time()))
        return identifier

    def list(self):
        with self.connect() as db:
            return [dict(r) for r in db.execute("SELECT id,state,attempts,next_at,error,approval_id,image_ready FROM queue ORDER BY created_at")]

    def retry_blocked(self, identifier):
        with self.connect() as db:
            db.execute("UPDATE queue SET state='pending',next_at=0,error=NULL,approval_id=NULL WHERE id=? AND state='blocked'", (identifier,))

    def drain(self, base_url, *, device_token, reviewer_token=None, transport=None, limit=100):
        parsed = urlparse(base_url)
        if parsed.scheme not in ("http", "https") or (parsed.scheme == "http" and parsed.hostname not in ("127.0.0.1", "localhost", "::1")):
            raise ValueError("https_or_loopback_gateway_required")
        results = []
        with httpx.Client(base_url=base_url.rstrip("/") + "/", timeout=30, transport=transport) as client:
            for _ in range(limit):
                now = time.time()
                with self.connect() as db:
                    db.execute("BEGIN IMMEDIATE")
                    db.execute("UPDATE queue SET state='pending',lease_until=NULL WHERE state='uploading' AND lease_until<?", (now,))
                    row = db.execute("SELECT * FROM queue WHERE state='pending' AND next_at<=? ORDER BY created_at LIMIT 1", (now,)).fetchone()
                    if not row:
                        break
                    db.execute("UPDATE queue SET state='uploading',lease_until=? WHERE id=?", (now + 120, row["id"]))
                try:
                    item = self.decode(row)
                    observation, path = item["observation"], workspace_path(item["image_path"], exists=True)
                    captured = datetime.fromisoformat(observation["captured_at"].replace("Z", "+00:00"))
                    if captured + timedelta(days=90) <= datetime.now(timezone.utc):
                        self._update(row["id"], "expired", "DATA_EXPIRED")
                        results.append({"id": row["id"], "state": "expired"})
                        continue
                    if sha256(path) != observation["image"]["sha256"]:
                        raise ValueError("reviewed_file_changed")
                    approval_id = row["approval_id"]
                    if not row["image_ready"]:
                        if not approval_id:
                            if not reviewer_token:
                                self._update(row["id"], "blocked", "REVIEWER_LOGIN_REQUIRED")
                                results.append({"id": row["id"], "state": "blocked"})
                                continue
                            payload = {"file_id": observation["image"]["file_id"], "device_id": observation["device_id"],
                                       "sha256": observation["image"]["sha256"], "size_bytes": path.stat().st_size,
                                       "mime": item["mime"], "captured_at": observation["captured_at"],
                                       "reviewed_at": item["reviewed_at"], "redaction_model_version": observation["model_version"]}
                            approval = client.post("image-approvals", json=payload, headers={"Authorization": "Bearer " + reviewer_token})
                            approval.raise_for_status()
                            approval_id = approval.json()["data"]["approval_id"]
                            with self.connect() as db:
                                db.execute("UPDATE queue SET approval_id=? WHERE id=?", (approval_id, row["id"]))
                        metadata = {"file_id": observation["image"]["file_id"], "approval_id": approval_id,
                                    "sha256": observation["image"]["sha256"], "captured_at": observation["captured_at"]}
                        with path.open("rb") as image_file:
                            uploaded = client.post("images", data={"metadata": json.dumps(metadata)},
                                                   files={"file": (path.name, image_file, item["mime"])},
                                                   headers={"Authorization": "Bearer " + device_token})
                        uploaded.raise_for_status()
                        with self.connect() as db:
                            db.execute("UPDATE queue SET image_ready=1 WHERE id=?", (row["id"],))
                    response = client.post("observations", json=observation, headers={"Authorization": "Bearer " + device_token})
                    response.raise_for_status()
                    if response.json().get("data", {}).get("observation_id") != row["id"]:
                        raise ValueError("ack_identifier_mismatch")
                    self._update(row["id"], "acked", None)
                    results.append({"id": row["id"], "state": "acked"})
                except httpx.HTTPStatusError as exc:
                    status = exc.response.status_code
                    if status == 410:
                        self._update(row["id"], "expired", "DATA_EXPIRED")
                    elif status in (408, 429) or status >= 500:
                        self._defer(row, "HTTP_" + str(status), exc.response.headers.get("Retry-After"))
                    else:
                        self._update(row["id"], "blocked", "HTTP_" + str(status))
                    results.append({"id": row["id"], "http_status": status})
                except (httpx.RequestError, OSError) as exc:
                    self._defer(row, type(exc).__name__)
                    results.append({"id": row["id"], "state": "pending"})
                except (ValueError, KeyError, InvalidTag) as exc:
                    self._update(row["id"], "blocked", str(exc))
                    results.append({"id": row["id"], "state": "blocked"})
        return results

    def _update(self, identifier, state, error):
        with self.connect() as db:
            db.execute("UPDATE queue SET state=?,error=?,lease_until=NULL WHERE id=?", (state, error, identifier))

    def _defer(self, row, error, retry_after=None):
        attempt = row["attempts"] + 1
        delay = (5, 15, 60)[attempt - 1] if attempt <= 3 else 300
        try:
            delay = max(delay, min(float(retry_after), 3600)) if retry_after else delay
        except ValueError:
            pass
        with self.connect() as db:
            db.execute("UPDATE queue SET state='pending',attempts=?,next_at=?,error=?,lease_until=NULL WHERE id=?",
                       (attempt, time.time() + delay, error, row["id"]))
