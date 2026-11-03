"""Pinned, checksum-verified OPTIONAL external YOLOX-S research baseline.

This is not the selected Nano production model and is never silently enabled.
"""
from datetime import datetime, timezone
import hashlib
import json
import httpx
from .paths import workspace_path, atomic_json, sha256

REPOSITORY = "AMD-PAVS-AI/pothole-detection"
REVISION = "cd5dd7a24988ac3cb8e58825a9ad477323787ed9"
MODEL_FILE = "pothole_yolox_s.onnx"
EXPECTED_SIZE = 35_774_753
EXPECTED_SHA256 = "9d55b8cc134f034b1885bd97aded14dcb17d887b8d5b7fbb3940e820d24c7660"


def download_external_baseline(directory="data/models/external"):
    directory = workspace_path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MODEL_FILE
    url = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{MODEL_FILE}"
    if not path.exists() or sha256(path) != EXPECTED_SHA256:
        temporary = path.with_suffix(".onnx.download")
        digest, size = hashlib.sha256(), 0
        try:
            with httpx.stream("GET", url, follow_redirects=True, timeout=90) as response:
                response.raise_for_status()
                with temporary.open("wb") as stream:
                    for chunk in response.iter_bytes():
                        size += len(chunk)
                        if size > EXPECTED_SIZE:
                            raise ValueError("baseline_exceeds_pinned_size")
                        digest.update(chunk)
                        stream.write(chunk)
            if size != EXPECTED_SIZE or digest.hexdigest() != EXPECTED_SHA256:
                raise ValueError("baseline_checksum_mismatch")
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()
    provenance = {"publisher": REPOSITORY, "revision": REVISION, "file": MODEL_FILE,
                  "sha256": EXPECTED_SHA256, "size_bytes": EXPECTED_SIZE, "download_url": url,
                  "model_card": f"https://huggingface.co/{REPOSITORY}/blob/{REVISION}/README.md",
                  "license": "Apache-2.0 (publisher model card)",
                  "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
                  "training_dataset": "jaygala24/pothole-detection",
                  "training_dataset_version": "publisher validation split of 186 images; exact training split hash not supplied",
                  "dataset_source": "https://github.com/jaygala24/pothole-detection",
                  "dataset_repository_license": "MIT; original repository describes web-collected images",
                  "publisher_metrics": {"validation_images": 186, "map50": .721, "map50_95": .344},
                  "project_acceptance": "not_validated_not_production_approved",
                  "selected_project_model": "YOLOX-Nano remains the target; this optional checkpoint is YOLOX-S",
                  "downloaded_at": datetime.now(timezone.utc).isoformat()}
    sources = {
        "MODEL_CARD.md": f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/README.md",
        "MODEL_LICENSE.txt": "https://www.apache.org/licenses/LICENSE-2.0.txt",
        "DATASET_LICENSE.txt": "https://raw.githubusercontent.com/jaygala24/pothole-detection/main/LICENSE",
    }
    provenance["source_documents"] = []
    for name, source_url in sources.items():
        destination = directory / name
        if not destination.exists():
            with httpx.stream("GET", source_url, follow_redirects=True, timeout=30) as response:
                response.raise_for_status()
                content = bytearray()
                for chunk in response.iter_bytes():
                    content.extend(chunk)
                    if len(content) > 1_000_000:
                        raise ValueError("source_document_exceeds_limit")
            destination.write_bytes(content)
        provenance["source_documents"].append({"file": name, "url": source_url, "sha256": sha256(destination)})
    atomic_json(directory / "provenance.json", provenance)
    manifest = {"format": "onnx", "task": "pothole", "classes": ["pothole"], "architecture": "YOLOX-S-external-baseline",
                "weights": path.relative_to(workspace_path(".")).as_posix(), "sha256": EXPECTED_SHA256,
                "target_trained": True, "dataset_release": "publisher:jaygala24/pothole-detection:186-validation",
                "input_size": [640, 640], "output_layout": "yolox_raw_xywh", "strides": [8, 16, 32],
                "confidence": .35, "nms_iou": .5, "production_approved": False,
                "provenance": (directory / "provenance.json").relative_to(workspace_path(".")).as_posix()}
    atomic_json(directory / "pothole_yolox_s.json", manifest)
    return manifest
