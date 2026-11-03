"""Real ONNX inference adapters with task-specific class and digest validation.

No packaged COCO model is treated as a pothole detector. Model absence is an
explicit blocked state, never a source of synthetic positive detections.
"""
import json
import math
import cv2
import numpy as np
from .paths import workspace_path, sha256


class ModelNotReady(RuntimeError):
    pass


def readiness(manifest_path):
    try:
        manifest_path = workspace_path(manifest_path, exists=True)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != "onnx":
            raise ValueError("only_onnx_runtime_supported")
        task = manifest.get("task")
        classes = manifest.get("classes")
        required = {"pothole": ["pothole"], "lane": ["background", "paint"],
                    "redaction": ["face", "license_plate"]}
        if task not in required or classes != required[task]:
            raise ValueError("task_class_mapping_mismatch")
        if not manifest.get("target_trained") or not manifest.get("dataset_release"):
            raise ValueError("target_training_provenance_required")
        weights = workspace_path(manifest["weights"], exists=True)
        if sha256(weights) != manifest.get("sha256"):
            raise ValueError("model_digest_mismatch")
        size = manifest.get("input_size")
        if not isinstance(size, list) or len(size) != 2 or any(type(x) is not int or x < 16 or x > 4096 for x in size):
            raise ValueError("invalid_input_size")
        if manifest.get("task") in ("pothole", "redaction") and manifest.get("output_layout") != "yolox_decoded_xywh":
            raise ValueError("export_decoded_yolox_required")
        return {"ready": True, "status": "ready", "manifest": manifest, "weights": str(weights)}
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as exc:
        return {"ready": False, "status": "model_not_ready", "reason": str(exc)}


class OnnxModel:
    def __init__(self, manifest_path):
        state = readiness(manifest_path)
        if not state["ready"]:
            raise ModelNotReady(state["reason"])
        self.manifest = state["manifest"]
        try:
            self.net = cv2.dnn.readNetFromONNX(state["weights"])
        except cv2.error as exc:
            raise ModelNotReady("onnx_load_failed: " + str(exc)) from exc
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def predict(self, frame):
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("bgr_frame_required")
        width, height = self.manifest["input_size"]
        source_h, source_w = frame.shape[:2]
        task = self.manifest["task"]
        if task in ("pothole", "redaction"):
            scale = min(width / source_w, height / source_h)
            resized = cv2.resize(frame, (max(1, round(source_w * scale)), max(1, round(source_h * scale))))
            image = np.full((height, width, 3), 114, np.uint8)
            image[:resized.shape[0], :resized.shape[1]] = resized
            # YOLOX reference preprocessing is BGR float32 with no /255.
            blob = image.transpose(2, 0, 1)[None].astype(np.float32)
        else:
            image = cv2.cvtColor(cv2.resize(frame, (width, height)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
            mean = np.array(self.manifest.get("mean", [.485, .456, .406]), np.float32)
            std = np.array(self.manifest.get("std", [.229, .224, .225]), np.float32)
            if np.any(std <= 0):
                raise ValueError("invalid_normalization_std")
            blob = ((image - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)
        self.net.setInput(blob)
        output = self.net.forward()
        if not np.isfinite(output).all():
            raise ModelNotReady("non_finite_model_output")
        if task == "lane":
            if output.ndim != 4 or output.shape[0] != 1 or output.shape[1] != 2:
                raise ModelNotReady("lane_output_must_be_N_2_H_W_logits")
            mask = np.argmax(output[0], axis=0).astype(np.uint8)
            return cv2.resize(mask, (source_w, source_h), interpolation=cv2.INTER_NEAREST)
        predictions = output[0] if output.ndim == 3 else output
        if predictions.ndim != 2 or predictions.shape[1] != 5 + len(self.manifest["classes"]):
            raise ModelNotReady("unexpected_yolox_output_shape")
        boxes, scores, labels = [], [], []
        threshold = float(self.manifest.get("confidence", .35))
        for row in predictions:
            label = int(np.argmax(row[5:]))
            confidence = float(row[4] * row[5 + label])
            if confidence < threshold:
                continue
            cx, cy, bw, bh = row[:4] / scale
            x, y = max(0., float(cx - bw / 2)), max(0., float(cy - bh / 2))
            right, bottom = min(source_w, float(cx + bw / 2)), min(source_h, float(cy + bh / 2))
            if right <= x or bottom <= y:
                continue
            boxes.append([x, y, right - x, bottom - y])
            scores.append(confidence)
            labels.append(label)
        results = []
        for label in set(labels):
            candidates = [i for i, value in enumerate(labels) if value == label]
            selected = cv2.dnn.NMSBoxes([boxes[i] for i in candidates], [scores[i] for i in candidates], threshold,
                                       float(self.manifest.get("nms_iou", .5)))
            for local_id in np.asarray(selected).reshape(-1):
                index = candidates[int(local_id)]
                results.append({"bbox_xywh_px": boxes[index], "confidence": scores[index],
                                "class": self.manifest["classes"][label]})
        return results


def redact(frame, boxes, padding=.15):
    result = frame.copy()
    h, w = result.shape[:2]
    for detection in boxes:
        x, y, bw, bh = detection["bbox_xywh_px"] if isinstance(detection, dict) else detection
        if not all(math.isfinite(v) for v in (x, y, bw, bh)) or bw <= 0 or bh <= 0:
            raise ValueError("invalid_redaction_box")
        x1, y1 = max(0, math.floor(x - bw * padding)), max(0, math.floor(y - bh * padding))
        x2, y2 = min(w, math.ceil(x + bw * (1 + padding))), min(h, math.ceil(y + bh * (1 + padding)))
        if x2 > x1 and y2 > y1:
            # Opaque redaction remains safe for tiny/low-resolution faces/plates.
            result[y1:y2, x1:x2] = 0
    return result


def frame_quality(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    reasons = []
    if float(cv2.Laplacian(gray, cv2.CV_64F).var()) < 30:
        reasons.append("blur_or_low_texture")
    if float((gray >= 250).mean()) > .5:
        reasons.append("overexposed")
    if float((gray <= 5).mean()) > .5:
        reasons.append("underexposed")
    return {"usable": not reasons, "reasons": reasons}
