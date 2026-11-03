"""Real ONNX inference adapters with task-specific class and digest validation.

No packaged COCO model is treated as a pothole detector. Model absence is an
explicit blocked state, never a source of synthetic positive detections.
"""
import json
import math
from datetime import datetime,timezone
import cv2
import numpy as np
from .paths import workspace_path, sha256


class ModelNotReady(RuntimeError):
    pass

class OrtNetwork:
    """CPU ORT, retaining the net.setInput/forward adapter interface."""
    backend="onnxruntime-cpu"
    def __init__(self,weights):
        try:import onnxruntime as ort
        except ImportError as exc:raise ValueError("onnxruntime_runtime_required") from exc
        self.backend_version=self.version=ort.__version__
        options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1;options.log_severity_level=3
        try:
            # A byte-loaded self-contained model cannot resolve external tensor
            # files against an arbitrary filesystem path.
            self.session=ort.InferenceSession(workspace_path(weights,exists=True).read_bytes(),sess_options=options,providers=["CPUExecutionProvider"])
        except Exception as exc:raise ValueError("onnx_load_failed: "+type(exc).__name__) from exc
        inputs=self.session.get_inputs();outputs=self.session.get_outputs()
        if len(inputs)!=1 or len(outputs)!=1 or inputs[0].type!="tensor(float)":raise ValueError("single_float_input_output_required")
        self.input_name=inputs[0].name;self.output_name=outputs[0].name;self.blob=None
    def setInput(self,value):
        self.blob=None
        if not isinstance(value,np.ndarray) or value.dtype!=np.float32 or value.ndim!=4 or value.shape[0]!=1 or value.shape[1]!=3 or not np.isfinite(value).all():raise ModelNotReady("invalid_model_input_tensor")
        self.blob=np.ascontiguousarray(value)
    def forward(self):
        if self.blob is None:raise ModelNotReady("model_input_required")
        try:output=self.session.run([self.output_name],{self.input_name:self.blob})[0]
        except Exception as exc:raise ModelNotReady("onnx_inference_failed: "+type(exc).__name__) from exc
        if not np.isfinite(output).all():raise ModelNotReady("non_finite_model_output")
        return output

class OpenCVClassicNetwork:
    backend="opencv-classic-diagnostic"
    def __init__(self,weights):
        self.backend_version=self.version=cv2.__version__
        if hasattr(cv2.dnn,"ENGINE_CLASSIC"):self.net=cv2.dnn.readNetFromONNX(str(weights),engine=cv2.dnn.ENGINE_CLASSIC)
        else:self.net=cv2.dnn.readNetFromONNX(str(weights))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV);self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    def setInput(self,value):self.net.setInput(value)
    def forward(self):return self.net.forward()

def load_network(weights,backend="onnxruntime"):
    if backend=="onnxruntime":return OrtNetwork(weights)
    if backend=="opencv-classic":return OpenCVClassicNetwork(weights)
    raise ValueError("unsupported_inference_backend")

def runtime_metadata(models):
    """Describe actual loaded adapters; test/custom factories remain explicit."""
    values={task:{"engine":getattr(model,"backend","custom-unverified"),"version":getattr(model,"backend_version",None)} for task,model in models.items()}
    engines={value["engine"] for value in values.values()};versions={value["version"] for value in values.values()}
    return {"engine":next(iter(engines)) if len(engines)==1 else "mixed","version":next(iter(versions)) if len(versions)==1 else None,"models":values,
            "device":"cpu" if engines and engines<={"onnxruntime-cpu","opencv-classic-diagnostic"} else "unspecified"}


def readiness(manifest_path,backend="onnxruntime"):
    try:
        manifest_path = workspace_path(manifest_path, exists=True)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest,dict):raise ValueError("model_manifest_object_required")
        if manifest.get("data_revoked") or manifest.get("needs_revalidation"):raise ValueError("model_training_data_revoked_needs_revalidation")
        if manifest.get("training_provenance"):
            provenance=json.loads(workspace_path(manifest["training_provenance"],exists=True).read_text(encoding="utf-8"))
            if not isinstance(provenance,dict):raise ValueError("training_provenance_object_required")
            if provenance.get("data_revoked") or provenance.get("needs_revalidation"):raise ValueError("model_training_data_revoked_needs_revalidation")
            from .governance import blocked_identities
            blocked=blocked_identities()
            if any(asset.get("revoked_at") or set(asset.get("identity_hashes",[]))&blocked or
                   datetime.fromisoformat(asset["expires_at"])<=datetime.now(timezone.utc) for asset in provenance.get("assets",[])):
                raise ValueError("model_training_data_expired_or_revoked")
        if manifest.get("format") != "onnx":
            raise ValueError("only_onnx_runtime_supported")
        task = manifest.get("task")
        classes = manifest.get("classes")
        required = {"pothole": ["pothole"], "lane": ["background", "paint"],
                    "redaction": ["face", "license_plate"]}
        if not isinstance(task,str) or task not in required or classes != required[task]:
            raise ValueError("task_class_mapping_mismatch")
        if manifest.get("target_trained") is not True or not isinstance(manifest.get("dataset_release"), str) or not manifest["dataset_release"].strip():
            raise ValueError("target_training_provenance_required")
        weights = workspace_path(manifest["weights"], exists=True)
        if sha256(weights) != manifest.get("sha256"):
            raise ValueError("model_digest_mismatch")
        size = manifest.get("input_size")
        if not isinstance(size, list) or len(size) != 2 or any(type(x) is not int or x < 16 or x > 4096 for x in size):
            raise ValueError("invalid_input_size")
        if manifest.get("task") in ("pothole", "redaction") and manifest.get("output_layout") not in ("yolox_decoded_xywh", "yolox_raw_xywh"):
            raise ValueError("supported_yolox_output_layout_required")
        for field in ("confidence", "nms_iou"):
            value = manifest.get(field, .35 if field == "confidence" else .5)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("invalid_" + field)
        if "paint_threshold" in manifest:
            threshold=manifest["paint_threshold"]
            if type(threshold) not in (int,float) or not math.isfinite(threshold) or not 0<threshold<1:raise ValueError("invalid_paint_threshold")
        try:
            network=load_network(weights,backend)
        except (cv2.error,ValueError) as exc:
            raise ValueError("onnx_load_failed: " + str(exc)) from exc
        return {"ready": True, "status": "ready", "research_only":manifest.get("research_only",False),
                "deployment_ready":manifest.get("production_approved") is True and not manifest.get("research_only",False)
                    and not manifest.get("deployment_blocked_reasons"),
                "inference_backend":getattr(network,"backend",backend),"inference_backend_version":getattr(network,"backend_version",None),"manifest": manifest, "weights": str(weights)}
    except (FileNotFoundError, KeyError, ValueError,TypeError,AttributeError, json.JSONDecodeError) as exc:
        return {"ready": False, "status": "model_not_ready", "reason": str(exc)}


class OnnxModel:
    def __init__(self, manifest_path,backend="onnxruntime"):
        state = readiness(manifest_path,backend)
        if not state["ready"]:
            raise ModelNotReady(state["reason"])
        self.manifest = state["manifest"]
        try:
            self.net = load_network(state["weights"],backend)
        except (cv2.error,ValueError) as exc:
            raise ModelNotReady("onnx_load_failed: " + str(exc)) from exc
        self.backend=getattr(self.net,"backend",backend)
        self.backend_version=getattr(self.net,"backend_version",None)

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
            scale = min(width / source_w, height / source_h)
            resized = cv2.resize(frame, (max(1, round(source_w * scale)), max(1, round(source_h * scale))))
            canvas = np.full((height, width, 3), 114, np.uint8)
            canvas[:resized.shape[0], :resized.shape[1]] = resized
            image = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
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
            threshold=self.manifest.get("paint_threshold",.5)
            mask = ((output[0,1]-output[0,0])>math.log(threshold/(1-threshold))).astype(np.uint8)
            crop_h = max(1, round(resized.shape[0] * mask.shape[0] / height))
            crop_w = max(1, round(resized.shape[1] * mask.shape[1] / width))
            mask = mask[:crop_h, :crop_w]
            return cv2.resize(mask, (source_w, source_h), interpolation=cv2.INTER_NEAREST)
        predictions = output[0] if output.ndim == 3 else output
        if predictions.ndim != 2 or predictions.shape[1] != 5 + len(self.manifest["classes"]):
            raise ModelNotReady("unexpected_yolox_output_shape")
        if self.manifest["output_layout"] == "yolox_raw_xywh":
            predictions = decode_yolox_raw(predictions, width, height, self.manifest.get("strides", [8, 16, 32]))
        threshold = float(self.manifest.get("confidence", .35))
        boxes, scores, labels = detection_candidates(predictions, scale, threshold)
        results = []
        for label in set(labels):
            candidates = [i for i, value in enumerate(labels) if value == label]
            selected = cv2.dnn.NMSBoxes([boxes[i] for i in candidates], [scores[i] for i in candidates], threshold,
                                       float(self.manifest.get("nms_iou", .5)))
            for local_id in np.asarray(selected).reshape(-1):
                index = candidates[int(local_id)]
                x, y, bw, bh = boxes[index]
                left, top = max(0., x), max(0., y)
                right, bottom = min(source_w, x + bw), min(source_h, y + bh)
                if right <= left or bottom <= top:
                    continue
                results.append({"bbox_xywh_px": [left, top, right - left, bottom - top], "confidence": scores[index],
                                "class": self.manifest["classes"][label]})
        return results


def detection_candidates(predictions, scale, threshold):
    """Vectorize selection while retaining the former scalar numeric contract."""
    labels = np.argmax(predictions[:, 5:], axis=1)
    # The old loop multiplied decoded scalars, converted the product to Python
    # float, then compared it to the Python threshold. Comparing float32 arrays
    # directly would round e.g. .35 to float32 and change tie selection.
    scores = (predictions[:, 4] * predictions[np.arange(len(predictions)), 5 + labels]).astype(np.float64)
    indices = np.flatnonzero(scores >= threshold)
    coordinates = predictions[indices, :4] / scale
    positive = (coordinates[:, 2] > 0) & (coordinates[:, 3] > 0)
    indices, coordinates = indices[positive], coordinates[positive]
    # NumPy1.x promotes float16/32 scalar / Python2 to float64; NumPy2 keeps
    # the scalar dtype. Array arithmetic alone has different promotion on1.x.
    geometry_dtype = np.asarray(coordinates.dtype.type(1) / 2).dtype
    geometry = coordinates.astype(geometry_dtype)
    xy = geometry[:, :2] - geometry[:, 2:4] / 2
    # Keep original row order and unclipped boxes for unchanged classwise NMS.
    boxes = np.concatenate((xy, geometry[:, 2:4]), axis=1).tolist()
    return boxes, scores[indices].tolist(), labels[indices].tolist()


def decode_yolox_raw(predictions, width, height, strides=(8, 16, 32)):
    grids, scales = [], []
    for stride in strides:
        if type(stride) is not int or stride <= 0 or width % stride or height % stride:
            raise ModelNotReady("invalid_yolox_stride")
        x, y = np.meshgrid(np.arange(width // stride), np.arange(height // stride))
        grid = np.stack((x, y), axis=-1).reshape(-1, 2)
        grids.append(grid)
        scales.append(np.full((len(grid), 1), stride, np.float32))
    grid, scale = np.concatenate(grids), np.concatenate(scales)
    if len(grid) != len(predictions):
        raise ModelNotReady("raw_yolox_grid_count_mismatch")
    decoded = predictions.copy()
    decoded[:, :2] = (decoded[:, :2] + grid) * scale
    with np.errstate(over="ignore", invalid="ignore"):
        decoded[:, 2:4] = np.exp(decoded[:, 2:4]) * scale
    if not np.isfinite(decoded).all():
        raise ModelNotReady("raw_yolox_non_finite_decoding")
    return decoded


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
