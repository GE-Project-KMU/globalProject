"""Explicit optional training/export entry point; never trains on the test split.

Requires a separate compatible PyTorch/torchvision/ONNX environment and approved
real masks. Importing agent or running the web application does not import Torch.
"""
import argparse
import json
import os
import random
import sys
import cv2
import numpy as np
from .dataset import validate_release
from .paths import ROOT, workspace_path, atomic_json, sha256


def prepare_pair(image_path, mask_path, size=512):
    image = cv2.imread(str(workspace_path(image_path, exists=True)))
    mask = cv2.imread(str(workspace_path(mask_path, exists=True)), cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None or mask.shape != image.shape[:2] or not np.isin(mask, [0, 1, 255]).all():
        raise ValueError("image_mask_shape_or_values_invalid")
    h, w = mask.shape
    scale = min(size / w, size / h)
    rw, rh = max(1, round(w * scale)), max(1, round(h * scale))
    canvas = np.full((size, size, 3), 114, np.uint8)
    labels = np.full((size, size), 255, np.uint8)
    canvas[:rh, :rw] = cv2.resize(image, (rw, rh))
    labels[:rh, :rw] = cv2.resize(mask, (rw, rh), interpolation=cv2.INTER_NEAREST)
    pixels = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    pixels = (pixels - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
    return pixels.transpose(2, 0, 1), labels.astype(np.int64)


def train(manifest_path, output, *, epochs=30, batch_size=2, learning_rate=.0001, seed=2026):
    if epochs < 1 or batch_size < 2 or not 0 < learning_rate < 1:
        raise ValueError("positive_epochs_batch_at_least_two_and_valid_lr_required")
    manifest_path, output = workspace_path(manifest_path, exists=True), workspace_path(output)
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation = validate_release(rows)
    if not validation["valid"]:
        raise ValueError("dataset_release_validation_failed:" + json.dumps(validation["errors"]))
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "validation"]
    if len(train_rows) < batch_size or not val_rows:
        raise ValueError("train_and_validation_samples_required")
    for row in train_rows + val_rows:
        if sha256(workspace_path(row["mask_path"], exists=True)) != row["mask_sha256"]:
            raise ValueError("mask_hash_mismatch")
    os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))
    try:
        import torch
        import torchvision
    except ImportError as exc:
        raise RuntimeError("isolated_training_runtime_required: install compatible torch,torchvision,onnx") from exc
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    class Pairs(torch.utils.data.Dataset):
        def __init__(self, entries): self.entries = entries
        def __len__(self): return len(self.entries)
        def __getitem__(self, index):
            row = self.entries[index]
            pixels, labels = prepare_pair(row["path"], row["mask_path"])
            return torch.from_numpy(pixels), torch.from_numpy(labels)

    model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(
        weights=None, weights_backbone=None, num_classes=2, aux_loss=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    loader = torch.utils.data.DataLoader(Pairs(train_rows), batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = torch.utils.data.DataLoader(Pairs(val_rows), batch_size=1, num_workers=0)
    output.mkdir(parents=True, exist_ok=True)
    best, history = -1., []
    checkpoint = output / "lane_best.pt"
    for epoch in range(epochs):
        model.train(); losses = []
        for pixels, labels in loader:
            if not (labels != 255).any():
                continue
            pixels, labels = pixels.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(pixels)["out"], labels)
            if not torch.isfinite(loss): raise RuntimeError("non_finite_training_loss")
            loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        if not losses: raise ValueError("no_valid_training_pixels")
        model.eval(); intersection = union = 0
        with torch.no_grad():
            for pixels, labels in val_loader:
                labels = labels.to(device)
                predicted = model(pixels.to(device))["out"].argmax(1)
                valid = labels != 255
                intersection += int(((predicted == 1) & (labels == 1) & valid).sum())
                union += int((((predicted == 1) | (labels == 1)) & valid).sum())
        if not union: raise ValueError("validation_has_no_paint_union")
        metric = intersection / union
        history.append({"epoch": epoch + 1, "loss": sum(losses) / len(losses), "validation_paint_iou": metric})
        if metric > best:
            best = metric
            torch.save(model.state_dict(), checkpoint)
        atomic_json(output / "training.json", {"seed": seed, "device": device, "manifest_sha256": sha256(manifest_path),
                    "torch": torch.__version__, "torchvision": torchvision.__version__, "history": history,
                    "test_split_used": False, "production_approved": False})
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)); model.eval().cpu()

    class Output(torch.nn.Module):
        def __init__(self, network): super().__init__(); self.network = network
        def forward(self, pixels): return self.network(pixels)["out"]

    weights = output / "lane.onnx"
    torch.onnx.export(Output(model), torch.zeros(1, 3, 512, 512), str(weights),
                      input_names=["images"], output_names=["logits"], opset_version=17)
    artifact = {"format": "onnx", "task": "lane", "classes": ["background", "paint"],
                "weights": weights.relative_to(ROOT).as_posix(), "sha256": sha256(weights), "target_trained": True,
                "dataset_release": "manifest-sha256:" + sha256(manifest_path), "input_size": [512, 512],
                "mean": [.485, .456, .406], "std": [.229, .224, .225], "production_approved": False,
                "validation_paint_iou": best, "test_split_used": False}
    atomic_json(output / "lane.json", artifact)
    return artifact


def main():
    parser = argparse.ArgumentParser(description="Train/export two-class paint model from approved masks")
    parser.add_argument("manifest"); parser.add_argument("output")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=.0001)
    args = parser.parse_args()
    try:
        print(json.dumps(train(args.manifest, args.output, epochs=args.epochs, batch_size=args.batch_size,
                               learning_rate=args.learning_rate), ensure_ascii=False, indent=2))
        return 0
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print(json.dumps({"status": "training_blocked", "reason": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__": sys.exit(main())
