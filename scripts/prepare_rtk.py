"""Acquire publisher RTK pixels for restricted offline paint-segmentation research.

The flattened publisher archive does not identify routes. Default research-only
split uses fixed contiguous index blocks and temporal guard bands. It is NOT a
route-independent holdout and cannot establish project acceptance.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import urllib.request
import uuid
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/public/rtk"
PAGE = "https://data.mendeley.com/datasets/hssswvmjwf/1"
SNAPSHOT = "https://data.mendeley.com/public-api/datasets/hssswvmjwf/snapshot/1"
DOWNLOAD = "https://data.mendeley.com/public-files/datasets/hssswvmjwf/files/2c72e75d-1f50-457a-86f0-a5e3a4174df9/file_downloaded"
ARCHIVE_SHA = "78803e8df89eae6130139ee9c48de947edd08015fe56826c30fdacb2599841db"
ARCHIVE_SIZE = 101609702


def local(value: str | Path) -> Path:
    value = Path(value)
    resolved = (ROOT / value).resolve() if not value.is_absolute() else value.resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError("path_outside_workspace")
    return resolved


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def fetch(url: str, target: Path) -> None:
    temp = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url, timeout=180) as response, temp.open("wb") as stream:
        while chunk := response.read(1024 * 1024):
            stream.write(chunk)
    temp.replace(target)


def perceptual_hash(path: Path) -> int:
    """64-bit DCT perceptual hash, grayscale32x32; no learned model or GPU."""
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L").resize((32,32), Image.Resampling.LANCZOS),dtype=np.float64)
    basis = np.cos(np.pi * np.arange(8)[:,None] * (2*np.arange(32)[None,:]+1)/64)
    low = basis @ gray @ basis.T
    bits = low.flatten() > np.median(low.flatten()[1:])
    return sum(int(value) << index for index,value in enumerate(bits))


def block_split(filename: str) -> str | None:
    index = int(Path(filename).stem)
    if 125<=index<=549:return "train"
    if 0<=index<=74:return "validation"
    if 630<=index<=700:return "test"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-plan", help="Workspace JSON containing documented per-image route groups/splits; never guessed numeric chunks")
    args = parser.parse_args()
    BASE.mkdir(parents=True, exist_ok=True)
    archive = BASE / "RTK.zip"
    if not archive.exists():
        fetch(DOWNLOAD, archive)
    if archive.stat().st_size != ARCHIVE_SIZE or sha(archive) != ARCHIVE_SHA:
        raise ValueError("publisher_archive_integrity_mismatch")
    snapshot = BASE / "source-metadata.json"
    if not snapshot.exists():
        fetch(SNAPSHOT, snapshot)
    metadata = json.loads(snapshot.read_text(encoding="utf-8"))
    if metadata.get("licence", {}).get("short_name") != "CC BY 4.0":
        raise ValueError("publisher_license_not_confirmed")
    acquisition_path = BASE / "acquisition.json"
    if acquisition_path.exists():
        acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
    else:
        acquired = datetime.now(timezone.utc)
        acquisition = {"acquired_at": acquired.isoformat(), "expires_at": (acquired + timedelta(days=90)).isoformat(), "archive_sha256": ARCHIVE_SHA}
        write_json(acquisition_path, acquisition)
    if datetime.fromisoformat(acquisition["expires_at"]) <= datetime.now(timezone.utc):
        raise ValueError("research_data_expired_no_automatic_ttl_extension")
    evidence_path = BASE / "license-evidence.json"
    write_json(evidence_path, {
        "research_allowed": True, "license_id": "CC-BY-4.0", "source_url": PAGE,
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "dataset_title": "Road Traversing Knowledge", "doi": "10.17632/hssswvmjwf.1",
        "authors": ["Thiago Rateke", "Aldo von Wangenheim", "Rafael Toledo"],
        "publisher": "Mendeley Data", "institution": "Universidade Federal de Santa Catarina",
        "publisher_snapshot_path": relative(snapshot), "publisher_snapshot_sha256": sha(snapshot),
        "archive_url": DOWNLOAD, "archive_sha256": ARCHIVE_SHA, "archive_size_bytes": ARCHIVE_SIZE,
        "modifications": "Index-4 roadMarking -> foreground 1; known other classes 1..12 -> background 0; publisher background 0/void 255 -> ignore 255. Original images and masks preserved.",
        "scope": "Restricted offline research; publisher pixel labels are not verified remaining-paint/wear-retention ground truth, nor evidence of human privacy acceptance.",
        "third_party_rights": "CC BY attribution does not waive privacy, personality, or identified third-party rights.",
    })
    plan = None
    if args.group_plan:
        plan = json.loads(local(args.group_plan).read_text(encoding="utf-8"))
        if not plan.get("source_evidence") or not isinstance(plan.get("images"), dict):
            raise ValueError("documented_group_plan_required")
    raw = BASE / "original"
    masks = BASE / "paint-masks"
    raw.mkdir(exist_ok=True)
    masks.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        for info in source.infolist():
            if info.is_dir():
                continue
            target = (raw / info.filename).resolve()
            if not target.is_relative_to(raw) or info.file_size > 8_000_000:
                raise ValueError("unsafe_archive_member")
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(source.read(info))
    codes = (raw / "RTK/codes.txt").read_text(encoding="utf-8")
    if "4,roadMarking" not in codes or "11,pothole" not in codes:
        raise ValueError("publisher_class_mapping_changed")
    publisher_splits = {}
    for split, filename in (("train", "trainpaths.txt"), ("validation", "valpaths.txt")):
        for line in (raw / "RTK" / filename).read_text().splitlines():
            publisher_splits[Path(line).name] = split
    rows, class_counts, positive_images = [], Counter(), 0
    positive_by_split, positive_pixels_by_split = Counter(), Counter()
    duplicate_splits, group_splits = {}, {}
    images = sorted((raw / "RTK/image").glob("*.png"))
    if len(images) != 701:
        raise ValueError("publisher_image_count_mismatch")
    candidates=[path for path in images if plan or block_split(path.name)]
    signatures=[(sha(path),perceptual_hash(path)) for path in candidates]
    parent=list(range(len(candidates)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,(digest_i,hash_i) in enumerate(signatures):
        for j in range(i):
            digest_j,hash_j=signatures[j]
            if digest_i==digest_j or (hash_i^hash_j).bit_count()<=6:
                parent[find(i)]=find(j)
    cluster_splits={}
    for i,path in enumerate(candidates):
        split=plan["images"][path.name]["split"] if plan else block_split(path.name)
        cluster_splits.setdefault(find(i),set()).add(split)
    dropped={candidates[i].name for i in range(len(candidates)) if len(cluster_splits[find(i)])>1}
    guard_dropped=[path.name for path in images if not plan and not block_split(path.name)]
    for path in images:
        if path.name in dropped or path.name in guard_dropped:
            continue
        label = raw / "RTK/label" / path.name
        with Image.open(path) as image:
            image.load()
            if image.size != (352, 288):
                raise ValueError("unexpected_image_size")
        with Image.open(label) as image:
            pixel = np.asarray(image)
        if pixel.shape != (288, 352) or not set(np.unique(pixel)).issubset(set(range(13)) | {255}):
            raise ValueError("invalid_publisher_pixel_labels")
        mask = np.full(pixel.shape, 255, dtype=np.uint8)
        mask[(pixel >= 1) & (pixel <= 12)] = 0
        mask[pixel == 4] = 1
        positive_images += int(np.any(mask == 1))
        values, counts = np.unique(mask, return_counts=True)
        class_counts.update(dict(zip(map(int, values), map(int, counts))))
        mask_path = masks / path.name
        Image.fromarray(mask).save(mask_path)
        mapping = plan["images"].get(path.name) if plan else None
        if plan and (not mapping or not mapping.get("group_id") or mapping.get("split") not in ("train", "validation", "test")):
            raise ValueError("incomplete_group_plan")
        split = mapping["split"] if mapping else block_split(path.name)
        positive_by_split[split] += int(np.any(mask == 1))
        positive_pixels_by_split[split] += int(np.sum(mask == 1))
        group = mapping["group_id"] if mapping else "RTK-block-"+split
        original_hash = sha(path)
        if original_hash in duplicate_splits and duplicate_splits[original_hash] != split:
            raise ValueError("identical_image_crosses_split")
        if group in group_splits and group_splits[group] != split:
            raise ValueError("route_group_crosses_split")
        duplicate_splits[original_hash] = split
        group_splits[group] = split
        rows.append({
            "sample_id": str(uuid.uuid5(uuid.NAMESPACE_URL, PAGE + "/image/" + path.name)),
            "source_id": "RTK-v1/image/" + path.name, "source_hash": original_hash,
            "leakage_group_id": group, "annotation_version": "RTK-publisher-v1",
            "path": relative(path), "sha256": original_hash,
            "mask_path": relative(mask_path), "mask_sha256": sha(mask_path), "split": split,
            "publisher_split": publisher_splits[path.name], "publisher_mask_path": relative(label), "publisher_mask_sha256": sha(label),
            "privacy_status": "restricted_research", "qc_status": "publisher_annotations_machine_checked",
            "release_scope": "offline_research", "human_privacy_reviewed": False,
            "not_for_deployment": True, "allowed_use": "CC BY 4.0 offline model research; no privacy acceptance",
            "split_independence":"externally_evidenced_group_plan" if plan else "blocked_frame_indices_with_guard_bands_route_unknown",
            "independent_route_verified":False, "project_acceptance_eligible":False,
            "expires_at": acquisition["expires_at"], "license_evidence_path": relative(evidence_path), "license_evidence_sha256": sha(evidence_path),
            "foreground_definition": "publisher class 4 roadMarking pixels; remaining wear holes and expected outline are not independently annotated",
        })
    write_json(BASE / "manifest.json", rows)
    write_json(BASE / "machine-qc.json", {
        "samples": len(rows), "positive_images": positive_images, "mask_pixel_counts": dict(class_counts),
        "positive_images_by_split":dict(positive_by_split),"positive_pixels_by_split":dict(positive_pixels_by_split),
        "warnings":[f"{split}_has_no_positive_paint_pixels" for split in ("train","validation","test") if positive_pixels_by_split[split]==0],
        "split_counts": dict(Counter(row["split"] for row in rows)), "block_group_count": len(group_splits), "route_group_count":None,
        "independent_holdout_available": False,
        "research_holdout_available": {"train", "validation", "test"} <= set(group_splits.values()),
        "split_independence":"externally_evidenced_group_plan" if plan else "blocked_frame_indices_with_guard_bands_route_unknown",
        "independent_route_verified":False,"project_acceptance_eligible":False,
        "split_blocks":{"train":[125,549],"validation":[0,74],"test":[630,700]},
        "guard_bands":[[75,124],[550,629]],"guard_band_dropped":len(guard_dropped),
        "split_plan_version":"RTK-research-blocks-v1-frozen-before-training",
        "split_plan_rationale":"Before any model training or evaluation, publisher class-support inspection found initial validation480..549 contained zero roadMarking pixels. Validation0..74 and train125..549 chosen once to ensure positive label support with guard bands75..124 and550..629. Test630..700 unchanged. This decision uses only labels, never model performance. No further split changes after training/test.",
        "near_duplicate_policy":"Exact SHA256 or grayscale32x32 DCT64 pHash Hamming<=6 connected clusters; remove ALL members of every cluster crossing partitions before release.",
        "cross_split_duplicate_dropped":len(dropped),"cross_split_duplicate_dropped_files":sorted(dropped),
        "group_plan": args.group_plan, "archive_sha256": ARCHIVE_SHA,
        "limitation": "Publisher filenames are flattened numeric IDs with no supplied route/video mapping or proof numeric order is chronological. Publisher interleaved 561/140 image split is preserved as metadata only. Blocked frame holdout with guard bands and duplicate checks is research validation, NOT verified independent routes or project acceptance. pHash cannot prove absence of shared road objects or routes.",
    })
    print(json.dumps({"manifest": relative(BASE / "manifest.json"), "samples": len(rows), "positive_images": positive_images, "splits": dict(Counter(row["split"] for row in rows)), "independent_route_verified":False,"cross_split_duplicate_dropped":len(dropped)}))


if __name__ == "__main__":
    main()
