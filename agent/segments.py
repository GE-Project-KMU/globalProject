"""Metric road-patch deduplication with GNSS/visual motion agreement."""
from datetime import datetime
import math
import uuid
import cv2
import numpy as np
from .lane import measure_patch, aggregate_segment, api_lane


class Distance:
    def __init__(self, scale):
        self.scale, self.previous, self.distance, self.epoch = scale, None, 0., 0
        self.broken=True

    def update(self, gray, at, speed_kmh, valid_fix):
        previous, self.previous = self.previous, (gray.copy(), at, speed_kmh, valid_fix)
        if previous is None: return None, "initial_motion_sample"
        image, before, speed, valid = previous
        dt = (datetime.fromisoformat(at) - datetime.fromisoformat(before)).total_seconds()
        reason = None
        if not valid_fix or not valid or speed_kmh is None or speed is None or not 0 < dt <= .5:
            reason = "gnss_or_time_gap"
        elif max(speed_kmh, speed) > 20: reason = "speed_out_of_scope"
        elif max(speed_kmh, speed) < .5:
            self.broken=False
            return self.distance, "stationary"
        else:
            (dx, dy), response = cv2.phaseCorrelate(image.astype(np.float32), gray.astype(np.float32))
            visual = abs(dy) * self.scale
            expected = (speed_kmh + speed) / 2 / 3.6 * dt
            if response < .15 or abs(dx) > max(2, abs(dy) * .5) or abs(visual - expected) > max(.08, expected * .5):
                reason = "visual_gnss_motion_disagreement"
            else:
                self.distance += visual
                self.broken=False
                return self.distance, None
        self.distance = 0.
        if not self.broken:self.epoch += 1
        self.broken=True
        return None, reason


class Segments:
    def __init__(self, run_id, profile):
        self.run_id, self.profile = run_id, profile
        self.cells = {}
        self.step = .1

    def add(self, paint, valid, distance, epoch, evidence, score):
        metric, c = self.profile["measurement"], self.profile["calibration"]
        scale = metric["meters_per_pixel"]
        near, far = c["visible_distance_m"]
        height = paint.shape[0]
        # BEV row 0 is far; bottom row is near. Metric cells define stable IDs.
        world = distance + far - (np.arange(height) + .5) * scale
        for track in metric["lane_tracks"]:
            left, right = map(int, track["bev_x_range"])
            for cell in np.unique(np.floor(world / self.step).astype(int)):
                rows = np.where(np.floor(world / self.step).astype(int) == cell)[0]
                midpoint = (cell + .5) * self.step
                if track["line_type"] == "dashed":
                    if (midpoint - track["phase_m"]) % (track["dash_length_m"] + track["gap_length_m"]) >= track["dash_length_m"]:
                        continue
                p, v = paint[rows, left:right], valid[rows, left:right]
                result = measure_patch(np.ones_like(p), np.zeros_like(p), p, v)
                key = (epoch, track["lane_track_id"], int(cell))
                # Keep one representative; repeated frames cannot inflate coverage/area.
                rank = (result["valid_area_ratio"] or 0, score, len(rows))
                if key not in self.cells or rank > self.cells[key]["rank"]:
                    self.cells[key] = {"rank": rank, "measurement": result, "evidence": evidence,
                                      "track": track, "patch_id": str(uuid.uuid5(uuid.UUID(self.run_id), str(key)))}

    def finish(self):
        grouped = {}
        for (epoch, lane, cell), value in self.cells.items():
            grouped.setdefault((epoch, lane, int(cell * self.step // 10)), []).append((cell, value))
        outputs = []
        for key, values in sorted(grouped.items()):
            epoch, lane, number = key
            track = values[0][1]["track"]
            start, end = number * 10, (number + 1) * 10
            centers = np.arange(start + self.step / 2, end, self.step)
            if track["line_type"] == "dashed":
                centers = centers[((centers - track["phase_m"]) % (track["dash_length_m"] + track["gap_length_m"])) < track["dash_length_m"]]
            patches = [v["measurement"] | {"patch_id": v["patch_id"]} for _, v in values]
            accepted = [p for p in patches if p["grade"] != "unassessable"]
            support = min(1., len(accepted) / len(centers)) if len(centers) else None
            total, effective = sum(p["expected_area"] for p in patches), sum(p["effective_area"] for p in patches)
            result = aggregate_segment(patches, segment_support_ratio=support, estimated_length_m=10.)
            segment_id = str(uuid.uuid5(uuid.UUID(self.run_id), "segment:" + str(key)))
            outputs.append({"lane": api_lane(result, segment_id=segment_id, color=track["color"],
                                             line_type=track["line_type"], outline_supported=True,
                                             valid_area_ratio=effective / total if total else None),
                            "evidence": max((v for _, v in values), key=lambda v: v["rank"])["evidence"],
                            "lane_track_id": lane, "distance_epoch": epoch, "start_m": start, "end_m": end})
        return outputs
