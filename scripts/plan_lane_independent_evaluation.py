"""Metadata-only feasibility check; never downloads pixels or evaluates a model."""
import json,sys,re,urllib.request,urllib.parse,xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import atomic_json,sha256
HOST="https://audi-autonomous-driving-dataset.s3.eu-central-1.amazonaws.com/"
NS={"s":"http://s3.amazonaws.com/doc/2006-03-01/"}

def main():
    sources=[ROOT/"data/public/a2d2/manifest.json",ROOT/"data/public/a2d2-improvement-v2/manifest.json"]
    used={r["recording"] for path in sources for r in json.loads(path.read_text(encoding="utf-8"))}
    parameters={"list-type":"2","prefix":"camera_lidar_semantic/","delimiter":"/","max-keys":"1000"}
    scenes=set();token=None
    while True:
        if token:parameters["continuation-token"]=token
        with urllib.request.urlopen(HOST+"?"+urllib.parse.urlencode(parameters),timeout=30) as response:
            root=ET.fromstring(response.read())
        for item in root.findall("s:CommonPrefixes/s:Prefix",NS):
            name=item.text.strip("/").split("/")[-1]
            if re.fullmatch(r"\d{8}_\d{6}",name):scenes.add(name)
        token=root.findtext("s:NextContinuationToken",namespaces=NS)
        if not token:break
    unused=scenes-used;dates={name[:8] for name in used};unused_dates={name[:8] for name in unused}-dates
    report={"format":"lane-independent-evaluation-feasibility-v1","checked_at":datetime.now(timezone.utc).isoformat(),
        "source":"https://registry.opendata.aws/aev-a2d2/","official_recordings_listed":len(scenes),
        "historically_used_recordings":len(used),"unused_recordings":len(unused),
        "unused_dates_outside_all_previous_train_validation_test":len(unused_dates),
        "new_recording_and_date_holdout_metadata_available":len(unused_dates)>=2,
        "source_manifest_hashes":[sha256(path) for path in sources],"pixels_or_new_GT_downloaded":False,"model_outputs_accessed":False,
        "new_final_test_already_frozen":False,"physical_route_independence_verified":False,"project_acceptance_eligible":False,
        "required_next_steps":[
            "Freeze chosen checkpoint/export expression/threshold before new images or predictions are examined",
            "Freeze at least two unused recording/date groups using metadata-only uniformly spaced frame indices",
            "Exclude exact SHA and connected DCT pHash<=6 duplicates against every historically used split, including both former TEST cohorts",
            "Do not replace excluded samples using model results; report actual retained support and uncertainty",
            "Pin publisher object hashes, class mapping and CC-BY-ND license evidence; private research only with no derived artifact redistribution approval",
            "Use a new source acquisition/expiry and inventory without extending any existing source deadline",
            "All earlier TEST cohorts are regression only; evaluate the new frozen holdout once after validation-only selection",
            "Obtain route/geospatial evidence to claim physical-road independence; recording/date separation alone is insufficient",
            "Independent wear/expected-outline/retention-ratio GT requires separate expert labeling; Solid/Dashed pixel masks do not supply it"],
        "limits":["Top-level listing proves metadata availability only; per-recording paired frame counts and lawful intended use must be verified before acquisition",
                  "New untouched source groups can support a new offline segmentation test; they do not establish Korean/ZED2i field performance"]}
    atomic_json(ROOT/"docs/reports/LANE-INDEPENDENT-EVALUATION-FEASIBILITY.json",report)
    atomic_json(ROOT/"data/training/lane-numeric-diagnostics-v3/unused-recording-metadata.json",
        {"used_recordings":sorted(used),"unused_recordings":sorted(unused),"unused_dates":sorted(unused_dates),"pixels_downloaded":False})
    print(json.dumps(report),flush=True)

if __name__=="__main__":main()
