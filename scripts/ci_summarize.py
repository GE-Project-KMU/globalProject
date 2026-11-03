"""Produce upload-safe CI summaries without raw logs, tokens, DBs or traces."""
from pathlib import Path
import json
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".cache" / "ci-artifacts"
OUT.mkdir(parents=True, exist_ok=True)
reports = []
for path in sorted((ROOT / "data").glob("*-junit.xml")):
    try:
        cases = ET.parse(path).getroot().findall(".//testcase")
        reports.append({"report": path.name, "tests": len(cases),
                        "failures": sum(c.find("failure") is not None for c in cases),
                        "errors": sum(c.find("error") is not None for c in cases),
                        "skipped": sum(c.find("skipped") is not None for c in cases),
                        "failed_tests": [c.attrib.get("name", "unknown") for c in cases
                                         if c.find("failure") is not None or c.find("error") is not None]})
    except (ET.ParseError, OSError):
        reports.append({"report": path.name, "error": "summary_parse_failed"})
logs = []
for path in sorted((ROOT / "data" / "logs").glob("*.log")):
    text = path.read_text(encoding="utf-8", errors="replace")
    logs.append({"service": path.stem,
                 "traceback_count": len(re.findall("Traceback", text)),
                 "unhandled_count": len(re.findall("unhandled", text)),
                 "http_5xx_count": len(re.findall(r'" 5\d\d ', text))})
summary = {"test_reports": reports, "service_log_counts": logs,
           "artifact_policy": "Counts and static test names only; no raw logs, credentials, screenshots, traces, images or databases."}
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
