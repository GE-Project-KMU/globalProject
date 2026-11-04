"""Probe safety/restore contracts; real PostgreSQL execution is opt-in CLI."""
from pathlib import Path
import uuid

from scripts import operational_probe as probe


def test_configuration_restore_applies_only_to_disposable_project():
    from scripts import config_backup
    original = config_backup.ROOT
    folder = probe.ROOT/".cache"/"operational-probe-tests"/uuid.uuid4().hex
    folder.mkdir(parents=True)
    before = {name:(original/name).read_bytes() for name in config_backup.ALLOWED if (original/name).is_file()}
    result = probe.config_roundtrip(folder)
    assert result["all_hashes_match"] and result["files_restored"] == len(before)
    assert result["detail_records_restored"] == 0 and not result["secrets_included"]
    assert config_backup.ROOT == original
    assert all((original/name).read_bytes() == body for name,body in before.items())


def test_bounded_nearest_rank_latency_statistic():
    assert probe.percentile(list(range(1,101)), .95) == 95
    assert probe.percentile([7], .95) == 7
    assert len(set(probe.PORTS.values())|{probe.PROXY_PORT}) == 7
    assert not ({8000,8101,8102,8103,8104,8105}&set(probe.PORTS.values()))
