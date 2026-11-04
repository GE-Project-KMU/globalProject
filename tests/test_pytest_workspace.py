from pathlib import Path
from types import SimpleNamespace
import tempfile
import pytest
from scripts.pytest_workspace import ROOT,configure,scoped_temporary

def test_external_temporary_root_rejected_without_creating_it():
    with pytest.raises(ValueError,match='outside_project'):scoped_temporary(Path(ROOT.anchor)/'outside-project-test-scratch')

def test_unique_default_roots_and_standard_tempfile_stay_inside_project():
    first=SimpleNamespace(option=SimpleNamespace(basetemp=None));second=SimpleNamespace(option=SimpleNamespace(basetemp=None))
    configure(first);configure(second)
    assert first.option.basetemp!=second.option.basetemp
    assert Path(first.option.basetemp).is_relative_to(ROOT/'.cache/pytest-tmp')
    with tempfile.TemporaryDirectory() as name:assert Path(name).is_relative_to(ROOT/'.cache/tmp/pytest')

def test_explicit_project_basetemp_preserved_and_external_rejected():
    local=ROOT/'.cache/pytest-explicit';config=SimpleNamespace(option=SimpleNamespace(basetemp=str(local)))
    configure(config);assert Path(config.option.basetemp)==local
    with pytest.raises(pytest.UsageError,match='outside_project'):configure(SimpleNamespace(option=SimpleNamespace(basetemp=str(Path(ROOT.anchor)/'outside-project-test-scratch'))))

@pytest.mark.parametrize('path',[ROOT,ROOT/'data/training',ROOT/'.cache',ROOT/'.cache/training',ROOT/'.cache/pytest-tmp'])
def test_basetemp_cannot_point_at_source_models_or_parallel_default_parent(path):
    with pytest.raises(pytest.UsageError,match='dedicated_scratch_namespace'):configure(SimpleNamespace(option=SimpleNamespace(basetemp=str(path))))
