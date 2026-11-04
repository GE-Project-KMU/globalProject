"""Project-local pytest scratch paths, with unique roots for parallel workers."""
import os,tempfile,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def scoped_temporary(value):
    # Reject lexically external paths before resolving or touching the filesystem.
    candidate=Path(os.path.abspath(os.path.normpath(value)))
    if not candidate.is_relative_to(ROOT):raise ValueError('pytest_temporary_path_outside_project')
    resolved=candidate.resolve()
    if not resolved.is_relative_to(ROOT):raise ValueError('pytest_temporary_link_outside_project')
    return resolved

def configure(config):
    import pytest
    try:
        shared=scoped_temporary(ROOT/'.cache/tmp/pytest');shared.mkdir(parents=True,exist_ok=True)
        requested=config.option.basetemp
        base=scoped_temporary(requested if requested is not None else ROOT/'.cache/pytest-tmp'/f'{os.getpid()}-{uuid.uuid4().hex}')
        try:relative=base.relative_to(ROOT/'.cache')
        except ValueError:raise ValueError('pytest_basetemp_requires_dedicated_scratch_namespace')
        if not relative.parts or not (relative.parts[0]=='pytest' or relative.parts[0].startswith('pytest-')) or relative.parts==('pytest-tmp',):raise ValueError('pytest_basetemp_requires_dedicated_scratch_namespace')
        base.parent.mkdir(parents=True,exist_ok=True)
    except ValueError as error:raise pytest.UsageError(str(error)) from error
    for key in ('TEMP','TMP','TMPDIR'):os.environ[key]=str(shared)
    tempfile.tempdir=str(shared);config.option.basetemp=str(base)
