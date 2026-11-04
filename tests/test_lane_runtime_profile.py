"""Runtime scope is a gate, not a relaxed floating point comparator."""
from types import SimpleNamespace
import pytest
from agent import models
def network(version='1.31.1',providers=None,backend='onnxruntime-cpu'):
    return SimpleNamespace(backend=backend,backend_version=version,session=SimpleNamespace(get_providers=lambda:providers if providers is not None else ['CPUExecutionProvider']))
@pytest.fixture(autouse=True)
def windows(monkeypatch):
    monkeypatch.setattr(models.platform,'system',lambda:'Windows');monkeypatch.setattr(models.platform,'machine',lambda:'AMD64')
def test_exact_supported_session():models.enforce_runtime_profile({'required_runtime_profile':dict(models.SUPPORTED_CPU_PROFILE)},network())
@pytest.mark.parametrize('version',['1.20.1','1.31.0','1.32.0','1.31.1rc1','1.31.1+custom',None])
def test_version_fails_closed(version):
    with pytest.raises(ValueError,match='profile_mismatch'):models.enforce_runtime_profile({'required_runtime_profile':dict(models.SUPPORTED_CPU_PROFILE)},network(version))
@pytest.mark.parametrize('providers',[[],['CUDAExecutionProvider'],['CPUExecutionProvider','CUDAExecutionProvider']])
def test_actual_session_providers(providers):
    with pytest.raises(ValueError,match='providers_mismatch'):models.enforce_runtime_profile({'required_runtime_profile':dict(models.SUPPORTED_CPU_PROFILE)},network(providers=providers))
@pytest.mark.parametrize('profile',[None,{},False,dict(models.SUPPORTED_CPU_PROFILE,version='1.20.1')])
def test_malformed_profile(profile):
    with pytest.raises(ValueError,match='unsupported_required'):models.enforce_runtime_profile({'required_runtime_profile':profile},network())
@pytest.mark.parametrize('field,value',[('system','Linux'),('machine','ARM64')])
def test_unsupported_platform(monkeypatch,field,value):
    monkeypatch.setattr(models.platform,field,lambda:value)
    with pytest.raises(ValueError,match='profile_mismatch'):models.enforce_runtime_profile({'required_runtime_profile':dict(models.SUPPORTED_CPU_PROFILE)},network())
def test_legacy_policy_unchanged():models.enforce_runtime_profile({},object())
@pytest.mark.parametrize('backend',['opencv-classic-diagnostic','custom'])
def test_backend_not_substituted(backend):
    with pytest.raises(ValueError,match='profile_mismatch'):models.enforce_runtime_profile({'required_runtime_profile':dict(models.SUPPORTED_CPU_PROFILE)},network(backend=backend))
def test_missing_loaded_session_rejected():
    n=network();n.session=None
    with pytest.raises(ValueError,match='providers_mismatch'):models.enforce_runtime_profile({'required_runtime_profile':dict(models.SUPPORTED_CPU_PROFILE)},n)
