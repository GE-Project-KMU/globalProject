from pathlib import Path
from types import SimpleNamespace as N
import shutil,uuid
import numpy as np
import pytest
from agent.paths import ROOT,sha256
from agent.recorded_zed import RecordedZedReader,RecordedZedError

@pytest.fixture
def files():
    folder=ROOT/'.cache/test-recorded-zed'/uuid.uuid4().hex;folder.mkdir(parents=True)
    raw=folder/'original';raw.mkdir();source=raw/'fake.svo2';source.write_bytes(b'FAKE SDK CONTRACT ONLY')
    settings=folder/'settings';settings.mkdir();(settings/'SN123.conf').write_text('contract fixture, not real calibration')
    yield dict(source=source,settings_dir=settings,work_dir=folder/'work',expected_serial=123)
    if not folder.resolve().is_relative_to(ROOT.resolve()):raise ValueError('unsafe_test_cleanup')
    shutil.rmtree(folder)

def sdk(errors=None):
    module=N(ERROR_CODE=N(SUCCESS=0,END_OF_SVOFILE_REACHED=9),UNIT=N(METER=1),DEPTH_MODE=N(NONE=0,ULTRA=1),
        VIEW=N(LEFT=0,RIGHT=1),TIME_REFERENCE=N(IMAGE=7),MEASURE=N(DEPTH=3),RuntimeParameters=lambda:N())
    class Init:
        def set_from_svo_file(self,path):self.source=path
    class Mat:
        def get_data(self):return self.data
    class Camera:
        def __init__(self):self.grabs=0;self.closed=0;self.errors=list(errors if errors is not None else [0,0,9]);self.serial=123;self.retrieve_error=0;self.timestamp=1730000000123456789;self.depth_error=0;module.camera=self
        def open(self,init):self.init=init;return getattr(module,'open_error',0)
        def close(self):self.closed+=1
        def get_sdk_version(self):return '5.fake-contract'
        def get_camera_information(self):return N(serial_number=getattr(module,'serial',self.serial),camera_model='ZED2i',camera_configuration=N(resolution=N(width=8,height=4),fps=30))
        def get_svo_number_of_frames(self):return 2
        def grab(self,params):self.grabs+=1;return self.errors.pop(0)
        def retrieve_image(self,mat,view):mat.data=np.full((4,8,4),10+view,np.uint8);self.last_image=mat.data;return self.retrieve_error
        def retrieve_measure(self,mat,measure):mat.data=np.full((4,8),2.,np.float32);mat.data[0,0]=np.inf;return self.depth_error
        def get_timestamp(self,reference):self.reference=reference;return N(get_nanoseconds=lambda:self.timestamp)
        def get_svo_position(self):return self.grabs-1
    module.InitParameters=Init;module.Mat=Mat;module.Camera=Camera
    return module

def test_original_timestamp_views_eof_and_readonly(files):
    before=sha256(files['source']);sl=sdk()
    with RecordedZedReader(**files,sl_module=sl) as reader:
        sample=reader.read();assert sample['timestamp_ns']==1730000000123456789
        assert sample['utc'] is None and sample['gnss'] is None and sample['depth_m'] is None
        assert sample['left'].shape==(4,8,3) and np.all(sample['left']==10) and np.all(sample['right']==11)
        sl.camera.last_image[:]=0;assert np.all(sample['right']==11)
        assert sl.camera.reference==sl.TIME_REFERENCE.IMAGE
        assert reader.read() is not None;assert reader.read() is None;assert reader.eof
        count=sl.camera.grabs;assert reader.read() is None and count==sl.camera.grabs
        assert reader.metadata['frame_count']==2 and reader.metadata['sdk_internal_scope_verified'] is False
    assert sl.camera.closed==1 and sha256(files['source'])==before
    with pytest.raises(RecordedZedError,match='closed'):reader.read()

def test_scoped_init_no_realtime_or_neural(files):
    sl=sdk()
    with RecordedZedReader(**files,sl_module=sl):
        init=sl.camera.init
        assert init.optional_settings_path.endswith(('/', '\\'))
        assert Path(init.sdk_verbose_log_file).is_relative_to(files['work_dir'])
        assert init.depth_mode==sl.DEPTH_MODE.NONE and init.svo_real_time_mode is False
        assert init.camera_disable_self_calib is True

def test_classical_depth_invalid_pixels_nan(files):
    sl=sdk()
    with RecordedZedReader(**files,sl_module=sl,depth=True) as reader:
        value=reader.read()['depth_m'];assert np.isnan(value[0,0]) and value[1,1]==2
        assert sl.camera.init.depth_mode==sl.DEPTH_MODE.ULTRA

@pytest.mark.parametrize('operation',['grab','retrieve_image','retrieve_depth'])
def test_errors_are_not_eof_and_release(files,operation):
    sl=sdk(errors=[4] if operation=='grab' else [0])
    with RecordedZedReader(**files,sl_module=sl,depth=operation=='retrieve_depth') as reader:
        if operation=='retrieve_image':sl.camera.retrieve_error=4
        if operation=='retrieve_depth':sl.camera.depth_error=4
        with pytest.raises(RecordedZedError,match='failed'):reader.read()
        assert reader.eof is False and sl.camera.closed==1

def test_timestamp_regression_rejected_without_fake_clock(files):
    sl=sdk()
    with RecordedZedReader(**files,sl_module=sl) as reader:
        reader.read();sl.camera.timestamp-=1
        with pytest.raises(RecordedZedError,match='timestamp'):reader.read()

@pytest.mark.parametrize('kind',['open','serial'])
def test_open_and_serial_failures_close(files,kind):
    sl=sdk()
    if kind=='open':sl.open_error=4
    else:sl.serial=999
    with pytest.raises(RecordedZedError):RecordedZedReader(**files,sl_module=sl)
    assert sl.camera.closed==1

def test_missing_calibration_before_sdk_open(files):
    sl=sdk();(files['settings_dir']/'SN123.conf').unlink()
    with pytest.raises(FileNotFoundError):RecordedZedReader(**files,sl_module=sl)
    assert not hasattr(sl,'camera')

def test_workspace_boundary_and_source_folder_write_guard(files):
    sl=sdk();bad=dict(files,work_dir=ROOT.parent/'forbidden')
    with pytest.raises(ValueError,match='outside_workspace'):RecordedZedReader(**bad,sl_module=sl)
    with pytest.raises(ValueError,match='modify_source'):RecordedZedReader(**dict(files,work_dir=files['source'].parent/'work'),sl_module=sl)
    assert not hasattr(sl,'camera')

@pytest.mark.parametrize('serial',[None,True,0,-1,'123'])
def test_verified_serial_required(files,serial):
    with pytest.raises(ValueError,match='serial_required'):RecordedZedReader(**dict(files,expected_serial=serial),sl_module=sdk())
