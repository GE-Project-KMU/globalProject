"""Optional recorded SVO/SVO2 SDK reader, separate from live capture.

No SDK installer, neural model API or clock/GNSS synthesis is invoked here.
There is no explicit calibration download request, but SDK open can perform
internal downloads: Python path checks are NOT a sandbox guarantee. Real
deployment needs an independently scoped SDK runtime.
"""
import os
from pathlib import Path
import numpy as np
from .paths import workspace_path,sha256

class RecordedZedError(RuntimeError):pass

class RecordedZedReader:
    def __init__(self,source,*,settings_dir,work_dir,expected_serial,depth=False,sl_module=None):
        self.camera=None;self.closed=False;self.eof=False;self.last_timestamp=None
        self.source=workspace_path(source,exists=True)
        if self.source.suffix.lower() not in ('.svo','.svo2') or not self.source.is_file():raise ValueError('recorded_svo_file_required')
        if type(expected_serial) is not int or expected_serial<=0:raise ValueError('verified_recorded_camera_serial_required')
        if type(depth) is not bool:raise ValueError('depth_boolean_required')
        self.settings=workspace_path(settings_dir,exists=True);self.work=workspace_path(work_dir)
        if not self.settings.is_dir():raise ValueError('settings_directory_required')
        if self.work.is_relative_to(self.source.parent):raise ValueError('work_directory_must_not_modify_source_directory')
        calibration=workspace_path(self.settings/f'SN{expected_serial}.conf',exists=True)
        if not calibration.is_file() or calibration.stat().st_size==0:raise ValueError('local_camera_calibration_required')
        if sl_module is None:
            try:import pyzed.sl as sl_module
            except ImportError as exc:raise RecordedZedError('zed_sdk_runtime_unavailable_no_automatic_install') from exc
        self.sl=sl_module;self.depth=depth;self.work.mkdir(parents=True,exist_ok=True)
        self.camera=self.sl.Camera()
        try:
            init=self.sl.InitParameters();init.set_from_svo_file(str(self.source));init.svo_real_time_mode=False
            init.optional_settings_path=str(self.settings)+os.sep
            init.sdk_verbose_log_file=str(self.work/'sdk-playback.log');init.sdk_verbose=1
            init.coordinate_units=self.sl.UNIT.METER
            init.depth_mode=self.sl.DEPTH_MODE.ULTRA if depth else self.sl.DEPTH_MODE.NONE
            init.camera_disable_self_calib=True
            error=self.camera.open(init)
            if error!=self.sl.ERROR_CODE.SUCCESS:raise RecordedZedError('svo_open_failed:'+str(error))
            info=self.camera.get_camera_information();configuration=info.camera_configuration
            if int(info.serial_number)!=expected_serial:raise RecordedZedError('recorded_serial_mismatch_download_safety_unverified')
            self.metadata={'source_format':'svo_sdk','sdk_version':str(self.camera.get_sdk_version()),'camera_serial':int(info.serial_number),
                'camera_model':str(info.camera_model),'width':int(configuration.resolution.width),'height':int(configuration.resolution.height),
                'fps':float(configuration.fps),'frame_count':int(self.camera.get_svo_number_of_frames()),
                'timestamp_reference':'SDK_TIME_REFERENCE_IMAGE_original_nanoseconds_epoch_unverified','utc':None,'gnss':None,
                'calibration_sha256':sha256(calibration),'calibration_source':'explicit_project_settings_file',
                'image_view':'rectified_LEFT_RIGHT','depth_mode':'classical_ULTRA' if depth else 'NONE',
                'depth_physical_validation':False,'sdk_internal_scope_verified':False}
        except BaseException:
            self.close();raise

    def _check(self,error,operation):
        if error!=self.sl.ERROR_CODE.SUCCESS:raise RecordedZedError('svo_'+operation+'_failed:'+str(error))

    def read(self):
        if self.closed:raise RecordedZedError('recorded_reader_closed')
        if self.eof:return None
        sl=self.sl
        try:
            error=self.camera.grab(sl.RuntimeParameters())
            if error==sl.ERROR_CODE.END_OF_SVOFILE_REACHED:self.eof=True;return None
            self._check(error,'grab')
            images=[]
            for view in (sl.VIEW.LEFT,sl.VIEW.RIGHT):
                mat=sl.Mat();self._check(self.camera.retrieve_image(mat,view),'retrieve_image')
                value=np.asarray(mat.get_data())
                if value.ndim!=3 or value.shape[2] not in (3,4) or value.dtype!=np.uint8:raise RecordedZedError('svo_image_shape_or_dtype_invalid')
                images.append(value[:,:,:3].copy())
            if images[0].shape!=images[1].shape:raise RecordedZedError('svo_stereo_shape_mismatch')
            timestamp=int(self.camera.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds())
            if timestamp<0 or (self.last_timestamp is not None and timestamp<self.last_timestamp):raise RecordedZedError('svo_image_timestamp_invalid_or_regressed')
            depth=None
            if self.depth:
                mat=sl.Mat();self._check(self.camera.retrieve_measure(mat,sl.MEASURE.DEPTH),'retrieve_depth')
                depth=np.asarray(mat.get_data()).copy()
                if depth.shape!=images[0].shape[:2] or depth.dtype.kind!='f':raise RecordedZedError('svo_depth_shape_or_dtype_invalid')
                depth[~np.isfinite(depth)|(depth<=0)]=np.nan
            self.last_timestamp=timestamp
            return {'left':images[0],'right':images[1],'depth_m':depth,'timestamp_ns':timestamp,
                'source_frame_index':int(self.camera.get_svo_position()),'utc':None,'gnss':None}
        except BaseException:
            self.close();raise

    def close(self):
        if not self.closed:
            self.closed=True
            if self.camera is not None:self.camera.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
