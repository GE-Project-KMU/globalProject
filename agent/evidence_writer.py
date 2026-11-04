"""Bounded optional PNG overlap; unchanged lossless encoding and fail-closed flush."""
from concurrent.futures import Future,ThreadPoolExecutor
from pathlib import Path
import os,uuid
import cv2
import numpy as np
from .paths import sha256
from .material_policy import safe_file,source_deadline


def writer_profile(value):
    if value not in ('default','overlap'):raise ValueError('unknown_evidence_io_profile')
    return value


class EvidenceWriter:
    def __init__(self,profile='default'):
        self.profile=writer_profile(profile);self.closed=False;self.pending=None;self.quality_pending=None;self.created=[]
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='road-png') if profile=='overlap' else None

    def __enter__(self):return self

    def __exit__(self,kind,error,traceback):
        if error is None:
            try:self.close()
            except BaseException as primary:
                self._cleanup_note(primary)
                raise
        else:
            try:self.close()
            except BaseException as secondary:error.add_note('Evidence writer flush also failed: '+type(secondary).__name__)
            self._cleanup_note(error)

    def _cleanup_note(self,primary):
        try:self._discard_owned()
        except BaseException as secondary:
            primary.add_note('Evidence cleanup failed; primary failure preserved: '+type(secondary).__name__)

    def submit(self,path,image,expires_at):
        if self.closed:raise RuntimeError('evidence_writer_closed')
        path=Path(path);path=safe_file(path,path.parent,exists=False,suffixes=('.png',))
        source_deadline(expires_at,required=True)
        if not isinstance(image,np.ndarray) or image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:
            raise ValueError('uint8_bgr_evidence_required')
        # At most one snapshot/write is in flight. A slow writer applies backpressure.
        if self.pending is not None:self.pending.result()
        if self.pool:self.pending=self.pool.submit(self._write,path,image.copy(),expires_at,True)
        else:
            self.pending=Future()
            try:self.pending.set_result(self._write(path,image,expires_at,False))
            except BaseException as error:self.pending.set_exception(error);raise
        return self.pending

    def quality(self,image,compute):
        if self.closed:raise RuntimeError('evidence_writer_closed')
        if self.quality_pending is not None:self.quality_pending.result()
        # The caller waits before advancing/reusing the frame. One quality job
        # plus one PNG snapshot is the complete bound, including queued work.
        if self.pool:self.quality_pending=self.pool.submit(compute,image)
        else:
            self.quality_pending=Future()
            try:self.quality_pending.set_result(compute(image))
            except BaseException as error:self.quality_pending.set_exception(error);raise
        return self.quality_pending

    def _write(self,path,image,expires_at,staged):
        source_deadline(expires_at,required=True)
        temporary=path.with_name(path.stem+'.pending-'+uuid.uuid4().hex+'.png') if staged else path
        if staged and path.exists():raise ValueError('overlap_requires_new_evidence_path')
        primary=None
        try:
            if not cv2.imwrite(str(temporary),image):raise RuntimeError('image_write_failed')
            digest=sha256(temporary)
            source_deadline(expires_at,required=True)
            if staged:
                # Atomic fail-if-exists on Windows and POSIX. rename would
                # replace a concurrently-created destination on POSIX.
                try:os.link(temporary,path)
                except FileExistsError:raise ValueError('overlap_requires_new_evidence_path') from None
                self.created.append((path,digest))
            return digest
        except BaseException as error:
            primary=error
            raise
        finally:
            if staged:
                try:
                    if temporary.exists():temporary.unlink()
                except BaseException as secondary:
                    if primary is None:raise
                    primary.add_note('Temporary PNG cleanup also failed: '+type(secondary).__name__)

    def _discard_owned(self):
        # Only newly created overlap files with unchanged bytes belong to this attempt.
        for path,digest in self.created:
            checked=safe_file(path,path.parent,exists=False,suffixes=('.png',))
            if checked.exists() and sha256(checked)==digest:checked.unlink()

    def close(self):
        if self.closed:return
        self.closed=True
        try:
            if self.pending is not None:self.pending.result()
            if self.quality_pending is not None:self.quality_pending.result()
        finally:
            if self.pool:self.pool.shutdown(wait=True,cancel_futures=True)
