"""Read-only device-memory observer and identity-bound own-child termination.

nvidia-smi device FB accounting under WDDM is sampled OS/driver evidence, not an
exact process sum or a hardware memory quota. No other process is controlled.
"""
from collections import deque
from decimal import Decimal,InvalidOperation
import os,re,subprocess,threading,time
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[1]
LIMIT=9*1024**3

def parse_used_mib(line):
 text=line.strip()
 if not re.fullmatch(r'\d+(?:\.\d+)?',text):raise ValueError('invalid_nvidia_device_memory_query')
 return int((Decimal(text)*1024**2).to_integral_value(rounding='ROUND_CEILING'))

class OwnedChild:
 def __init__(self,process):
  self.process=process;self.pid=process.pid;info=psutil.Process(self.pid);self.creation_time=info.create_time();self.parent_pid=os.getpid()
  if info.ppid()!=self.parent_pid:raise ValueError('direct_own_child_required')
 def identity_matches(self):
  if self.process.pid!=self.pid or self.process.poll() is not None:return False
  try:
   info=psutil.Process(self.pid)
   return info.create_time()==self.creation_time and info.ppid()==self.parent_pid and info.is_running()
  except psutil.NoSuchProcess:return False
 def stop(self):
  if not self.identity_matches():return False
  self.process.terminate()
  try:self.process.wait(timeout=10)
  except subprocess.TimeoutExpired:
   if not self.identity_matches():raise ValueError('child_identity_changed_before_force_stop')
   self.process.kill();self.process.wait(timeout=10)
  return True

class DeviceMemoryObserver:
 def __init__(self,log_path,*,command=None,limit=LIMIT):
  path=Path(log_path).resolve()
  if not path.is_relative_to(ROOT):raise ValueError('observer_log_outside_workspace')
  path.parent.mkdir(parents=True,exist_ok=True);self.path=path
  self.command=command or ['nvidia-smi','--id=0','--query-gpu=memory.used','--format=csv,noheader,nounits','--loop-ms=100'];self.limit=limit
  self.samples=deque(maxlen=100000);self.total_samples=0;self.max_used=0;self.errors=[];self.violation=False;self.lock=threading.Lock();self.started_ns=None;self.owned=None;self.thread=None;self.closing=False
 def start(self,timeout=5):
  self.log=self.path.open('w',encoding='utf-8');self.started_ns=time.perf_counter_ns()
  try:
   process=subprocess.Popen(self.command,cwd=ROOT,stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1);self.owned=OwnedChild(process)
   self.thread=threading.Thread(target=self._read,daemon=True);self.thread.start();end=time.monotonic()+timeout
   while time.monotonic()<end:
    self.raise_if_unsafe(require_sample=False)
    if self.total_samples>=2:return self
    if process.poll() is not None:raise ValueError('device_observer_exited_before_ready')
    time.sleep(.01)
   raise ValueError('device_observer_first_samples_timeout')
  except BaseException:
   if self.owned is None and 'process' in locals() and process.poll() is None:
    # Exact Popen handle only: no process search or unverified external PID.
    process.terminate();process.wait(timeout=10)
   self.close();raise
 def _read(self):
  try:
   for line in self.owned.process.stdout:
    used=parse_used_mib(line);entry={'received_elapsed_ns':time.perf_counter_ns()-self.started_ns,'used_bytes':used}
    with self.lock:
     self.samples.append(entry);self.total_samples+=1;self.max_used=max(self.max_used,used);self.violation=self.violation or used>self.limit
  except BaseException as error:
   with self.lock:self.errors.append(type(error).__name__+':'+str(error))
 def raise_if_unsafe(self,*,require_sample=True,max_age_seconds=2):
  with self.lock:
   if self.errors:raise ValueError('device_observer_parse_or_runtime_failed')
   if self.violation:raise ValueError('device_reported_global_memory_over_9GiB')
   latest=self.samples[-1]['received_elapsed_ns'] if self.samples else None
  if require_sample and latest is None:raise ValueError('no_device_memory_samples')
  if latest is not None and (time.perf_counter_ns()-self.started_ns-latest)>max_age_seconds*1e9:raise ValueError('stale_device_memory_samples')
  if self.owned and self.owned.process.poll() is not None and not self.closing:raise ValueError('device_observer_unexpected_exit')
 def wait_additional_samples(self,count=2,timeout=3):
  target=self.total_samples+count;end=time.monotonic()+timeout
  while self.total_samples<target:
   self.raise_if_unsafe()
   if time.monotonic()>=end:raise ValueError('device_post_samples_timeout')
   time.sleep(.01)
  self.raise_if_unsafe()
 def summary(self):
  import statistics
  with self.lock:rows=list(self.samples);maximum=self.max_used;count=self.total_samples;errors=self.errors[:];violation=self.violation
  gaps=[(b['received_elapsed_ns']-a['received_elapsed_ns'])/1e6 for a,b in zip(rows,rows[1:])]
  return {'scope':'nvidia-smi device-level reported FB memory; WDDM OS/driver accounting, not exact process sum','query_interval_requested_ms':100,'timestamps_are_receive_times_not_hardware_sample_times':True,'cannot_guarantee_no_transient_excess_between_samples':True,'total_samples':count,'retained_samples':len(rows),'max_device_reported_used_bytes':maximum,'observed_receive_interval_ms':{'min':min(gaps),'median':statistics.median(gaps),'max':max(gaps)} if gaps else None,'over_budget_observed':violation,'errors':errors}
 def sample_rows(self):
  with self.lock:return list(self.samples)
 def close(self):
  self.closing=True
  if self.owned:self.owned.stop()
  if self.thread:self.thread.join(timeout=10)
  if getattr(self,'log',None):self.log.close()
