"""Bounded local SVO SDK probe. No images are written or uploaded; no training."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.recorded_zed import RecordedZedReader
from agent.paths import atomic_json
def main():
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--settings',required=True);p.add_argument('--work-dir',required=True);p.add_argument('--serial',type=int,required=True);p.add_argument('--max-frames',type=int,default=3);p.add_argument('--classical-depth',action='store_true');p.add_argument('--output',required=True);a=p.parse_args()
    if not 1<=a.max_frames<=100:raise ValueError('bounded_probe_frames_1_to_100_required')
    with RecordedZedReader(a.source,settings_dir=a.settings,work_dir=a.work_dir,expected_serial=a.serial,depth=a.classical_depth) as reader:
        frames=[]
        for _ in range(a.max_frames):
            sample=reader.read()
            if sample is None:break
            frames.append({'source_frame_index':sample['source_frame_index'],'timestamp_ns':sample['timestamp_ns'],'shape':list(sample['left'].shape),'depth_available':sample['depth_m'] is not None})
        result={'status':'sdk_recorded_probe_completed','metadata':reader.metadata,'frames':frames,'reached_eof':reader.eof,'images_saved':False,'uploaded':False,'training_started':False}
        atomic_json(a.output,result)
    print(json.dumps({'status':result['status'],'frames':len(frames),'reached_eof':result['reached_eof']}))
if __name__=='__main__':
    try:main()
    except Exception as exc:print(json.dumps({'status':'blocked_or_failed','reason':str(exc)}));sys.exit(2)
