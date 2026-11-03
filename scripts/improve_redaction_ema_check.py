"""CPU exact-equivalence evidence for caching an EMA source state dictionary."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from agent.redaction_network import build_redaction_network
from agent.paths import atomic_json,sha256

def main():
    torch.set_num_threads(2);torch.manual_seed(20261104)
    source=build_redaction_network().face;old=deepcopy(source);new=deepcopy(source)
    with torch.no_grad():
        for value in source.state_dict().values():
            if value.dtype.is_floating_point:value.add_(.001)
        decay=.9998*(1-__import__('math').exp(-123/2000))
        start=time.perf_counter()
        for key,value in old.state_dict().items():
            incoming=source.state_dict()[key]
            if value.dtype.is_floating_point:value.mul_(decay).add_(incoming,alpha=1-decay)
            else:value.copy_(incoming)
        old_seconds=time.perf_counter()-start
        start=time.perf_counter();incoming_state=source.state_dict()
        for key,value in new.state_dict().items():
            incoming=incoming_state[key]
            if value.dtype.is_floating_point:value.mul_(decay).add_(incoming,alpha=1-decay)
            else:value.copy_(incoming)
        new_seconds=time.perf_counter()-start
    a,b=old.state_dict(),new.state_dict();equal=all(torch.equal(a[k],b[k]) for k in a)
    result={'device':'CPU','state_tensor_count':len(a),'optimizer_or_ema_algorithm_changed':False,'ema_decay':decay,'repeated_state_dict_seconds':old_seconds,'cached_state_dict_seconds':new_seconds,'all_ema_tensors_bit_identical':equal,'factory_sha256':sha256(ROOT/'agent/redaction_network.py'),'corrected_trainer_sha256':sha256(ROOT/'scripts/improve_redaction_train.py'),'scope':'Dictionary overhead and exact EMA update equivalence only; not an inference throughput or model metric.'}
    atomic_json(ROOT/'docs/reports/REDACTION-V4-EMA-OVERHEAD.json',result)
    if not equal:raise AssertionError('EMA_numerical_change_not_allowed')
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
