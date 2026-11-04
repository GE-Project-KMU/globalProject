"""Official API transport retry; frozen blind selection source is not modified."""
from __future__ import annotations
import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import atomic_json,sha256
from scripts import prepare_rdd2022_blind as selection

SELECTION_SHA='199dde6f939a23423d6d8db2db16d6564bb92b0dcf7d9994bdc59f72446260b3'
PROTOCOL_SHA='586417a773e22d07cda40bbc91a103de4d194de4695d770fbbfc16a40bdd47c2'
OFFICIAL_API='https://api.figshare.com/v2/file/download/38030910'

def run():
    folder=selection.scoped_output(selection.DEST)
    if sha256(Path(selection.__file__))!=SELECTION_SHA:raise ValueError('frozen_selection_source_changed')
    if sha256(folder/'selection-protocol.json')!=PROTOCOL_SHA:raise ValueError('frozen_selection_protocol_changed')
    source=json.loads((folder/'source.json').read_text(encoding='utf-8'))
    expiry=selection.aware(source['expires_at'])
    if expiry<=datetime.now(timezone.utc):raise ValueError('original_source_expired')
    receipt=folder/'official-api-transport-attempt.json'
    if receipt.exists():raise ValueError('attempt_receipt_already_exists')
    record={'started_at':datetime.now(timezone.utc).isoformat(),'transport_url':OFFICIAL_API,
            'reason':'Original official ndownloader entry returned HTTP403; official API entry returned exact HTTP206 range.',
            'selection_source_sha256':SELECTION_SHA,'selection_protocol_sha256':PROTOCOL_SHA,
            'transport_source_sha256':sha256(Path(__file__)),'source_expires_at':expiry.isoformat(),
            'selection_unchanged':True,'model_inferences':0,'whole_outer_MD5_verified':False,'status':'running'}
    atomic_json(receipt,record)
    # Only the public official transport entry changes in this process. Range,
    # redirect host, size, CRC, identity and selection checks remain frozen.
    selection.URL=OFFICIAL_API
    try:
        selection.prepare()
    except BaseException as error:
        record.update(status='failed',error_type=type(error).__name__,finished_at=datetime.now(timezone.utc).isoformat())
        atomic_json(receipt,record)
        raise
    record.update(status='completed',finished_at=datetime.now(timezone.utc).isoformat())
    atomic_json(receipt,record)

if __name__=='__main__':run()
