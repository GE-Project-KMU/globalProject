"""Make a private local-only real-road reference pack, never observation evidence.

Publisher semantic road pixels select conservative bottom crops. This machine
criterion is NOT human privacy review, detection, wear assessment or approval.
Original images, research splits, source expiry and inventories are unchanged.
"""
import base64,hashlib,io,json,sys
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from agent.paths import atomic_json,sha256

def main():
    source=ROOT/'data/public/rtk';dest=ROOT/'data/demo-reference';pack_path=dest/'road-reference-pack.json'
    if pack_path.exists():raise ValueError('Reference pack already frozen; original expiry is never renewed')
    acquisition=source/'acquisition.json';evidence=source/'license-evidence.json';expiry=json.loads(acquisition.read_text())['expires_at']
    if datetime.fromisoformat(expiry)<=datetime.now(timezone.utc):raise ValueError('Source data expired')
    if json.loads(evidence.read_text())['license_id']!='CC-BY-4.0':raise ValueError('License mismatch')
    original=source/'original/RTK';entries=[];used=set();dest.mkdir(parents=True,exist_ok=True);(dest/'crops').mkdir(exist_ok=True)
    for target in (4,11,None):
        selected=None
        for index in range(125,550):
            if index in used:continue
            image=original/'image'/f'{index:06d}.png';label=original/'label'/f'{index:06d}.png'
            if not image.exists():
                image=next((p for p in (original/'image').glob('*.png') if int(p.stem)==index),None)
                label=(original/'label'/image.name) if image else None
            if not image or not label or not label.exists():continue
            with Image.open(label) as mask:codes=np.asarray(mask)
            for y0 in (144,160,176,192):
                for x0 in (32,64,96,128,160):
                    box=(x0,y0,x0+160,y0+64);region=codes[y0:y0+64,x0:x0+160]
                    if region.shape!=(64,160) or not np.isin(region,np.arange(1,13)).all():continue
                    if target is not None and int((region==target).sum())<8:continue
                    selected=(image,label,box,index);break
                if selected:break
            if selected:break
        if not selected:raise ValueError('No suitable machine-selected TRAIN road crop')
        image,label,box,index=selected;used.add(index)
        with Image.open(image) as photo:crop=photo.convert('RGB').crop(box)
        stream=io.BytesIO();crop.save(stream,format='JPEG',quality=90);raw=stream.getvalue();number=len(entries)+1
        crop_path=dest/'crops'/f'road-reference-{number}.jpg';crop_path.write_bytes(raw)
        entries.append({'image':{'title':f'실노면 연구 참고 {number}','data_url':'data:image/jpeg;base64,'+base64.b64encode(raw).decode(),'width':crop.width,'height':crop.height},'evidence':{'source_path':image.relative_to(ROOT).as_posix(),'source_sha256':sha256(image),'publisher_label_path':label.relative_to(ROOT).as_posix(),'publisher_label_sha256':sha256(label),'crop_xyxy_px':list(box),'derived_path':crop_path.relative_to(ROOT).as_posix(),'derived_sha256':hashlib.sha256(raw).hexdigest(),'source_split':'train','expires_at':expiry,'human_privacy_reviewed':False,'inference_performed':False,'machine_selection_only':'All retained pixels have publisher road-surface/road-marking/damage class1..12; does not replace privacy review.'}})
    pack={'schema_version':1,'pack_name':'실노면 로컬 참고사진','source_name':'Road Traversing Knowledge (RTK), UFSC / Mendeley Data','attribution':'Thiago Rateke, Aldo von Wangenheim, Rafael Toledo · DOI 10.17632/hssswvmjwf.1','license_id':'CC-BY-4.0','modifications':'원본 하단 도로 영역 자르기 및 JPEG 변환','expires_at':expiry,'source_expires_at':expiry,'local_only':True,'human_privacy_reviewed':False,'inference_performed':False,'images':[r['image'] for r in entries]}
    atomic_json(pack_path,pack)
    atomic_json(dest/'preview-items-receipt.json',{'format':'local-reference-preview-provenance-v1','created_at':datetime.now(timezone.utc).isoformat(),'expires_at':expiry,'source_expiry_not_extended':True,'source_acquisition_path':acquisition.relative_to(ROOT).as_posix(),'source_acquisition_sha256':sha256(acquisition),'license_evidence_path':evidence.relative_to(ROOT).as_posix(),'license_evidence_sha256':sha256(evidence),'pack_sha256':sha256(pack_path),'original_records_modified':False,'local_only':True,'human_privacy_reviewed':False,'inference_performed':False,'items':[r['evidence'] for r in entries]})
    print(json.dumps({'local_only':True,'images':len(entries),'expires_at':expiry,'original_records_modified':False,'pack_sha256':sha256(pack_path),'human_privacy_reviewed':False,'inference_performed':False}))
if __name__=='__main__':main()
