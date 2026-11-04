"""Exact shared1920 -> four960 views; no Torch import or silent GT truncation."""
from __future__ import annotations
from datetime import datetime,timezone
import hashlib,json
import cv2,numpy as np
from agent.paths import workspace_path,sha256

SHARED=1920;TILE=960;MAX_LABELS=512
DEVELOPMENT='data/training/research-redaction-cycle2-v7/development.json'

def source_boxes(boxes,width,height):
    if type(width) is not int or type(height) is not int or min(width,height)<=0:raise ValueError('positive_original_dimensions_required')
    result=np.asarray(boxes,dtype=np.float64).reshape(-1,4).copy()
    if len(result)>MAX_LABELS:raise ValueError('original_GT_over512_rejected_no_truncation')
    if not np.isfinite(result).all() or (result[:,:2]<0).any() or (result[:,2:]<=0).any():raise ValueError('valid_original_face_boxes_required')
    if (result[:,0]+result[:,2]>width+1e-6).any() or (result[:,1]+result[:,3]>height+1e-6).any():raise ValueError('original_face_box_outside_image')
    return result

def shared_view(image,boxes,*,flip=False,hsv=None):
    if image.ndim!=3 or image.shape[2]!=3 or image.dtype!=np.uint8:raise ValueError('uint8_BGR_original_required')
    height,width=image.shape[:2];boxes=source_boxes(boxes,width,height)
    pixels=image
    if hsv is not None:
        hue,saturation,value=hsv
        if not all(np.isfinite(x) for x in hsv):raise ValueError('finite_HSV_required')
        converted=cv2.cvtColor(pixels,cv2.COLOR_BGR2HSV).astype(np.float32)
        converted[:,:,0]=(converted[:,:,0]+hue)%180
        converted[:,:,1]=np.clip(converted[:,:,1]*saturation,0,255)
        converted[:,:,2]=np.clip(converted[:,:,2]*value,0,255)
        pixels=cv2.cvtColor(converted.astype(np.uint8),cv2.COLOR_HSV2BGR)
    if flip:
        pixels=pixels[:,::-1].copy();boxes[:,0]=width-boxes[:,0]-boxes[:,2]
    scale=min(SHARED/width,SHARED/height)
    resized=cv2.resize(pixels,(max(1,round(width*scale)),max(1,round(height*scale))))
    canvas=np.full((SHARED,SHARED,3),114,np.uint8);canvas[:resized.shape[0],:resized.shape[1]]=resized
    # Nominal scale matches application inverse coordinates. Raster dimensions
    # round exactly as application preprocessing; GT stays float64 until packing.
    return canvas,boxes*scale,{'nominal_scale':scale,'resized_width':resized.shape[1],'resized_height':resized.shape[0]}

def tile_targets(scaled_boxes,tile):
    if type(tile) is not int or not 0<=tile<4:raise ValueError('quadrant0_to3_required')
    boxes=np.asarray(scaled_boxes,dtype=np.float64).reshape(-1,4)
    if len(boxes)>MAX_LABELS:raise ValueError('original_GT_over512_rejected_no_truncation')
    if not np.isfinite(boxes).all() or (boxes[:,2:]<=0).any():raise ValueError('finite_positive_shared_boxes_required')
    left,top=(tile%2)*TILE,(tile//2)*TILE
    packed=np.zeros((MAX_LABELS,5),np.float32);parents=[]
    for index,(x,y,w,h) in enumerate(boxes):
        x1,y1=max(0.,x-left),max(0.,y-top);x2,y2=min(float(TILE),x+w-left),min(float(TILE),y+h-top)
        if x2<=x1 or y2<=y1:continue
        if len(parents)>=MAX_LABELS:raise ValueError('visible_GT_over512_rejected_no_truncation')
        packed[len(parents)]=[0.,(x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1];parents.append(index)
    return packed,parents

def tile_view(canvas,scaled_boxes,tile):
    if canvas.shape!=(SHARED,SHARED,3) or canvas.dtype!=np.uint8:raise ValueError('exact_shared1920_canvas_required')
    labels,parents=tile_targets(scaled_boxes,tile);left,top=(tile%2)*TILE,(tile//2)*TILE
    pixels=np.ascontiguousarray(canvas[top:top+TILE,left:left+TILE].transpose(2,0,1),dtype=np.float32)
    return pixels,labels,parents

def check_conservation(scaled_boxes):
    boxes=np.asarray(scaled_boxes,np.float64).reshape(-1,4);area=np.zeros(len(boxes),np.float64);seen=set();counts=[]
    for tile in range(4):
        labels,parents=tile_targets(boxes,tile);counts.append(len(parents));seen.update(parents)
        for target,parent in zip(labels,parents):area[parent]+=float(target[3])*float(target[4])
    expected=boxes[:,2]*boxes[:,3]
    if seen!=set(range(len(boxes))) or not np.allclose(area,expected,rtol=2e-6,atol=1e-4):raise ValueError('tile_GT_coverage_or_area_not_conserved')
    return {'original_GT':len(boxes),'visible_GT_per_tile':counts,'all_original_GT_covered':True,'clipped_area_conserved':True}

def load_development():
    path=workspace_path(DEVELOPMENT,exists=True);rows=json.loads(path.read_text(encoding='utf-8'))
    if any(r['split'] not in ('train','validation') for r in rows):raise ValueError('development_only_source_required')
    train=[r for r in rows if r['split']=='train' and r['evaluation_class']=='face'];validation=[r for r in rows if r['split']=='validation']
    if len(train)!=3000 or sum(r['evaluation_class']=='face' for r in validation)!=400 or sum(r['evaluation_class']=='license_plate' for r in validation)!=108:raise ValueError('frozen_TRAIN3000_VAL508_required')
    for row in train+validation:
        if row.get('ignore_bboxes_xywh_px') or row.get('known_label_coverage')!=[row['evaluation_class']]:raise ValueError('explicit_publisher_complete_knownclass_required')
        if row.get('human_privacy_reviewed') is not False or row.get('not_for_deployment') is not True:raise ValueError('restricted_research_required')
        expiry=datetime.fromisoformat(row['expires_at'].replace('Z','+00:00'))
        if expiry.tzinfo is None or expiry<=datetime.now(timezone.utc):raise ValueError('original_source_expired')
        boxes=[o['bbox_xywh_px'] for o in row['objects'] if o['class']==row['evaluation_class']]
        source_boxes(boxes,row['image']['width'],row['image']['height'])
    return sorted(train,key=lambda r:r['sample_id']),sorted(validation,key=lambda r:r['sample_id']),sha256(path)

def read_image(row):
    expiry=datetime.fromisoformat(row['expires_at'].replace('Z','+00:00'))
    if expiry.tzinfo is None or expiry<=datetime.now(timezone.utc):raise ValueError('original_source_expired_before_pixel_read')
    path=workspace_path(row['path'],exists=True)
    if not 0<path.stat().st_size<=32*1024*1024:raise ValueError('bounded_publisher_image_required')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=row['sha256']:raise ValueError('original_image_SHA_changed')
    image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
    if image is None or image.shape[:2]!=(row['image']['height'],row['image']['width']):raise ValueError('original_image_geometry_changed')
    return image
