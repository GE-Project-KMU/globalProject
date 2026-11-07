"""Create a private offline AI-draft review page without reading image bytes.

The browser deadline check is an auxiliary client-clock check, not authorization
or a deletion mechanism. File copies and a changed client clock cannot be revoked
by a static HTML page. Keep HTML and labels private and within the approved TTL.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.validate_visual_annotation_bundle import MAX_BYTES, scoped_file, validate_bundle

OUTPUT_ROOT = ROOT / 'data' / 'recorded-drive-ai-labeling' / 'session-v1'

TEMPLATE = r'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src 'self' file:; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'none'; font-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'">
<title>AI 라벨 초안 검수</title><style>
:root{color-scheme:dark;font:16px system-ui,sans-serif;background:#101522;color:#edf1fa}body{margin:0 auto;max-width:1400px;padding:20px}h1{font-size:24px}.notice{background:#243047;padding:14px;border-radius:8px;line-height:1.6}.controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:16px 0}button,select{font:inherit;padding:9px;border:1px solid #7284a5;border-radius:5px;background:#202c44;color:inherit}button:disabled{opacity:.4}.stage{position:relative;width:100%;background:#080c14;min-height:100px}.stage img{display:block;width:100%;height:auto}.stage svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}.legend{display:flex;gap:16px;flex-wrap:wrap;margin:12px 0}.key{border-left:6px solid var(--c);padding-left:7px}#details{white-space:pre-wrap;overflow-wrap:anywhere;background:#1b2538;padding:15px;line-height:1.65}.held{color:#ffd36b}#expired{color:#ff8d91;font-weight:bold}label{display:flex;gap:6px;align-items:center}footer{margin-top:18px;color:#b9c7dd;font-size:14px;line-height:1.6}svg text{paint-order:stroke;stroke:#101522;stroke-width:2px;fill:white;font:16px sans-serif}
</style></head><body>
<h1>AI 시각 라벨 초안 · 사람 검수용</h1>
<div class="notice">미검수 AI 초안입니다. 학습 내보내기·독립 평가 정답·확정 음성으로 사용할 수 없습니다.<br>
이 묶음은 <strong id="count"></strong>개의 대표 키프레임이며 전체 프레임을 검수한 결과가 아닙니다. 같은 주행·스테레오 자료는 하나의 누출 방지 그룹입니다.<br>
처리·보관 기한: <strong id="deadline"></strong> · 기한 이후 이미지 로드를 중단합니다. 원본 파일 삭제나 기한 변경은 수행하지 않습니다.</div>
<p id="expired" role="alert"></p>
<div class="controls"><button id="prev" type="button">← 이전</button><button id="next" type="button">다음 →</button><span id="position" aria-live="polite"></span>
<label>상태 <select id="status"><option value="all">전체</option><option value="held">보류 held</option><option value="draft">초안 draft</option></select></label>
<label>영역 <select id="kind"><option value="all">전체</option><option value="potholes">포트홀</option><option value="hard_negatives">유사 음성 후보</option><option value="uncertain_regions">불확실 영역</option><option value="privacy_boxes">얼굴·번호판</option><option value="lane_polygons">차선 다각형</option><option value="lane_masks">차선 마스크 있음</option></select></label></div>
<div class="legend"><span class="key" style="--c:#ff6565">포트홀</span><span class="key" style="--c:#78dfa4">유사 음성 후보</span><span class="key" style="--c:#ffd05f">불확실</span><span class="key" style="--c:#d6a0ff">개인정보</span><span class="key" style="--c:#6ccfff">차선 다각형</span></div>
<div class="stage"><img id="frame" alt="선택한 대표 키프레임" data-src=""><svg id="overlay" xmlns="http://www.w3.org/2000/svg" aria-label="원해상도 라벨 영역"></svg></div>
<h2 id="heading"></h2><div id="details"></div>
<footer>좌우 화살표로 필터 내 키프레임을 이동합니다. 좌표는 원본 이미지 기준이며 화면 크기에 맞춰 함께 축소됩니다.
차선 RLE는 존재 여부와 원해상도 크기를 표시하고, 이 화면은 bbox·polygon을 겹쳐 보여줍니다. uncertain은 확정 라벨이 아닙니다. 검수 의견은 원본 초안을 보존한 별도 검수 기록으로 남겨 주세요.</footer>
<script id="bundle-data" type="application/json">__DATA__</script>
<script>
'use strict';
const data=JSON.parse(document.getElementById('bundle-data').textContent);
const $=id=>document.getElementById(id), image=$('frame'), svg=$('overlay');
const colors={potholes:'#ff6565',hard_negatives:'#78dfa4',uncertain_regions:'#ffd05f',privacy_boxes:'#d6a0ff',lane_polygons:'#6ccfff'};
let selected=[],cursor=0,locked=false;
$('count').textContent=String(data.samples.length);$('deadline').textContent=data.expires_at;
function alive(row){const end=Math.min(Date.parse(data.expires_at),row?Date.parse(row.expires_at):Infinity);return Number.isFinite(end)&&Date.now()<end;}
function unload(){image.removeAttribute('src');image.dataset.src='';svg.replaceChildren();}
function guard(){if(!alive(selected[cursor])){locked=true;unload();$('expired').textContent='승인된 처리·보관 기한이 지났거나 기한 정보가 유효하지 않습니다. 이미지를 로드하지 않습니다.';return false;}return true;}
function shape(tag,attrs){const el=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [k,v] of Object.entries(attrs))el.setAttribute(k,String(v));svg.appendChild(el);return el;}
function render(){unload();$('prev').disabled=cursor<=0;$('next').disabled=cursor>=selected.length-1;
if(!selected.length){$('position').textContent='필터 결과 없음';$('heading').textContent='표시할 프레임이 없습니다.';$('details').textContent='';return;}
const row=selected[cursor];$('position').textContent=`${cursor+1} / ${selected.length}`;
$('heading').textContent=`대표 프레임 ${row.ordinal} · ${row.annotation_status==='held'?'보류 held':'초안 draft'}`;$('heading').className=row.annotation_status==='held'?'held':'';
const notes=[`원해상도 ${row.image.width} × ${row.image.height}`, '사람 검수: 미완료 · 학습 내보내기: 불가 · 독립 평가 정답: 불가'];
svg.setAttribute('viewBox',`0 0 ${row.image.width} ${row.image.height}`);
for(const kind of Object.keys(colors)){for(const item of row[kind]||[]){if(kind==='lane_polygons'){shape('polygon',{points:item.points.map(p=>p.join(',')).join(' '),fill:'none',stroke:colors[kind],'stroke-width':2});}
else{const [x,y,w,h]=item.bbox_xywh_px;shape('rect',{x,y,width:w,height:h,fill:'none',stroke:colors[kind],'stroke-width':2,'stroke-dasharray':item.certainty==='uncertain'||kind==='uncertain_regions'?'8 5':'none'});}
const label=item.class||item.type||item.target||item.kind||'pothole';notes.push(`${label}${item.certainty?' · '+item.certainty:''}: ${item.reason||'AI 시각 판단 초안'}`);}}
if(row.has_lane_masks)notes.push('원해상도 차선 RLE 마스크 포함 (outline/gap/paint/valid).');
if(row.note)notes.push(`메모: ${row.note}`);$('details').textContent=notes.join('\n');
if(locked||!guard())return;image.dataset.src=row.image.relative_path;if(!guard())return;image.src=image.dataset.src;
}
function filter(){selected=data.samples.filter(row=>($('status').value==='all'||row.annotation_status===$('status').value)&&($('kind').value==='all'||($('kind').value==='lane_masks'?row.has_lane_masks:(row[$('kind').value]||[]).length>0)));cursor=0;render();}
$('prev').onclick=()=>{if(cursor>0){cursor--;render();}};$('next').onclick=()=>{if(cursor<selected.length-1){cursor++;render();}};
$('status').onchange=filter;$('kind').onchange=filter;
document.addEventListener('keydown',e=>{if(['SELECT','INPUT','TEXTAREA'].includes(document.activeElement.tagName))return;if(e.key==='ArrowLeft')$('prev').click();if(e.key==='ArrowRight')$('next').click();});
window.addEventListener('pagehide',unload);setInterval(guard,1000);filter();
</script></body></html>'''


def generate(bundle_path, output_path, *, now=None):
    now = now or datetime.now(timezone.utc)
    source = scoped_file(str(bundle_path))
    base = OUTPUT_ROOT.resolve()
    if not source.is_relative_to(base):
        raise ValueError('bundle_outside_session')
    if source.stat().st_size > MAX_BYTES:
        raise ValueError('bundle_too_large')
    with source.open('rb') as stream:
        raw = stream.read(MAX_BYTES+1)
    if len(raw) > MAX_BYTES:
        raise ValueError('bundle_too_large')
    bundle = json.loads(raw)
    if not validate_bundle(bundle, now=now)['valid']:
        raise ValueError('invalid_annotation_bundle')
    # Both source labels and the new private HTML must stay in the exact session.
    value = os.path.abspath(output_path)
    if os.path.commonpath([os.path.normcase(str(base)),os.path.normcase(value)]) != os.path.normcase(str(base)):
        raise ValueError('output_outside_session')
    output = Path(value)
    if output.suffix.lower() != '.html' or not output.parent.is_dir():
        raise ValueError('invalid_html_output')
    for parent in (output, *output.parents):
        if parent == base.parent:
            break
        if parent.is_symlink() or (hasattr(parent,'is_junction') and parent.is_junction()):
            raise ValueError('linked_output_forbidden')
    if not output.resolve().is_relative_to(base):
        raise ValueError('output_outside_session')
    samples=[]
    for i,row in enumerate(bundle['samples'],1):
        path=scoped_file(row['image']['path'])
        if not path.is_relative_to(base) or path.suffix.lower() not in ('.png','.jpg','.jpeg','.webp'):
            raise ValueError('frame_outside_session_or_unsupported_type')
        # Only local relative sidecars; no absolute path, frame hash, or sample ID.
        relative=quote(os.path.relpath(path,output.parent).replace('\\','/'),safe='/')
        samples.append({'ordinal':i,'annotation_status':row['annotation_status'],'expires_at':row['expires_at'],
                        'image':{'relative_path':relative,'width':row['image']['width'],'height':row['image']['height']},
                        'has_lane_masks':bool(row.get('lane_masks')),'note':row.get('note',''),
                        **{key:row.get(key,[]) for key in ('potholes','hard_negatives','uncertain_regions','privacy_boxes','lane_polygons')}})
    payload=json.dumps({'expires_at':bundle['expires_at'],'samples':samples},ensure_ascii=False,allow_nan=False)
    payload=payload.replace('&','\\u0026').replace('<','\\u003c').replace('>','\\u003e').replace('\u2028','\\u2028').replace('\u2029','\\u2029')
    html=TEMPLATE.replace('__DATA__',payload)
    # Exclusive creation prevents replacing an earlier review or attacker-created path.
    with output.open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(html)
    return {'status':'created','samples':len(samples),'human_reviewed':False,'training_export_enabled':False,
            'evaluation_ground_truth_eligible':False,'image_bytes_read':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    try:
        result=generate(args.bundle,args.output)
    except (ValueError,OSError,TypeError,KeyError):
        print(json.dumps({'status':'rejected','code':'invalid_or_existing_private_review_output'}))
        return 1
    print(json.dumps(result))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
