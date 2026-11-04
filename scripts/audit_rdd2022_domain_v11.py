"""Post-MD5/CRC annotation and historical-domain metadata audit, no image decode.

Only XML payloads are opened after the complete archive receipt passes. No split,
training, inference, or permanent blacklist mutation is performed.
"""
from __future__ import annotations
import argparse,hashlib,json,re,sys,time,zipfile
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path,PurePosixPath
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import sha256,workspace_path
from scripts import acquire_rdd2022_domain_v11 as acquire
from scripts import probe_rdd2022_domain_v11 as prior

DEST=prior.DEST;PLAN=DEST/'annotation-audit-plan.json'
HISTORY=ROOT/'data/research/rdd2022-blind-v7/history-exclusions.json'
US100=ROOT/'data/training/research-pothole-rdd2022-us100-v7/release.json'
COUNTRIES=('Japan','India','Czech','Norway','United_States','China_MotorBike','China_Drone')


def read(path):
    path=workspace_path(path,exists=True)
    if path.stat().st_size>40_000_000:raise ValueError('bounded_metadata_required')
    return json.loads(path.read_text(encoding='utf-8'))


def pointer(path):return {'path':path.relative_to(ROOT).as_posix(),'sha256':sha256(path)}


def verified_receipt():
    receipt=read(DEST/'full-transfer-results.json')
    if receipt.get('status')!='verified_archive_structure_CRC' or receipt.get('usable_source') is not True or receipt.get('all_members_CRC_verified') is not True or receipt.get('md5')!=prior.MD5 or receipt.get('bytes')!=prior.SIZE or receipt.get('source_expires_at')!=prior.EXPIRY:raise ValueError('whole_official_MD5_CRC_required_before_metadata_audit')
    if datetime.fromisoformat(prior.EXPIRY)<=datetime.now(timezone.utc):raise ValueError('original_source_expired')
    archive=workspace_path(acquire.ARCHIVE,exists=True);before=archive.stat()
    if before.st_size!=prior.SIZE or sha256(archive)!=receipt['sha256']:raise ValueError('verified_archive_changed')
    after=archive.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('archive_changed_while_hashing')
    return receipt


def prepare():
    if PLAN.exists():raise ValueError('new_annotation_plan_required')
    receipt=verified_receipt();history=read(HISTORY)
    if history.get('all_splits_excluded') is not True or history.get('never_reenter_training') is not True:raise ValueError('permanent_historical_exclusions_required')
    refs=[]
    for item in history['releases']:
        path=workspace_path(item['path'],exists=True)
        if sha256(path)!=item['sha256']:raise ValueError('original_historical_release_changed')
        refs.append({'path':item['path'],'sha256':item['sha256']})
    plan={'format':'rdd2022-domain-v11-annotation-metadata-plan-v1','created_at':datetime.now(timezone.utc).isoformat(),
        'audit_source_sha256':sha256(Path(__file__)),'transfer_receipt':pointer(DEST/'full-transfer-results.json'),
        'archive_sha256':receipt['sha256'],'history_snapshot':pointer(HISTORY),'historical_releases':refs,
        'US100_never_train':pointer(US100),'source_expires_at':prior.EXPIRY,'source_expiry_extended':False,
        'countries':list(COUNTRIES),'XML_maximum_bytes':2*1024**2,'XML_DOCTYPE_or_ENTITY_allowed':False,
        'image_decode':False,'image_model_inference':False,'new_TRAIN_split':False,'training_updates':0,
        'all_historical_TEST_permanently_excluded_from_future_training':True}
    prior.new_json(PLAN,plan);return {'status':'annotation_metadata_plan_fixed','plan_sha256':sha256(PLAN)}


def annotation(raw):
    if len(raw)>2*1024**2 or b'<!DOCTYPE' in raw.upper() or b'<!ENTITY' in raw.upper():raise ValueError('bounded_plain_XML_required')
    root=ET.fromstring(raw)
    try:width=int(root.findtext('size/width','0'));height=int(root.findtext('size/height','0'))
    except ValueError:width=height=0
    result={'width':width,'height':height,'D40_boxes':0,'invalid_D40_boxes':0,'invalid_boxes':0,'all_boxes':0,
        'invalid_dimensions':not(0<width<=32768 and 0<height<=32768),'declared_filename':root.findtext('filename','')}
    for obj in root.findall('object'):
        label=(obj.findtext('name') or '').strip();result['all_boxes']+=1
        if label=='D40':result['D40_boxes']+=1
        try:
            coords=[float(obj.findtext('bndbox/'+key,'')) for key in ('xmin','ymin','xmax','ymax')]
            x0,y0,x1,y1=coords
            # Publisher VOC may use zero or one based lower coordinates; this
            # validity audit reports only finite ordered bounds, no conversion.
            valid=not result['invalid_dimensions'] and 0<=x0<x1<=width and 0<=y0<y1<=height
        except (ValueError,TypeError):valid=False
        if not valid:
            result['invalid_boxes']+=1
            if label=='D40':result['invalid_D40_boxes']+=1
    return result


def split_of(name):
    parts=PurePosixPath(name).parts
    matches=[label for label in ('train','test') if label in parts]
    return matches[0] if len(matches)==1 else 'unknown'


def country_counts(archive):
    infos=archive.infolist();images=Counter();extensions=Counter();xmls=[];XML_keys=Counter()
    for item in infos:
        if item.is_dir():continue
        path=PurePosixPath(item.filename);split=split_of(item.filename)
        if path.suffix.lower() in ('.jpg','.jpeg','.png'):
            images[(split,path.stem)]+=1;extensions[(split,path.suffix.lower())]+=1
        elif path.suffix.lower()=='.xml':xmls.append(item)
    totals=Counter(images=sum(images.values()),JPG=sum(n for (_,ext),n in extensions.items() if ext in ('.jpg','.jpeg')),
        PNG=sum(n for (_,ext),n in extensions.items() if ext=='.png'),XML=len(xmls),duplicate_image_keys=sum(n-1 for n in images.values() if n>1))
    matched=set();by_split=defaultdict(Counter);positive=defaultdict(set);negative=defaultdict(set)
    for item in xmls:
        split=split_of(item.filename);key=(split,PurePosixPath(item.filename).stem);counts=by_split[split];counts['XML']+=1
        XML_keys[key]+=1
        if key in images:counts['image_annotation_matches']+=1;matched.add(key)
        else:counts['annotation_without_image']+=1
        try:
            if item.file_size>2*1024**2:raise ValueError('XML_size_exceeded')
            with archive.open(item) as stream:raw=stream.read(2*1024**2+1)
            parsed=annotation(raw)
            declared=PurePosixPath(parsed['declared_filename']).stem
            if not declared:counts['missing_declared_filename']+=1
            elif declared!=key[1]:counts['declared_filename_differs_from_XML_stem']+=1
            if declared and (split,declared) not in images:counts['declared_filename_without_image']+=1
            counts['D40_positive_annotation_images' if parsed['D40_boxes'] else 'noD40_annotation_images']+=1
            if key in images:(positive if parsed['D40_boxes'] else negative)[split].add(key)
            for field in ('D40_boxes','invalid_D40_boxes','invalid_boxes','all_boxes'):counts[field]+=parsed[field]
            counts['invalid_dimensions']+=parsed['invalid_dimensions']
        except (ValueError,ET.ParseError):counts['invalid_XML']+=1
    for (split,stem),count in images.items():
        by_split[split]['images']+=count
        if (split,stem) not in matched:by_split[split]['image_without_annotation']+=count
    for (split,ext),count in extensions.items():by_split[split]['JPG' if ext in ('.jpg','.jpeg') else 'PNG']+=count
    for split,counts in by_split.items():
        counts.update(D40_positive_matched_images=len(positive[split]),noD40_matched_images=len(negative[split]-positive[split]),
            contradictory_D40_annotation_keys=len(positive[split]&negative[split]),unique_images_with_annotation=sum(key[0]==split for key in matched),
            duplicate_XML_keys=sum(n-1 for key,n in XML_keys.items() if key[0]==split and n>1))
    return {'totals':dict(totals),'publisher_splits':{split:dict(counts) for split,counts in sorted(by_split.items())}}


def country_label(row):
    explicit=row.get('country')
    if isinstance(explicit,str) and explicit:return explicit,'country_field'
    found=set()
    for field in ('source_member','original_member'):
        text=row.get(field)
        if not isinstance(text,str):continue
        for country in COUNTRIES:
            if country in PurePosixPath(text).parts or re.search(r'(?:^|[/\\])'+re.escape(country)+r'_[0-9]+\.(?:jpg|jpeg|png)$',text,re.I):found.add(country)
    return (next(iter(found)),'publisher_member_label') if len(found)==1 else ('unknown','missing_or_ambiguous_metadata')


def historical_counts(rows):
    result=defaultdict(Counter);unique=defaultdict(set);split_unique=defaultdict(set);bases=Counter();invalid_sha=0
    for row in rows:
        country,basis=country_label(row);split=row.get('split','unknown');bases[basis]+=1
        key=(country,split);result[key]['rows']+=1
        if re.fullmatch('[0-9a-f]{64}',str(row.get('sha256',''))):unique[key].add(row['sha256']);split_unique[split].add(row['sha256'])
        else:invalid_sha+=1
        result[key]['positive_pothole_rows' if any(obj.get('class')=='pothole' for obj in row.get('objects',[])) else 'no_pothole_rows']+=1
    countries=defaultdict(dict)
    for (country,split),counts in sorted(result.items()):countries[country][split]=dict(counts)|{'unique_sha256_images':len(unique[(country,split)])}
    return {'countries':dict(countries),'country_evidence_basis':dict(bases),'invalid_or_missing_image_SHA':invalid_sha,
        'historical_any_TEST_unique_SHA':len(split_unique['test']),
        'cross_historical_release_split_overlap_unique_SHA':{
            'train_test':len(split_unique['train']&split_unique['test']),
            'validation_test':len(split_unique['validation']&split_unique['test']),
            'train_validation':len(split_unique['train']&split_unique['validation'])},
        'country_is_publisher_metadata_not_verified_physical_location':True}


def run():
    plan=read(PLAN)
    if plan['audit_source_sha256']!=sha256(Path(__file__)) or plan['transfer_receipt']!=pointer(DEST/'full-transfer-results.json') or plan['history_snapshot']!=pointer(HISTORY) or plan['US100_never_train']!=pointer(US100):raise ValueError('frozen_metadata_audit_inputs_changed')
    receipt=verified_receipt()
    if receipt['sha256']!=plan['archive_sha256']:raise ValueError('frozen_verified_archive_changed')
    started=DEST/'annotation-audit-started.json'
    if started.exists():raise ValueError('annotation_audit_already_started')
    prior.new_json(started,{'plan_sha256':sha256(PLAN),'started_at':datetime.now(timezone.utc).isoformat()})
    began=time.monotonic();countries={}
    with zipfile.ZipFile(acquire.ARCHIVE) as outer:
        for item in outer.infolist():
            if item.is_dir() or PurePosixPath(item.filename).suffix.lower()!='.zip':continue
            country=PurePosixPath(item.filename).stem
            if country not in COUNTRIES or country in countries:raise ValueError('exact_seven_publisher_country_archives_required')
            with outer.open(item) as stream,zipfile.ZipFile(stream) as nested:countries[country]=country_counts(nested)
    if set(countries)!=set(COUNTRIES):raise ValueError('all_seven_country_archives_required')
    historical=[];combined=[]
    for ref in plan['historical_releases']:
        if sha256(ROOT/ref['path'])!=ref['sha256']:raise ValueError('historical_release_changed_during_audit')
        rows=read(ROOT/ref['path']);combined.extend(rows)
        historical.append({'release_sha256':ref['sha256'],'rows':len(rows),**historical_counts(rows)})
    old_test=read(US100)
    if len(old_test)!=100 or any(row.get('split')!='test' for row in old_test):raise ValueError('unchanged_US100_permanent_TEST_required')
    for ref in [plan['history_snapshot'],plan['US100_never_train'],*plan['historical_releases']]:
        if sha256(ROOT/ref['path'])!=ref['sha256']:raise ValueError('metadata_changed_during_audit')
    result={'format':'rdd2022-domain-v11-annotation-metadata-results-v1','plan_sha256':sha256(PLAN),'archive_sha256':receipt['sha256'],
        'official_whole_MD5_verified':True,'whole_archive_CRC_verified':True,'source_expires_at':prior.EXPIRY,'source_expiry_extended':False,
        'countries':countries,'publisher_country_archive_groups':7,'distinct_publisher_countries':6,
        'China_MotorBike_and_Drone_are_separate_archive_groups_not_separate_countries':True,
        'historical_release_count':len(historical),'historical_per_release':historical,
        'historical_combined_with_duplicate_rows':historical_counts(combined),'US100_never_train':historical_counts(old_test),
        'US100_rows_permanently_excluded':100,'all_historical_TEST_permanently_excluded':True,
        'image_decodes':0,'model_inferences':0,'new_splits':0,'training_updates':0,'GPU_used':False,
        'wall_seconds':time.monotonic()-began,'limits':['Publisher country label is not verified physical road identity',
            'Annotation bounds are machine checks; publisher omissions and real pothole completeness are unverified',
            'Historical releases include interrupted/quarantined experiments; aggregate overlaps do not establish current single-release leakage',
            'No future training membership or independent holdout has been created']}
    prior.new_json(DEST/'annotation-audit-results.json',result)
    return {'status':'metadata_audited','countries':len(countries),'historical_releases':len(historical),'image_decodes':0,'model_inferences':0}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','audit']);args=parser.parse_args()
    try:print(json.dumps(prepare() if args.action=='prepare' else run()))
    except Exception as error:print(json.dumps({'status':'metadata_audit_stopped','error_type':type(error).__name__}));return 1
    return 0


if __name__=='__main__':raise SystemExit(main())
