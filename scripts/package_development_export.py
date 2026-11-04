"""Add an explicit dev-only export contract without editing executed evidence."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import workspace_path,sha256,atomic_json

def package(folder):
    folder=workspace_path(folder,exists=True)
    if not folder.is_relative_to(ROOT/'data/training'):raise ValueError('training_experiment_required')
    parent=folder/'model.json';freeze=folder/'selection-freeze.json';out=folder/'model-development.json'
    if out.exists():raise ValueError('preserve_existing_development_package')
    model=json.loads(parent.read_text());selection=json.loads(freeze.read_text());release=workspace_path(model['source_release_path'],exists=True);rows=json.loads(release.read_text())
    if (selection.get('manifest_sha256')!=sha256(parent) or selection.get('TEST_inference_performed') is not False or not rows or set(r['split'] for r in rows)!={'train','validation'}):raise ValueError('completed_preTEST_dev_export_required')
    if sha256(workspace_path(model['weights'],exists=True))!=model['sha256'] or sha256(folder/'best.pt')!=selection['checkpoint_sha256']:raise ValueError('selected_artifacts_changed')
    derived=model|{'development_only_release':True,'executed_export_manifest_sha256':sha256(parent),'selection_freeze_sha256':sha256(freeze),'packaging_source_sha256':sha256(Path(__file__))}
    if derived.get('exported_before_final_test') is not True:raise ValueError('preTEST_export_declaration_required')
    # Validation before writing uses the same contract as the final manifest.
    from scripts.verify_trained_models import validate_metadata
    atomic_json(out,derived);validate_metadata(out)
    result={'format':'development-export-package-v1','parent_manifest_sha256':sha256(parent),'manifest_sha256':sha256(out),'source_sha256':sha256(Path(__file__)),'weights_changed':False,'training_source_changed':False,'TEST_inference_performed':False,'split_counts':{s:sum(r['split']==s for r in rows) for s in ('train','validation','test')}}
    atomic_json(folder/'development-packaging.json',result);print(json.dumps(result))
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('folder');package(p.parse_args().folder)
